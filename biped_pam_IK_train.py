import os
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch
import glob
import re
from datetime import datetime
import json
#import pickle

# Import your environments
from bided_pam_IK import PAMIKBipedEnv  # Assuming this is your PAM environment
from Archivos_Apoyo.Configuraciones_adicionales import cargar_posible_normalizacion, phase_trainig_preparations

class UnifiedBipedTrainer:
    """
        Entrenamiento de marcha de robot bípedo para entornos PAM + IK.
    """
    
    def __init__(self, 
                 env_type='pam',  # 'standard' or 'pam'
                 total_timesteps=5000000,
                 n_envs=8,
                 learning_rate=5e-4,
                 use_wandb=False,
                 resume_from=None,
                 action_space="hybrid" # "pam o hybrid"
                 ):
        # Atributos de entrada
        self.env_type = env_type
        self.action_space=action_space
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.resume_from = resume_from

        # Genero la configuración de entrenamiento del modelo
        self.configuracion_modelo_entrenamiento
        
       
        
    def create_training_env(self):
        """
            Creación de entorno de entrenamiento vectorizado y 
            normalizado para darle robustez (al entrenamiento).
        """
        
        config = self.env_configs[self.env_type]
        env_class = config['env_class']
        # Intenta crear el entorno con el modo de renderizado adecuado
        def make_env():
            def _init():
                env = env_class(render_mode='human' if self.n_envs == 1 else 'direct', action_space=self.action_space)
                env = Monitor(env, self.logs_dir)
                return env
            return _init
        # Generación de entornos paralelos en caso de n_env>1
        env = self.create_parallel_env(make_env=make_env, config=config)

        return env
    
    def create_eval_env(self):
        """
            Se genera un entorno de evaluación 
            con la misma estructura del entrenamiento
            Sin palalelización
        """
        
        config = self.env_configs[self.env_type]
        env_class = config['env_class']
        
        def make_eval_env():
            def _init():
                env = env_class(render_mode='direct', action_space=self.action_space)
                env = Monitor(env, os.path.join(self.logs_dir, "eval"))
                return env
            return _init
        
        eval_env = DummyVecEnv([make_eval_env()])
        eval_env = VecNormalize(eval_env, 
                               norm_obs=True, 
                               norm_reward=True, 
                               clip_obs=config['clip_obs'], 
                               training=False)

        return eval_env
    
    def cargar_creacion_modelo(self, resume_path, env):
        """Load existing model if available"""
        if resume_path and os.path.exists(resume_path):
            print(f"🔄 Loading model from checkpoint: {resume_path}")
            try:
                model = RecurrentPPO.load(resume_path, env=env)
                
                # Update learning rate if different
                if hasattr(model, 'learning_rate') and model.learning_rate != self.learning_rate:
                    print(f"📝 Updating learning rate: {model.learning_rate} -> {self.learning_rate}")
                    model.learning_rate = self.learning_rate
                    
                return model
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("🔄 Creating new model instead...")
                return None
        return None
    
    def create_model(self, env, resume_path=None):
        """
            Creación de modelo RecurrentPPO. 
            Debería de estar optimizado para el entorno seleccionado.
        """

        # Carga un podelo creado anteriormente
        model = self.cargar_creacion_modelo(resume_path=resume_path, env=env)
        if model is not None:
            return model
        
        print("🧠 Creating new RecurrentPPO model...")
        
        # Arquitectura de redes neuronales LSTM para el RL
        policy_kwargs_lstm={
                            'lstm_hidden_size': 128,
                            'n_lstm_layers': 1,
                            'shared_lstm': False
                            }
        
        # Hiperparámetros de entorno y modelo
        # PAM-specific optimizations
        model_params = {
            'learning_rate': self.learning_rate,
            'gamma': 0.99,
            'max_grad_norm': 0.5,
            'ent_coef': 0.01,
        }
        
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_steps=512,
            batch_size=512,
            n_epochs=5,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs_lstm,
            verbose=1,
            tensorboard_log=self.logs_dir,
            device='auto',
            **model_params
        )
        
        return model
    
    def prep_modelo_anterior(self, resume):
        """Carga el modelo anterior teniendo en cuenta los timesteps restantes"""
        # Determine starting point
        resume_path = None
        resume_timesteps = 0

        if resume and self.resume_from:
            # Resume from specific checkpoint
            if os.path.exists(self.resume_from):
                resume_path = self.resume_from
                match = re.search(r'_(\d+)_steps\.zip$', self.resume_from)
                if match:
                    resume_timesteps = int(match.group(1))
                print(f"📂 Resuming from specified checkpoint: {self.resume_from}")
            else:
                print(f"⚠️ Specified checkpoint not found: {self.resume_from}")
        elif resume:
            # Search for latest checkpoint automatically
            latest_checkpoint, resume_timesteps = self.find_latest_checkpoint()
            if latest_checkpoint:
                resume_path = latest_checkpoint
                print(f"📂 Found latest checkpoint: {latest_checkpoint}")
                print(f"📊 Resuming from timestep: {resume_timesteps:,}")
            else:
                print("📝 No previous checkpoints found, starting fresh training")
        
        # Calculate remaining timesteps
        remaining_timesteps = max(0, self.total_timesteps - resume_timesteps)

        print(f"🎯 Training plan:")
        print(f"   - Completed: {resume_timesteps:,} timesteps")
        print(f"   - Remaining: {remaining_timesteps:,} timesteps")
        print(f"   - Total target: {self.total_timesteps:,} timesteps")

        if remaining_timesteps == 0:
            print(f"✅ Training already completed! ({resume_timesteps:,}/{self.total_timesteps:,} timesteps)")
            return None, None, None
        
        return resume_path, resume_timesteps, remaining_timesteps
    
    def train(self, resume=True):
        """Ejecución del entrenamiento RecurrentPPO."""
        
        config = self.env_configs[self.env_type]
        
        print(f"🤖 Starting {self.env_type.upper()} biped training with RecurrentPPO (LSTM)...")
        
        resume_path, resume_timesteps, remaining_timesteps = self.prep_modelo_anterior(resume)
        if resume_path is None and remaining_timesteps is None:
            return

        if self.env_type == 'pam':
            print("📊 PAM Training Features:")
            print("   - Hybrid IK + PAM control")
            print("   - Realistic tensile forces")
            print("   - LSTM memory for muscle coordination")
            print("   - Energy efficiency optimization")
        
        # Create environments
        print(f"📦 Creating {self.n_envs} training environments...")
        # train_env=self.create_training_env_with_walking_cycle()
        train_env = self.create_training_env()

        def setup_walking_cycle(env_wrapper):
            """Configura el ciclo de paso en entornos vectorizados"""
            if hasattr(env_wrapper, 'envs'):
                # VecEnv - configurar cada entorno
                for env in env_wrapper.envs:
                    if hasattr(env, 'env'):  # Wrapper
                        base_env = env.env
                    else:
                        base_env = env
                    
                    if hasattr(base_env, 'set_walking_cycle'):
                        base_env.set_walking_cycle(True)
                        print(f"   ✅ Walking cycle enabled for training env")
            else:
                # Entorno simple
                if hasattr(env_wrapper, 'set_walking_cycle'):
                    env_wrapper.set_walking_cycle(True)

        setup_walking_cycle(train_env)
        
        # Validate VecNormalize dimensions
        if isinstance(train_env, VecNormalize):
            obs_dim = train_env.observation_space.shape[0]
            rms_dim = train_env.obs_rms.mean.shape[0]
            if obs_dim != rms_dim:
                print(f"⚠️ Dimension mismatch: env={obs_dim}, rms={rms_dim}. Resetting normalization.")
                train_env.reset()

        # Cargaa las normalizaciones existentes si las hay
        train_env = cargar_posible_normalizacion(self.model_dir, resume_path, config, train_env)
        
        eval_env = self.create_eval_env()

        # También configurar en entorno de evaluación
        setup_walking_cycle(eval_env)
        
        # Create model
        print("🧠 Creating RecurrentPPO model with LSTM architecture...")
        model = self.create_model(train_env, resume_path)
        
        # Setup callbacks
        callbacks = self.setup_callbacks(eval_env)

        phase1_timesteps, phase2_timesteps, phase3_timesteps=self.generar_timesteps(remaining_timesteps, config, model)

        # Record training start
        self.training_info['training_start_time'] = datetime.now().isoformat()
        self.training_info['completed_timesteps'] = resume_timesteps
        
        try:
            # ==============================================
            #  ENTRENAMIENTO POR FASES
            # ==============================================
            current_timesteps = 0

            # FASE 1: Entrenamiento con ciclo base (solo ajustes finos)
            if phase1_timesteps > 0:
                print(f"🚀 Starting training for {remaining_timesteps:,} steps...")
                model, current_timesteps, phase1_timesteps=phase_trainig_preparations(self.model_dir, remaining_timesteps, 
                                                                                      train_env, eval_env, current_timesteps,
                                                                                      model, callbacks, phase1_timesteps, config, 1)

            # FASE 2: Entrenamiento con modulación del ciclo
            if phase2_timesteps > 0:
                print(f"🚀 Phase 2: Cycle modulation training ({phase2_timesteps:,} steps)...")
                model, current_timesteps, phase2_timesteps=phase_trainig_preparations(self.model_dir, remaining_timesteps, train_env, eval_env,
                                                                                      current_timesteps, model, callbacks, phase2_timesteps, config, 2)

            # FASE 3: Entrenamiento con control libre
            if phase3_timesteps > 0:
                print(f"🚀 Phase 3: Full RL control training ({phase3_timesteps:,} steps)...")
                
                # Desactivar ciclo de paso
                def disable_walking_cycle(env_wrapper):
                    if hasattr(env_wrapper, 'envs'):
                        for env in env_wrapper.envs:
                            base_env = env.env if hasattr(env, 'env') else env
                            if hasattr(base_env, 'set_walking_cycle'):
                                base_env.set_walking_cycle(False)
                    else:
                        if hasattr(env_wrapper, 'set_walking_cycle'):
                            env_wrapper.set_walking_cycle(False)
                
                disable_walking_cycle(train_env)
                disable_walking_cycle(eval_env)
                
                model.learn(
                    total_timesteps=phase3_timesteps,
                    callback=callbacks,
                    tb_log_name=f"{config['model_prefix']}_phase3",
                    reset_num_timesteps=False
                )
                current_timesteps += phase3_timesteps

            # Update training info
            self.training_info['completed_timesteps'] = self.total_timesteps
            
            # Save final model
            final_model_path = os.path.join(self.model_dir, f"{config['model_prefix']}_final")
            model.save(final_model_path)
            print(f"✅ Final model saved at: {final_model_path}")
            
            # Save normalization if used
            if isinstance(train_env, VecNormalize):
                norm_path = os.path.join(self.model_dir, f"{config['model_prefix']}_normalize.pkl")
                train_env.save(norm_path)
                print(f"✅ Normalization saved at: {norm_path}")
            
            # Save training info
            self.save_training_info()
                
        except KeyboardInterrupt:
            print("⏸️ Training interrupted by user")
            
        finally:
            train_env.close()
            eval_env.close()
                
        print("🎉 Training completed!")
    
    def test_model(self, model_path, episodes=10, normalization_path=None):
        """Hacer test de modelo RecurrentPPO ."""
        
        config = self.env_configs[self.env_type]
        env_class = config['env_class']
        
        print(f"Testing {self.env_type.upper()} RecurrentPPO model: {model_path}")
        
        # Load model
        model = RecurrentPPO.load(model_path)
        
        # Create test environment
        env = env_class(render_mode='human', action_space=self.action_space)
        
        # Handle normalization if used
        if normalization_path and os.path.exists(normalization_path):
            print("Loading normalization statistics...")
            test_env = DummyVecEnv([lambda: Monitor(env, os.path.join(self.logs_dir, "test"))])
            test_env = VecNormalize.load(normalization_path, test_env)
            test_env.training = False
            test_env.norm_reward = False
            
            # Test with normalization
            for episode in range(episodes):
                obs = test_env.reset()
                episode_reward = 0
                steps = 0
                
                lstm_states = None
                episode_start = np.ones((test_env.num_envs,), dtype=bool)
                
                while True:
                    action, lstm_states = model.predict(
                        obs, 
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True
                    )
                    
                    obs, reward, done, info = test_env.step(action)
                    episode_reward += reward[0]
                    steps += 1
                    episode_start = np.array([False])
                    
                    if done[0]:
                        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
                        break
            
            test_env.close()
        else:
            # Test without normalization
            for episode in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                steps = 0
                
                lstm_states = None
                episode_start = np.ones((1,), dtype=bool)
                
                while True:
                    # Wrap obs for LSTM prediction
                    obs_wrapped = obs.reshape(1, -1)
                    action, lstm_states = model.predict(
                        obs_wrapped, 
                        state=lstm_states,
                        episode_start=episode_start,
                        deterministic=True
                    )
                    
                    obs, reward, terminated, truncated, info = env.step(action[0])
                    episode_reward += reward
                    steps += 1
                    episode_start = np.array([False])
                    
                    done = terminated or truncated
                    if done:
                        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
                        break
                    
        env.close()

    def find_latest_checkpoint(self, model_prefix=None):
        """
            Find the most recent checkpoint based on timesteps.
        """
        if model_prefix is None:
            model_prefix = self.env_configs[self.env_type]['model_prefix']
        
        # Search for checkpoint files
        checkpoint_pattern = os.path.join(self.checkpoints_dir, f"{model_prefix}_checkpoint_*_steps.zip")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None, 0
        
        # Extract timestep numbers from filenames
        timesteps = []
        for file in checkpoint_files:
            match = re.search(r'_(\d+)_steps\.zip$', file)
            if match:
                timesteps.append((int(match.group(1)), file))
        
        if not timesteps:
            return None, 0
        
        # Find checkpoint with most timesteps
        latest_timesteps, latest_file = max(timesteps, key=lambda x: x[0])
        
        return latest_file, latest_timesteps
    
    def save_training_info(self):
        """Save training state information."""
        config = self.env_configs[self.env_type]
        info_path = os.path.join(self.model_dir, f"{config['model_prefix']}_training_info.json")
        
        training_data = {
            'environment': self.env_type,
            'completed_timesteps': self.training_info['completed_timesteps'],
            'total_timesteps_target': self.total_timesteps,
            'last_checkpoint': self.training_info['last_checkpoint'],
            'training_start': self.training_info['training_start_time'],
            'total_training_time_hours': self.training_info['total_training_time'],
            'n_envs': self.n_envs,
            'learning_rate': self.learning_rate
        }
        
        with open(info_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"💾 Training info saved to: {info_path}")

    def load_training_info(self):
        """Load previous training state information."""
        config = self.env_configs[self.env_type]
        info_path = os.path.join(self.model_dir, f"{config['model_prefix']}_training_info.json")
        
        if not os.path.exists(info_path):
            return False
        
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
                
            self.training_info['completed_timesteps'] = data.get('completed_timesteps', 0)
            self.training_info['last_checkpoint'] = data.get('last_checkpoint', None)
            self.training_info['training_start_time'] = data.get('training_start', None)
            self.training_info['total_training_time'] = data.get('total_training_time_hours', 0)
            
            return True
        except Exception as e:
            print(f"⚠️ Error loading training info: {e}")
            return False
        
    def setup_walking_cycle_config(self):
        """Configura parámetros del ciclo de paso"""
        self.walking_cycle_config = {
            'phase1_ratio': 0.3,    # 30% del tiempo con ciclo base
            'phase2_ratio': 0.4,    # 40% con modulación
            'phase3_ratio': 0.3,    # 30% control libre
            'modulation_factor': 0.3,  # Intensidad de modulación en fase 2
            'enable_gradual_transition': True  # Transición gradual entre fases
        }

    def create_training_env_with_walking_cycle(self):
        # De momento no lo uso, pero lo dejo por si acaso
        """Versión modificada del create_training_env que incluye configuración del ciclo"""
        # Tu código actual de create_training_env
        env = self.create_training_env()
        
        # Añadir configuración del ciclo
        if hasattr(env, 'envs'):
            for single_env in env.envs:
                base_env = single_env.env if hasattr(single_env, 'env') else single_env
                if hasattr(base_env, 'setup_walking_cycle_config'):
                    base_env.setup_walking_cycle_config(self.walking_cycle_config)
        
        return env
    
    @property
    def configuracion_modelo_entrenamiento(self):
        # Creación de carpetas donde se guardan los modelos
        self.model_dir = "./models"
        self.logs_dir = "./logs"
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Configuraciones especificas para el modelo de entrenamiento
        self.env_configs = {
            'pam': {
                'env_class': PAMIKBipedEnv,
                'clip_obs': 20.0,
                'clip_reward': 15.0,
                'net_arch': dict(pi=[256, 256, 128], vf=[256, 256, 128]),
                'model_prefix': 'biped_pam'
            }
        }

        # Información de entrenamiento (en caso de no continuar con uno previo)
        self.training_info = {
            'completed_timesteps': 0,
            'last_checkpoint': None,
            'training_start_time': None,
            'total_training_time': 0
        }

    def create_parallel_env(self,make_env,config):
        # Creación de entornos paralelos para entrenamiento n_envs > 1
        if self.n_envs > 1:
            base_env = SubprocVecEnv([make_env() for _ in range(self.n_envs)])
            env = VecNormalize(base_env, 
                             norm_obs=True, 
                             norm_reward=True, 
                             clip_obs=config['clip_obs'],
                             clip_reward=config['clip_reward'])
        else:
            base_env = DummyVecEnv([make_env()])
            env = VecNormalize(base_env, 
                             norm_obs=True, 
                             norm_reward=True, 
                             clip_obs=config['clip_obs'],
                             clip_reward=config['clip_reward'])

        return env
    
    def setup_callbacks(self, eval_env):
        """Configures callbacks for monitoring and model saving."""
        
        config = self.env_configs[self.env_type]
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoints_dir,
            name_prefix=f'{config["model_prefix"]}_checkpoint'
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=os.path.join(self.logs_dir, "eval"),
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=5
        )
        
        return CallbackList([checkpoint_callback, eval_callback])
    
    def generar_timesteps(self, remaining_timesteps, config, model):
        # Dividir entrenamiento en fases
        phase1_timesteps = int(remaining_timesteps * 0.3)  # 30% con ciclo base
        phase2_timesteps = int(remaining_timesteps * 0.4)  # 40% con modulación
        phase3_timesteps = remaining_timesteps - phase1_timesteps - phase2_timesteps  # 30% control libre

        print(f"📊 Training Plan:")
        print(f"   - Phase 1: {phase1_timesteps:,} steps (Walking cycle base)")
        print(f"   - Phase 2: {phase2_timesteps:,} steps (Cycle + RL modulation)")
        print(f"   - Phase 3: {phase3_timesteps:,} steps (Full RL control)")
        
        # Show model information
        print(f"📊 Model Configuration:")
        print(f"   - Algorithm: RecurrentPPO")
        print(f"   - Policy: MlpLstmPolicy")
        print(f"   - Environment: {self.env_type.upper()}")
        print(f"   - Architecture: {config['net_arch']}")
        print(f"   - Learning Rate: {model.learning_rate}")
        print(f"   - Device: {model.device}")

        return phase1_timesteps, phase2_timesteps, phase3_timesteps
    


    
