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
from Archivos_Apoyo.Generate_Prints import log_training_plan, error_load_model, \
                                info_latest_checkpoint, print_info_env_pam
from bided_pam_IK import PAMIKBipedEnv  # Assuming this is your PAM environment
from Archivos_Apoyo.Configuraciones_adicionales import cargar_posible_normalizacion, phase_trainig_preparations
from Curriculum_generator.Curriculum_Manager import ExpertCurriculumManager 

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
                 action_space="hybrid", # "pam o hybrid"
                 enable_expert_curriculum=True
                 ):
        # Atributos de entrada
        self.env_type = env_type
        self.action_space=action_space
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.learning_rate = learning_rate
        self.use_wandb = use_wandb
        self.resume_from = resume_from
        self.enable_expert_curriculum = enable_expert_curriculum

        # Inicializar gestor de currículo experto
        if self.enable_expert_curriculum:
            self.curriculum_manager = ExpertCurriculumManager(total_timesteps)
            print("🎓 Expert curriculum enabled with {} phases".format(len(self.curriculum_manager.phases)))
        else:
            self.curriculum_manager = None

        # Genero la configuración de entrenamiento del modelo
        self.configuracion_modelo_entrenamiento

        #  Métricas del currículo experto
        self.expert_metrics = {
            'imitation_losses': [],
            'phase_rewards': [],
            'expert_action_similarity': [],
            'phase_success_rates': []
        }

    def _configure_environment_for_phase(self, env_wrapper, phase_info):
        """
        Configura el entorno para una fase específica del currículo.
        Aquí es donde establecemos los parámetros específicos de cada fase.
        """
        
        def configure_single_env(base_env, phase_info):
            """Configura un entorno individual"""
            if hasattr(base_env, 'set_training_phase'):
                # Configurar fase en el entorno
                base_env.set_training_phase(
                    phase=phase_info['phase_id'],
                    phase_timesteps=phase_info['timesteps']
                )
                
                # Configurar peso de imitación
                if hasattr(base_env, 'imitation_weight'):
                    base_env.imitation_weight = phase_info['expert_weight']
                
                # Configurar modo de control
                if hasattr(base_env, 'control_mode'):
                    base_env.control_mode = phase_info['control_mode']
                
                # Configurar controlador de paso según la fase
                if hasattr(base_env, 'walking_controller') and base_env.walking_controller:
                    if phase_info['phase_id'] <= 2:  # Fases iniciales usan trayectorias
                        base_env.walking_controller.mode = "trajectory"
                    elif phase_info['phase_id'] == 3:  # Fase híbrida
                        base_env.walking_controller.mode = "blend"
                        base_env.walking_controller.blend_factor = 0.5
                    else:  # Fases avanzadas usan presiones PAM
                        base_env.walking_controller.mode = "pressure"
                
                # Configurar dificultad ambiental
                self._set_environmental_difficulty(base_env, phase_info['difficulty_level'])
                
                print(f"   ✅ Environment configured for Phase {phase_info['phase_id']}: {phase_info['name']}")
        
        # Aplicar configuración a entornos vectorizados
        if hasattr(env_wrapper, 'envs'):
            for env in env_wrapper.envs:
                base_env = env.env if hasattr(env, 'env') else env
                configure_single_env(base_env, phase_info)
        else:
            configure_single_env(env_wrapper, phase_info)
    
    def _set_environmental_difficulty(self, env, difficulty_level):
        """
        Ajusta la dificultad del entorno según el nivel especificado.
        """
        if difficulty_level == 1:  # Básico
            # Condiciones ideales: superficie plana, sin perturbaciones
            if hasattr(env, 'sistema_recompensas'):
                env.sistema_recompensas.target_forward_velocity = 0.6  # Velocidad baja
                
        elif difficulty_level == 2:  # Intermedio
            # Pequeñas variaciones en superficie y velocidad objetivo
            if hasattr(env, 'sistema_recompensas'):
                env.sistema_recompensas.target_forward_velocity = 0.8
                
        elif difficulty_level == 3:  # Avanzado
            # Velocidad normal con más exigencia de estabilidad
            if hasattr(env, 'sistema_recompensas'):
                env.sistema_recompensas.target_forward_velocity = 1.0
                
        elif difficulty_level == 4:  # Experto
            # Alta velocidad y requisitos de eficiencia
            if hasattr(env, 'sistema_recompensas'):
                env.sistema_recompensas.target_forward_velocity = 1.2
                
        elif difficulty_level == 5:  # Maestría
            # Máxima dificultad con desafíos adicionales
            if hasattr(env, 'sistema_recompensas'):
                env.sistema_recompensas.target_forward_velocity = 1.4

    def _calculate_expert_action_similarity(self, env, rl_action, expert_action):
        """
        Calcula la similitud entre las acciones RL y las acciones expertas.
        Esto nos ayuda a monitorear qué tan bien está aprendiendo el agente.
        """
        if expert_action is None or len(expert_action) == 0:
            return 0.0
            
        # Normalizar acciones si tienen diferentes escalas
        rl_normalized = np.array(rl_action) / (np.linalg.norm(rl_action) + 1e-8)
        expert_normalized = np.array(expert_action) / (np.linalg.norm(expert_action) + 1e-8)
        
        # Calcular similitud coseno
        similarity = np.dot(rl_normalized, expert_normalized)
        return max(0.0, similarity)  # Clamp to [0, 1]
    
    def _compute_phase_specific_rewards(self, env, phase_info, base_reward):
        """
        Calcula recompensas específicas para cada fase del currículo.
        """
        phase_reward = base_reward
        
        if phase_info['phase_id'] <= 2:  # Fases de imitación
            # Bonificar similitud con acciones expertas
            if hasattr(env, 'walking_controller') and env.walking_controller:
                expert_action = env.walking_controller.get_expert_action_pressures()
                if expert_action is not None:
                    # Obtener última acción RL del entorno
                    rl_action = getattr(env, 'last_rl_action', None)
                    if rl_action is not None:
                        similarity = self._calculate_expert_action_similarity(
                            env, rl_action, expert_action
                        )
                        imitation_bonus = similarity * 2.0  # Bonificación por imitación
                        phase_reward += imitation_bonus
                        
                        # Registrar métrica
                        self.expert_metrics['expert_action_similarity'].append(similarity)
        
        elif phase_info['phase_id'] >= 4:  # Fases de RL avanzado
            # Bonificar eficiencia energética y suavidad de movimientos
            if hasattr(env, 'pam_states'):
                # Penalizar uso excesivo de presión
                pressure_penalty = np.mean(env.pam_states['pressures']) * 0.001
                phase_reward -= pressure_penalty
                
                # Bonificar movimientos suaves
                if hasattr(env, 'previous_action'):
                    action_smoothness = 1.0 / (1.0 + np.linalg.norm(
                        np.array(env.last_rl_action) - np.array(env.previous_action)
                    ))
                    phase_reward += action_smoothness * 0.5
        
        return phase_reward

    def seleccion_entrenamiento_fases(self, resume_timesteps, config, callbacks, model,
                                      train_env, eval_env):
        """
        Método mejorado para selección y ejecución de fases de entrenamiento con currículo experto.
        """
        
        if not self.enable_expert_curriculum or self.action_space == "pam":
            # Entrenamiento tradicional sin currículo
            return self._traditional_training(resume_timesteps, config, callbacks, 
                                           model, train_env, eval_env)
        
        print("\n🎓 Starting Expert Curriculum Training")
        print("=" * 60)
        
        current_timesteps = 0
        completed_phases = []
        
        # Iterar a través de todas las fases del currículo
        for phase_idx, phase in enumerate(self.curriculum_manager.phases):
            
            phase_info = self.curriculum_manager.get_phase_info(phase_idx)
            phase_timesteps = phase_info['timesteps']
            
            # Verificar si esta fase ya fue completada en un entrenamiento previo
            if current_timesteps + phase_timesteps <= resume_timesteps:
                current_timesteps += phase_timesteps
                completed_phases.append(phase_idx)
                continue
            
            # Ajustar timesteps si estamos resumiendo desde medio de una fase
            if current_timesteps < resume_timesteps:
                remaining_in_phase = phase_timesteps - (resume_timesteps - current_timesteps)
                phase_timesteps = remaining_in_phase
                current_timesteps = resume_timesteps
                
            if phase_timesteps <= 0:
                continue
            
            print(f"\n🚀 Phase {phase_idx}: {phase.name}")
            print(f"   Description: {phase.description}")
            print(f"   Duration: {phase_timesteps:,} timesteps")
            print(f"   Expert Weight: {phase.expert_weight:.2f}")
            print(f"   Control Mode: {phase.control_mode}")
            print(f"   Difficulty: {phase.difficulty_level}/5")
            
            # Configurar entornos para esta fase
            self._configure_environment_for_phase(train_env, phase_info)
            self._configure_environment_for_phase(eval_env, phase_info)
            
            # Crear callbacks específicos para esta fase
            phase_callbacks = self._create_phase_callbacks(phase_info, callbacks)
            
            # Ejecutar entrenamiento para esta fase
            try:
                model.learn(
                    total_timesteps=phase_timesteps,
                    callback=phase_callbacks,
                    tb_log_name=f"{config['model_prefix']}_phase_{phase_idx}",
                    reset_num_timesteps=(phase_idx == 0 and current_timesteps == 0)
                )
                
                current_timesteps += phase_timesteps
                completed_phases.append(phase_idx)
                
                # Guardar modelo de fase
                phase_model_path = os.path.join(
                    self.model_dir, 
                    f"{config['model_prefix']}_phase_{phase_idx}"
                )
                model.save(phase_model_path)
                print(f"✅ Phase {phase_idx} completed and saved: {phase_model_path}")
                
                # Avanzar el gestor de currículo
                self.curriculum_manager.advance_phase()
                
                # Registrar métricas de la fase
                self._record_phase_metrics(phase_info, model, train_env)
                
            except Exception as e:
                print(f"❌ Error in Phase {phase_idx}: {e}")
                raise e
        
        # Actualizar información de entrenamiento
        self.training_info['completed_timesteps'] = current_timesteps
        self.training_info['completed_phases'] = completed_phases
        
        print(f"\n🎉 Expert Curriculum Training Completed!")
        print(f"   Total phases completed: {len(completed_phases)}")
        print(f"   Total timesteps: {current_timesteps:,}")
        
        return model, config, train_env, eval_env
    
    def _create_phase_callbacks(self, phase_info, base_callbacks):
        """
        Crea callbacks específicos para cada fase del currículo.
        """
        class PhaseCallback:
            def __init__(self, phase_info, trainer):
                self.phase_info = phase_info
                self.trainer = trainer
                self.phase_rewards = []
                
            def on_step(self, locals_, globals_):
                # Registrar recompensas específicas de la fase
                if 'rewards' in locals_:
                    self.phase_rewards.extend(locals_['rewards'])
                    
                # Ajustar dinámicamente el peso de imitación (opcional)
                if self.phase_info['phase_id'] in [2, 4]:  # Fases de transición
                    progress = locals_.get('num_timesteps', 0) / self.phase_info['timesteps']
                    dynamic_weight = self.phase_info['expert_weight'] * (1 - progress * 0.3)
                    
                    # Aplicar a entornos si es posible
                    # (esto requeriría implementación adicional en el entorno)
                    
                return True
        
        phase_callback = PhaseCallback(phase_info, self)
        
        # Combinar con callbacks base
        if isinstance(base_callbacks, CallbackList):
            base_callbacks.callbacks.append(phase_callback)
            return base_callbacks
        else:
            return CallbackList([base_callbacks, phase_callback])
    
    def _record_phase_metrics(self, phase_info, model, env):
        """
        Registra métricas específicas de cada fase para análisis posterior.
        """
        metrics = {
            'phase_id': phase_info['phase_id'],
            'phase_name': phase_info['name'],
            'expert_weight_used': phase_info['expert_weight'],
            'control_mode': phase_info['control_mode'],
            'difficulty_level': phase_info['difficulty_level'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Calcular métricas de rendimiento si están disponibles
        if hasattr(env, 'get_episode_rewards'):
            recent_rewards = env.get_episode_rewards()[-100:]  # Últimos 100 episodios
            if recent_rewards:
                metrics['mean_reward'] = np.mean(recent_rewards)
                metrics['std_reward'] = np.std(recent_rewards)
        
        self.expert_metrics['phase_rewards'].append(metrics)
        
        # Guardar métricas en archivo
        metrics_path = os.path.join(
            self.model_dir, 
            f"expert_curriculum_metrics_phase_{phase_info['phase_id']}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _traditional_training(self, resume_timesteps, config, callbacks, 
                            model, train_env, eval_env):
        """
        Método de entrenamiento tradicional (sin currículo) como fallback.
        """
        print("🔄 Using traditional training approach (no expert curriculum)")
        
        remaining_timesteps = max(0, self.total_timesteps - resume_timesteps)
        
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                tb_log_name=f"{config['model_prefix']}_traditional",
                reset_num_timesteps=(resume_timesteps == 0)
            )
        
        self.training_info['completed_timesteps'] = self.total_timesteps
        
        return model, config, train_env, eval_env
    
    def save_training_info(self):
        """
        Versión extendida que incluye información del currículo experto.
        """
        config = self.env_configs[self.env_type]
        info_path = os.path.join(self.model_dir, f"{config['model_prefix']}_training_info.json")
        
        training_data = {
            'environment': self.env_type,
            'action_space': self.action_space,
            'completed_timesteps': self.training_info['completed_timesteps'],
            'total_timesteps_target': self.total_timesteps,
            'last_checkpoint': self.training_info['last_checkpoint'],
            'training_start': self.training_info['training_start_time'],
            'total_training_time_hours': self.training_info['total_training_time'],
            'n_envs': self.n_envs,
            'learning_rate': self.learning_rate,
            'expert_curriculum_enabled': self.enable_expert_curriculum
        }
        
        # Añadir información del currículo si está habilitado
        if self.enable_expert_curriculum and self.curriculum_manager:
            training_data['curriculum_info'] = {
                'total_phases': len(self.curriculum_manager.phases),
                'completed_phases': self.training_info.get('completed_phases', []),
                'phase_transitions': self.curriculum_manager.phase_transitions,
                'expert_metrics_summary': {
                    'total_similarity_measurements': len(self.expert_metrics['expert_action_similarity']),
                    'mean_similarity': np.mean(self.expert_metrics['expert_action_similarity']) if self.expert_metrics['expert_action_similarity'] else 0.0
                }
            }
        
        with open(info_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"💾 Enhanced training info saved to: {info_path}")
        
# ================================================================================================================================================================== #
# ===================================================Versión antigua de entrenamiento curriculo expert============================================================== #
# ================================================================================================================================================================== #
        
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
                error_load_model(e)
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
            resume_path=info_latest_checkpoint(latest_checkpoint, resume_timesteps)
        
        # Calculate remaining timesteps
        remaining_timesteps = max(0, self.total_timesteps - resume_timesteps)

        log_training_plan(resume_timesteps, remaining_timesteps, self.total_timesteps)

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

        print_info_env_pam(self)
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

        # Record training start
        self.training_info['training_start_time'] = datetime.now().isoformat()
        self.training_info['completed_timesteps'] = resume_timesteps
        
        try:
            # ==============================================
            #  ENTRENAMIENTO POR FASES
            # ==============================================
            
            model, config, \
            train_env, eval_env\
            =self.seleccion_entrenamiento_fases(resume_timesteps, config, callbacks,model,
                                                train_env, eval_env)
            
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

    def seleccion_entrenamiento_fases(self, resume_timesteps, config, callbacks,model,
                                      train_env, eval_env):
        if self.action_space == "pam":
            phase0_timesteps = 0
            phase1_timesteps = 0
            phase2_timesteps = 0
            phase3_timesteps = self.total_timesteps
        else:
            total_phase0_timesteps,\
            total_phase1_timesteps, \
            total_phase2_timesteps, \
            total_phase3_timesteps=self.generar_timesteps(self.total_timesteps, config, model)
            phase0_timesteps = np.max(total_phase1_timesteps-resume_timesteps, 0)

            if phase0_timesteps == 0:
                phase1_timesteps = np.max(total_phase1_timesteps-(resume_timesteps-total_phase0_timesteps), 0)
            else:
                phase1_timesteps = np.max(total_phase1_timesteps, 0)
            if phase1_timesteps == 0:
                phase2_timesteps = np.max(total_phase3_timesteps-(resume_timesteps-total_phase0_timesteps-
                                                                  total_phase1_timesteps), 0)
            else:
                phase2_timesteps = np.max(total_phase3_timesteps, 0)
            if phase2_timesteps == 0:
                phase3_timesteps = np.max(total_phase3_timesteps-(resume_timesteps-total_phase1_timesteps-
                                                                  total_phase2_timesteps- total_phase0_timesteps), 0)
            else:
                phase3_timesteps = np.max(total_phase3_timesteps, 0)
        current_timesteps = 0

        if phase0_timesteps > 0:
            print(f"🚀 Starting training phase 1 for {phase0_timesteps:,} steps...")
            model, current_timesteps, phase0_timesteps=phase_trainig_preparations(self.model_dir,
                                                                                    train_env, eval_env, current_timesteps,
                                                                                    model, callbacks, phase0_timesteps, config, 0)

        # FASE 1: Entrenamiento con ciclo base (solo ajustes finos)
        if phase1_timesteps > 0:
            print(f"🚀 Starting training phase 1 for {phase1_timesteps:,} steps...")
            model, current_timesteps, phase1_timesteps=phase_trainig_preparations(self.model_dir,
                                                                                    train_env, eval_env, current_timesteps,
                                                                                    model, callbacks, phase1_timesteps, config, 1)
        else:
            current_timesteps += phase1_timesteps
        # FASE 2: Entrenamiento con modulación del ciclo
        if phase2_timesteps > 0:
            print(f"🚀 Phase 2: Cycle modulation training ({phase2_timesteps:,} steps)...")
            model, current_timesteps, phase2_timesteps=phase_trainig_preparations(self.model_dir, train_env, eval_env,
                                                                                    current_timesteps, model, callbacks, phase2_timesteps, config, 2)
        else:
            current_timesteps += phase2_timesteps
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
        return model,config, train_env, eval_env
    
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
                'model_prefix': 'biped_pam_expert' if self.enable_expert_curriculum else 'biped_pam'
            }
        }

        # Información de entrenamiento (en caso de no continuar con uno previo)
        self.training_info = {
            'completed_timesteps': 0,
            'completed_phases': [],  # Nueva: fases completadas
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
        phase0_timesteps = int(remaining_timesteps * 0.1)   # 10% para equilibrio
        phase1_timesteps = int(remaining_timesteps * 0.2)  # 30% con ciclo base
        phase2_timesteps = int(remaining_timesteps * 0.4)  # 40% con modulación
        phase3_timesteps = remaining_timesteps - phase1_timesteps - phase2_timesteps - phase0_timesteps  # 30% control libre

        print(f"📊 Training Plan:")
        print(f"   - Phase 1: {phase0_timesteps:,} steps (Walking cycle base)")
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

        return phase0_timesteps, phase1_timesteps, phase2_timesteps, phase3_timesteps
    


    
