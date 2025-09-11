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

# Import your environments
from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv  # Nuevo entorno mejorado
from Archivos_Apoyo.Configuraciones_adicionales import cargar_posible_normalizacion
from Archivos_Apoyo.simple_log_redirect import log_print, both_print

class Simplified_Lift_Leg_Trainer:
    """
    Entrenador unificado mejorado para sistemas de robot bípedo con músculos PAM antagónicos.
    
    Esta versión mejorada del entrenador entiende la diferencia fundamental entre:
    - Sistemas de músculos simples (4 PAMs independientes)
    - Sistemas de músculos antagónicos (6 PAMs coordinados biomecánicamente)
    
    Es como la diferencia entre entrenar a alguien para mover músculos individuales
    versus entrenar a un atleta para coordinar grupos musculares de manera natural.
    """
    
    def __init__(self, 
                 total_timesteps=2000000,
                 n_envs=4,
                 learning_rate=3e-4,  # Ligeramente reducido para mayor estabilidad
                 resume_from=None,
                 ):
        
        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.env_type = "simplified_lift_legs"
        self.action_space = "pam"
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.learning_rate = learning_rate
        self.resume_from = resume_from
        

        # Configurar el entorno y modelo según el tipo de sistema
        self._configuracion_modelo_entrenamiento()

        log_print(f"🤖 simplified_lift_legs Trainer initialized")
        log_print(f"   Target: Balance + Sentadillas con 6 PAMs")
        log_print(f"   Timesteps: {self.total_timesteps:,}")
        log_print(f"   Parallel envs: {self.n_envs}")
        log_print(f"   Learning rate: {self.learning_rate}")

        # MANTENER EN CONSOLA SOLO CONFIRMACIÓN
        print(f"🤖 Trainer ready")
    

    def _configuracion_modelo_entrenamiento(self):
        """
            Configuración del modelo de entrenamiento adaptada para sistemas antagónicos.
            
            Esta función establece todas las configuraciones específicas que necesita
            cada tipo de sistema. Es como configurar diferentes programas de entrenamiento
            para atletas de diferentes niveles.
        """

        # ===== CONFIGURACIÓN DE DIRECTORIOS =====
        
        # Crear directorios base
        self.model_dir = "./models_lift_leg"
        self.logs_dir = "./logs_lift_leg"
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")

        # Creacion directorios si no existen
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # ===== CONFIGURACIONES ENTORNO =====
        
        # Configuración optimizada para sistemas de 6 músculos antagónicos
        self.env_config = {
                'clip_obs': 10.0,      
                'clip_reward': 15.0,   
                'model_prefix': 'single_leg_balance_pam',
                'description': 'lift_legs with 6 PAMs + Auto Knee Control'
        }
        # También mantener el plural para compatibilidad interna
        self.env_configs = self.env_config

        # ===== ARQUITECTURA LSTM Sencilla=====
        
        # Sistemas antagónicos requieren más memoria temporal y capacidad de procesamiento
        self.policy_kwargs_lstm = {
            'lstm_hidden_size': 128,  # Más Simplifico memoria
            'n_lstm_layers': 1,       # Capas simples
            'shared_lstm': False,     # LSTMs separados para policy y value
        }

        # ===== INFORMACIÓN DE ENTRENAMIENTO =====
        
        self.training_info = {
            'completed_timesteps': 0,
            'last_checkpoint': None,
            'training_start_time': None,
            'total_training_time': 0
        }

    def create_training_env(self):
        """
        Creación de entorno de entrenamiento para sentadillas.
        
        """
        
        config = self.env_configs
        
        log_print(f"🏗️ Creating training environment: {config['description']}")
        
        def make_env():
            def _init():
                # Crear el entorno con la configuración apropiada
                env = Simple_Lift_Leg_BipedEnv(
                    render_mode='human' if self.n_envs == 1 else 'direct', 
                    action_space=self.action_space,
                    enable_curriculum=True
                    
                )
                
                env = Monitor(env, self.logs_dir)
                return env
            return _init
        
        # Crear entornos paralelos con configuración optimizada
        env = self._create_parallel_env(make_env=make_env, config=config)
        
        return env

    
    def create_eval_env(self):
        """
            Crear entorno de evaluación.
       
        """
        
        def make_eval_env():
            def _init():
                env = Simple_Lift_Leg_BipedEnv(render_mode='direct', 
                                             action_space=self.action_space
                                            )  # Fase de evaluación es balance
                env = Monitor(env, os.path.join(self.logs_dir, "eval"))
                return env
            return _init
        
        eval_env = DummyVecEnv([make_eval_env()])
        eval_env = VecNormalize(eval_env, 
                               norm_obs=True, 
                               norm_reward=True, 
                               clip_obs=self.env_configs['clip_obs'], 
                               training=False)

        return eval_env
    
    def create_model(self, env, resume_path=None):
        """
            Crear modelo RecurrentPPO optimizado para sistemas antagónicos.
            
            Los sistemas de músculos antagónicos requieren arquitecturas de red
            más sofisticadas y hiperparámetros ajustados para manejar la complejidad
            adicional de la coordinación muscular.
        """
        
        # Intentar cargar modelo existente. Se escribe así para ahorrar espacio y legibilidad.
        model = self.cargar_creacion_modelo(resume_path=resume_path, env=env)
        if model is not None:
            return model

        print(f"🧠 Creating new RecurrentPPO model for lift_leg...")
        # ===== CREACIÓN DEL MODELO =====
        model_params = {
            'learning_rate': self.learning_rate,
            'gamma': 0.99,             # Estándar
            'max_grad_norm': 0.5,      # Estándar
            'ent_coef': 0.01,          # Exploración moderada
            'n_steps': 256,            # Reducido
            'batch_size': 128,         # Reducido
            'n_epochs': 4,             # Reducido
            'gae_lambda': 0.95,        # Estándar
            'clip_range': 0.2,         # Estándar
            'vf_coef': 0.5,            # Estándar
        }
        
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            policy_kwargs=self.policy_kwargs_lstm,
            verbose=1,
            tensorboard_log=self.logs_dir,
            device='auto',
            **model_params
        )
        
        print(f"✅ Model created with {self.policy_kwargs_lstm['lstm_hidden_size']} LSTM units")
        
        return model
    
    def setup_callbacks(self, eval_env):
        """
            Configurar callbacks optimizados para sistemas antagónicos.
            
            Los sistemas antagónicos requieren monitoreo más frecuente y métricas
            más sofisticadas para entender el progreso del entrenamiento.
        """
        
        config = self.env_configs
        
        # ===== CHECKPOINT CALLBACK =====
        
        # Sistemas antagónicos se benefician de checkpoints más frecuentes
        checkpoint_freq = 100000//self.n_envs  # Cada 100k timesteps dividido por el número de entornos
        
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=self.checkpoints_dir,
            name_prefix=f'{config["model_prefix"]}_checkpoint',
            verbose=1
        )
        
        # ===== EVALUATION CALLBACK =====
        
        # Evaluación más frecuente para sistemas complejos
        eval_freq = 50000 //self.n_envs
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=os.path.join(self.logs_dir, "eval"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        callbacks = CallbackList([checkpoint_callback, eval_callback])
        
        
        return callbacks
        
    
    def train(self, resume=True):
        """
            Entrenamiento DIRECTO y SIMPLIFICADO para balance/sentadillas.
            
            Sin fases complejas, sin curriculum, solo entrenamiento RecurrentPPO directo.
        """
        
        print(f"🚀 Starting lift_leg training with RecurrentPPO...")

        # DETALLES AL LOG
        log_print("🚀 Training session started")
        log_print(f"   Resume: {resume}")
        
        # ===== PREPARACIÓN DEL ENTRENAMIENTO =====
        
        resume_path, resume_timesteps, remaining_timesteps = self.prep_modelo_anterior(resume)
        if resume_path is None and remaining_timesteps is None:
            print("   ✅ Training already completed!")
            return

        # ===== CREACIÓN DE ENTORNOS =====
        
        train_env = self.create_training_env()
        eval_env = self.create_eval_env()
        
        # Cargar normalizaciones existentes si las hay
        train_env = cargar_posible_normalizacion(self.model_dir, resume_path, self.env_configs, train_env)
        
        # ===== CREACIÓN DEL MODELO =====
        
        model = self.create_model(train_env, resume_path)
        
        # ===== CONFIGURACIÓN DE CALLBACKS =====
        
        callbacks = self.setup_callbacks(eval_env)
        
        # ===== REGISTRO DE INICIO =====
        
        self.training_info['training_start_time'] = datetime.now().isoformat()
        self.training_info['completed_timesteps'] = resume_timesteps
        
        try:
            # ===== EJECUTAR ENTRENAMIENTO =====
            
            # Entrenamiento tradicional sin currículo
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                tb_log_name=f"{self.env_configs['model_prefix']}_training",
                reset_num_timesteps=(resume_timesteps == 0)
            )
            self.training_info['completed_timesteps'] = self.total_timesteps
            
            # ===== GUARDAR MODELO FINAL =====
            
            final_model_path = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_final")
            model.save(final_model_path)
            print(f"✅ Final model saved at: {final_model_path}")
            
            # Guardar normalizaciones
            if isinstance(train_env, VecNormalize):
                norm_path = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_normalize.pkl")
                train_env.save(norm_path)
                print(f"✅ Normalization saved at: {norm_path}")
            
            # ===== GUARDAR INFORMACIÓN DE ENTRENAMIENTO =====

            self._save_training_info()
            
            print(f"\n🎉 lift_leg training completed successfully!")
            print(f"   Total timesteps: {self.total_timesteps:,}")
            print(f"   Model saved in: {self.model_dir}")
            # ÉXITO: A AMBOS
            both_print(f"✅ Training completed successfully!")
            
            return model
                
        except KeyboardInterrupt:
            both_print("⏸️ Training interrupted by user")

        except Exception as e:
            both_print(f"❌ Training failed: {e}")
            
        finally:
            train_env.close()
            eval_env.close()
    

    def cargar_creacion_modelo(self, resume_path, env):
        """Cargando el modelo, movio aqui por legibilidad"""
        if resume_path and os.path.exists(resume_path):
            print(f"🔄 Loading model from checkpoint: {resume_path}")
            try:
                model = RecurrentPPO.load(resume_path, env=env)
                
                if hasattr(model, 'learning_rate') and model.learning_rate != self.learning_rate:
                    print(f"📝 Updating learning rate: {model.learning_rate} -> {self.learning_rate}")
                    model.learning_rate = self.learning_rate
                    
                return model
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                print("🔄 Creating new model instead...")

                
        return None
    
    def _save_training_info(self):
        """Guardar información básica del entrenamiento"""
        
        info_path = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_info.json")
        
        training_data = {
            'objective': 'lift_leg with 6 PAM muscles',
            'environment': self.env_type,
            'action_space': self.action_space,
            'completed_timesteps': self.training_info['completed_timesteps'],
            'total_timesteps_target': self.total_timesteps,
            'training_start': self.training_info['training_start_time'],
            'n_envs': self.n_envs,
            'learning_rate': self.learning_rate,
            'lstm_config': self.policy_kwargs_lstm,
            'simplified_trainer': True
        }
        
        with open(info_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"💾 Training info saved: {info_path}")

    def prep_modelo_anterior(self, resume):
        """Carga el modelo anterior teniendo en cuenta los timesteps restantes"""
        resume_path = None
        resume_timesteps = 0

        if resume and self.resume_from:
            if os.path.exists(self.resume_from):
                resume_path = self.resume_from
                match = re.search(r'_(\d+)_steps\.zip$', self.resume_from)
                if match:
                    resume_timesteps = int(match.group(1))
                print(f"📂 Resuming from specified checkpoint: {self.resume_from}")
            else:
                print(f"⚠️ Specified checkpoint not found: {self.resume_from}")
        elif resume:
            latest_checkpoint, resume_timesteps = self._find_latest_checkpoint()
            if latest_checkpoint:
                resume_path = latest_checkpoint
        
        remaining_timesteps = max(0, self.total_timesteps - resume_timesteps)

        if remaining_timesteps == 0:
            print(f"✅ Training already completed! ({resume_timesteps:,}/{self.total_timesteps:,} timesteps)")
            return None, None, None
        
        print(f"📊 Training plan:")
        print(f"   Completed: {resume_timesteps:,}")
        print(f"   Remaining: {remaining_timesteps:,}")
        print(f"   Total target: {self.total_timesteps:,}")
        
        return resume_path, resume_timesteps, remaining_timesteps

    def _find_latest_checkpoint(self):
        """Find the most recent checkpoint based on timesteps."""
        model_prefix = self.env_configs['model_prefix']
        
        checkpoint_pattern = os.path.join(self.checkpoints_dir, f"{model_prefix}_checkpoint_*_steps.zip")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            return None, 0
        
        timesteps = []
        for file in checkpoint_files:
            match = re.search(r'_(\d+)_steps\.zip$', file)
            if match:
                timesteps.append((int(match.group(1)), file))
        
        if not timesteps:
            return None, 0
        
        latest_timesteps, latest_file = max(timesteps, key=lambda x: x[0])
        return latest_file, latest_timesteps
    
# ========================================================================================================= #
# =====================================creacion de entornos paralelos====================================== #
# ========================================================================================================= #

    def _create_parallel_env(self, make_env, config):
        """Creación de entornos paralelos optimizada"""
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
    
    # Ya que he creado clase sin RL para testing, también creo con este un modelo de entrenamiento sin RL

def create_balance_leg_trainer_no_curriculum(total_timesteps=1000000, n_envs=4, learning_rate=3e-4):
    """
    Función para crear fácilmente un entrenador SIN curriculum
    """
    
    trainer = Simplified_Lift_Leg_Trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=learning_rate
    )
    
    # Modificar solo la creación del entorno para deshabilitar curriculum
    original_create_training = trainer.create_training_env
    original_create_eval = trainer.create_eval_env
    
    def create_training_env_no_curriculum():
        config = trainer.env_configs
        log_print(f"🏗️ Creating NO-CURRICULUM training environment: {config['description']}")
        
        def make_env():
            def _init():
                env = Simple_Lift_Leg_BipedEnv(
                    render_mode='human' if trainer.n_envs == 1 else 'direct', 
                    action_space=trainer.action_space,
                    enable_curriculum=False  # ⭐ CLAVE: Deshabilitar curriculum
                )
                env = Monitor(env, trainer.logs_dir)
                return env
            return _init
        
        return trainer._create_parallel_env(make_env=make_env, config=config)
    
    def create_eval_env_no_curriculum():
        def make_eval_env():
            def _init():
                env = Simple_Lift_Leg_BipedEnv(
                    render_mode='direct', 
                    action_space=trainer.action_space,
                    enable_curriculum=False  # ⭐ CLAVE: Deshabilitar curriculum
                )
                env = Monitor(env, os.path.join(trainer.logs_dir, "eval"))
                return env
            return _init
        
        eval_env = DummyVecEnv([make_eval_env()])
        eval_env = VecNormalize(eval_env, 
                               norm_obs=True, 
                               norm_reward=True, 
                               clip_obs=trainer.env_configs['clip_obs'], 
                               training=False)
        return eval_env
    
    # Reemplazar métodos
    trainer.create_training_env = create_training_env_no_curriculum
    trainer.create_eval_env = create_eval_env_no_curriculum
    
    print(f"✅ Trainer created (NO CURRICULUM)")
    print(f"   Focus: Balance básico con RL puro")
    print(f"   Expert help: 0% (assist=0 siempre)")
    print(f"   Architecture: RecurrentPPO with {trainer.policy_kwargs_lstm['lstm_hidden_size']} LSTM units")
    
    return trainer
    

# ===== FUNCIONES DE UTILIDAD =====

def create_balance_leg_trainer(total_timesteps=2000000, n_envs=4, learning_rate=3e-4):
    """
    Función para crear fácilmente un entrenador simplificado
    """
    
    trainer = Simplified_Lift_Leg_Trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=learning_rate
    )
    
    print(f"✅ Simplified Trainer created")
    print(f"   Focus: Balance de pie + Sentadillas")
    print(f"   Architecture: RecurrentPPO with {trainer.policy_kwargs_lstm['lstm_hidden_size']} LSTM units")
    
    return trainer

def train_balance_and_lift_legs(total_timesteps=2000000, n_envs=4, resume=True):
    """
    Función principal para entrenar balance y sentadillas
    """
    
    print("🎯 SIMPLIFIED lift_leg TRAINING")
    print("=" * 60)
    print("Objetivo específico:")
    print("  ✅ Mantener equilibrio de pie estático")
    print("  ✅ Realizar sentadillas controladas")
    print("  ✅ Usar 6 músculos PAM antagónicos eficientemente")
    print("=" * 60)
    
    trainer = create_balance_leg_trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📁 Modelo guardado en: {trainer.model_dir}")
        print(f"📊 Logs disponibles en: {trainer.logs_dir}")
    
    return trainer, model