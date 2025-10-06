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
import csv
from stable_baselines3.common.callbacks import BaseCallback

# Import your environments
from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv  # Nuevo entorno mejorado
from Archivos_Apoyo.Configuraciones_adicionales import cargar_posible_normalizacion
from Archivos_Apoyo.simple_log_redirect import log_print, both_print


class SimpleCsvKpiCallback(BaseCallback):
    """
    Lee info['kpi'] por step e info['ep_kpi'] al finalizar episodios
    y los vuelca a CSV en self.logs_dir.
    """
    def __init__(self, logs_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.logs_dir = logs_dir
        self._step_f = None
        self._ep_f = None
        self._step_writer = None
        self._ep_writer = None

    def _on_training_start(self) -> None:
        os.makedirs(self.logs_dir, exist_ok=True)
        self._step_f = open(os.path.join(self.logs_dir, "kpi_step.csv"), "w", newline="")
        self._ep_f   = open(os.path.join(self.logs_dir, "kpi_episode.csv"), "w", newline="")
        self._step_writer = csv.DictWriter(self._step_f,
            fieldnames=["timesteps","env_idx","reward","roll","pitch",
                        "left_down","right_down","F_L","F_R",
                        "zmp_x","zmp_y","com_x","com_y","com_z"])
        self._ep_writer = csv.DictWriter(self._ep_f,
            fieldnames=["timesteps","env_idx","ep_return","ep_len","done_reason"])
        self._step_writer.writeheader()
        self._ep_writer.writeheader()

    def _on_step(self) -> bool:
        # infos por cada env en este step
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            kpi = info.get("kpi")
            if kpi:
                row = {
                    "timesteps": int(self.num_timesteps),
                    "env_idx": env_idx,
                    "reward": kpi.get("reward", 0.0),
                    "roll": kpi.get("roll", 0.0),
                    "pitch": kpi.get("pitch", 0.0),
                    "left_down": kpi.get("left_down", 0),
                    "right_down": kpi.get("right_down", 0),
                    "F_L": kpi.get("F_L", 0.0),
                    "F_R": kpi.get("F_R", 0.0),
                    "zmp_x": kpi.get("zmp_x", 0.0),
                    "zmp_y": kpi.get("zmp_y", 0.0),
                    "com_x": kpi.get("com_x", 0.0),
                    "com_y": kpi.get("com_y", 0.0),
                    "com_z": kpi.get("com_z", 0.0),
                }
                self._step_writer.writerow(row)
        return True

    def _on_rollout_end(self) -> None:
        # algunos episodios terminan aquí; SB3 no siempre pasa ep_info,
        # pero nosotros miramos en infos por si hay 'ep_kpi'
        ep_infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(ep_infos):
            ep_kpi = info.get("ep_kpi")
            if ep_kpi:
                row = {
                    "timesteps": int(self.num_timesteps),
                    "env_idx": env_idx,
                    "ep_return": ep_kpi.get("ep_return", 0.0),
                    "ep_len": ep_kpi.get("ep_len", 0),
                    "done_reason": ep_kpi.get("done_reason", None),
                }
                self._ep_writer.writerow(row)

    def _on_training_end(self) -> None:
        try:
            if self._step_f: self._step_f.close()
            if self._ep_f: self._ep_f.close()
        except Exception:
            pass

"""
KPI CHECKLIST (entrenamiento PAM – bípede)
- Éxito L3 por pierna (%), duración single-support (media/p95)
- Tilt p95 / pico, ZMP margin p05 (>0 ideal), rate de toe-touch (<5%)
- No-support time máx (<0.2 s), slip del pie de soporte
- Saturación PAM (<25%), co-contracción, ||Δu|| (suavidad)
- Cap-hit rate (L3=7.0) < 30%, acciones en borde < 30%
- PPO: entropy, approx_kl, clip_fraction, explained_variance
Alertas (3 ventanas seguidas):
- éxito ↓ 5pp, tilt_p95 > 20°, KL > 2×obj o clip_fraction > 0.6, FPS cae >30%
"""
class Simplified_Lift_Leg_Trainer:
    """
    Entrenador unificado mejorado para sistemas de robot bípedo con músculos PAM antagónicos.
    
    Esta versión mejorada del entrenador entiende la diferencia fundamental entre:
    - Sistemas de músculos simples (4 PAMs independientes)
    - Sistemas de músculos antagónicos (12 PAMs coordinados biomecánicamente)
    
    Es como la diferencia entre entrenar a alguien para mover músculos individuales
    versus entrenar a un atleta para coordinar grupos musculares de manera natural.
    """
    
    def __init__(self, 
                 total_timesteps=2000000,
                 n_envs=4,
                 learning_rate=3e-4,  # Ligeramente reducido para mayor estabilidad
                 resume_from=None,
                 enable_curriculum=True,
                 logger=None
                 ):
        
        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.env_type = "simplified_lift_legs"
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.logger=logger
        self.learning_rate = learning_rate
        self.resume_from = resume_from
        self.enable_curriculum=enable_curriculum
        

        # Configurar el entorno y modelo según el tipo de sistema
        self._configuracion_modelo_entrenamiento()
        if self.logger:
            self.logger.log("main",f"🤖 simplified_lift_legs Trainer initialized")
            self.logger.log("main",f"   Target: Balance + Sentadillas con 6 PAMs")
            self.logger.log("main",f"   Timesteps: {self.total_timesteps:,}")
            self.logger.log("main",f"   Parallel envs: {self.n_envs}")
            self.logger.log("main",f"   Learning rate: {self.learning_rate}")

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
        n_envs_local = int(self.n_envs)
        enable_curr_local = bool(self.enable_curriculum)
        logger_local = (self.logger if (self.logger and n_envs_local == 1) else None)
        if logger_local and n_envs_local==1:
            self.logger.log("main",f"🏗️ Creating training environment: {config['description']}")
        def make_env(logger=logger_local,
                     enable_curriculum=enable_curr_local,
                     n_envs=n_envs_local):
            def _init():
                # Crear el entorno con la configuración apropiada
                env = Simple_Lift_Leg_BipedEnv(
                    logger=logger,
                    render_mode='human' if n_envs == 1 else 'direct', 
                    enable_curriculum=enable_curriculum,
                    print_env="TRAIN"  # Para diferenciar en logs
                    
                )
                #Eliminado escribir , os.path.join(self.logs_dir, f"train_worker_{rank}" acelero entrenamiento
                env = Monitor(env)
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
                                             enable_curriculum=False,  # Evaluación sin curriculum
                                            print_env="EVAL",
                                            logger=self.logger
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
        # NEW: cargar stats de normalización del train si existen
        norm_path = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_normalize.pkl")
        if os.path.exists(norm_path):
            try:
                eval_env = VecNormalize.load(norm_path, eval_env)
                eval_env.training = False          # importante para que no actualice stats
                eval_env.norm_reward = False       # opcional: no normalizar reward en eval
            except Exception as e:
                print(f"⚠️ Could not load eval normalization: {e}")

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
        checkpoint_freq = 90000//self.n_envs  # Cada 100k timesteps dividido por el número de entornos
        
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=self.checkpoints_dir,
            name_prefix=f'{config["model_prefix"]}_checkpoint',
            verbose=1
        )
        
        # ===== EVALUATION CALLBACK =====
        
        # Evaluación más frecuente para sistemas complejos
        eval_freq = 30000 //self.n_envs
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_dir,
            log_path=os.path.join(self.logs_dir, "eval"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=1
        )
        kpi_csv_cb = SimpleCsvKpiCallback(logs_dir=self.logs_dir, verbose=0)
        callbacks = CallbackList([checkpoint_callback, eval_callback, kpi_csv_cb])
        
        
        return callbacks
        
    
    def train(self, resume=True):
        """
            Entrenamiento DIRECTO y SIMPLIFICADO para balance/sentadillas.
            
            Sin fases complejas, sin curriculum, solo entrenamiento RecurrentPPO directo.
        """
        
        print(f"🚀 Starting lift_leg training with RecurrentPPO...")

        # DETALLES AL LOG
        if self.logger:
            self.logger.log("main","🚀 Training session started")
            self.logger.log("main",f"   Resume: {resume}")
        
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
            base_env = SubprocVecEnv([make_env() for i in range(self.n_envs)])
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

def create_balance_leg_trainer_no_curriculum(total_timesteps=1000000, n_envs=4, learning_rate=3e-4,logger=None):
    """
    Función para crear fácilmente un entrenador SIN curriculum
    """
    
    trainer = Simplified_Lift_Leg_Trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=learning_rate,
        enable_curriculum=False,
        logger=logger
    )
    
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
        learning_rate=learning_rate,
        enable_curriculum=True
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