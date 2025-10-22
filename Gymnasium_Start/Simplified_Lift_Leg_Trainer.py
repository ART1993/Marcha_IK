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

from stable_baselines3.common.callbacks import BaseCallback


# Import your environments
from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv  # Nuevo entorno mejorado
from Archivos_Apoyo.Configuraciones_adicionales import cargar_posible_normalizacion
from Archivos_Apoyo.simple_log_redirect import log_print, both_print
from Archivos_Apoyo.CSVLogger import CSVLogger

class SaveVecNormalizeOnStep(BaseCallback):
    """
    Guarda VecNormalize emparejado con cada checkpoint:
    <prefix>_checkpoint_<N>_steps.zip -> <prefix>_checkpoint_<N>_steps_normalize.pkl
    """
    def __init__(self, vecnorm_env, save_dir, name_prefix, save_every_steps, verbose=0):
        super().__init__(verbose)
        self.vecnorm_env = vecnorm_env
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.save_every_steps = save_every_steps

    def _on_step(self) -> bool:
        if self.save_every_steps and (self.num_timesteps % self.save_every_steps == 0):
            if isinstance(self.vecnorm_env, VecNormalize):
                path = os.path.join(
                    self.save_dir,
                    f"{self.name_prefix}_{self.num_timesteps}_steps_normalize.pkl"
                )
                try:
                    self.vecnorm_env.save(path)
                    print(f"💾 Saved CHECKPOINT VecNormalize -> {path}")
                except Exception as e:
                    print(f"⚠️ Could not save CHECKPOINT VecNormalize: {e}")
        return True


class EvalCallbackSaveVecnorm(EvalCallback):
    """
    Amplía EvalCallback: cuando guarda un nuevo best_model.zip,
    guarda también <prefix>_best_normalize.pkl en model_dir.
    """
    def __init__(self, *args, train_vecnorm=None, model_prefix="model", **kwargs):
        super().__init__(*args, **kwargs)
        self._train_vecnorm = train_vecnorm
        self._best_seen = -float("inf")
        self._model_prefix = model_prefix

    def _on_step(self) -> bool:
        prev = self._best_seen
        ok = super()._on_step()
        cur = getattr(self, "best_mean_reward", -float("inf"))
        if cur > prev:
            self._best_seen = cur
            try:
                if isinstance(self._train_vecnorm, VecNormalize):
                    best_norm_path = os.path.join(
                        self.best_model_save_path,
                        f"{self._model_prefix}_best_normalize.pkl"
                    )
                    self._train_vecnorm.save(best_norm_path)
                    print(f"💾 Saved BEST VecNormalize -> {best_norm_path}")
            except Exception as e:
                print(f"⚠️ Could not save BEST VecNormalize: {e}")
        return ok

class Simplified_Lift_Leg_Trainer:
    """
    Entrenador unificado mejorado para sistemas de robot bípedo con músculos PAM antagónicos.
    
    Esta versión mejorada del entrenador entiende la diferencia fundamental entre:
    - Sistemas de músculos antagónicos (20 PAMs coordinados biomecánicamente)
    
    Es como la diferencia entre entrenar a alguien para mover músculos individuales
    versus entrenar a un atleta para coordinar grupos musculares de manera natural.
    """
    
    def __init__(self, 
                 total_timesteps=2000000,
                 n_envs=4,
                 learning_rate=3e-4,  # Ligeramente reducido para mayor estabilidad
                 resume_from=None,
                 logger=None,
                 csvlog=None,
                 robot_name="2_legged_human_like_robot16DOF",
                 _simple_reward_mode="progressive",  # Modo de recompensa por defecto
                 _allow_hops=False,
                 _vx_target=0.6
                 ):
        
        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.env_type = "simplified_lift_legs"
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.logger=logger
        self.learning_rate = learning_rate
        self.resume_from = resume_from
        self.csvlog=csvlog
        self.simple_reward_mode = _simple_reward_mode
        self.allow_hops = _allow_hops
        self.vx_target=_vx_target
        self.robot_name = robot_name

        # Configurar el entorno y modelo según el tipo de sistema
        self._configuracion_modelo_entrenamiento()
        if self.logger:
            self.logger.log("main",f"🤖 simplified_lift_legs Trainer initialized")
            self.logger.log("main",f"   Target: Balance + Sentadillas con 6 PAMs")
            self.logger.log("main",f"   Timesteps: {self.total_timesteps:,}")
            self.logger.log("main",f"   Parallel envs: {self.n_envs}")
            self.logger.log("main",f"   Learning rate: {self.learning_rate}")
            self.logger.log("main",f"   Metodo de recompensa: {self.simple_reward_mode}")
            self.logger.log("main",f"   Permite saltos: {self.allow_hops}")
            self.logger.log("main",f"   Velocidad en caso de marcha: {self.vx_target}")

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
                'description': 'lift_legs with 20 PAMs + Auto Knee Control'
        }
        # También mantener el plural para compatibilidad interna
        self.env_configs = self.env_config

        # ===== ARQUITECTURA LSTM Sencilla=====
        
        # Sistemas antagónicos requieren más memoria temporal y capacidad de procesamiento
        self.policy_kwargs_lstm = {
            'lstm_hidden_size': 128,  # 128 Más Simplifico memoria, subir a 256 para mayor capacidad temporal
            'n_lstm_layers': 1,       # Capas simples
            'shared_lstm': False,     # LSTMs separados para policy y value
            # 'net_arch': [256, 256],       # MLP antes/después del LSTM
            # 'activation_fn': torch.nn.SiLU,
            # 'ortho_init': False           # con SiLU suele ir mejor desactivarlo
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
        logger_local = (self.logger if (self.logger and n_envs_local == 1) else None)
        # Se puede pasar puramente a csvlog
        csvlog_local= (self.csvlog if (self.csvlog and n_envs_local == 1) else None)
        _simple_reward_mode_local=self.simple_reward_mode
        _allow_hops_local=self.allow_hops
        _vx_target_local=self.vx_target
        robot_name_local=self.robot_name
        if logger_local and n_envs_local==1:
            self.logger.log("main",f"🏗️ Creating training environment: {config['description']}")
        def make_env(logger=logger_local, csvlog=csvlog_local,
                     n_envs=n_envs_local,_simple_reward_mode=_simple_reward_mode_local,
                     _allow_hops=_allow_hops_local, _vx_target=_vx_target_local,
                     robot_name=robot_name_local):
            def _init():
                # Crear el entorno con la configuración apropiada
                env = Simple_Lift_Leg_BipedEnv(
                    logger=logger,
                    csvlog=csvlog,
                    render_mode='human' if n_envs == 1 else 'direct', 
                    print_env="TRAIN",  # Para diferenciar en logs
                    simple_reward_mode=_simple_reward_mode,
                    allow_hops=_allow_hops,
                    vx_target=_vx_target,
                    robot_name=robot_name
                    
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
        eval_csvlog = CSVLogger(
            timestamp=getattr(self.csvlog, "timestamp", None),  # reutiliza el TS si ya existe
            only_workers=False                                  # ← importante: que escriba en main
        )
        def make_eval_env():
            def _init():
                env = Simple_Lift_Leg_BipedEnv(render_mode='direct', 
                                            print_env="EVAL",
                                            logger=self.logger,
                                            csvlog=eval_csvlog,
                                            simple_reward_mode=self.simple_reward_mode,
                                            allow_hops=self.allow_hops,
                                            vx_target=self.vx_target,
                                            robot_name=self.robot_name

                                            )  # Fase de evaluación es balance
                env = Monitor(env, os.path.join(self.logs_dir, "eval"))
                return env
            return _init
        
        eval_env = DummyVecEnv([make_eval_env()])
        eval_env = VecNormalize(eval_env, 
                               norm_obs=True, 
                               norm_reward=True, 
                               clip_obs=self.env_configs['clip_obs'],
                               clip_reward=self.env_configs['clip_reward'],
                               training=False)
        # # NEW: cargar stats de normalización del train si existen
        # norm_path = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_normalize.pkl")
        # if os.path.exists(norm_path):
        #     try:
        #         eval_env = VecNormalize.load(norm_path, eval_env)
        #         eval_env.training = False          # importante para que no actualice stats
        #         eval_env.norm_reward = False       # opcional: no normalizar reward en eval
        #         print("eval normalization cargado")
        #     except Exception as e:
        #         print(f"⚠️ Could not load eval normalization: {e}")
        # Preferencia: usar primero el BEST normalize, luego el genérico
        best_norm = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_best_normalize.pkl")
        generic_norm = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_normalize.pkl")
        for cand in (best_norm, generic_norm):
            if os.path.exists(cand):
                try:
                    eval_env = VecNormalize.load(cand, eval_env)
                    eval_env.training = False     # no actualizar estadísticas en eval
                    eval_env.norm_reward = False  # opcional: no normalizar reward en eval
                    print(f"📊 Eval normalization loaded from: {cand}")
                except Exception as e:
                    print(f"⚠️ Could not load eval normalization from {cand}: {e}")
                break

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
            'ent_coef': 0.01,          # Exploración moderada subir a 0.02 para mayor exploración
            'n_steps': 256,            # 265 es bajo, subir a 1024 para secuencias más largas
            'batch_size': 128,         # 128 subir a 512, multiplo de n_envs
            'n_epochs': 4,             # con n_epoch=3 hay menos pasadas
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
             (checkpoints + eval + guardado de VecNormalize emparejado)
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
        eval_freq = 25000 //self.n_envs
         # ⚠️ De momento, no podemos pasar train_vecnorm aquí porque aún no tenemos el train_env definitivo.
        eval_callback = EvalCallbackSaveVecnorm(
                eval_env,
                best_model_save_path=self.model_dir,
                log_path=os.path.join(self.logs_dir, "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=10,
                deterministic=False,
                render=False,
                verbose=1,
                train_vecnorm=None,                   # ← Se “inyecta” en train()
                model_prefix=config["model_prefix"],  # ← Para nombrar <prefix>_best_normalize.pkl
            )
        # eval_callback = EvalCallback(
        #     eval_env,
        #     best_model_save_path=self.model_dir,
        #     log_path=os.path.join(self.logs_dir, "eval"),
        #     eval_freq=eval_freq,
        #     n_eval_episodes=10,
        #     deterministic=False,
        #     render=False,
        #     verbose=1
        # )
        #kpi_csv_cb = SimpleCsvKpiCallback(logs_dir=self.logs_dir, verbose=0)
        # Guardado de normalize emparejado con cada checkpoint
        save_vecnorm_cb = SaveVecNormalizeOnStep(
            vecnorm_env=None,  # ← Se “inyecta” en train()
            save_dir=self.checkpoints_dir,
            name_prefix=f'{config["model_prefix"]}_checkpoint',
            save_every_steps=checkpoint_freq
        )
        # callbacks = CallbackList([checkpoint_callback, eval_callback])
        callbacks = CallbackList([checkpoint_callback, eval_callback, save_vecnorm_cb])
        
        
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
        train_env = cargar_posible_normalizacion(self.model_dir, resume_path, self.env_configs, train_env,
                                                 checkpoints_dir=self.checkpoints_dir)
        
        # ===== CREACIÓN DEL MODELO =====
        
        model = self.create_model(train_env, resume_path)
        
        # ===== CONFIGURACIÓN DE CALLBACKS =====
        
        callbacks = self.setup_callbacks(eval_env)

        # Inyectar el VecNormalize real a los nuevos callbacks
        for cb in callbacks.callbacks:
            if hasattr(cb, "_train_vecnorm") and cb._train_vecnorm is None:
                cb._train_vecnorm = train_env
            if hasattr(cb, "vecnorm_env") and cb.vecnorm_env is None:
                cb.vecnorm_env = train_env
        
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
            
            # ===== GUARDAR MODELO FINAL + NORMALIZE EMPAREJADO =====
            final_model_path = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_final")
            model.save(final_model_path)
            print(f"✅ Final model saved at: {final_model_path}")
            if isinstance(train_env, VecNormalize):
                # guardado genérico por compatibilidad
                norm_path = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_normalize.pkl")
                train_env.save(norm_path)
                print(f"✅ Normalization saved at: {norm_path}")
                # guardado emparejado con el 'final'
                final_norm = os.path.join(self.model_dir, f"{self.env_configs['model_prefix']}_final_normalize.pkl")
                train_env.save(final_norm)
                print(f"💾 Saved FINAL VecNormalize -> {final_norm}")
            
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

def create_balance_leg_trainer_no_curriculum(total_timesteps=1000000, n_envs=4, learning_rate=3e-4, logger=None, csvlog=None):
    """
    Función para crear fácilmente un entrenador SIN curriculum
    """
    
    trainer = Simplified_Lift_Leg_Trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=learning_rate,
        logger=logger,
        csvlog=csvlog
    )
    
    print(f"✅ Trainer created (NO CURRICULUM)")
    print(f"   Focus: Balance básico con RL puro")
    print(f"   Accion personalizada (estirar hacia los lados)")
    print(f"   Architecture: RecurrentPPO with {trainer.policy_kwargs_lstm['lstm_hidden_size']} LSTM units")
    
    return trainer

# ===== Helpers para lanzar entrenos por modo =====
#Ejemplos de uso, (No parece buena idea usarlos así, mejor creo versiones mias
def create_march_in_place_trainer(total_timesteps=1_500_000, n_envs=4, learning_rate=3e-4, 
                                  logger=None, csvlog=None, robot_name="2_legged_human_like_robot20DOF"):
    trainer = Simplified_Lift_Leg_Trainer(total_timesteps=total_timesteps, n_envs=n_envs, 
                                          learning_rate=learning_rate, logger=logger, 
                                          csvlog=csvlog, robot_name=robot_name,_simple_reward_mode='march_in_place',
                                          _allow_hops=True, _vx_target=0.0)
    print(f"✅ Trainer created (NO CURRICULUM)")
    print(f"   Focus: Balance básico con RL puro")
    print(f"   Expert help: 0% (assist=0 siempre)")
    print(f"   Architecture: RecurrentPPO with {trainer.policy_kwargs_lstm['lstm_hidden_size']} LSTM units")
    
    return trainer


def create_walk3d_trainer(total_timesteps=2_000_000, n_envs=4, learning_rate=3e-4, vx_target=0.6, 
                          logger=None, csvlog=None, robot_name="2_legged_human_like_robot20DOF", _simple_reward_mode="walk3d"):
    trainer = Simplified_Lift_Leg_Trainer(total_timesteps=total_timesteps, n_envs=n_envs, 
                                          learning_rate=learning_rate, logger=logger, 
                                          csvlog=csvlog, _simple_reward_mode=_simple_reward_mode,
                                          _allow_hops=True, _vx_target=vx_target, robot_name=robot_name)
    
    print(f"✅ Trainer created (NO CURRICULUM)")
    print(f"   Focus: Balance básico con RL puro")
    print(f"   Expert help: 0% (assist=0 siempre)")
    print(f"   Architecture: RecurrentPPO with {trainer.policy_kwargs_lstm['lstm_hidden_size']} LSTM units")
    return trainer