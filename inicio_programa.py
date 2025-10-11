import multiprocessing as mp

from Gymnasium_Start.Simplified_Lift_Leg_Trainer import create_balance_leg_trainer_no_curriculum, create_march_in_place_trainer
from Archivos_Apoyo.simple_log_redirect import  MultiLogRedirect
from datetime import datetime
from Archivos_Apoyo.CSVLogger import CSVLogger

def _setup_multiprocessing_simple():
    # Prepara el multiprocesado para el caso de n_env>1
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        print("⚠️ Multiprocessing already initialized with different method")


def train_balance_pure_rl(total_timesteps=1000000, n_envs=4, resume=True, with_logger=True):
    """
    Función principal para entrenar balance con RL PURO (sin ayuda experta)
    """
    if with_logger:
        logger = MultiLogRedirect()
        RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")  # una sola vez por ejecución
        csvlog = CSVLogger(timestamp=RUN_TS, only_workers=True)  # el main no escribe; solo los env workers
    else:
        logger=None
        csvlog=None
    
    print("🎯 PURE RL BALANCE TRAINING")
    print("=" * 60)
    print("Objetivo específico:")
    print("  ✅ Mantener equilibrio básico de pie")
    print("  ✅ Sin ayuda experta (assist=0)")
    print("  ✅ Sin progression de niveles")
    print("  ✅ RL puro - el modelo aprende solo")
    print("=" * 60)
    _setup_multiprocessing_simple()
    trainer = create_balance_leg_trainer_no_curriculum(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        logger=logger,
        csvlog=csvlog
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        print("\n🎉 ¡Entrenamiento RL puro completado!")
        print(f"📁 Modelo guardado en: {trainer.model_dir}")
        print(f"📊 Logs disponibles en: {trainer.logs_dir}")
        print("🤖 El modelo aprendió sin ayuda experta")
    logger.close()
    return trainer, model

def train_balance_march_in_place(total_timesteps=1000000, n_envs=4, resume=True, 
                                 with_logger=True, robot_name="2_legged_human_like_robot18DOF"):
    """
    Función principal para entrenar balance con RL PURO (sin ayuda experta)
    """
    if with_logger:
        logger = MultiLogRedirect()
        RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")  # una sola vez por ejecución
        csvlog = CSVLogger(timestamp=RUN_TS, only_workers=True)  # el main no escribe; solo los env workers
    else:
        logger=None
        csvlog=None
    
    print("🎯 PURE RL BALANCE TRAINING")
    print("=" * 60)
    print("Objetivo específico:")
    print("  ✅ Mantener equilibrio básico de pie")
    print("  ✅ Sin ayuda experta (assist=0)")
    print("  ✅ Sin progression de niveles")
    print("  ✅ RL puro - el modelo aprende solo")
    print("=" * 60)
    _setup_multiprocessing_simple()
    trainer = create_march_in_place_trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        logger=logger,
        csvlog=csvlog,
        robot_name=robot_name
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        print("\n🎉 ¡Entrenamiento RL puro completado!")
        print(f"📁 Modelo guardado en: {trainer.model_dir}")
        print(f"📊 Logs disponibles en: {trainer.logs_dir}")
        print("🤖 El modelo aprendió sin ayuda experta")
    logger.close()
    return trainer, model