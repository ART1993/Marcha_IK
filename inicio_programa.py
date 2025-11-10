import multiprocessing as mp

from Gymnasium_Start.Simplified_Lift_Leg_Trainer import create_walk3d_trainer
from Archivos_Apoyo.simple_log_redirect import  MultiLogRedirect
from datetime import datetime
from Archivos_Apoyo.CSVLogger import CSVLogger

def _setup_multiprocessing_simple():
    # Prepara el multiprocesado para el caso de n_env>1
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        print("⚠️ Multiprocessing already initialized with different method")

def train_balance_walk_3d(total_timesteps=1000000, n_envs=4, resume=True, 
                                 with_logger=True, robot_name="2_legged_human_like_robot12DOF_done",
                                 simple_reward_mode="walk3d"):
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
    trainer = create_walk3d_trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        logger=logger,
        csvlog=csvlog,
        robot_name=robot_name,
        _simple_reward_mode=simple_reward_mode
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        print("\n🎉 ¡Entrenamiento RL puro completado!")
        print(f"📁 Modelo guardado en: {trainer.model_dir}")
        print(f"📊 Logs disponibles en: {trainer.logs_dir}")
        print("🤖 El modelo aprendió sin ayuda experta")
    logger.close()
    return trainer, model


if __name__ == "__main__":
    train_balance_walk_3d()