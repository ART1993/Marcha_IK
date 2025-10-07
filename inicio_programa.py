import multiprocessing as mp
import os
from collections import deque
import numpy as np
import pybullet as p

from Gymnasium_Start.Simplified_Lift_Leg_Trainer import create_balance_leg_trainer, create_balance_leg_trainer_no_curriculum
from Archivos_Apoyo.simple_log_redirect import init_simple_logging, log_print, both_print, MultiLogRedirect
from datetime import datetime
from Archivos_Apoyo.CSVLogger import CSVLogger

def _setup_multiprocessing_simple():
    # Prepara el multiprocesado para el caso de n_env>1
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        print("⚠️ Multiprocessing already initialized with different method")


def train_single_leg_balance(total_timesteps=2000000, n_envs=4, resume=True):
    """
    Función principal para entrenar balance y sentadillas
    """

    # INICIALIZAR LOGGING SIMPLE (1 LÍNEA)
    #logger = init_simple_logging()
    logger = MultiLogRedirect()
    
    
    # PRINTS QUE MANTENER EN CONSOLA (para ver progreso)
    print("🎯 SINGLE LEG BALANCE TRAINING")
    print("=" * 60)
    print("📝 Details → training_YYYYMMDD_HHMMSS.txt")
    print("🖥️  Progress → console")
    print("=" * 60)
    
    # DETALLES AL LOG (cambiar print por log_print)
    logger.console("Objetivo específico:")
    logger.both("main", "Objetivo específico:")
    logger.log("main","  ✅ Mantener equilibrio en una sola pierna")
    logger.log("main","  ✅ Alternar entre pierna izquierda y derecha")
    logger.log("main","  ✅ Control automático de altura de rodilla")
    logger.log("main","  ✅ Tobillos fijos para mayor estabilidad")
    _setup_multiprocessing_simple()
    trainer = create_balance_leg_trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        logger=logger
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        # EVENTOS IMPORTANTES A AMBOS LUGARES
        logger.both("main","🎉 ¡Entrenamiento completado exitosamente!")
        logger.both("main",f"📁 Modelo guardado en: {trainer.model_dir}")
        
        # DETALLES SOLO AL LOG
        logger.both("main",f"📊 Logs disponibles en: {trainer.logs_dir}")
    #logger.close()  # CERRAR AL FINAL
    logger.close()
    return trainer, model


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