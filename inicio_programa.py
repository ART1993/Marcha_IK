import multiprocessing as mp
import os
from collections import deque
import numpy as np
import pybullet as p

from Gymnasium_Start.Simplified_Lift_Leg_Trainer import create_balance_leg_trainer, create_balance_leg_trainer_no_curriculum
from Archivos_Apoyo.simple_log_redirect import init_simple_logging, log_print, both_print

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
    logger = init_simple_logging()
    
    # PRINTS QUE MANTENER EN CONSOLA (para ver progreso)
    print("🎯 SINGLE LEG BALANCE TRAINING")
    print("=" * 60)
    print("📝 Details → training_YYYYMMDD_HHMMSS.txt")
    print("🖥️  Progress → console")
    print("=" * 60)
    
    # DETALLES AL LOG (cambiar print por log_print)
    log_print("Objetivo específico:")
    log_print("Objetivo específico:")
    log_print("  ✅ Mantener equilibrio en una sola pierna")
    log_print("  ✅ Alternar entre pierna izquierda y derecha")
    log_print("  ✅ Control automático de altura de rodilla")
    log_print("  ✅ Tobillos fijos para mayor estabilidad")
    _setup_multiprocessing_simple()
    trainer = create_balance_leg_trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        # EVENTOS IMPORTANTES A AMBOS LUGARES
        both_print("🎉 ¡Entrenamiento completado exitosamente!")
        both_print(f"📁 Modelo guardado en: {trainer.model_dir}")
        
        # DETALLES SOLO AL LOG
        log_print(f"📊 Logs disponibles en: {trainer.logs_dir}")
    logger.close()  # CERRAR AL FINAL
    return trainer, model


def train_balance_pure_rl(total_timesteps=1000000, n_envs=4, resume=True):
    """
    Función principal para entrenar balance con RL PURO (sin ayuda experta)
    """
    logger = init_simple_logging()
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
        n_envs=n_envs
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        print("\n🎉 ¡Entrenamiento RL puro completado!")
        print(f"📁 Modelo guardado en: {trainer.model_dir}")
        print(f"📊 Logs disponibles en: {trainer.logs_dir}")
        print("🤖 El modelo aprendió sin ayuda experta")
    logger.close()
    return trainer, model