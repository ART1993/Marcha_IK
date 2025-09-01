import multiprocessing as mp
import os
from collections import deque
import numpy as np
import pybullet as p

from Gymnasium_Start.Simplified_BalanceSquat_Trainer import create_balance_squat_trainer
from Archivos_Apoyo.simple_log_redirect import init_simple_logging, log_print, both_print

def _setup_multiprocessing_simple():
    # Prepara el multiprocesado para el caso de n_env>1
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        print("⚠️ Multiprocessing already initialized with different method")


def train_balance_and_squats(total_timesteps=2000000, n_envs=4, resume=True):
    """
    Función principal para entrenar balance y sentadillas
    """

    # INICIALIZAR LOGGING SIMPLE (1 LÍNEA)
    logger = init_simple_logging()
    
    # PRINTS QUE MANTENER EN CONSOLA (para ver progreso)
    print("🎯 SIMPLIFIED BALANCE & SQUAT TRAINING")
    print("=" * 60)
    print("📝 Details → training_YYYYMMDD_HHMMSS.txt")
    print("🖥️  Progress → console")
    print("=" * 60)
    
    # DETALLES AL LOG (cambiar print por log_print)
    log_print("Objetivo específico:")
    log_print("  ✅ Mantener equilibrio de pie estático") 
    log_print("  ✅ Realizar sentadillas controladas")
    log_print("  ✅ Usar 6 músculos PAM antagónicos eficientemente")
    _setup_multiprocessing_simple()
    trainer = create_balance_squat_trainer(
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