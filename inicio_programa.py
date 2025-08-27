import multiprocessing as mp
import os
from collections import deque
import numpy as np
import pybullet as p

from Gymnasium_Start.Simplified_BalanceSquat_Trainer import create_balance_squat_trainer

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
    
    print("🎯 SIMPLIFIED BALANCE & SQUAT TRAINING")
    print("=" * 60)
    print("Objetivo específico:")
    print("  ✅ Mantener equilibrio de pie estático")
    print("  ✅ Realizar sentadillas controladas")
    print("  ✅ Usar 6 músculos PAM antagónicos eficientemente")
    print("=" * 60)
    _setup_multiprocessing_simple()
    trainer = create_balance_squat_trainer(
        total_timesteps=total_timesteps,
        n_envs=n_envs
    )
    
    model = trainer.train(resume=resume)
    
    if model:
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print(f"📁 Modelo guardado en: {trainer.model_dir}")
        print(f"📊 Logs disponibles en: {trainer.logs_dir}")
    
    return trainer, model