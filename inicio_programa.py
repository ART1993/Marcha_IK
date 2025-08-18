import multiprocessing as mp
import os
from collections import deque
import numpy as np
import pybullet as p

from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import Simple_BalanceSquat_BipedEnv
from Gymnasium_Start.Enhanced_UnifiedBipedTrainer import Enhanced_UnifiedBipedTrainer

from Archivos_Mejorados.Setup_multiprocessing import setup_multiprocessing_for_training

def _setup_multiprocessing():
    # Prepara el multiprocesado para el caso de n_env>1
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        print("⚠️ Multiprocessing already initialized with different method")


def start_training_min():
    """Función optimizada para iniciar entrenamiento"""
    # PASO CRÍTICO: Configurar multiprocessing ANTES de crear cualquier objeto
    target_envs = 1
    
    # Crear trainer con configuración óptima
    trainer = Enhanced_UnifiedBipedTrainer(
        env_type='enhanced_pam',
        system_version='enhanced', 
        total_timesteps=500000,
        n_envs=target_envs,
        learning_rate=3e-4,
        action_space='pam',
        enable_expert_curriculum=True
    )
    
    # Entrenar con resume automático
    try:
        trainer.train(resume=True)
        return trainer
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")


def start_training():
    """Función optimizada para iniciar entrenamiento"""
    # PASO CRÍTICO: Configurar multiprocessing ANTES de crear cualquier objeto
    target_envs = 6
    mp_success = setup_multiprocessing_for_training(n_envs=target_envs, verbose=True)
    
    if not mp_success:
        print("⚠️ Problemas con configuración de multiprocessing detectados")
        print("🔧 Cambiando a configuración más conservadora...")
        target_envs = 1  # Fallback a un solo entorno
    
    # Crear trainer con configuración óptima
    trainer = Enhanced_UnifiedBipedTrainer(
        env_type='enhanced_pam',
        system_version='enhanced', 
        total_timesteps=5_000_000,
        n_envs=target_envs,
        learning_rate=3e-4,
        action_space='pam',
        enable_expert_curriculum=True
    )
    
    # Entrenar con resume automático
    try:
        trainer.train(resume=True)
        return trainer
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")
        
        # Si falla con múltiples entornos, intentar con uno solo
        #if target_envs > 1:
        #    print("🔄 Reintentando con un solo entorno...")
        #    trainer.n_envs = 1
        #    trainer.train(resume=True)
        #    return 
        #else:
        #    raise

# ===================================================================================================================================================== #
# =================================================Testeo de la calidad de PAMBipedENV================================================================= #
# ===================================================================================================================================================== #

def test_enhanced_6pam_system():
    """Script de prueba para verificar el sistema de 6 PAMs"""
    
    print("🔧 Testing Enhanced PAM System (6 actuators)")
    
    env = Simple_BalanceSquat_BipedEnv(render_mode='human', action_space="pam")
    
    obs, info = env.reset()
    print(f"✅ Environment created successfully")
    print(f"   - Action space: {env.action_space.shape}")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Active PAMs: {env.num_active_pams}")
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"   Step {step}: Reward = {reward:.2f}")
            if 'pam_pressures' in info:
                print(f"      PAM pressures = {info['pam_pressures']}")
        
        if done:
            print(f"   Episode ended at step {step}")
            obs, info = env.reset()
    
    env.close()
    print("🎉 Test completed successfully!")


# Main execution
if __name__ == "__main__":
    start_training()
    # Choose training type
#    training_type = 'pam'  # Change to 'standard' for standard biped
    
#    if training_type == 'pam':
#        train_pam_biped()
#    else:
#        test_pam_biped()