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
    
    print("🎯 PURE RL BALANCE TRAINING")
    print("=" * 60)
    print("Objetivo específico:")
    print("  ✅ Mantener equilibrio básico de pie")
    print("  ✅ Sin ayuda experta (assist=0)")
    print("  ✅ Sin progression de niveles")
    print("  ✅ RL puro - el modelo aprende solo")
    print("=" * 60)
    
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
    
    return trainer, model

# def probe_level3_expert_rollout(env, seconds=3.0, target='right'):
    
#     obs, info = env.reset()
#     env.enable_curriculum = True
#     env.simple_reward_system.level = 3
#     env.probe_expert_only = True
#     if hasattr(env, "action_selector"):
#         env.action_selector.expert_help_ratio = 1.0

#     # fija pierna objetivo para que sea reproducible
#     if hasattr(env.simple_reward_system, "target_leg"):
#         env.simple_reward_system.target_leg = target

#     steps = int(seconds * env.frequency_simulation)
#     total_r = 0.0
#     for _ in range(steps):
#         # acción dummy; será ignorada
#         a = np.zeros(env.num_active_pams, dtype=np.float32)
#         obs, r, done, _, info = env.step(a)
#         total_r += r
#         if done:
#             break
#     print(f"[PROBE] level=3, leg={target}, steps={info.get('step_count', 0)}, reward={total_r:.2f}")´


# def probe_level1_expert_rollout(env, seconds=3.0):
    
#     # reset estilo Gymnasium
#     obs, info = env.reset()

#     # Aseguramos modo con curriculum y nivel 1
#     env.enable_curriculum = True
#     if hasattr(env, "simple_reward_system"):
#         env.simple_reward_system.level = 1

#     # (opcional) ignorar acción del agente y usar solo experto
#     if hasattr(env, "action_selector"):
#         # fuerza ayuda experta total
#         env.action_selector.expert_help_ratio = 1.0
#     # si tu entorno usa este flag para ignorar la acción externa:
#     if hasattr(env, "probe_expert_only"):
#         env.probe_expert_only = True

#     steps = int(seconds * env.frequency_simulation)
#     total_r = 0.0
#     for _ in range(steps):
#         # acción dummy; será ignorada si probe_expert_only=True
#         a = np.zeros(env.num_active_pams, dtype=np.float32)
#         obs, r, done, _, info = env.step(a)
#         total_r += r
#         if done:
#             break

#     print(f"[PROBE] level=1, steps={info.get('step_count', 0)}, reward={total_r:.2f}")


def probe_expert_rollout(env, level=1, seconds=3.0, target=None):
    """
    Prueba rápida del CONTROL EXPERTO en un nivel dado.
    - level: 1, 2 o 3
    - target: 'left' o 'right' (solo aplica en level 3)
    """
    obs, info = env.reset()

    env.enable_curriculum = True
    env.simple_reward_system.level = level
    env.probe_expert_only = True
    if hasattr(env, "action_selector"):
        env.action_selector.expert_help_ratio = 1.0

    # Solo nivel 3 requiere fijar pierna objetivo
    if level == 3 and target is not None and hasattr(env.simple_reward_system, "target_leg"):
        env.simple_reward_system.target_leg = target

    steps = int(seconds * env.frequency_simulation)
    total_r = 0.0
    for _ in range(steps):
        a = np.zeros(env.num_active_pams, dtype=np.float32)
        obs, r, done, _, info = env.step(a)
        total_r += r
        if done:
            break

    msg = f"[PROBE] level={level}"
    if level == 3 and target is not None:
        msg += f", leg={target}"
    msg += f", steps={info.get('step_count', 0)}, reward={total_r:.2f}"
    print(msg)


# Atajos específicos
def probe_level1_expert_rollout(env, seconds=3.0):
    probe_expert_rollout(env, level=1, seconds=seconds)

def probe_level3_expert_rollout(env, seconds=3.0, target='right'):
    probe_expert_rollout(env, level=3, seconds=seconds, target=target)