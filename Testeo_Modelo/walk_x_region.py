# test_march_in_place.py
import os, numpy as np
import math
import pybullet as p
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv
from Archivos_Apoyo.CSVLogger import CSVLogger

MODEL_DIR = "./models_lift_leg"
BEST_PATH = os.path.join(MODEL_DIR, "best_model.zip")
NORM_PATH = os.path.join(MODEL_DIR, "Walker_6DOF_3D_normalize.pkl")

# --- Estado de cámara (suavizado) ---
_CAM_TARGET = None
_CAM_ALPHA = 0.15  # 0..1; mayor = menos suavizado

def _get_base_env(vec_or_norm):
    """Devuelve el env base aunque esté envuelto por VecNormalize."""
    return vec_or_norm.venv.envs[0] if hasattr(vec_or_norm, "venv") else vec_or_norm.envs[0]

def _update_camera_follow(base_env, dist=2.0, pitch=-30.0, yaw_offset=0.0):
    """
    Coloca la cámara detrás/encima de la pelvis y la suaviza levemente.
    - dist: distancia de cámara
    - pitch: inclinación (grados; negativa = mirando hacia abajo)
    - yaw_offset: offset sobre el heading del robot
    """
    global _CAM_TARGET
    # Pose de referencia (pelvis/base) que expone tu env
    pos = base_env.pos                # (x, y, z)
    roll, pitch_body, yaw_body = base_env.euler

    # Punto objetivo ligeramente por encima de la pelvis
    target = np.array([pos[0], pos[1], pos[2] + 0.35], dtype=float)
    if _CAM_TARGET is None:
        _CAM_TARGET = target
    else:
        _CAM_TARGET = (1.0 - _CAM_ALPHA) * _CAM_TARGET + _CAM_ALPHA * target

    # Yaw mirando en la dirección del cuerpo
    yaw_deg = math.degrees(yaw_body) + yaw_offset

    # Aplicar en la GUI de PyBullet
    p.resetDebugVisualizerCamera(
        cameraDistance=float(dist),
        cameraYaw=float(yaw_deg),
        cameraPitch=float(pitch),
        cameraTargetPosition=_CAM_TARGET.tolist(),
    )

def make_env(render=True, robot_name="2_legged_human_like_robot20DOF", 
             simple_reward_mode='march_in_place', vx=1.2):
    csvlog = CSVLogger(only_workers=False)  # escribe desde el main
    env = Simple_Lift_Leg_BipedEnv(
        render_mode='human' if render else 'direct',
        print_env="TEST",
        simple_reward_mode=simple_reward_mode,  # 👈 modo de marcha en el sitio
        allow_hops=True,                      # 👈 vuelos permitidos
        vx_target=vx,                        # 👈 sin avance longitudinal
        csvlog=csvlog,
        robot_name=robot_name
    )
    return env

def run_test(episodes=5, render=True, max_steps=6000, best_model=BEST_PATH,deterministic=True,
             robot_name="2_legged_human_like_robot20DOF", simple_reward_mode='march_in_place'):
    # 1) Crear env
    base = DummyVecEnv([lambda: make_env(render=render, robot_name=robot_name, simple_reward_mode=simple_reward_mode)])

    # 2) Cargar normalización si existe
    if os.path.exists(NORM_PATH):
        base = VecNormalize.load(NORM_PATH, base)
        base.training = False
        base.norm_reward = False

    # 3) Cargar modelo
    model = RecurrentPPO.load(best_model, env=base, device='auto')
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0) 
    # 4) Rollouts deterministas
    for ep in range(episodes):
        obs = base.reset()
        state = None
        episode_starts = np.ones((base.num_envs,), dtype=bool)
        ep_ret = 0.0
        for t in range(max_steps):
            action, state = model.predict(
                obs, state=state, episode_start=episode_starts, deterministic=deterministic
            )
            obs, reward, dones, infos = base.step(action)
            episode_starts = dones
            ep_ret += float(reward[0])
            # --- Seguimiento de cámara (solo si GUI) ---
            if render:
                base_env = _get_base_env(base)
                _update_camera_follow(base_env, dist=2.0, pitch=-30.0, yaw_offset=0.0)

            if dones[0]:
                reason = infos[0].get("ep_kpi", {}).get("done_reason")
                print(f"EP {ep+1}: return={ep_ret:.1f} | len={t+1} | reason={reason}")
                break

    base.close()

if __name__ == "__main__":
    run_test(episodes=5, render=True)
