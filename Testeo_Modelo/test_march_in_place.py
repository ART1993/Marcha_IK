# test_march_in_place.py
import os, numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv
from Archivos_Apoyo.CSVLogger import CSVLogger

MODEL_DIR = "./models_lift_leg"
BEST_PATH = os.path.join(MODEL_DIR, "best_model.zip")
NORM_PATH = os.path.join(MODEL_DIR, "single_leg_balance_pam_best_normalize.pkl")

def make_env(render=True, robot_name="2_legged_human_like_robot20DOF", simple_reward_mode='march_in_place'):
    csvlog = CSVLogger(only_workers=False)  # escribe desde el main
    env = Simple_Lift_Leg_BipedEnv(
        render_mode='human' if render else 'direct',
        print_env="TEST",
        simple_reward_mode=simple_reward_mode,  # 👈 modo de marcha en el sitio
        allow_hops=True,                      # 👈 vuelos permitidos
        vx_target=0.0,                        # 👈 sin avance longitudinal
        csvlog=csvlog,
        robot_name=robot_name
    )
    return env

def run_test(episodes=5, render=True, max_steps=6000, best_model=BEST_PATH,
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

    # 4) Rollouts deterministas
    for ep in range(episodes):
        obs = base.reset()
        state = None
        episode_starts = np.ones((base.num_envs,), dtype=bool)
        ep_ret = 0.0
        for t in range(max_steps):
            action, state = model.predict(
                obs, state=state, episode_start=episode_starts, deterministic=True
            )
            obs, reward, dones, infos = base.step(action)
            episode_starts = dones
            ep_ret += float(reward[0])

            if dones[0]:
                reason = infos[0].get("ep_kpi", {}).get("done_reason")
                print(f"EP {ep+1}: return={ep_ret:.1f} | len={t+1} | reason={reason}")
                break

    base.close()

if __name__ == "__main__":
    run_test(episodes=5, render=True)
