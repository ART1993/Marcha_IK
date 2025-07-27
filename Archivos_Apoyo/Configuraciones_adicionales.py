import os
from stable_baselines3.common.vec_env import VecNormalize


def cargar_posible_normalizacion(model_dir, resume_path, config, train_env):
        """Load normalization statistics if they exist"""
        if resume_path and isinstance(train_env, VecNormalize):
            norm_path = os.path.join(model_dir, f"{config['model_prefix']}_normalize.pkl")
            if os.path.exists(norm_path):
                print(f"📊 Loading normalization statistics from: {norm_path}")
                try:
                    # Load normalization statistics
                    train_env = VecNormalize.load(norm_path, train_env)
                    # Keep normalization training active
                    train_env.training = True
                    train_env.norm_reward = True
                except Exception as e:
                    print(f"⚠️ Warning: Could not load normalization stats: {e}")
                    print("   Continuing with fresh normalization...")
        return train_env

def set_env_phase(env_wrapper, phase, phase_timesteps):
    if hasattr(env_wrapper, 'envs'):
        for env in env_wrapper.envs:
            base_env = env.env if hasattr(env, 'env') else env
            if hasattr(base_env, 'set_training_phase'):
                base_env.set_training_phase(phase, phase_timesteps)
    else:
        if hasattr(env_wrapper, 'set_training_phase'):
            env_wrapper.set_training_phase(phase, phase_timesteps)

def phase_trainig_preparations(model_dir, train_env, eval_env, current_timesteps,
                                model, callbacks, phase_timesteps, config, num_phase:int):
    # Configurar entornos para usar ciclo base
    set_env_phase(train_env, num_phase, phase_timesteps)
    set_env_phase(eval_env, num_phase, phase_timesteps)
    if num_phase >1:
        model.learn(
            total_timesteps=phase_timesteps,
            callback=callbacks,
            tb_log_name=f"{config['model_prefix']}_training",
            reset_num_timesteps=False
        )
    else:
        model.learn(
            total_timesteps=phase_timesteps,
            callback=callbacks,
            tb_log_name=f"{config['model_prefix']}_training")
    
    current_timesteps += phase_timesteps

    # Guardar modelo de fase i
    phase_path = os.path.join(model_dir, f"{config['model_prefix']}_phase1")
    model.save(phase_path)
    print(f"✅ Phase {num_phase} model saved at: {phase_path}")

    return model, current_timesteps, phase_timesteps