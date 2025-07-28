import numpy as np
import pybullet as p

def log_training_plan(resume_timesteps, remaining_timesteps, total_timesteps):
    print(f"🎯 Training plan:")
    print(f"   - Completed: {resume_timesteps:,} timesteps")
    print(f"   - Remaining: {remaining_timesteps:,} timesteps")
    print(f"   - Total target: {total_timesteps:,} timesteps")

def error_load_model(e):
    print(f"❌ Error loading model: {e}")
    print("🔄 Creating new model instead...")


def info_latest_checkpoint(latest_checkpoint, resume_timesteps):
    if latest_checkpoint:
        resume_path = latest_checkpoint
        print(f"📂 Found latest checkpoint: {latest_checkpoint}")
        print(f"📊 Resuming from timestep: {resume_timesteps:,}")
        return resume_path
    else:
        print("📝 No previous checkpoints found, starting fresh training")


def print_info_env_pam(self):
    if self.env_type == 'pam':
        print("📊 PAM Training Features:")
        print("   - Hybrid IK + PAM control")
        print("   - Realistic tensile forces")
        print("   - LSTM memory for muscle coordination")
        print("   - Energy efficiency optimization")
    
    # Create environments
    print(f"📦 Creating {self.n_envs} training environments...")
    # train_env=self.create_training_env_with_walking_cycle()

def setup_walking_cycle_config(self):
        """Configura parámetros del ciclo de paso"""
        self.walking_cycle_config = {
            'phase1_ratio': 0.3,    # 30% del tiempo con ciclo base
            'phase2_ratio': 0.4,    # 40% con modulación
            'phase3_ratio': 0.3,    # 30% control libre
            'modulation_factor': 0.3,  # Intensidad de modulación en fase 2
            'enable_gradual_transition': True  # Transición gradual entre fases
        }

def create_training_env_with_walking_cycle(self):
    # De momento no lo uso, pero lo dejo por si acaso
    """Versión modificada del create_training_env que incluye configuración del ciclo"""
    # Tu código actual de create_training_env
    env = self.create_training_env()
    
    # Añadir configuración del ciclo
    if hasattr(env, 'envs'):
        for single_env in env.envs:
            base_env = single_env.env if hasattr(single_env, 'env') else single_env
            if hasattr(base_env, 'setup_walking_cycle_config'):
                base_env.setup_walking_cycle_config(self.walking_cycle_config)

    return env


def foot_bezier_parabola(start, end, ctrl1, ctrl2, alpha, height):
    """
        Trayectoria mixta: curva Bézier para (x, y), parábola para z.
        - start, end: posiciones 3D (inicio y fin del swing)
        - ctrl1, ctrl2: puntos de control Bézier (3D)
        - alpha: progreso de swing [0, 1]
        - height: altura máxima de swing (aplica a z)
    """
    # Bézier para X e Y (puedes usarlo también para Z base si quieres)
    def bezier(p0, p1, p2, p3, t):
        return ((1 - t)**3) * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    # Bézier en x, y (z puede quedarse lineal o usar solo inicio y fin)
    x = bezier(start[0], ctrl1[0], ctrl2[0], end[0], alpha)
    y = bezier(start[1], ctrl1[1], ctrl2[1], end[1], alpha)

    # Parábola para z
    z_base = (1 - alpha) * start[2] + alpha * end[2]
    z = z_base + height * 4 * alpha * (1 - alpha)

    return [x, y, z]