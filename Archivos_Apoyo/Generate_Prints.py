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

def set_training_phase_print(phase, use_walking_cycle, imitation_weight):
    print(f"   🎮 Environment configuration for phase {phase}:")
    print(f"      Use walking cycle: {use_walking_cycle}")
    print(f"      Imitation weight: {imitation_weight}")
    print(f"   🏁 Phase {phase} configuration completed\n")


def log_integration_status(self, reward, reward_components):
    """
    Log de estado de integración para debugging y análisis.
    
    Este método es como el "informe de progreso" que un entrenador da
    periódicamente para mostrar cómo va el entrenamiento del atleta.
    """
    
    if self.step_count % 500 == 0:  # Log cada 500 pasos (menos frecuente)
        print(f"\n📊 Integration Status Report (Step {self.step_count})")
        print(f"   Total reward: {reward:.3f}")
        
        # Mostrar componentes de recompensa
        if reward_components:
            print("   Reward breakdown:")
            for component, value in reward_components.items():
                print(f"      {component}: {value:.3f}")
        
        # Estado de los PAMs
        if hasattr(self, 'pam_states') and self.pam_states is not None:
            pressures = self.pam_states['pressures']
            normalized_pressures = pressures / self.max_pressure
            
            print("   PAM status:")
            muscle_names = ['L_hip_flex', 'L_hip_ext', 'R_hip_flex', 'R_hip_ext', 'L_knee', 'R_knee']
            for i, (name, pressure) in enumerate(zip(muscle_names, normalized_pressures)):
                print(f"      {name}: {pressure:.2f}")
        
        # Estado del currículo
        curriculum_info = self.reward_system_integration_status
        print(f"   Curriculum phase: {curriculum_info['curriculum_phase']}")
        print(f"   Imitation weight: {self.imitation_weight}")
        
        # Métricas de coordinación si están disponibles
        if hasattr(self.sistema_recompensas, 'coordination_metrics'):
            coords = self.sistema_recompensas.coordination_metrics
            print("   Coordination metrics:")
            for metric, value in coords.items():
                if value != 0.0:  # Solo mostrar métricas activas
                    print(f"      {metric}: {value:.3f}")
        
        print("   ✅ Integration status: HEALTHY\n")

# =========================================================================================================================================================================== #
# ===================================================== Métodos de trayectorias y curvas Bézier caduco ====================================================================== #
# =========================================================================================================================================================================== #

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