import os

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize

from Archivos_Apoyo.dinamica_pam import PAMMcKibben


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
    if num_phase >0:
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

# ===================================================================================================================================================================================================== #
# ===================================================== Métodos de impresión y logging de Enhanced_PAMIKBipedEnv ====================================================================================== #
# ===================================================================================================================================================================================================== #

def joint_forces():
    """Define las fuerzas específicas para cada articulación del robot bípedo."""
    # Fuerzas específicas para cada articulación (en Newtons)
    return {
        'left_hip_joint': 150,    # Cadera necesita más fuerza
        'left_knee_joint': 120,   # Rodilla fuerza moderada
        'right_hip_joint': 150,
        'right_knee_joint': 120,
    }

def foot_workspace():
    """Define el espacio de trabajo de los pies del robot bípedo."""
    return {
        'x_range': (-1.2, 1.2),    # Adelante/atrás
        'y_range': (-0.8, 0.8),    # Izquierda/derecha 
        'z_range': (-0.3, 1.0)     # Altura del suelo
    }

def reward_system_integration_status(curriculum_phase):
    """Check if the reward system integration is active."""
    return {
            'initialized': True,
            'pam_states_linked': False,
            'curriculum_phase': curriculum_phase,
            'last_reward_components': None
        }

def PAM_McKibben():
    return {
            # Caderas - Control antagónico completo
            'left_hip_flexor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),   # Más potente
            'left_hip_extensor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            'right_hip_flexor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            'right_hip_extensor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            
            # Rodillas - Solo flexores (extensión pasiva)
            'left_knee_flexor': PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4),
            'right_knee_flexor': PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4),
        }

def joint_limits():
    """Define los límites de las articulaciones del robot bípedo."""
    return {
            'left_hip_joint': (-1.2, 1.0),      # Rango ampliado para flexión/extensión
            'left_knee_joint': (0.0, 1.571),    # Solo flexión (extensión por resorte)
            'right_hip_joint': (-1.2, 1.0),
            'right_knee_joint': (0.0, 1.571),
            'left_ankle_joint': (-0.5, 0.5),    # Controlado por resortes
            'right_ankle_joint': (-0.5, 0.5),
        }

def pam_states_init():
    """Initialize the PAM states for the robot."""
    return {
            'pressures': np.zeros(6),
            'contractions': np.zeros(6),
            'forces': np.zeros(6)
        }

def setup_passive_elements():
        """Configurar parámetros de elementos pasivos (resortes)"""
        
        return {
            # Resortes extensores de rodilla (devuelven la rodilla a posición neutra)
            'left_knee_extensor': {
                'k_spring': 15.0,      # Rigidez del resorte (Nm/rad)
                'rest_angle': 0.1,     # Ángulo de reposo (ligeramente flexionado)
                'damping': 2.0         # Amortiguación
            },
            'right_knee_extensor': {
                'k_spring': 15.0,
                'rest_angle': 0.1, 
                'damping': 2.0
            },
            
            # Resortes de tobillo (estabilización pasiva) # ¿Debería de usarse para el flexor y el extensor?
            'left_ankle_spring': {
                'k_spring': 8.0,       # Más suave para permitir adaptación al terreno
                'rest_angle': 0.0,     # Pie horizontal
                'damping': 1.5
            },
            'right_ankle_spring': {
                'k_spring': 8.0,
                'rest_angle': 0.0,
                'damping': 1.5
            }
        }

def relacionar_musculo_con_joint(muscle_name, joint_data):
        """
            Se usa para relacionar musculos flexores y extensores con sus correspondientes articulaciones.
            Devuelve el ID de la articulación y su estado actual.
        """
        # ===== PASO 1: DETERMINAR QUÉ ARTICULACIÓN AFECTA ESTE MÚSCULO =====
            
        if muscle_name in ['left_hip_flexor', 'left_hip_extensor']:
            joint_id = 0  # left_hip_joint
            joint_state = joint_data[0]
        elif muscle_name in ['right_hip_flexor', 'right_hip_extensor']:
            joint_id = 3  # right_hip_joint (en PyBullet)
            joint_state = joint_data[3]
        elif muscle_name == 'left_knee_flexor':
            joint_id = 1  # left_knee_joint
            joint_state = joint_data[1]
        elif muscle_name == 'right_knee_flexor':
            joint_id = 4  # right_knee_joint (en PyBullet)
            joint_state = joint_data[4]
        else:
            raise ValueError(f"Unknown muscle name: {muscle_name}")

        return joint_id, joint_state

def calculate_joint_torques_from_pam_forces(pam_forces):
    """
    Convierte fuerzas PAM individuales en torques articulares netos
    considerando pares antagónicos
    """
    joint_torques = []
    
    # Cadera izquierda: combinar flexor y extensor
    left_hip_torque = pam_forces[0] + pam_forces[1]  # flexor + extensor (extensor ya es negativo)
    joint_torques.append(left_hip_torque)
    
    # Rodilla izquierda: solo flexor
    left_knee_torque = pam_forces[4]
    joint_torques.append(left_knee_torque)
    
    # Cadera derecha: combinar flexor y extensor  
    right_hip_torque = pam_forces[2] + pam_forces[3]
    joint_torques.append(right_hip_torque)
    
    # Rodilla derecha: solo flexor
    right_knee_torque = pam_forces[5]
    joint_torques.append(right_knee_torque)
    
    return joint_torques

def expected_joints_fun():
    return {
            'left_hip': 0,
            'left_knee': 1, 
            'right_hip': 3,
            'right_knee': 4
            }

def phase_configs_method():
    return {
            0: {'imitation_weight': 0.0, 'use_walking_cycle': False},    # Equilibrio libre
            1: {'imitation_weight': 0.9, 'use_walking_cycle': True},     # Imitación equilibrio
            2: {'imitation_weight': 0.3, 'use_walking_cycle': True},     # Exploración equilibrio
            3: {'imitation_weight': 0.85, 'use_walking_cycle': True},    # Imitación sentadilla parcial
            4: {'imitation_weight': 0.5, 'use_walking_cycle': True},     # Exploración sentadilla
            5: {'imitation_weight': 0.9, 'use_walking_cycle': True},     # Imitación levantar izq.
            6: {'imitation_weight': 0.6, 'use_walking_cycle': True},     # Exploración levantar izq.
            7: {'imitation_weight': 0.9, 'use_walking_cycle': True},     # Imitación levantar der.
            8: {'imitation_weight': 0.6, 'use_walking_cycle': True},     # Exploración levantar der.
            9: {'imitation_weight': 0.8, 'use_walking_cycle': True},     # Imitación paso izq.
            10: {'imitation_weight': 0.5, 'use_walking_cycle': True},    # Exploración paso izq.
            11: {'imitation_weight': 0.8, 'use_walking_cycle': True},    # Imitación paso der.
            12: {'imitation_weight': 0.2, 'use_walking_cycle': True},    # Maestría bilateral
        }