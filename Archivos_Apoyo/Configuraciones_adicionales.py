import os

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
import pybullet as p
from pathlib import Path
from enum import Enum
from os import listdir

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

def buscar_archivo(nombre_archivo, ruta_base="/", select_multiple=False):
    for root, dirs, files in os.walk(ruta_base):
        if select_multiple:
            similar_files=[]
            for f in files:
                if nombre_archivo in f:
                    similar_files.append(os.path.join(root, f))
            return similar_files
        else:
            if nombre_archivo in files:
                return os.path.join(root, nombre_archivo)
    return None

def find_project_root(marker_files=(".git", "pyproject.toml", "setup.py", "requirements.txt", ".gitignore")) -> Path:
    try:
        base = Path(__file__).resolve().parent  # funciona en scripts/módulos
    except NameError:
        base = Path.cwd().resolve()             # fallback para Jupyter/REPL

    for parent in [base] + list(base.parents):
        if any((parent / m).exists() for m in marker_files):
            return parent
    return base

class Rutas_Archivos(Enum):
    """
    Contiene los nombres de las rutas de los archivos a usar sin extensiones
    """
    _ignore_ = ['archivo', 'tmp_rutas', 'ruta_proyecto']
    ruta_proyecto=find_project_root()
    rutas_robots = {}
    for archivo in listdir("Robots_Versiones"):
        rutas_robots[archivo] = buscar_archivo(archivo, ruta_proyecto)
    #ruta_KREI=buscar_archivo("WP5 RURACTIVE KREI List.xlsx", ruta_proyecto)


# ===================================================================================================================================================================================================== #
# ===================================================== Métodos de impresión y logging de Enhanced_PAMIKBipedEnv ====================================================================================== #
# ===================================================================================================================================================================================================== #


def PAM_McKibben():

    return {
        # Caderas roll - Control antagónico completo
        'left_hip_roll_flexor': PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4), # probar con 0.035
        'left_hip_roll_extensor': PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4),# de 0.035 a 0.032
        'right_hip_roll_flexor': PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4), # de 0.03 a 0.04
        'right_hip_roll_extensor': PAMMcKibben(L0=0.6, r0=0.032, alpha0=np.pi/4),
        # caderas pitch
        'left_hip_pitch_flexor': PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4),
        'left_hip_pitch_extensor': PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4),
        'right_hip_pitch_flexor': PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4),
        'right_hip_pitch_extensor': PAMMcKibben(L0=0.6, r0=0.035, alpha0=np.pi/4),
        
        # Rodillas - Control antagónico completo
        'left_knee_flexor': PAMMcKibben(L0=0.5, r0=0.03, alpha0=np.pi/4),
        'left_knee_extensor': PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4),# de 0.04 a 0.036
        'right_knee_flexor': PAMMcKibben(L0=0.5, r0=0.03, alpha0=np.pi/4),
        'right_knee_extensor': PAMMcKibben(L0=0.5, r0=0.036, alpha0=np.pi/4),

        # Tobillos - Control antagónico completo
        'left_ankle_flexor': PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4),
        'left_ankle_extensor': PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4),
        'right_ankle_flexor': PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4),
        'right_ankle_extensor': PAMMcKibben(L0=0.3, r0=0.035, alpha0=np.pi/4),
    }
    
        # ---------- helper: co-contracción torque-neutral para cadera ----------
def split_cocontraction_torque_neutral(Fco: float, Rf: float, Re: float, R_min: float = 1e-3):
    """
    Reparte la co-fuerza basal Fco entre flexor/extensor de forma que
    Fco_flex * Rf == Fco_ext * Re  (no genera par neto).
    Mantiene Fco_flex + Fco_ext == Fco.
    """
    Rf = max(Rf, R_min); Re = max(Re, R_min)
    s = Rf + Re
    Fco_flex = Fco * (Re / s)
    Fco_ext  = Fco * (Rf / s)
    return Fco_flex, Fco_ext
    

# ========================================================================================================================================================================================= #
# ===================================================== Calculo de las presiones y torques de modelo ====================================================================================== #
# ========================================================================================================================================================================================= #
def eps_from(env,theta, R_abs,muscle_name):
        # θ0=0 como referencia; el método ya hace clip a [0, εmax]
        return env.pam_muscles[muscle_name].epsilon_from_angle(theta, 0.0, max(abs(R_abs), 1e-9))

def obtener_pam_forces_flexor_extensor(env, angulo_rad_articulacion, P, indice_flexor,indice_extensor,
                                       flexor_moment_function, extensor_moment_function):
    nombre_articulacion_flexor, nombre_articulacion_extensor=env.muscle_names[indice_flexor], env.muscle_names[indice_extensor]
    #Momentos de brazos de articulación
    # R_flexion_articulacion = env.hip_flexor_moment_arm(angulo_rad_articulacion)
    # R_extension_articulacion  = env.hip_extensor_moment_arm(angulo_rad_articulacion)
    R_flexion_articulacion = flexor_moment_function(angulo_rad_articulacion)
    R_extension_articulacion  = extensor_moment_function(angulo_rad_articulacion)
    eps_flex_L = eps_from(env,angulo_rad_articulacion, R_flexion_articulacion,nombre_articulacion_flexor)
    eps_ext_L  = eps_from(env, angulo_rad_articulacion, R_extension_articulacion, nombre_articulacion_extensor)
    # pam_forces[0] = env.pam_muscles[flexor_cadera_L].force_model_new(P[0], eps_flex_L)  # flexor L
    # pam_forces[1] = env.pam_muscles[extensor_cadera_L].force_model_new(P[1], eps_ext_L)   # extensor L
    pam_forces_flexor=env.pam_muscles[nombre_articulacion_flexor].force_model_new(P[indice_flexor], eps_flex_L)
    pam_forces_extensor =env.pam_muscles[nombre_articulacion_extensor].force_model_new(P[indice_extensor], eps_ext_L)
    return pam_forces_flexor, pam_forces_extensor, R_flexion_articulacion, R_extension_articulacion


def calculate_robot_specific_joint_torques_16_pam(env, pam_pressures):
    
    """
        Calcular torques básicos de articulaciones desde presiones PAM.
        
        Este método reemplaza la parte inicial de _apply_pam_forces
        antes del control automático de rodilla.
    """
    
    # Obtener estados articulares (solo joints activos: caderas y rodillas)
    joint_states = p.getJointStates(env.robot_id, env.joint_indices)  
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [s[1] for s in joint_states]
    
    # Calcular fuerzas PAM reales
    pam_forces = np.zeros(env.num_active_pams, dtype=float)
    
    P = np.array([env.pam_muscles[muscle_names].real_pressure_PAM(u) for muscle_names,u 
                    in zip(env.muscle_names, pam_pressures)], dtype=float)

    # Cadera izquierda roll
    pam_forces[0], pam_forces[1], R_flex_roll_L, R_ext_roll_L=obtener_pam_forces_flexor_extensor(env, joint_positions[0], P, 0, 1,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)

    # Cadera derecha roll
    pam_forces[2], pam_forces[3], R_flex_roll_R, R_ext_roll_R=obtener_pam_forces_flexor_extensor(env, joint_positions[4], P, 2, 3,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)
    # Cadera izquierda pitch
    pam_forces[4], pam_forces[5], R_flex_pitch_L, R_ext_pitch_L=obtener_pam_forces_flexor_extensor(env, joint_positions[1], P, 4, 5,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Cadera derecha pitch 
    pam_forces[6], pam_forces[7], R_flex_pitch_R, R_ext_pitch_R=obtener_pam_forces_flexor_extensor(env, joint_positions[5], P, 6, 7,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Rodilla izquierda 
    pam_forces[8], pam_forces[9], R_knee_flex_L, R_knee_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[2], P, 8, 9,
                                                                                     env.knee_flexor_moment_arm,
                                                                                     env.knee_extensor_moment_arm)

    # Rodilla derecha
    pam_forces[10], pam_forces[11], R_knee_flex_R, R_knee_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[6], P, 10, 11,
                                                                                                 env.knee_flexor_moment_arm,
                                                                                                 env.knee_extensor_moment_arm)
    
    # Tobillo izquierdo 
    pam_forces[12], pam_forces[13], R_ankle_flex_L, R_ankle_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[3], P, 12, 13,
                                                                                                    env.ankle_flexor_moment_arm,
                                                                                                    env.ankle_extensor_moment_arm)

    # Tobillo derecho
    pam_forces[14], pam_forces[15], R_ankle_flex_R, R_ankle_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[7], P, 14, 15,
                                                                                                    env.ankle_flexor_moment_arm,
                                                                                                    env.ankle_extensor_moment_arm)

    # Convertir a torques articulares
    joint_torques = np.zeros(len(joint_states))
    # TODO Ver que cambios tengo que aplicar para los torques ya queno tiene por que tener mismo torque en x que en y
    # Cadera izquierda roll: flexión positiva por flexor, extensión por extensor
    joint_torques[0] = ( pam_forces[0] * R_flex_roll_L) + (-pam_forces[1] * R_ext_roll_L)
    # Cadera izquierda_pitch: flexión positiva por flexor, extensión por extensor
    joint_torques[1] = ( pam_forces[4] * R_flex_pitch_L) + (-pam_forces[5] * R_ext_pitch_L)

    # Rodilla izquierda: flexor + resorte/damping pasivos
    joint_torques[2] = (pam_forces[8] * R_knee_flex_L) + (-pam_forces[9] * R_knee_ext_L)
    # tobillo izquierda: flexor + resorte/damping pasivos
    joint_torques[3] = (pam_forces[12] * R_ankle_flex_L) + (-pam_forces[13] * R_ankle_ext_L)
    # Cadera derecha roll
    joint_torques[4] = ( pam_forces[2] * R_flex_roll_R) + (-pam_forces[3] * R_ext_roll_R)
    # Cadera derecha_pitch
    joint_torques[5] = ( pam_forces[6] * R_flex_pitch_R) + (-pam_forces[7] * R_ext_pitch_R)
    # Rodilla derecha
    joint_torques[6] = (pam_forces[10] * R_knee_flex_R) + (-pam_forces[11] * R_knee_ext_R) 
    # tobillo derecha
    joint_torques[7] = (pam_forces[14] * R_ankle_flex_R) + (-pam_forces[15] * R_ankle_ext_R) 

    # PAra reducir el movimiento agresivo del robot
    # # Para rodillas
    # joint_torques[2] -= env.DAMPING_COEFFICIENT * joint_velocities[2]
    # joint_torques[6] -= env.DAMPING_COEFFICIENT * joint_velocities[6]
    # #Para tobillos
    # joint_torques[3] -= 0.5*env.DAMPING_COEFFICIENT * joint_velocities[3]
    # joint_torques[7] -= 0.5*env.DAMPING_COEFFICIENT * joint_velocities[7]
    joint_torques = np.clip(joint_torques, -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE)

    # ===== PASO 6: ACTUALIZAR ESTADOS PARA DEBUGGING =====
    
    env.pam_states = {
        'pressures': pam_pressures.copy(),
        'forces': np.abs(pam_forces),
        'raw_forces': pam_forces,
        'joint_torques': joint_torques.copy(),
        'moment_arms': {
             # Cadera eje y
            'left_hip_roll_flexor': R_flex_roll_L,
            'left_hip_roll_extensor': R_ext_roll_L,
            'right_hip_roll_flexor': R_flex_roll_R,
            'right_hip_roll_extensor': R_ext_roll_R,
            #
            'left_hip_pitch_flexor': R_flex_pitch_L,
            'left_hip_pitch_extensor': R_ext_pitch_L,
            'right_hip_pitch_flexor': R_flex_pitch_R,
            'right_hip_pitch_extensor': R_ext_pitch_R,

            'left_knee_flexor': R_knee_flex_L,
            'left_knee_extensor': R_knee_ext_L,
            'right_knee_flexor': R_knee_flex_R,
            'right_knee_extensor': R_knee_ext_R,

            'left_ankle_flexor': R_ankle_flex_L,
            'left_ankle_extensor': R_ankle_ext_L,
            'right_ankle_flexor': R_ankle_flex_R,
            'right_ankle_extensor': R_ankle_ext_R
        },
        'inhibition_applied': False,
        'robot_specific_params': True
    }
    
    return joint_torques


def apply_reciprocal_inhibition(flexor_force, extensor_force, INHIBITION_FACTOR):
    """
        Inhibición recíproca calibrada para tu robot.
        Basada en estudios neurológicos: cuando un músculo se activa fuerte,
        el sistema nervioso inhibe parcialmente su antagonista.
    """
    total_activation = flexor_force + extensor_force
    if total_activation > 0:
        # Reducir la fuerza del músculo menos activo
        flexor_ratio = flexor_force / total_activation
        extensor_ratio = extensor_force / total_activation

        if flexor_ratio > 0.6:
            extensor_force *= (1.0 - INHIBITION_FACTOR * flexor_ratio)
        elif extensor_ratio > 0.6:
            flexor_force *= (1.0 - INHIBITION_FACTOR * extensor_ratio)
    
    return flexor_force, extensor_force