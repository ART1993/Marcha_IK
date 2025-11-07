import os, re

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
import pybullet as p
from pathlib import Path
from enum import Enum
from os import listdir

from Archivos_Apoyo.SIstemasPamRobot import Sistema_Musculos_PAM_16, Sistema_Musculos_PAM_20, Sistema_Musculos_PAM_12, Sistema_Musculos_PAM_12_done




def cargar_posible_normalizacion(model_dir, resume_path, config, train_env):
        """Load normalization statistics if they exist"""
        if resume_path and isinstance(train_env, VecNormalize):
            prefix = config.get("model_prefix", "rl_model")
            steps_match = re.search(r"_(\d+)_steps\.zip$", os.path.basename(resume_path))
            candidate_paths = []
            if steps_match:
                steps = steps_match.group(1)
                candidate_paths.append(os.path.join(model_dir, f"{prefix}_normalize_{steps}_steps.pkl"))
            else:
                # Fallback generico, por si por motivos extraños no se encuentra el normalize
                norm_path = os.path.join(model_dir, f"{config['model_prefix']}_normalize.pkl")
                candidate_paths.append(norm_path)
            if os.path.exists(candidate_paths[0]):
                print(f"📊 Loading normalization statistics from: {candidate_paths[0]}")
                try:
                    # Load normalization statistics
                    train_env = VecNormalize.load(candidate_paths[0], train_env)
                    # Keep normalization training active
                    train_env.training = True
                    train_env.norm_reward = True
                except Exception as e:
                    print(f"⚠️ Warning: Could not load normalization stats: {e}")
                    print("   Continuing with fresh normalization...")
        return train_env

def cargar_posible_normalizacion_nuevo(model_dir, resume_path, config, train_env):
    """Carga SOLO el normalize emparejado con el checkpoint; si falla, usa el normalize final."""
    if not resume_path or not isinstance(train_env, VecNormalize):
        return train_env

    prefix = config.get("model_prefix", "rl_model")
    # 1) Construye candidatos en orden de prioridad
    cand = []

    # a) Emparejado por steps si el checkpoint lo tiene
    m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(resume_path))
    if m:
        steps = m.group(1)
        cand.append(os.path.join(model_dir, f"{prefix}_normalize_{steps}_steps.pkl"))

    # b) Fallback final
    cand.append(os.path.join(model_dir, f"{prefix}_normalize.pkl"))

    # 2) Intenta primero el emparejado (si hay), luego el final
    last_error = None
    for norm_path in cand:
        if not os.path.exists(norm_path):
            continue
        print(f"📊 Loading normalization statistics from: {norm_path}")
        try:
            train_env = VecNormalize.load(norm_path, train_env)
            # En TRAIN queremos seguir actualizando stats y normalizando recompensas
            train_env.training = True
            train_env.norm_reward = True
            print("✅ VecNormalize loaded.")
            return train_env
        except Exception as e:
            last_error = e
            print(f"⚠️ Warning: Could not load normalization stats from {norm_path}: {e}")

    # 3) Si no hubo éxito, seguimos con stats frescas
    if last_error:
        print("   Continuing with fresh normalization (no matching/fallback stats loaded).")
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
    rutas_jsons = {}
    for archivo in listdir("Robots_Versiones"):
        rutas_robots[archivo.split(".")[0]] = buscar_archivo(archivo, ruta_proyecto)
        rutas_jsons[archivo.split(".")[0]] = buscar_archivo(archivo.replace(".urdf",".json"), ruta_proyecto)
    #ruta_KREI=buscar_archivo("WP5 RURACTIVE KREI List.xlsx", ruta_proyecto)


# ===================================================================================================================================================================================================== #
# ===================================================== Métodos de impresión y logging de Enhanced_PAMIKBipedEnv ====================================================================================== #
# ===================================================================================================================================================================================================== #


def PAM_McKibben(robot_name="2_legged_human_like_robot16DOF", control_joint_names=None, max_pressure=5):
    if "2_legged_human_like_robot16DOF" in robot_name:
        return Sistema_Musculos_PAM_16(control_joint_names,max_pressure)
    elif "2_legged_human_like_robot20DOF" in robot_name:
        return Sistema_Musculos_PAM_20(control_joint_names,max_pressure)
    elif "2_legged_human_like_robot12DOF" in robot_name:
        return Sistema_Musculos_PAM_12(control_joint_names,max_pressure)
    #elif "2_legged_human_like_robot12DOF_done" in robot_name:
    #    return Sistema_Musculos_PAM_12_done(control_joint_names)
    else:
        raise ValueError(f"Robot '{robot_name}' no soportado para sistema PAM.")
    
    
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
    R_flexion_articulacion = flexor_moment_function(angulo_rad_articulacion)
    R_extension_articulacion  = extensor_moment_function(angulo_rad_articulacion)
    # epsilon
    eps_flex_L = eps_from(env,angulo_rad_articulacion, R_flexion_articulacion,nombre_articulacion_flexor)
    eps_ext_L  = eps_from(env, angulo_rad_articulacion, R_extension_articulacion, nombre_articulacion_extensor)
    # pam forces
    pam_forces_flexor=env.pam_muscles[nombre_articulacion_flexor].force_model_new(P[indice_flexor], eps_flex_L)
    pam_forces_extensor =env.pam_muscles[nombre_articulacion_extensor].force_model_new(P[indice_extensor], eps_ext_L)
    return pam_forces_flexor, pam_forces_extensor, R_flexion_articulacion, R_extension_articulacion


def calculate_robot_specific_joint_torques_12_pam(env, pam_pressures):
    
    """
        Calcular torques básicos de articulaciones desde presiones PAM.
        Se usa para robots de 12 PAMs
    """
    
    # Obtener estados articulares (solo joints activos: caderas y rodillas)
    joint_states = p.getJointStates(env.robot_id, env.joint_indices)  
    joint_positions = [state[0] for state in joint_states]
    
    # Calcular fuerzas PAM reales
    pam_forces = np.zeros(env.num_active_pams, dtype=float)
    
    # Ahora los nombres de los músculos siguen el orden del URDF
    P = np.array([env.pam_muscles[muscle_names].real_pressure_PAM(u) for muscle_names,u 
                    in zip(env.muscle_names, pam_pressures)], dtype=float)
    
    print(env.muscle_names)
    print(env.joint_names)
    # Cadera izquierda pitch
    pam_forces[0], pam_forces[1], R_flex_pitch_L, R_ext_pitch_L=obtener_pam_forces_flexor_extensor(env, joint_positions[0], P, 0, 1,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)
    
    # Rodilla izquierda 
    pam_forces[2], pam_forces[3], R_knee_flex_L, R_knee_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[1], P, 2, 3,
                                                                                     env.knee_flexor_moment_arm,
                                                                                     env.knee_extensor_moment_arm)
    
    # Tobillo izquierdo pitch
    pam_forces[4], pam_forces[5], R_ankle_pitch_flex_L, R_ankle_pitch_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[2], P, 4, 5,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)

    # Cadera derecha pitch 
    pam_forces[6], pam_forces[7], R_flex_pitch_R, R_ext_pitch_R=obtener_pam_forces_flexor_extensor(env, joint_positions[3], P, 6, 7,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Rodilla derecha
    pam_forces[8], pam_forces[9], R_knee_flex_R, R_knee_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[4], P, 8, 9,
                                                                                                 env.knee_flexor_moment_arm,
                                                                                                 env.knee_extensor_moment_arm)
    
    # Tobillo derecho pitch
    pam_forces[10], pam_forces[11], R_ankle_pitch_flex_R, R_ankle_pitch_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[5], P, 10, 11,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)
    
    assert env.num_active_pams == env.action_space.shape[0] == 12, \
    f"{env.num_active_pams=} {env.action_space.shape=}"
    
    # Convertir a torques articulares
    joint_torques = np.zeros(len(joint_states))
    # Cadera izquierda pitch: flexión positiva por flexor, extensión por extensor
    joint_torques[0] = ( pam_forces[0] * R_flex_pitch_L) + (-pam_forces[1] * R_ext_pitch_L)

    # Rodilla izquierda: flexor + resorte/damping pasivos
    joint_torques[1] = (pam_forces[2] * R_knee_flex_L) + (-pam_forces[3] * R_knee_ext_L)

    # tobillo izquierdo pitch: flexor + resorte/damping pasivos
    joint_torques[2] = (pam_forces[4] * R_ankle_pitch_flex_L) + (-pam_forces[5] * R_ankle_pitch_ext_L)

    # Cadera derecha roll
    joint_torques[3] = ( pam_forces[6] * R_flex_pitch_R) + (-pam_forces[7] * R_ext_pitch_R)

    # Rodilla derecha
    joint_torques[4] = (pam_forces[8] * R_knee_flex_R) + (-pam_forces[9] * R_knee_ext_R) 

    # tobillo derecho pitch
    joint_torques[5] = (pam_forces[10] * R_ankle_pitch_flex_R) + (-pam_forces[11] * R_ankle_pitch_ext_R)

    # joint_tau_max_force
    # ======= REEMPLAZO DE CLIP GLOBAL POR CLIP ANGULAR =======
    # if hasattr(env, "tau_limit_interp") and isinstance(env.tau_limit_interp, dict) and len(env.tau_limit_interp) > 0:
    #     # Usamos las posiciones articulares ya calculadas para interpolar τ_max(θ)
    #     for i, jid in enumerate(env.joint_indices):
    #         th_i = float(joint_positions[i])
    #         lims = env.tau_limit_interp.get(jid, None)
    #         if lims is not None:
    #             tau_flex_max = max(0.0, lims["flex"](th_i))
    #             tau_ext_max  = max(0.0, lims["ext"](th_i))
    #             joint_torques[i] = float(np.clip(joint_torques[i], -tau_ext_max, +tau_flex_max))
    #         else:
    #             joint_torques[i] = float(np.clip(joint_torques[i], -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE))
    # else:
    #     # en caso de que tau_limit_interp
    for i, index in enumerate(env.joint_indices):
        #torque_i=env.joint_tau_max_force[env.joint_indices[i]]
        joint_torques[i] = np.clip(joint_torques[i], -env.joint_tau_max_force[index], env.joint_tau_max_force[index])

    # ===== PASO 6: ACTUALIZAR ESTADOS PARA DEBUGGING =====
    env.pam_states = {
        'pressures': pam_pressures.copy(),
        'forces': np.abs(pam_forces),
        'raw_forces': pam_forces,
        'joint_torques': joint_torques.copy(),
        'moment_arms': {
            #
            'left_hip_pitch_flexor': R_flex_pitch_L,
            'left_hip_pitch_extensor': R_ext_pitch_L,
            'right_hip_pitch_flexor': R_flex_pitch_R,
            'right_hip_pitch_extensor': R_ext_pitch_R,

            'left_knee_flexor': R_knee_flex_L,
            'left_knee_extensor': R_knee_ext_L,
            'right_knee_flexor': R_knee_flex_R,
            'right_knee_extensor': R_knee_ext_R,

            'left_ankle_pitch_flexor': R_ankle_pitch_flex_L,
            'left_ankle_pitch_extensor': R_ankle_pitch_ext_L,
            'right_ankle_pitch_flexor': R_ankle_pitch_flex_R,
            'right_ankle_pitch_extensor': R_ankle_pitch_ext_R,
        },
        'inhibition_applied': False,
        'robot_specific_params': True
    }
    
    return np.array(joint_torques)


def calculate_robot_specific_joint_torques_16_pam(env, pam_pressures):
    
    """
        Calcular torques básicos de articulaciones desde presiones PAM.
        
        Este método reemplaza la parte inicial de _apply_pam_forces
        antes del control automático de rodilla.
    """
    
    # Obtener estados articulares (solo joints activos: caderas y rodillas)
    joint_states = p.getJointStates(env.robot_id, env.joint_indices)  
    joint_positions = [state[0] for state in joint_states]
    # joint_velocities = [s[1] for s in joint_states]
    
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
                                                                                                    env.ankle_roll_flexor_moment_arm,
                                                                                                    env.ankle_roll_extensor_moment_arm)

    # Tobillo derecho
    pam_forces[14], pam_forces[15], R_ankle_flex_R, R_ankle_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[7], P, 14, 15,
                                                                                                    env.ankle_roll_flexor_moment_arm,
                                                                                                    env.ankle_roll_extensor_moment_arm)

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

    # ======= REEMPLAZO DE CLIP GLOBAL POR CLIP ANGULAR =======
    # if hasattr(env, "tau_limit_interp") and isinstance(env.tau_limit_interp, dict) and len(env.tau_limit_interp) > 0:
    #     # Usamos las posiciones articulares ya calculadas para interpolar τ_max(θ)
    #     for i, jid in enumerate(env.joint_indices):
    #         th_i = float(joint_positions[i])
    #         lims = env.tau_limit_interp.get(jid, None)
    #         if lims is not None:
    #             tau_flex_max = max(0.0, lims["flex"](th_i))
    #             tau_ext_max  = max(0.0, lims["ext"](th_i))
    #             joint_torques[i] = float(np.clip(joint_torques[i], -tau_ext_max, +tau_flex_max))
    #         else:
    #             joint_torques[i] = float(np.clip(joint_torques[i], -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE))
    # else:
    for i, max_torque in enumerate(env.joint_tau_max_force.values()):
        joint_torques[i] = np.clip(joint_torques, -max_torque, max_torque)

    

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

def calculate_robot_specific_joint_torques_12_pam_old(env, pam_pressures):
    
    """
        Calcular torques básicos de articulaciones desde presiones PAM.
        Se usa para robots de 12 PAMs
    """
    
    # Obtener estados articulares (solo joints activos: caderas y rodillas)
    joint_states = p.getJointStates(env.robot_id, env.joint_indices)  
    joint_positions = [state[0] for state in joint_states]
    
    # Calcular fuerzas PAM reales
    pam_forces = np.zeros(env.num_active_pams, dtype=float)
    
    # Ahora los nombres de los músculos siguen el orden del URDF
    P = np.array([env.pam_muscles[muscle_names].real_pressure_PAM(u) for muscle_names,u 
                    in zip(env.muscle_names, pam_pressures)], dtype=float)
    
    # Cadera izquierda pitch
    pam_forces[0], pam_forces[1], R_flex_pitch_L, R_ext_pitch_L=obtener_pam_forces_flexor_extensor(env, joint_positions[0], P, 0, 1,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)
    
    # Rodilla izquierda 
    pam_forces[4], pam_forces[5], R_knee_flex_L, R_knee_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[1], P, 4, 5,
                                                                                     env.knee_flexor_moment_arm,
                                                                                     env.knee_extensor_moment_arm)
    
    # Tobillo izquierdo pitch
    pam_forces[8], pam_forces[9], R_ankle_pitch_flex_L, R_ankle_pitch_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[2], P, 8, 9,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)

    # Cadera derecha pitch 
    pam_forces[2], pam_forces[3], R_flex_pitch_R, R_ext_pitch_R=obtener_pam_forces_flexor_extensor(env, joint_positions[3], P, 2, 3,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Rodilla derecha
    pam_forces[6], pam_forces[7], R_knee_flex_R, R_knee_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[4], P, 6, 7,
                                                                                                 env.knee_flexor_moment_arm,
                                                                                                 env.knee_extensor_moment_arm)
    
    # Tobillo derecho pitch
    pam_forces[10], pam_forces[11], R_ankle_pitch_flex_R, R_ankle_pitch_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[5], P, 10, 11,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)
    
    assert env.num_active_pams == env.action_space.shape[0] == 12, \
    f"{env.num_active_pams=} {env.action_space.shape=}"
    
    # Convertir a torques articulares
    joint_torques = np.zeros(len(joint_states))
    # Cadera izquierda pitch: flexión positiva por flexor, extensión por extensor
    joint_torques[0] = ( pam_forces[0] * R_flex_pitch_L) + (-pam_forces[1] * R_ext_pitch_L)

    # Rodilla izquierda: flexor + resorte/damping pasivos
    joint_torques[1] = (pam_forces[4] * R_knee_flex_L) + (-pam_forces[5] * R_knee_ext_L)

    # tobillo izquierdo pitch: flexor + resorte/damping pasivos
    joint_torques[2] = (pam_forces[8] * R_ankle_pitch_flex_L) + (-pam_forces[9] * R_ankle_pitch_ext_L)

    # Cadera derecha roll
    joint_torques[3] = ( pam_forces[2] * R_flex_pitch_R) + (-pam_forces[3] * R_ext_pitch_R)

    # Rodilla derecha
    joint_torques[4] = (pam_forces[6] * R_knee_flex_R) + (-pam_forces[7] * R_knee_ext_R) 

    # tobillo derecho pitch
    joint_torques[5] = (pam_forces[10] * R_ankle_pitch_flex_R) + (-pam_forces[11] * R_ankle_pitch_ext_R)

    # joint_tau_max_force
    # ======= REEMPLAZO DE CLIP GLOBAL POR CLIP ANGULAR =======
    # if hasattr(env, "tau_limit_interp") and isinstance(env.tau_limit_interp, dict) and len(env.tau_limit_interp) > 0:
    #     # Usamos las posiciones articulares ya calculadas para interpolar τ_max(θ)
    #     for i, jid in enumerate(env.joint_indices):
    #         th_i = float(joint_positions[i])
    #         lims = env.tau_limit_interp.get(jid, None)
    #         if lims is not None:
    #             tau_flex_max = max(0.0, lims["flex"](th_i))
    #             tau_ext_max  = max(0.0, lims["ext"](th_i))
    #             joint_torques[i] = float(np.clip(joint_torques[i], -tau_ext_max, +tau_flex_max))
    #         else:
    #             joint_torques[i] = float(np.clip(joint_torques[i], -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE))
    # else:
    #     # en caso de que tau_limit_interp
    for i, max_torque in enumerate(env.joint_tau_max_force.values()):
        #torque_i=env.joint_tau_max_force[env.joint_indices[i]]
        joint_torques[i] = np.clip(joint_torques, -max_torque, max_torque)

    # ===== PASO 6: ACTUALIZAR ESTADOS PARA DEBUGGING =====
    env.pam_states = {
        'pressures': pam_pressures.copy(),
        'forces': np.abs(pam_forces),
        'raw_forces': pam_forces,
        'joint_torques': joint_torques.copy(),
        'moment_arms': {
            #
            'left_hip_pitch_flexor': R_flex_pitch_L,
            'left_hip_pitch_extensor': R_ext_pitch_L,
            'right_hip_pitch_flexor': R_flex_pitch_R,
            'right_hip_pitch_extensor': R_ext_pitch_R,

            'left_knee_flexor': R_knee_flex_L,
            'left_knee_extensor': R_knee_ext_L,
            'right_knee_flexor': R_knee_flex_R,
            'right_knee_extensor': R_knee_ext_R,

            'left_ankle_pitch_flexor': R_ankle_pitch_flex_L,
            'left_ankle_pitch_extensor': R_ankle_pitch_ext_L,
            'right_ankle_pitch_flexor': R_ankle_pitch_flex_R,
            'right_ankle_pitch_extensor': R_ankle_pitch_ext_R,
        },
        'inhibition_applied': False,
        'robot_specific_params': True
    }
    
    return joint_torques
    


def calculate_robot_specific_joint_torques_20_pam(env, pam_pressures):
    
    """
        Calcular torques básicos de articulaciones desde presiones PAM.
        Se usa para robots de 20 PAMs
    """
    
    # Obtener estados articulares (solo joints activos: caderas y rodillas)
    joint_states = p.getJointStates(env.robot_id, env.joint_indices)  
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [s[1] for s in joint_states]
    
    # Calcular fuerzas PAM reales
    pam_forces = np.zeros(env.num_active_pams, dtype=float)
    
    P = np.array([env.pam_muscles[muscle_names].real_pressure_PAM(u) for muscle_names,u 
                    in zip(env.muscle_names, pam_pressures)], dtype=float)
    
    # Cadera izquierda pitch
    pam_forces[0], pam_forces[1], R_flex_pitch_L, R_ext_pitch_L=obtener_pam_forces_flexor_extensor(env, joint_positions[0], P, 0, 1,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Cadera derecha pitch 
    pam_forces[2], pam_forces[3], R_flex_pitch_R, R_ext_pitch_R=obtener_pam_forces_flexor_extensor(env, joint_positions[5], P, 2, 3,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Cadera izquierda roll
    pam_forces[4], pam_forces[5], R_flex_roll_L, R_ext_roll_L=obtener_pam_forces_flexor_extensor(env, joint_positions[1], P, 4, 5,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)

    # Cadera derecha roll
    pam_forces[6], pam_forces[7], R_flex_roll_R, R_ext_roll_R=obtener_pam_forces_flexor_extensor(env, joint_positions[6], P, 6, 7,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)

    # Rodilla izquierda 
    pam_forces[8], pam_forces[9], R_knee_flex_L, R_knee_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[2], P, 8, 9,
                                                                                     env.knee_flexor_moment_arm,
                                                                                     env.knee_extensor_moment_arm)

    # Rodilla derecha
    pam_forces[10], pam_forces[11], R_knee_flex_R, R_knee_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[7], P, 10, 11,
                                                                                                 env.knee_flexor_moment_arm,
                                                                                                 env.knee_extensor_moment_arm)
    
    # Tobillo izquierdo pitch
    pam_forces[12], pam_forces[13], R_ankle_pitch_flex_L, R_ankle_pitch_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[3], P, 12, 13,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)

    # Tobillo derecho pitch
    pam_forces[14], pam_forces[15], R_ankle_pitch_flex_R, R_ankle_pitch_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[8], P, 14, 15,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)
    
    # Tobillo izquierdo roll
    pam_forces[16], pam_forces[17], R_ankle_roll_flex_L, R_ankle_roll_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[4], P, 16, 17,
                                                                                                    env.ankle_roll_flexor_moment_arm,
                                                                                                    env.ankle_roll_extensor_moment_arm)

    # Tobillo derecho roll
    pam_forces[18], pam_forces[19], R_ankle_roll_flex_R, R_ankle_roll_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[9], P, 18, 19,
                                                                                                    env.ankle_roll_flexor_moment_arm,
                                                                                                    env.ankle_roll_extensor_moment_arm)
    
    assert env.num_active_pams == env.action_space.shape[0] == 20, \
    f"{env.num_active_pams=} {env.action_space.shape=}"
    
    # Convertir a torques articulares
    joint_torques = np.zeros(len(joint_states))
    # TODO Ver que cambios tengo que aplicar para los torques ya queno tiene por que tener mismo torque en x que en y
    # Cadera izquierda pitch: flexión positiva por flexor, extensión por extensor
    joint_torques[0] = ( pam_forces[0] * R_flex_pitch_L) + (-pam_forces[1] * R_ext_pitch_L)
    # Cadera izquierda roll: flexión positiva por flexor, extensión por extensor
    joint_torques[1] = ( pam_forces[4] * R_flex_roll_L) + (-pam_forces[5] * R_ext_roll_L)

    # Rodilla izquierda: flexor + resorte/damping pasivos
    joint_torques[2] = (pam_forces[8] * R_knee_flex_L) + (-pam_forces[9] * R_knee_ext_L)

    # tobillo izquierdo pitch: flexor + resorte/damping pasivos
    joint_torques[3] = (pam_forces[12] * R_ankle_pitch_flex_L) + (-pam_forces[13] * R_ankle_pitch_ext_L)
    # tobillo izquierdo roll: flexor + resorte/damping pasivos
    joint_torques[4] = (pam_forces[16] * R_ankle_roll_flex_L) + (-pam_forces[17] * R_ankle_roll_ext_L)

    # Cadera derecha pitch
    joint_torques[5] = ( pam_forces[2] * R_flex_pitch_R) + (-pam_forces[3] * R_ext_pitch_R)
    # Cadera derecha roll
    joint_torques[6] = ( pam_forces[6] * R_flex_roll_R) + (-pam_forces[7] * R_ext_roll_R)

    # Rodilla derecha
    joint_torques[7] = (pam_forces[10] * R_knee_flex_R) + (-pam_forces[11] * R_knee_ext_R) 

    # tobillo derecho pitch
    joint_torques[8] = (pam_forces[14] * R_ankle_pitch_flex_R) + (-pam_forces[15] * R_ankle_pitch_ext_R)
    # tobillo derecho roll
    joint_torques[9] = (pam_forces[18] * R_ankle_roll_flex_R) + (-pam_forces[19] * R_ankle_roll_ext_R) 

    # PAra reducir el movimiento agresivo del robot
    # # Para rodillas
    # joint_torques[2] -= env.DAMPING_COEFFICIENT * joint_velocities[2]
    # joint_torques[6] -= env.DAMPING_COEFFICIENT * joint_velocities[6]
    # #Para tobillos
    # joint_torques[3] -= 0.5*env.DAMPING_COEFFICIENT * joint_velocities[3]
    # joint_torques[7] -= 0.5*env.DAMPING_COEFFICIENT * joint_velocities[7]
    # ======= REEMPLAZO DE CLIP GLOBAL POR CLIP ANGULAR =======
    if hasattr(env, "tau_limit_interp") and isinstance(env.tau_limit_interp, dict) and len(env.tau_limit_interp) > 0:
        # Usamos las posiciones articulares ya calculadas para interpolar τ_max(θ)
        for i, jid in enumerate(env.joint_indices):
            th_i = float(joint_positions[i])
            lims = env.tau_limit_interp.get(jid, None)
            if lims is not None:
                tau_flex_max = max(0.0, lims["flex"](th_i))
                tau_ext_max  = max(0.0, lims["ext"](th_i))
                joint_torques[i] = float(np.clip(joint_torques[i], -tau_ext_max, +tau_flex_max))
            else:
                joint_torques[i] = float(np.clip(joint_torques[i], -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE))
    else:
        # en caso de que tau_limit_interp
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

            'left_ankle_pitch_flexor': R_ankle_pitch_flex_L,
            'left_ankle_pitch_extensor': R_ankle_pitch_ext_L,
            'right_ankle_pitch_flexor': R_ankle_pitch_flex_R,
            'right_ankle_pitch_extensor': R_ankle_pitch_ext_R,

            'left_ankle_roll_flexor': R_ankle_roll_flex_L,
            'left_ankle_roll_extensor': R_ankle_roll_ext_L,
            'right_ankle_roll_flexor': R_ankle_roll_flex_R,
            'right_ankle_roll_extensor': R_ankle_roll_ext_R
        },
        'inhibition_applied': False,
        'robot_specific_params': True
    }
    
    return joint_torques


def calculate_robot_specific_joint_torques_24_pam(env, pam_pressures, robot_name):
    
    """
        Calcular torques básicos de articulaciones desde presiones PAM.
        Se usa para robots de 20 PAMs
    """
    
    # Obtener estados articulares (solo joints activos: caderas y rodillas)
    joint_states = p.getJointStates(env.robot_id, env.joint_indices)  
    joint_positions = [state[0] for state in joint_states]
    # Calcular fuerzas PAM reales
    pam_forces = np.zeros(env.num_active_pams, dtype=float)
    
    P = np.array([env.pam_muscles[muscle_names].real_pressure_PAM(u) for muscle_names,u 
                    in zip(env.muscle_names, pam_pressures)], dtype=float)
    
    # Cadera izquierda pitch
    pam_forces[0], pam_forces[1], R_flex_pitch_L, R_ext_pitch_L=obtener_pam_forces_flexor_extensor(env, joint_positions[0], P, 0, 1,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Cadera derecha pitch 
    pam_forces[2], pam_forces[3], R_flex_pitch_R, R_ext_pitch_R=obtener_pam_forces_flexor_extensor(env, joint_positions[6], P, 2, 3,
                                                                                     env.hip_pitch_flexor_moment_arm,
                                                                                     env.hip_pitch_extensor_moment_arm)

    # Cadera izquierda roll
    pam_forces[4], pam_forces[5], R_flex_roll_L, R_ext_roll_L=obtener_pam_forces_flexor_extensor(env, joint_positions[1], P, 4, 5,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)

    # Cadera derecha roll
    pam_forces[6], pam_forces[7], R_flex_roll_R, R_ext_roll_R=obtener_pam_forces_flexor_extensor(env, joint_positions[7], P, 6, 7,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)
    
    # Cadera izquierda yaw
    pam_forces[8], pam_forces[9], R_flex_yaw_L, R_ext_yaw_L=obtener_pam_forces_flexor_extensor(env, joint_positions[2], P, 8, 9,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)

    # Cadera derecha yaw
    pam_forces[10], pam_forces[11], R_flex_yaw_R, R_ext_yaw_R=obtener_pam_forces_flexor_extensor(env, joint_positions[8], P, 10, 11,
                                                                                     env.hip_roll_flexor_moment_arm,
                                                                                     env.hip_roll_extensor_moment_arm)

    # Rodilla izquierda 
    pam_forces[12], pam_forces[13], R_knee_flex_L, R_knee_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[3], P, 12, 13,
                                                                                     env.knee_flexor_moment_arm,
                                                                                     env.knee_extensor_moment_arm)

    # Rodilla derecha
    pam_forces[14], pam_forces[15], R_knee_flex_R, R_knee_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[9], P, 14, 15,
                                                                                                 env.knee_flexor_moment_arm,
                                                                                                 env.knee_extensor_moment_arm)
    
    # Tobillo izquierdo pitch
    pam_forces[16], pam_forces[17], R_ankle_pitch_flex_L, R_ankle_pitch_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[4], P, 16, 17,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)

    # Tobillo derecho pitch
    pam_forces[18], pam_forces[19], R_ankle_pitch_flex_R, R_ankle_pitch_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[10], P, 18, 19,
                                                                                                    env.ankle_pitch_flexor_moment_arm,
                                                                                                    env.ankle_pitch_extensor_moment_arm)
    
    # Tobillo izquierdo roll
    pam_forces[20], pam_forces[21], R_ankle_roll_flex_L, R_ankle_roll_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[5], P, 20, 21,
                                                                                                    env.ankle_roll_flexor_moment_arm,
                                                                                                    env.ankle_roll_extensor_moment_arm)

    # Tobillo derecho roll
    pam_forces[22], pam_forces[23], R_ankle_roll_flex_R, R_ankle_roll_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[11], P, 22, 23,
                                                                                                    env.ankle_roll_flexor_moment_arm,
                                                                                                    env.ankle_roll_extensor_moment_arm)
    
    assert env.num_active_pams == env.action_space.shape[0] == 20, \
    f"{env.num_active_pams=} {env.action_space.shape=}"
    
    # Convertir a torques articulares
    joint_torques = np.zeros(len(joint_states))
    # TODO Ver que cambios tengo que aplicar para los torques ya queno tiene por que tener mismo torque en x que en y
    # Cadera izquierda pitch: flexión positiva por flexor, extensión por extensor
    joint_torques[0] = ( pam_forces[0] * R_flex_pitch_L) + (-pam_forces[1] * R_ext_pitch_L)
    # Cadera izquierda roll: flexión positiva por flexor, extensión por extensor
    joint_torques[1] = ( pam_forces[4] * R_flex_roll_L) + (-pam_forces[5] * R_ext_roll_L)
    # Cadera izquierda yaw: flexión positiva por flexor, extensión por extensor
    joint_torques[2] = ( pam_forces[8] * R_flex_yaw_L) + (-pam_forces[9] * R_ext_yaw_L)

    # Rodilla izquierda: flexor + resorte/damping pasivos
    joint_torques[3] = (pam_forces[12] * R_knee_flex_L) + (-pam_forces[13] * R_knee_ext_L)

    # tobillo izquierdo pitch: flexor + resorte/damping pasivos
    joint_torques[4] = (pam_forces[16] * R_ankle_pitch_flex_L) + (-pam_forces[17] * R_ankle_pitch_ext_L)
    # tobillo izquierdo roll: flexor + resorte/damping pasivos
    joint_torques[5] = (pam_forces[20] * R_ankle_roll_flex_L) + (-pam_forces[21] * R_ankle_roll_ext_L)

    # Cadera derecha pitch
    joint_torques[6] = ( pam_forces[2] * R_flex_pitch_R) + (-pam_forces[3] * R_ext_pitch_R)
    # Cadera derecha roll
    joint_torques[7] = ( pam_forces[6] * R_flex_roll_R) + (-pam_forces[7] * R_ext_roll_R)
    # Cadera derecha yaw
    joint_torques[8] = ( pam_forces[10] * R_flex_yaw_R) + (-pam_forces[11] * R_ext_yaw_R)

    # Rodilla derecha
    joint_torques[9] = (pam_forces[14] * R_knee_flex_R) + (-pam_forces[15] * R_knee_ext_R) 

    # tobillo derecho pitch
    joint_torques[10] = (pam_forces[18] * R_ankle_pitch_flex_R) + (-pam_forces[19] * R_ankle_pitch_ext_R)
    # tobillo derecho roll
    joint_torques[11] = (pam_forces[22] * R_ankle_roll_flex_R) + (-pam_forces[23] * R_ankle_roll_ext_R) 

    # ======= REEMPLAZO DE CLIP GLOBAL POR CLIP ANGULAR =======
    if hasattr(env, "tau_limit_interp") and isinstance(env.tau_limit_interp, dict) and len(env.tau_limit_interp) > 0:
        # Usamos las posiciones articulares ya calculadas para interpolar τ_max(θ)
        for i, jid in enumerate(env.joint_indices):
            th_i = float(joint_positions[i])
            lims = env.tau_limit_interp.get(jid, None)
            if lims is not None:
                tau_flex_max = max(0.0, lims["flex"](th_i))
                tau_ext_max  = max(0.0, lims["ext"](th_i))
                joint_torques[i] = float(np.clip(joint_torques[i], -tau_ext_max, +tau_flex_max))
            else:
                joint_torques[i] = float(np.clip(joint_torques[i], -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE))
    else:
        # en caso de que tau_limit_interp
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

            #
            'left_hip_yaw_flexor': R_flex_yaw_L,
            'left_hip_yaw_extensor': R_ext_yaw_L,
            'right_hip_yaw_flexor': R_flex_yaw_R,
            'right_hip_yaw_extensor': R_ext_yaw_R,

            'left_knee_flexor': R_knee_flex_L,
            'left_knee_extensor': R_knee_ext_L,
            'right_knee_flexor': R_knee_flex_R,
            'right_knee_extensor': R_knee_ext_R,

            'left_ankle_pitch_flexor': R_ankle_pitch_flex_L,
            'left_ankle_pitch_extensor': R_ankle_pitch_ext_L,
            'right_ankle_pitch_flexor': R_ankle_pitch_flex_R,
            'right_ankle_pitch_extensor': R_ankle_pitch_ext_R,

            'left_ankle_roll_flexor': R_ankle_roll_flex_L,
            'left_ankle_roll_extensor': R_ankle_roll_ext_L,
            'right_ankle_roll_flexor': R_ankle_roll_flex_R,
            'right_ankle_roll_extensor': R_ankle_roll_ext_R
        },
        'inhibition_applied': False,
        'robot_specific_params': True
    }
    
    return joint_torques


def seleccionar_funcion_calculo_torques(env, pam_pressures):
    if "16DOF" in env.robot_name:
        return calculate_robot_specific_joint_torques_16_pam(env, pam_pressures)
    elif "20DOF" in env.robot_name:
        return calculate_robot_specific_joint_torques_20_pam(env, pam_pressures)
    elif "12DOF" in env.robot_name:
        return calculate_robot_specific_joint_torques_12_pam(env, pam_pressures)
    else:
        raise ValueError(f"Robot '{env.robot_name}' no soportado para sistema PAM.")




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


import csv
from stable_baselines3.common.callbacks import BaseCallback
class SimpleCsvKpiCallback(BaseCallback):
    """
    Lee info['kpi'] por step e info['ep_kpi'] al finalizar episodios
    y los vuelca a CSV en self.logs_dir.
    """
    def __init__(self, logs_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.logs_dir = logs_dir
        self._step_f = None
        self._ep_f = None
        self._step_writer = None
        self._ep_writer = None

    def _on_training_start(self) -> None:
        os.makedirs(self.logs_dir, exist_ok=True)
        self._step_f = open(os.path.join(self.logs_dir, "kpi_step.csv"), "w", newline="")
        self._ep_f   = open(os.path.join(self.logs_dir, "kpi_episode.csv"), "w", newline="")
        self._step_writer = csv.DictWriter(self._step_f,
            fieldnames=["timesteps","env_idx","reward","roll","pitch",
                        "left_down","right_down","F_L","F_R",
                        "zmp_x","zmp_y","com_x","com_y","com_z"])
        self._ep_writer = csv.DictWriter(self._ep_f,
            fieldnames=["timesteps","env_idx","ep_return","ep_len","done_reason"])
        self._step_writer.writeheader()
        self._ep_writer.writeheader()

    def _on_step(self) -> bool:
        # infos por cada env en este step
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for env_idx, (d, info) in enumerate(zip(dones, infos)):
            kpi = info.get("kpi")
            if kpi:
                row = {
                    "timesteps": int(self.num_timesteps),
                    "env_idx": env_idx,
                    "reward": kpi.get("reward", 0.0),
                    "roll": kpi.get("roll", 0.0),
                    "pitch": kpi.get("pitch", 0.0),
                    "left_down": kpi.get("left_down", 0),
                    "right_down": kpi.get("right_down", 0),
                    "F_L": kpi.get("F_L", 0.0),
                    "F_R": kpi.get("F_R", 0.0),
                    "zmp_x": kpi.get("zmp_x", 0.0),
                    "zmp_y": kpi.get("zmp_y", 0.0),
                    "com_x": kpi.get("com_x", 0.0),
                    "com_y": kpi.get("com_y", 0.0),
                    "com_z": kpi.get("com_z", 0.0),
                }
                self._step_writer.writerow(row)
            if d and "ep_kpi" in info:
                ep = info["ep_kpi"]
                self._ep_writer.writerow({
                    "timesteps": int(self.num_timesteps), "env_idx": env_idx,
                    "ep_return": ep.get("ep_return", 0.0),
                    "ep_len": ep.get("ep_len", 0),
                    "done_reason": ep.get("done_reason", None),
                })
        return True

    def _on_rollout_end(self) -> None:
        # algunos episodios terminan aquí; SB3 no siempre pasa ep_info,
        # pero nosotros miramos en infos por si hay 'ep_kpi'
        ep_infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(ep_infos):
            ep_kpi = info.get("ep_kpi")
            if ep_kpi:
                row = {
                    "timesteps": int(self.num_timesteps),
                    "env_idx": env_idx,
                    "ep_return": ep_kpi.get("ep_return", 0.0),
                    "ep_len": ep_kpi.get("ep_len", 0),
                    "done_reason": ep_kpi.get("done_reason", None),
                }
                self._ep_writer.writerow(row)

    def _on_training_end(self) -> None:
        try:
            if self._step_f: self._step_f.close()
            if self._ep_f: self._ep_f.close()
        except Exception:
            pass