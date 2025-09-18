import os

import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
import pybullet as p

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


# ===================================================================================================================================================================================================== #
# ===================================================== Métodos de impresión y logging de Enhanced_PAMIKBipedEnv ====================================================================================== #
# ===================================================================================================================================================================================================== #


def PAM_McKibben(include_knee_extensors: bool = False):

    if include_knee_extensors == True:
        return {
            # Caderas - Control antagónico completo
            'left_hip_flexor': PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4),
            'left_hip_extensor': PAMMcKibben(L0=0.6, r0=0.04, alpha0=np.pi/4),
            'right_hip_flexor': PAMMcKibben(L0=0.6, r0=0.030, alpha0=np.pi/4),
            'right_hip_extensor': PAMMcKibben(L0=0.6, r0=0.04, alpha0=np.pi/4),
            
            # Rodillas - Control antagónico completo
            'left_knee_flexor': PAMMcKibben(L0=0.5, r0=0.03, alpha0=np.pi/4),
            'left_knee_extensor': PAMMcKibben(L0=0.5, r0=0.04, alpha0=np.pi/4),
            'right_knee_flexor': PAMMcKibben(L0=0.5, r0=0.03, alpha0=np.pi/4),
            'right_knee_extensor': PAMMcKibben(L0=0.5, r0=0.04, alpha0=np.pi/4),
        }
    else:
        return {
            # Caderas - Control antagónico completo
            'left_hip_flexor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            'left_hip_extensor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            'right_hip_flexor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            'right_hip_extensor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            
            # Rodillas - Solo flexores (extensión pasiva)
            'left_knee_flexor': PAMMcKibben(L0=0.5, r0=0.035, alpha0=np.pi/4),
            'right_knee_flexor': PAMMcKibben(L0=0.5, r0=0.035, alpha0=np.pi/4),
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


def calculate_robot_specific_joint_torques_8_pam(env, pam_pressures):
    
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

    # Cadera izquierda j0
    pam_forces[0], pam_forces[1], R_flex_L, R_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[0], P, 0, 1,
                                                                                     env.hip_flexor_moment_arm,
                                                                                     env.hip_extensor_moment_arm)

    # Cadera derecha j2
    pam_forces[2], pam_forces[3], R_flex_R, R_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[2], P, 2, 3,
                                                                                     env.hip_flexor_moment_arm,
                                                                                     env.hip_extensor_moment_arm)

    # RODILLAS (solo flexores activos)  (j1 = left_knee, j3 = right_knee)
    pam_forces[4], pam_forces[5], R_knee_flex_L, R_knee_ext_L=obtener_pam_forces_flexor_extensor(env, joint_positions[1], P, 4, 5,
                                                                                     env.knee_flexor_moment_arm,
                                                                                     env.knee_extensor_moment_arm)

    # flexor_rodilla_R, extensor_rodilla_R=env.muscle_names[6], env.muscle_names[7]
    pam_forces[6], pam_forces[7], R_knee_flex_R, R_knee_ext_R=obtener_pam_forces_flexor_extensor(env, joint_positions[3], P, 6, 7,
                                                                                                 env.knee_flexor_moment_arm,
                                                                                                 env.knee_extensor_moment_arm)

    # Aplicar a las caderas y rodillas (tienen músculos antagónicos)
    pam_forces[0], pam_forces[1] = apply_reciprocal_inhibition(pam_forces[0], 
                                                                pam_forces[1],
                                                                env.INHIBITION_FACTOR)  # Cadera izq
    pam_forces[2], pam_forces[3] = apply_reciprocal_inhibition(pam_forces[2], 
                                                                pam_forces[3],
                                                                env.INHIBITION_FACTOR)  # Cadera der
    pam_forces[4], pam_forces[5] = apply_reciprocal_inhibition(pam_forces[4], 
                                                                pam_forces[5],
                                                                env.INHIBITION_FACTOR)  # rodilla izq
    pam_forces[6], pam_forces[7] = apply_reciprocal_inhibition(pam_forces[6],
                                                                pam_forces[7],
                                                                env.INHIBITION_FACTOR)  # rodilla der

    # Convertir a torques articulares
    joint_torques = np.zeros(4)

    # Cadera izquierda: flexión positiva por flexor, extensión por extensor
    joint_torques[0] = ( pam_forces[0] * R_flex_L) + (-pam_forces[1] * R_ext_L)

    # Rodilla izquierda: flexor + resorte/damping pasivos
    joint_torques[1] = (pam_forces[4] * R_knee_flex_L) + (-pam_forces[5] * R_knee_ext_L)
    # Cadera derecha
    joint_torques[2] = ( pam_forces[2] * R_flex_R) + (-pam_forces[3] * R_ext_R)
    # Rodilla derecha
    joint_torques[3] = (pam_forces[6] * R_knee_flex_R) + (-pam_forces[7] * R_knee_ext_R) 

    joint_torques[1] -= env.DAMPING_COEFFICIENT * joint_velocities[1]
    joint_torques[3] -= env.DAMPING_COEFFICIENT * joint_velocities[3]
    joint_torques = np.clip(joint_torques, -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE)

    # ===== PASO 6: ACTUALIZAR ESTADOS PARA DEBUGGING =====
    
    env.pam_states = {
        'pressures': pam_pressures.copy(),
        'forces': np.abs(pam_forces),
        'raw_forces': pam_forces,
        'joint_torques': joint_torques.copy(),
        'moment_arms': {
            'left_hip_flexor': R_flex_L,
            'left_hip_extensor': R_ext_L,
            'right_hip_flexor': R_flex_R,
            'right_hip_extensor': R_ext_R,
            'left_knee_flexor': R_knee_flex_L,
            'left_knee_extensor': R_knee_ext_L,
            'right_knee_flexor': R_knee_flex_R,
            'right_knee_extensor': R_knee_ext_R
        },
        'inhibition_applied': True,
        'robot_specific_params': True
    }
    
    return joint_torques

def calculate_robot_specific_joint_torques_6_pam(env, pam_pressures):
    """
    Calcular torques básicos de articulaciones desde presiones PAM.
    
    Este método reemplaza la parte inicial de _apply_pam_forces
    antes del control automático de rodilla.
    """
    
    # Obtener estados articulares (solo joints activos: caderas y rodillas)
    joint_states = p.getJointStates(env.robot_id, env.joint_indices)  
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    
    # Calcular fuerzas PAM reales
    pam_forces = np.zeros(env.num_active_pams, dtype=float)
    
    def eps_from(theta, R_abs,muscle_name):
        # θ0=0 como referencia; el método ya hace clip a [0, εmax]
        return env.pam_muscles[muscle_name].epsilon_from_angle(theta, 0.0, max(abs(R_abs), 1e-9))
    
    P = np.array([env.pam_muscles[muscle_names].real_pressure_PAM(u) 
                  for muscle_names,u in zip(env.muscle_names, pam_pressures)], dtype=float)

    # Cadera izquierda j0
    flexor_cadera_L, extensor_cadera_L=env.muscle_names[0], env.muscle_names[1]
    thL = joint_positions[0]
    R_flex_L = env.hip_flexor_moment_arm(thL)
    R_ext_L  = env.hip_extensor_moment_arm(thL)
    eps_flex_L = eps_from(thL, R_flex_L,flexor_cadera_L)
    eps_ext_L  = eps_from(thL, R_ext_L, extensor_cadera_L)
    pam_forces[0] = env.pam_muscles[flexor_cadera_L].force_model_new(P[0], eps_flex_L)  # flexor L
    pam_forces[1] = env.pam_muscles[extensor_cadera_L].force_model_new(P[1], eps_ext_L)   # extensor L

    # Cadera derecha j2
    flexor_cadera_R, extensor_cadera_R=env.muscle_names[2], env.muscle_names[3]
    thR = joint_positions[2]
    R_flex_R = env.hip_flexor_moment_arm(thR)
    R_ext_R  = env.hip_extensor_moment_arm(thR)
    eps_flex_R = eps_from(thR, R_flex_R,flexor_cadera_R)
    eps_ext_R  = eps_from(thR, R_ext_R,extensor_cadera_R)
    pam_forces[2] = env.pam_muscles[flexor_cadera_R].force_model_new(P[2], eps_flex_R)  # flexor L
    pam_forces[3] = env.pam_muscles[extensor_cadera_R].force_model_new(P[3], eps_ext_R)   # extensor L

    # RODILLAS (solo flexores activos)  (j1 = left_knee, j3 = right_knee)
    flexor_rodilla_L, flexor_rodilla_R=env.muscle_names[4], env.muscle_names[5]
    thK_L = joint_positions[1]
    thK_R = joint_positions[3]
    R_knee_L = env.knee_flexor_moment_arm(thK_L)
    R_knee_R = env.knee_flexor_moment_arm(thK_R)
    eps_knee_L = eps_from(thK_L, R_knee_L,flexor_rodilla_L)
    eps_knee_R = eps_from(thK_R, R_knee_R,flexor_rodilla_R)
    pam_forces[4] = env.pam_muscles[flexor_rodilla_L].force_model_new(P[4], eps_knee_L)  # flexor rodilla L
    pam_forces[5] = env.pam_muscles[flexor_rodilla_R].force_model_new(P[5], eps_knee_R)  # flexor rodilla R

    

    # Aplicar a las caderas (tienen músculos antagónicos)
    pam_forces[0], pam_forces[1] = apply_reciprocal_inhibition(pam_forces[0], 
                                                                pam_forces[1],
                                                                env.INHIBITION_FACTOR)  # Cadera izq
    pam_forces[2], pam_forces[3] = apply_reciprocal_inhibition(pam_forces[2], 
                                                                pam_forces[3],
                                                                env.INHIBITION_FACTOR)  # Cadera der


        
    # Convertir a torques articulares
    joint_torques = np.zeros(4)

    # Cadera izquierda: flexión positiva por flexor, extensión por extensor
    joint_torques[0]  = ( pam_forces[0] * R_flex_L) + (-pam_forces[1] * R_ext_L)

    # Rodilla izquierda: flexor + resorte/damping pasivos
    left_contact  = env.contact_with_force(env.left_foot_link_id)
    right_contact = env.contact_with_force(env.right_foot_link_id)
    #left_contact, right_contact=env.contacto_pies
    spring_scale_L   = 1.0 if left_contact else 0.3
    passive_springL  = - env.PASSIVE_SPRING_STRENGTH * np.sin(thK_L)*spring_scale_L
    passive_damping = - env.DAMPING_COEFFICIENT    * joint_velocities[1]
    joint_torques[1]  = (pam_forces[4] * R_knee_L) + passive_springL + passive_damping
    
    # Cadera derecha
    joint_torques[2]  = ( pam_forces[2] * R_flex_R) + (-pam_forces[3] * R_ext_R)
    
    # Rodilla derecha
    spring_scale_R    = 1.0 if right_contact else 0.3
    passive_spring_R  = - env.PASSIVE_SPRING_STRENGTH * np.sin(thK_R) * spring_scale_R
    passive_damping_R = - env.DAMPING_COEFFICIENT    * joint_velocities[3]
    joint_torques[3]  = (pam_forces[5] * R_knee_R) + passive_spring_R + passive_damping_R
    
    joint_torques = np.clip(joint_torques, -env.MAX_REASONABLE_TORQUE, env.MAX_REASONABLE_TORQUE)

    # ===== PASO 6: ACTUALIZAR ESTADOS PARA DEBUGGING =====
    
    env.pam_states = {
        'pressures': pam_pressures.copy(),
        'forces': np.abs(pam_forces),
        'raw_forces': pam_forces,
        'joint_torques': joint_torques.copy(),
        'moment_arms': {
            'left_hip_flexor': R_flex_L,
            'left_hip_extensor': R_ext_L,
            'right_hip_flexor': R_flex_R,
            'right_hip_extensor': R_ext_R,
            'left_knee_flexor': R_knee_L,
            'right_knee_flexor': R_knee_R
        },
        'inhibition_applied': True,
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