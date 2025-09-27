import pybullet as p
import numpy as np


from Archivos_Mejorados.RewardSystemSimple import SingleLegActionType
from Archivos_Apoyo.Configuraciones_adicionales import split_cocontraction_torque_neutral

class AngleBasedExpertController:
    """
    Control experto que trabaja con ángulos objetivo en lugar de presiones PAM directas.
    
    Mucho más intuitivo: "levanta la pierna 40°" vs "presión PAM 0.7"
    """
    
    def __init__(self, env):
        self.robot_id = env.robot_id
        self.env=env
        # ===== PARÁMETROS DEL CONTROLADOR PD =====
        self.kp = self.env.KP   # Ganancia proporcional
        self.kd = self.env.KD   # Ganancia derivativa
        self.max_torque = self.env.MAX_REASONABLE_TORQUE  # Torque máximo por articulación
        self.use_ik = False  # Usar IK para levantar pierna (más natural)

        self.support_side = None          # 'left' | 'right'
        self._phase_lock_until = 0.0      # tiempo sim hasta el que no se puede cambiar
        self._prep_until = 0.0            # ventana de pre-shift (COM) tras cambio
        # Tuning rápido (ajusta si hace falta)
        self.PREP_DURATION = 0.20         # s de extensión extra del nuevo soporte
        self.PHASE_MIN_HOLD = 0.30        # s mínimos antes de poder cambiar
        self.CONTACT_ENTER_N = 25.0       # N para "entrar" en apoyo
        self.CONTACT_EXIT_N  = 17.0       # N para "salir" de apoyo

        
        # ===== ÁNGULOS OBJETIVO SEGÚN TAREA =====
        self.target_angles = {
            # NIVEL 1: Solo balance básico
            'level_1_balance': {
                'left_hip': -0.05,
                'left_knee': 0.05,
                'right_hip': -0.05,
                'right_knee': 0.05,
                'description': 'Posición erguida básica'
            },
            
            # NIVEL 2: Balance estable con micro-ajustes
            'level_2_balance': {
                'left_hip': -0.05,    # Ligera flexión para estabilidad
                'left_knee': 0.05,
                'right_hip': -0.05,
                'right_knee': 0.05,
                'description': 'Balance estable con micro-flexión'
            },
            
            # NIVEL 3: Equilibrio en una pierna
            'level_3_left_support': {
                'left_hip': 0.1,     # Pierna izq: soporte
                'left_knee': 0.1,
                'right_hip': -1.0,    # Pierna der: levantada 34°
                'right_knee': 0.6,
                'description': 'Pierna derecha levantada'
            },
            
            'level_3_right_support': {
                'left_hip': -1.0,     # Pierna izq: levantada 34°
                'left_knee': 0.6,
                'right_hip': 0.1,    # Pierna der: soporte
                'right_knee': 0.1,
                'description': 'Pierna izquierda levantada'
            }
        }
    
    def get_target_angles_for_task(self, current_task):
        """
        Obtener ángulos objetivo según la tarea actual
        
        Args:
            current_task: SingleLegActionType enum
            
        Returns:
            dict: Ángulos objetivo para cada articulación
        """
        
        if current_task == SingleLegActionType.BALANCE_LEFT_SUPPORT:
            base = self.target_angles['level_3_left_support'].copy()
            lift_side = 'right' 
        elif current_task == SingleLegActionType.BALANCE_RIGHT_SUPPORT:
            base = self.target_angles['level_3_right_support'].copy()  # soporta DER, levanta IZQ
            lift_side = 'left'
        elif current_task in (SingleLegActionType.TRANSITION,
                          SingleLegActionType.TRANSITION_TO_LEFT,
                          SingleLegActionType.TRANSITION_TO_RIGHT):
            # Postura intermedia estable para transiciones
            return self.target_angles['level_2_balance']
        else:  # Transiciones
            return self.target_angles['level_1_balance']
        # return base
        if self.use_ik:
            return self.get_target_angles_via_ik(lift_side=('right' if current_task==SingleLegActionType.BALANCE_LEFT_SUPPORT else 'left'), dz=0.08)
        
        return self.medir_progreso_cadera_rodilla(base, lift_side)

    def medir_progreso_cadera_rodilla(self, base, lift_side):
        joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        hip_now = joint_states[5][0] if lift_side=='right' else joint_states[1][0]
        # progreso 0..1 entre 0.25 y 0.60 rad aprox
        hip_prog = float(np.clip((abs(hip_now) - 0.25)/0.35, 0.0, 1.0))
        # rodilla acompaña a cadera (mínimo suave para despegar)
        knee_key = f"{lift_side}_knee"
        hip_key  = f"{lift_side}_hip"
        #base[knee_key] = np.sign(base[knee_key]) * (0.20 + 0.60*hip_prog)   # ≤ ~0.8 rad solo si la cadera progresa
        #base[hip_key]  = np.sign(base[hip_key])  * max(0.45, 0.80*hip_prog) # empuja que la cadera lidere
        # Limita la cadera del swing a env.swing_hip_target (p.ej. 0.35–0.40)
        hip_cap = float(getattr(self.env, "swing_hip_target", 0.40))
        base[hip_key]  = np.sign(base[hip_key])  * (hip_cap * hip_prog)      # 0 → hip_cap gradual
        # Rodilla acompaña, pero dentro del rango env.swing_knee_[lo,hi]
        knee_lo = float(getattr(self.env, "swing_knee_lo", 0.45))
        knee_hi = float(getattr(self.env, "swing_knee_hi", 0.75))
        knee_cmd = 0.20 + (knee_hi - 0.20) * hip_prog
        knee_cmd = float(np.clip(knee_cmd, knee_lo, knee_hi))
        base[knee_key] = np.sign(base[knee_key]) * knee_cmd
        
        return base
    
    def get_target_angles_via_ik(self, lift_side='right', dz=0.08):
        # Link del pie objetivo
        foot_id = self.env.right_foot_link_id if lift_side=='right' else self.env.left_foot_link_id
        # Posición actual del pie
        foot_pos = p.getLinkState(self.robot_id, foot_id)[0]
        tgt = (foot_pos[0], foot_pos[1], foot_pos[2] + dz)

        # IK para todos los joints; luego cogemos hip/knee del lado correspondiente
        joint_sol = p.calculateInverseKinematics(self.robot_id, foot_id, tgt)

        # En tu robot, hip/knee indices son [0,1] (izq) y [3,4] (der)
        if lift_side=='right':
            return {'left_hip': -0.05, 'left_knee': 0.05,
                    'right_hip': float(joint_sol[3]),
                    'right_knee': float(joint_sol[4])}
        else:
            return {'left_hip': float(joint_sol[0]),
                    'left_knee': float(joint_sol[1]),
                    'right_hip': -0.05, 'right_knee': 0.05}
    
    def calculate_pd_torques(self, target_angles_dict):
        """
        Calcular torques usando control PD hacia ángulos objetivo
        
        Args:
            target_angles_dict: Diccionario con ángulos objetivo
            
        Returns:
            numpy.array: Torques para [left_hip, left_knee, right_hip, right_knee]
        """
        
        # Obtener estados actuales de articulaciones
        joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        current_angles = [state[0] for state in joint_states] # q en DK o PD
        current_velocities = [state[1] for state in joint_states]# qDot
        
        # Ángulos objetivo en orden correcto
        target_angles_array = np.array([
            target_angles_dict['left_hip'],
            target_angles_dict['left_knee'], 
            target_angles_dict['right_hip'],
            target_angles_dict['right_knee']
        ], dtype=float)
        
        # Calcular errores
        angle_errors = np.array(target_angles_array) - np.array(current_angles)
        velocity_errors = -np.array(current_velocities)  # Queremos velocidad 0
        
        # Control PD
        pd_torques = self.kp * angle_errors + self.kd * velocity_errors
        # Limitar torques
        # pd_torques = np.clip(pd_torques, -self.max_torque, self.max_torque)

        # Antibloqueo + rescate cruzado de caderas
        # pd_torques = self._anti_stall_pd(pd_torques, target_angles_array, current_angles, current_velocities)
        # pd_torques = self._cross_hip_rescue(pd_torques, target_angles_array, current_angles, current_velocities)
        # Limitar torques
        pd_torques = np.clip(pd_torques, -self.max_torque, self.max_torque)

        
        return pd_torques
    
    def torques_to_pam_pressures(self, desired_torques, target_angles):
        """
        Convertir torques deseados en presiones PAM equivalentes
        
        Esta es la función INVERSA de _apply_pam_forces()
        
        Args:
            desired_torques: Array de torques [left_hip, left_knee, right_hip, right_knee]
            
        Returns:
            numpy.array: Presiones PAM normalizadas [0,1] para 6 PAMs
        """
        
        # # Salida
        pam_pressures = np.zeros(self.env.num_active_pams)
        
        # ===== CONVERSIÓN TORQUE → PRESIONES PAM =====
        pam = self.torques_to_pam_pressures_for_16_pam(desired_torques, pam_pressures)
        #pam = self._hip_complementary_routing(pam, desired_torques)       # si ya lo tienes
        #pam = self._ensure_opposition(pam, desired_torques, target_angles) # ⬅️ NUEVO
        

        
        # Asegurar rango [0, 1]
        pam = np.clip(pam, 0.0, 1.0)
        
        return pam
    
    def eps_from(self, theta, R_abs, R_min, muscle_name):
            return self.env.pam_muscles[muscle_name].epsilon_from_angle(theta, 0.0, max(abs(R_abs), R_min))
    
    def P_to_u(self, P, muscle_name):
        return float(np.clip(self.env.pam_muscles[muscle_name].normalized_pressure_PAM(P), 0.0, 1.0))
    
    def par_presiones_flexor_extensor(self, env, muscle_flexor_name, muscle_extensor_name,
                                      desired_torque_i, thetas_i, R_min_ligament, F_co, ma_flex, ma_ext):

        R_flexor = ma_flex(thetas_i)
        R_extensor = ma_ext(thetas_i)
        eps_flex_L = self.eps_from(thetas_i, R_flexor, R_min_ligament, muscle_flexor_name)
        eps_ext_L  = self.eps_from(thetas_i, R_extensor, R_min_ligament, muscle_extensor_name)
        Fco_flex, Fco_ext = split_cocontraction_torque_neutral(F_co, R_flexor, R_extensor, 1e-3)
        if desired_torque_i >= 0.0:  # flexión
            F_main = desired_torque_i / max(R_flexor, R_min_ligament)
            P_flexor = env.pam_muscles[muscle_flexor_name].pressure_from_force_and_contraction(Fco_flex + F_main, eps_flex_L)
            P_extensor = env.pam_muscles[muscle_extensor_name].pressure_from_force_and_contraction(max(Fco_ext - 0.5*F_main, 0.0), eps_ext_L)
        else:              # extensión
            F_main = (-desired_torque_i) / max(R_extensor, R_min_ligament)
            P_flexor = env.pam_muscles[muscle_flexor_name].pressure_from_force_and_contraction(max(Fco_flex - 0.5*F_main, 0.0), eps_flex_L)
            P_extensor = env.pam_muscles[muscle_extensor_name].pressure_from_force_and_contraction(Fco_ext + F_main, eps_ext_L)
        
        # Pressure from flexor and extensor
        return self.P_to_u(P_flexor,muscle_flexor_name), self.P_to_u(P_extensor, muscle_extensor_name)

    
    def torques_to_pam_pressures_for_16_pam(self,desired_torques, pam_pressures):
        env = self.env
        muscle_names=env.muscle_names
        
        # Obtener estados actuales para cálculos biomecánicos
        joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        thetas = [state[0] for state in joint_states] # [left_hip, left_knee, right_hip, right_knee]

        R_min_base = 1e-3 # Para evitar división por cero
        R_min_hip_pitch=1e-3
        R_min_knee  = 1e-2    # ↑ mayor que base
        # Antes era el min de 30
        F_co_hip = 20.0 #Genero rigidez y evito saturación
        F_co_hip_pitch = 30.0
        F_co_knee   = 50.0    # nueva rigidez basal de rodilla
        
        # ------ CADERA IZQUIERDA Roll ------
        pam_pressures[0], pam_pressures[1] = self.par_presiones_flexor_extensor(env, muscle_names[0], muscle_names[1],
                                                                                desired_torques[0], thetas[0],
                                                                                R_min_base, F_co_hip,
                                                                                env.hip_roll_flexor_moment_arm,env.hip_roll_extensor_moment_arm)

        
        # ------ CADERA DERECHA roll ------
        pam_pressures[2], pam_pressures[3] = self.par_presiones_flexor_extensor(env, muscle_names[2], muscle_names[3],
                                                                                desired_torques[4], thetas[4],
                                                                                R_min_base, F_co_hip,
                                                                                env.hip_roll_flexor_moment_arm,env.hip_roll_extensor_moment_arm)
        
        # ------ CADERA IZQUIERDA pitch ------
        pam_pressures[4], pam_pressures[5] = self.par_presiones_flexor_extensor(env, muscle_names[4], muscle_names[5],
                                                                                desired_torques[1], thetas[1],
                                                                                R_min_hip_pitch, F_co_hip_pitch,
                                                                                env.hip_pitch_flexor_moment_arm,env.hip_pitch_extensor_moment_arm)

        
        # ------ CADERA DERECHA pitch ------
        pam_pressures[6], pam_pressures[7] = self.par_presiones_flexor_extensor(env, muscle_names[6], muscle_names[7],
                                                                                desired_torques[5], thetas[5],
                                                                                R_min_hip_pitch, F_co_hip_pitch,
                                                                                env.hip_pitch_flexor_moment_arm,env.hip_pitch_extensor_moment_arm)

        # ------ RODILLA IZQUIERDA  ------
        pam_pressures[8], pam_pressures[9] = self.par_presiones_flexor_extensor(env, muscle_names[8], muscle_names[9],
                                                                                desired_torques[2], thetas[2],
                                                                                R_min_knee, F_co_knee,
                                                                                env.knee_flexor_moment_arm,env.knee_extensor_moment_arm)

        # ------ RODILLA Derecha  ------
        pam_pressures[10], pam_pressures[11] = self.par_presiones_flexor_extensor(env, muscle_names[10], muscle_names[11],
                                                                                desired_torques[6], thetas[6],
                                                                                R_min_knee, F_co_knee,
                                                                                env.knee_flexor_moment_arm,env.knee_extensor_moment_arm)
        
        # ------ Tobillo IZQUIERDA  ------
        pam_pressures[12], pam_pressures[13] = self.par_presiones_flexor_extensor(env, muscle_names[12], muscle_names[13],
                                                                                desired_torques[3], thetas[3],
                                                                                R_min_knee, F_co_knee,
                                                                                env.anckle_flexor_moment_arm,
                                                                                env.anckle_extensor_moment_arm)

        # ------ Tobillo Derecha  ------
        pam_pressures[14], pam_pressures[15] = self.par_presiones_flexor_extensor(env, muscle_names[14], muscle_names[15],
                                                                                desired_torques[7], thetas[7],
                                                                                R_min_knee, F_co_knee,
                                                                                env.anckle_flexor_moment_arm,
                                                                                env.anckle_extensor_moment_arm)

        return pam_pressures