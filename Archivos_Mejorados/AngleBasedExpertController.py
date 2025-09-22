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
                'left_hip': -0.0,     # Pierna izq: soporte
                'left_knee': 0.00,
                'right_hip': 1.0,    # Pierna der: levantada 34°
                'right_knee': 0.6,
                'description': 'Pierna derecha levantada'
            },
            
            'level_3_right_support': {
                'left_hip': -1.0,     # Pierna izq: levantada 34°
                'left_knee': 0.6,
                'right_hip': -0.0,    # Pierna der: soporte
                'right_knee': 0.00,
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
        hip_now = joint_states[2][0] if lift_side=='right' else joint_states[0][0]
        # progreso 0..1 entre 0.25 y 0.60 rad aprox
        hip_prog = float(np.clip((abs(hip_now) - 0.25)/0.35, 0.0, 1.0))
        # rodilla acompaña a cadera (mínimo suave para despegar)
        knee_key = f"{lift_side}_knee"
        hip_key  = f"{lift_side}_hip"
        base[knee_key] = np.sign(base[knee_key]) * (0.20 + 0.60*hip_prog)   # ≤ ~0.8 rad solo si la cadera progresa
        base[hip_key]  = np.sign(base[hip_key])  * max(0.45, 0.80*hip_prog) # empuja que la cadera lidere
        
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
        pam = self.torques_to_pam_pressures_for_8_pam(desired_torques, pam_pressures)
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
        if desired_torque_i >= 0.0:  # flexión
            F_main = desired_torque_i / max(R_flexor, R_min_ligament)
            P_flexor = env.pam_muscles[muscle_flexor_name].pressure_from_force_and_contraction(F_co + F_main, eps_flex_L)
            P_extensor = env.pam_muscles[muscle_extensor_name].pressure_from_force_and_contraction(max(F_co - 0.5*F_main, 0.0), eps_ext_L)
        else:              # extensión
            F_main = (-desired_torque_i) / max(R_extensor, R_min_ligament)
            P_flexor = env.pam_muscles[muscle_flexor_name].pressure_from_force_and_contraction(max(F_co - 0.5*F_main, 0.0), eps_flex_L)
            P_extensor = env.pam_muscles[muscle_extensor_name].pressure_from_force_and_contraction(F_co + F_main, eps_ext_L)
        
        # Pressure from flexor and extensor
        return self.P_to_u(P_flexor,muscle_flexor_name), self.P_to_u(P_extensor, muscle_extensor_name)

    
    def torques_to_pam_pressures_for_8_pam(self,desired_torques, pam_pressures):
        env = self.env
        muscle_names=env.muscle_names
        
        # Obtener estados actuales para cálculos biomecánicos
        joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        thetas = [state[0] for state in joint_states] # [left_hip, left_knee, right_hip, right_knee]

        R_min_base = 1e-3 # Para evitar división por cero
        R_min_knee  = 1e-2    # ↑ mayor que base
        # Antes era el min de 30
        F_co_hip = 20.0 #Genero rigidez y evito saturación
        F_co_knee   = 50.0    # nueva rigidez basal de rodilla
        
        # ------ CADERA IZQUIERDA (antagónica: PAM0 flexor, PAM1 extensor) ------
        pam_pressures[0], pam_pressures[1] = self.par_presiones_flexor_extensor(env, muscle_names[0], muscle_names[1],
                                                                                desired_torques[0], thetas[0],
                                                                                R_min_base, F_co_hip,
                                                                                env.hip_flexor_moment_arm,env.hip_extensor_moment_arm)

        
        # ------ CADERA DERECHA (PAM2 flexor, PAM3 extensor) ------
        pam_pressures[2], pam_pressures[3] = self.par_presiones_flexor_extensor(env, muscle_names[2], muscle_names[3],
                                                                                desired_torques[2], thetas[2],
                                                                                R_min_base, F_co_hip,
                                                                                env.hip_flexor_moment_arm,env.hip_extensor_moment_arm)

        # ------ RODILLA IZQUIERDA (antagónica: PAM4 flexor, PAM5 extensor) ------
        pam_pressures[4], pam_pressures[5] = self.par_presiones_flexor_extensor(env, muscle_names[4], muscle_names[5],
                                                                                desired_torques[1], thetas[1],
                                                                                R_min_knee, F_co_knee,
                                                                                env.knee_flexor_moment_arm,env.knee_extensor_moment_arm)

        # ------ RODILLA Derecha (antagónica: PAM6 flexor, PAM7 extensor) ------
        pam_pressures[6], pam_pressures[7] = self.par_presiones_flexor_extensor(env, muscle_names[6], muscle_names[7],
                                                                                desired_torques[3], thetas[3],
                                                                                R_min_knee, F_co_knee,
                                                                                env.knee_flexor_moment_arm,env.knee_extensor_moment_arm)

        return pam_pressures


    # ========================================================================================================================================================================================================================================================= #
    # ============================================================================================================= CROSS CORRECTION ========================================================================================================================== #
    # ========================================================================================================================================================================================================================================================= #

    def limite_interior_co_contraction(self):
        pass
        # --- limite inferior de co-contracción torque-neutral por cadera ---
        # def co_contraction_floor(i_flex, i_ext, theta, Rf, Re, error_abs):
        #     # Si error grande y ambas presiones ~0, sube Fco mínimo
        #     if error_abs < 0.08: 
        #         return
        #     if (pam_pressures[i_flex] + pam_pressures[i_ext]) < 0.05:
        #         # fuerza basal en N, mapeada con tu modelo inverso
        #         Fco = 30.0
        #         Fco_flex, Fco_ext = split_cocontraction_torque_neutral(Fco, Rf, Re, 1e-3)
        #         eps_flex = self.eps_from(theta, Rf, 1e-3, self.env.muscle_names[i_flex])
        #         eps_ext  = self.eps_from(theta, Re, 1e-3, self.env.muscle_names[i_ext])
        #         Pflex = self.env.pam_muscles[self.env.muscle_names[i_flex]].pressure_from_force_and_contraction(Fco_flex, eps_flex)
        #         Pext  = self.env.pam_muscles[self.env.muscle_names[i_ext ]].pressure_from_force_and_contraction(Fco_ext , eps_ext )
        #         pam_pressures[i_flex] = max(pam_pressures[i_flex], self.P_to_u(Pflex, self.env.muscle_names[i_flex]))
        #         pam_pressures[i_ext ] = max(pam_pressures[i_ext ], self.P_to_u(Pext , self.env.muscle_names[i_ext ]))

        # # Lectura rápida del estado actual para brazos de momento
        # joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        # thetas = [s[0] for s in joint_states]

        # Cadera izquierda (índices PAM 0/1 ↔ joint 0)
        # Rf_L = self.env.hip_flexor_moment_arm(thetas[0]); Re_L = self.env.hip_extensor_moment_arm(thetas[0])
        # # PAM flexor/extensor_id=0,1
        # co_contraction_floor(0, 1, thetas[0], Rf_L, Re_L, error_abs=abs(desired_torques[0])/max(self.max_torque,1e-9))

        # # Cadera derecha (índices PAM 2/3 ↔ joint 2)
        # Rf_R = self.env.hip_flexor_moment_arm(thetas[2]); Re_R = self.env.hip_extensor_moment_arm(thetas[2])
        # # PAM flexor/extensor_id=2,3
        # co_contraction_floor(2, 3, thetas[2], Rf_R, Re_R, error_abs=abs(desired_torques[2])/max(self.max_torque,1e-9))
        # ¿Y que sucede para 0,3 y 1,2?


    def _error_and_trend(self, target, q, qdot):
        e = float(target - q)
        de = float(-qdot)  # queremos qdot→0; signo consistente con PD
        # "acercándose" si e*de > 0 (error y su corrección van a favor)
        approaching = (e * de) > 0.0
        return e, de, approaching
    
    # 2) NUEVO: acoplamiento cruzado de caderas (rescate)
    def _cross_hip_rescue(self, pd_torques, targets, q, qdot):
        # Orden: [L_hip, L_knee, R_hip, R_knee]
        eL, deL, appL = self._error_and_trend(targets[0], q[0], qdot[0])
        eR, deR, appR = self._error_and_trend(targets[2], q[2], qdot[2])

        # Condición: una se acerca y la otra se aleja; error notable en la que se aleja
        E_MIN = 0.06   # ~3.5°
        BOOST = 0.35   # ganancia de rescate (ajusta si hace falta)
        TAU_MAX = self.max_torque

        if appR and (not appL) and abs(eL) > E_MIN:
            pd_torques[0] += np.clip(BOOST * eL + 0.05 * deL, -0.5*TAU_MAX, 0.5*TAU_MAX)
        elif appL and (not appR) and abs(eR) > E_MIN:
            pd_torques[2] += np.clip(BOOST * eR + 0.05 * deR, -0.5*TAU_MAX, 0.5*TAU_MAX)

        return pd_torques
    
    # 3) NUEVO: antibloqueo si hay error grande pero |tau_PD| pequeño
    def _anti_stall_pd(self, pd_torques, targets, q, qdot):
        BIG_E   = 0.10   # ~5.7°
        LOW_TAU = 1.0    # Nm
        KP_BOOST = 0.6
        for i in (0, 2):  # solo caderas
            e = float(targets[i] - q[i])
            if abs(e) > BIG_E and abs(pd_torques[i]) < LOW_TAU:
                pd_torques[i] += np.clip(KP_BOOST * e - 0.05 * qdot[i], -3.0, 3.0)
        return pd_torques
    
    def _hip_complementary_routing(self, pam, desired_torques):
        """
            Fuerza un patrón complementario explícito de presiones por cadera:
            - Si τ_L > 0 (flexión), L_flex = P_req(τ_L), L_ext = piso
            - Si τ_L < 0 (extensión), L_ext = P_req(|τ_L|), L_flex = piso
            - Si τ_R > 0 (flexión), R_flex = P_req(τ_R), R_ext = piso
            - Si τ_R < 0 (extensión), R_ext = P_req(|τ_R|), R_flex = piso

            'piso' es co-contracción mínima (si la quieres). El cálculo de P_req es físico:
            usa torque → fuerza (R) → presión con tu modelo PAM.
        """
        # Índices PAM: [0 L_flex, 1 L_ext, 2 R_flex, 3 R_ext, ...]
        # Estados actuales para calcular ε a partir de R(θ):
        joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        theta_L = joint_states[0][0]; theta_R = joint_states[2][0]

        # Brazos de momento actuales
        Rf_L = self.env.hip_flexor_moment_arm(theta_L); Re_L = self.env.hip_extensor_moment_arm(theta_L)
        Rf_R = self.env.hip_flexor_moment_arm(theta_R); Re_R = self.env.hip_extensor_moment_arm(theta_R)

        # Contracciones (ε) para los 4 PAM de cadera
        eps_Lf = self.eps_from(theta_L, Rf_L, 1e-3, self.env.muscle_names[0])
        eps_Le = self.eps_from(theta_L, Re_L, 1e-3, self.env.muscle_names[1])
        eps_Rf = self.eps_from(theta_R, Rf_R, 1e-3, self.env.muscle_names[2])
        eps_Re = self.eps_from(theta_R, Re_R, 1e-3, self.env.muscle_names[3])

        # Piso de co-contracción (muy pequeño; sube si quieres más rigidez)
        piso = 0.00

        # Utilidad para convertir τ → presión normalizada en el PAM correcto
        def P_req_norm(tau, R, eps, muscle_idx):
            if abs(R) < 1e-6 or abs(tau) < 1e-6:
                return 0.0
            F = abs(tau) / max(abs(R), 1e-9)
            return float(self.env.pam_muscles[self.env.muscle_names[muscle_idx]].pressure_normalized_from_force_and_contraction(F, eps))

        # Izquierda
        tau_L = float(desired_torques[0])
        if tau_L >= 0.0:  # flexión
            pam[0] = max(pam[0], P_req_norm(tau_L, Rf_L, eps_Lf, 0));  pam[1] = max(piso, 0.0)
        else:             # extensión
            pam[1] = max(pam[1], P_req_norm(tau_L, Re_L, eps_Le, 1));  pam[0] = max(piso, 0.0)

        # Derecha
        tau_R = float(desired_torques[2])
        if tau_R >= 0.0:  # flexión
            pam[2] = max(pam[2], P_req_norm(tau_R, Rf_R, eps_Rf, 2));  pam[3] = max(piso, 0.0)
        else:             # extensión
            pam[3] = max(pam[3], P_req_norm(tau_R, Re_R, eps_Re, 3));  pam[2] = max(piso, 0.0)

        return np.clip(pam, 0.0, 1.0)
    
    def _ensure_opposition(self, pam, desired_torques, target_angles):
        """
        Si una cadera tiene objetivo ~0 rad (soporte) y la otra está lejos (swing),
        forzar activación opuesta mínima en la de soporte para sostener 0 rad
        y en la de swing para empujar al objetivo, usando el inverso PAM.
        """
        # Índices y nombres
        Lf, Le, Rf, Re = 0, 1, 2, 3

        # Estados actuales
        js = p.getJointStates(self.robot_id, self.env.joint_indices)
        qL, qR = js[0][0], js[2][0]

        # Momento y contracción en tiempo real
        RfL = self.env.hip_flexor_moment_arm(qL); ReL = self.env.hip_extensor_moment_arm(qL)
        RfR = self.env.hip_flexor_moment_arm(qR); ReR = self.env.hip_extensor_moment_arm(qR)
        epsLf = self.env.pam_muscles[self.env.muscle_names[Lf]].epsilon_from_angle(qL, 0.0, max(abs(RfL),1e-9))
        epsLe = self.env.pam_muscles[self.env.muscle_names[Le]].epsilon_from_angle(qL, 0.0, max(abs(ReL),1e-9))
        epsRf = self.env.pam_muscles[self.env.muscle_names[Rf]].epsilon_from_angle(qR, 0.0, max(abs(RfR),1e-9))
        epsRe = self.env.pam_muscles[self.env.muscle_names[Re]].epsilon_from_angle(qR, 0.0, max(abs(ReR),1e-9))

        # Utilidad inversa normalizada ya presente en tu modelo PAM ✅
        def Pnorm_req(F, eps, muscle_name):
            return float(self.env.pam_muscles[muscle_name].pressure_normalized_from_force_and_contraction(F, eps))

        # Torques PD que ya calculaste para hip L/R
        tauL = float(desired_torques[0])
        tauR = float(desired_torques[2])

        # Objetivos vigentes (nivel 3): una ~0 y la otra lejos
        tgtL = float(target_angles['left_hip'])
        tgtR = float(target_angles['right_hip'])
        near0 = lambda x: abs(x) < 0.08  # 0.08 rad ~ 4.6°, margen

        # Magnitudes de par requeridas para sostener o empujar (usa el mismo PD, no constantes)
        # Convertimos τ → F = |τ|/|R| → Pnorm con el PAM correspondiente
        def inject_ext(side, tau, R, eps, flex_idx, ext_idx):
            if abs(R) < 1e-6 or abs(tau) < 1e-6: return
            F = abs(tau)/max(abs(R),1e-9)
            if tau >= 0:  # flexión
                pam[flex_idx] = max(pam[flex_idx], Pnorm_req(F, eps['flex'], self.env.muscle_names[flex_idx]))
                pam[ext_idx]  = max(pam[ext_idx],  0.0)
            else:         # extensión
                pam[ext_idx]  = max(pam[ext_idx],  Pnorm_req(F, eps['ext'],  self.env.muscle_names[ext_idx]))
                pam[flex_idx] = max(pam[flex_idx], 0.0)

        # Lógica de oposición: si L es swing (|tgtL|≫|tgtR|) y R soporte (~0), impone:
        #   L: seguir su τ_L (signo→músculo correcto); R: seguir su τ_R para sostener 0
        if (abs(tgtL) > 0.25 and near0(tgtR)) or (abs(tgtR) > 0.25 and near0(tgtL)):
            # Emparejar ambos lados (no dejar uno en ~0 si el otro es alto)
            # Lado izquierdo
            inject_ext('L', tauL, (RfL if tauL>=0 else ReL),
                    {'flex': epsLf, 'ext': epsLe}, Lf, Le)
            # Lado derecho
            inject_ext('R', tauR, (RfR if tauR>=0 else ReR),
                    {'flex': epsRf, 'ext': epsRe}, Rf, Re)

            # (Opcional) Piso de co-contracción torque-neutral para amortiguar sin par
            # Fco = k * (|tauL|+|tauR|)  -> reparte con split_cocontraction_torque_neutral(...)
            # y conviértelo a presiones invertidas en ambos PAMs de cada cadera.
        return np.clip(pam, 0.0, 1.0)