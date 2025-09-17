# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

from collections import deque

from Archivos_Apoyo.simple_log_redirect import log_print, both_print
from Archivos_Apoyo.Configuraciones_adicionales import split_cocontraction_torque_neutral

class SingleLegActionType(Enum):
    """Acciones para equilibrio en una pierna"""
    BALANCE_LEFT_SUPPORT = "balance_left_support"    # Equilibrio con pie izquierdo
    BALANCE_RIGHT_SUPPORT = "balance_right_support"  # Equilibrio con pie derecho
    # Lo dejo en caso de que encuentre una forma más explicita de realizar la transición
    TRANSITION = "transition"              # Transición entre piernas
    TRANSITION_TO_LEFT = "transition_to_left"        # Transición hacia apoyo izquierdo
    TRANSITION_TO_RIGHT = "transition_to_right"      # Transición hacia apoyo derecho
    
class SingleLegActionSelector:
    """
        Selector de acciones expertas
        PAM 0: left_hip_flexor
        PAM 1: left_hip_extensor
        PAM 2: right_hip_flexor
        PAM 3: right_hip_extensor
        PAM 4: left_knee_flexor
        PAM 5: right_knee_flexor

    """
    
    def __init__(self, env):
        self.env = env
        self.episode_count = 0
        
        # ===== PARÁMETROS DE CURRICULUM =====
        if hasattr(env, 'enable_curriculum') and not env.enable_curriculum:
            # MODO SIN CURRICULUM: RL puro
            self.expert_help_ratio = 0.0  # ⭐ CLAVE: Sin ayuda experta
            self.min_expert_help = 0.0
            self.curriculum_enabled = False
            log_print(f"🎯 Action Selector: CURRICULUM DISABLED - Pure RL mode")
        else:
            # MODO CON CURRICULUM: comportamiento normal
            self.expert_help_ratio = 0.85
            self.min_expert_help = 0.0
            self.curriculum_enabled = True
            log_print(f"🎯 Action Selector: CURRICULUM ENABLED - Expert help starts at {self.expert_help_ratio:.1%}")

        # Solo crear si realmente necesitamos acciones expertas
        # ===== NUEVO: CONTROLADOR BASADO EN ÁNGULOS =====
        if self.expert_help_ratio > 0.0:
            self.angle_controller = AngleBasedExpertController(env)
            log_print(f"   Control method: Target angles → PD torques → PAM pressures")
        else:
            self.angle_controller = None
            log_print(f"   Control method: Pure RL (no expert controller needed)")
        
        
        # Musculos PAM McKibben usarlos para generar presiones y angulos
        self.pam_muscle=env.pam_muscles
        
        # ===== ESTADO INTERNO =====
        self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
        self.last_10_rewards = deque(maxlen=10)
        self.time_in_current_stance = 0
        self.target_switch_time = env.switch_interval
        frequency = env.frequency_simulation
        switch_time_seconds = self.target_switch_time / frequency
        print(f"🤖 Action Selector synchronized:")
        print(f"   Switch interval: {self.target_switch_time} steps ({switch_time_seconds:.1f}s)")
        
        log_print(f"✅ Angle-Based Single Leg Action Selector initialized")
        log_print(f"   Control method: Target angles → PD torques → PAM pressures")
        log_print(f"   Leg raise angle: 40° (0.7 rad)")
    
    def get_expert_action(self):
        """
        NUEVO: Obtener acción experta usando control basado en ángulos objetivo
        
        Proceso:
        1. Definir ángulos objetivo según tarea actual
        2. Calcular torques PD necesarios para alcanzar esos ángulos
        3. Convertir torques a presiones PAM equivalentes
        4. Añadir pequeñas correcciones por estabilidad
        
        Returns:
            numpy.array: Presiones PAM normalizadas [0,1]
        """
        level = self.env.simple_reward_system.level
        if level == 1:
            target_angles = self.angle_controller.target_angles['level_1_balance']
        elif level == 2:
            target_angles = self.angle_controller.target_angles['level_2_balance']
        else:
            # 🔗 Sincroniza con la pierna objetivo del sistema de recompensas
            if hasattr(self.env, "simple_reward_system"):
                self.current_action = (SingleLegActionType.BALANCE_LEFT_SUPPORT
                                    if self.env.simple_reward_system.target_leg == 'right'
                                else SingleLegActionType.BALANCE_RIGHT_SUPPORT)
            # PASO 1: Obtener ángulos objetivo
            target_angles = self.angle_controller.get_target_angles_for_task(self.current_action)
        # PASO 1: Obtener ángulos objetivo
        target_angles = self.angle_controller.get_target_angles_for_task(self.current_action)
        
        # PASO 2: Calcular torques PD
        pd_torques = self.angle_controller.calculate_pd_torques(target_angles)
        
        # PASO 3: Convertir a presiones PAM
        base_pressures = self.angle_controller.torques_to_pam_pressures(pd_torques)
        
        # PASO 4: Añadir correcciones por estabilidad (opcional)
        corrected_pressures = self._add_stability_corrections(base_pressures)
        
        # PASO 5: Variación natural pequeña
        noise = np.random.normal(0, 0.02, size=self.env.num_active_pams)
        final_pressures = corrected_pressures + noise
        
        return np.clip(final_pressures, 0.0, 1.0)
    
    def _add_stability_corrections(self, base_pressures):
        """
        Añadir correcciones pequeñas por inclinación para mayor estabilidad
        """
        
        # Obtener orientación actual
        pos, orn = p.getBasePositionAndOrientation(self.env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch = euler[0], euler[1]
        
        corrected = base_pressures.copy()

        # Ganancias muy pequeñas para no dominar la política
        k_pitch = 0.10   # corrige inclinación hacia delante/atrás
        k_roll  = 0.08   # corrige caída lateral

        if self.current_action == SingleLegActionType.BALANCE_LEFT_SUPPORT:
            corrected[1] += k_pitch * max(pitch, 0.0)  # L_hip_ext ↑
            corrected[0] -= k_pitch * max(pitch, 0.0)  # L_hip_flex ↓
        elif self.current_action == SingleLegActionType.BALANCE_RIGHT_SUPPORT:
            corrected[3] += k_pitch * max(pitch, 0.0)  # R_hip_ext ↑
            corrected[2] -= k_pitch * max(pitch, 0.0)  # R_hip_flex ↓

        # Roll: empuja a “plantar” más la cadera del lado que sube
        corrected[0] += k_roll * max(-roll, 0.0)  # roll<0 → flexión cadera izq
        corrected[2] += k_roll * max( roll, 0.0)  # roll>0 → flexión cadera dcha
        
        return np.clip(corrected, 0.0, 1.0)
    
    def decide_current_action(self):
        """Decidir qué acción tomar basado en el contexto"""
        
        self.time_in_current_stance += 1
        
        # Si no hay suficiente historial, mantener acción actual
        if len(self.last_10_rewards) < 5:
            return
        
        recent_performance = np.mean(list(self.last_10_rewards)[-3:])

        if self.time_in_current_stance >= self.target_switch_time:
            if self.current_action == SingleLegActionType.BALANCE_LEFT_SUPPORT:
                self.current_action = SingleLegActionType.BALANCE_RIGHT_SUPPORT
            else:
                self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
            self.time_in_current_stance = 0
        
        # ===== LÓGICA DE CAMBIO DE PIERNA =====
        
        # Si el rendimiento es malo, volver a posición estable
        elif recent_performance < 1.0:
            if self.current_action not in [SingleLegActionType.BALANCE_LEFT_SUPPORT, 
                                          SingleLegActionType.BALANCE_RIGHT_SUPPORT]:
                self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
                self.time_in_current_stance = 0
    
    def update_after_step(self, reward):
        """Actualizar después de cada step"""
        self.last_10_rewards.append(reward)
        self.decide_current_action()
    
    def on_episode_end(self, total_episode_reward):
        """Actualizar después de cada episodio"""
        self.episode_count += 1
        log_print(f"📊 Episode {self.episode_count} ended with total reward: {total_episode_reward:.1f}")
        if self.curriculum_enabled:
            # Curriculum más conservador para tarea difícil
            if total_episode_reward > 80:  # Episodio muy exitoso
                self.expert_help_ratio *= 0.90  # Reducción más gradual
            elif total_episode_reward > 40:  # Episodio moderadamente exitoso
                self.expert_help_ratio *= 0.95
            else:  # Episodio problemático
                self.expert_help_ratio = min(0.97, self.expert_help_ratio * 1.02)
            
            self.expert_help_ratio = max(self.min_expert_help, self.expert_help_ratio)
            log_print(f"   Expert help ratio updated to: {self.expert_help_ratio:.1%}")
        else:
            log_print(f"   Pure RL mode - no expert help adjustment")
        
        # Reset para nuevo episodio
        self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
        self.time_in_current_stance = 0

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
                'left_hip': -0.00,     # Pierna izq: soporte
                'left_knee': 0.00,
                'right_hip': -1.0,    # Pierna der: levantada 34°
                'right_knee': 0.6,
                'description': 'Pierna derecha levantada'
            },
            
            'level_3_right_support': {
                'left_hip': -1.0,     # Pierna izq: levantada 34°
                'left_knee': 0.6,
                'right_hip': -0.00,    # Pierna der: soporte
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
        current_angles = [state[0] for state in joint_states]
        current_velocities = [state[1] for state in joint_states]
        
        # Ángulos objetivo en orden correcto
        target_angles = [
            target_angles_dict['left_hip'],
            target_angles_dict['left_knee'], 
            target_angles_dict['right_hip'],
            target_angles_dict['right_knee']
        ]
        
        # Calcular errores
        angle_errors = np.array(target_angles) - np.array(current_angles)
        velocity_errors = -np.array(current_velocities)  # Queremos velocidad 0
        
        # Control PD
        pd_torques = self.kp * angle_errors + self.kd * velocity_errors
        
        # Limitar torques
        pd_torques = np.clip(pd_torques, -self.max_torque, self.max_torque)
        
        return pd_torques
    
    def torques_to_pam_pressures(self, desired_torques):
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
        if self.env.use_knee_extensor_pams:
            pam_pressures = self.torques_to_pam_pressures_for_8_pam(desired_torques, pam_pressures)
        else:
            pam_pressures = self.torques_to_pam_pressures_for_6_pam(desired_torques, pam_pressures)

   
        # Asegurar rango [0, 1]
        pam_pressures = np.clip(pam_pressures, 0.0, 1.0)
        
        return pam_pressures
    
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
        F_co_hip = 18.0 #Genero rigidez y evito saturación
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

    
    def torques_to_pam_pressures_for_6_pam(self,desired_torques, pam_pressures):
        env = self.env
        muscle_names=self.env.muscle_names
        pam_muscles = self.env.pam_muscles
        joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        thetas = [state[0] for state in joint_states] # [left_hip, left_knee, right_hip, right_knee]

        R_min_base = 1e-3 # Para evitar división por cero
        R_min_knee  = 1e-2    # ↑ mayor que base
        F_co_hip = 18.0 #Genero rigidez y evito saturación
        F_co_knee   = 50.0    # nueva rigidez basal de rodilla

        def eps_from(theta, R_abs, R_min, muscle_name):
            return pam_muscles[muscle_name].epsilon_from_angle(theta, 0.0, max(abs(R_abs), R_min))
        def P_to_u(P,muscle_name):
            return float(np.clip(pam_muscles[muscle_name].normalized_pressure_PAM(P), 0.0, 1.0))
        # ------ CADERA IZQUIERDA (antagónica: PAM0 flexor, PAM1 extensor) ------
        flexor_cadera_L, extensor_cadera_L=muscle_names[0], muscle_names[1]
        tau_LH = desired_torques[0]
        thL    = thetas[0]
        RfL = env.hip_flexor_moment_arm(thL)
        ReL = env.hip_extensor_moment_arm(thL)
        eps_flex_L = eps_from(thL, RfL, R_min_base, flexor_cadera_L)
        eps_ext_L  = eps_from(thL, ReL, R_min_base, extensor_cadera_L)
        if tau_LH >= 0.0:  # flexión
            F_main = tau_LH / max(RfL, R_min_base)
            P0 = pam_muscles[flexor_cadera_L].pressure_from_force_and_contraction(F_co_hip + F_main, eps_flex_L)
            P1 = pam_muscles[extensor_cadera_L].pressure_from_force_and_contraction(max(F_co_hip - 0.5*F_main, 0.0), eps_ext_L)
        else:              # extensión
            F_main = (-tau_LH) / max(ReL, R_min_base)
            P0 = pam_muscles[flexor_cadera_L].pressure_from_force_and_contraction(max(F_co_hip - 0.5*F_main, 0.0), eps_flex_L)
            P1 = pam_muscles[extensor_cadera_L].pressure_from_force_and_contraction(F_co_hip + F_main, eps_ext_L)
        pam_pressures[0], pam_pressures[1] = P_to_u(P0,flexor_cadera_L), P_to_u(P1, extensor_cadera_L)
        
        # ------ CADERA DERECHA (PAM2 flexor, PAM3 extensor) ------
        flexor_cadera_R, extensor_cadera_R=muscle_names[2], muscle_names[3]
        tau_RH = desired_torques[2]
        thR    = thetas[2]
        RfR = env.hip_flexor_moment_arm(thR)
        ReR = env.hip_extensor_moment_arm(thR)
        eps_flex_R = eps_from(thR, RfR, R_min_base, flexor_cadera_R)
        eps_ext_R  = eps_from(thR, ReR, R_min_base, extensor_cadera_R )
        if tau_RH >= 0.0:
            F_main = tau_RH / max(RfR, R_min_base)
            P2 = pam_muscles[flexor_cadera_R].pressure_from_force_and_contraction(F_co_hip + F_main, eps_flex_R)
            P3 = pam_muscles[extensor_cadera_R].pressure_from_force_and_contraction(max(F_co_hip - 0.5*F_main, 0.0), eps_ext_R)
        else:
            F_main = (-tau_RH) / max(ReR, R_min_base)
            P2 = pam_muscles[flexor_cadera_R].pressure_from_force_and_contraction(max(F_co_hip - 0.5*F_main, 0.0), eps_flex_R)
            P3 = pam_muscles[extensor_cadera_R].pressure_from_force_and_contraction(F_co_hip + F_main, eps_ext_R)
        pam_pressures[2], pam_pressures[3] = P_to_u(P2,flexor_cadera_R), P_to_u(P3,extensor_cadera_R)
        
        
        # ------ RODILLA IZQUIERDA (solo flexor activo: PAM4) ------
        flexor_rodilla_L, flexor_rodilla_R=muscle_names[4], muscle_names[5]
        tau_LK = desired_torques[1]
        thKL   = thetas[1]
        RkL = env.knee_flexor_moment_arm(thKL)
        eps_kL = eps_from(thKL, RkL,R_min_knee, flexor_rodilla_L)
        if tau_LK > 0.0:  # flexión
            Fk = tau_LK / max(RkL, R_min_knee)
            P4 = pam_muscles[flexor_rodilla_L].pressure_from_force_and_contraction(F_co_knee + Fk, eps_kL)
        else:
            Fk = (-tau_LK) / max(RkL, R_min_knee)
            P4 = pam_muscles[flexor_rodilla_L].pressure_from_force_and_contraction(max(F_co_knee - 0.5*Fk, 0.0), eps_kL)
        pam_pressures[4] = P_to_u(P4,flexor_rodilla_L)
        
        # ------ RODILLA DERECHA (solo flexor activo: PAM5) ------
        tau_RK = desired_torques[3]
        thKR   = thetas[3]
        RkR = env.knee_flexor_moment_arm(thKR)
        eps_kR = eps_from(thKR, RkR,R_min_knee,flexor_rodilla_R)
        if tau_RK > 0.0:
            Fk = tau_RK / max(RkR, R_min_knee)
            P5 = pam_muscles[flexor_rodilla_R].pressure_from_force_and_contraction(F_co_knee  +Fk, eps_kR)
        else:
            Fk = (-tau_RK) / max(RkR, R_min_knee)
            P5 = pam_muscles[flexor_rodilla_R].pressure_from_force_and_contraction(max(F_co_knee - 0.5*Fk, 0.0), eps_kR)
        pam_pressures[5] = P_to_u(P5,flexor_rodilla_R)

        return pam_pressures
    

    
# =============================================================================
# SISTEMA DE RECOMPENSAS PROGRESIVO SIMPLE
# Solo 3 niveles, fácil de entender y modificar
# =============================================================================
    
class SimpleProgressiveReward:
    """
    Sistema súper simple: 3 niveles que van aumentando la dificultad y las recompensas
    
    NIVEL 1: Solo mantenerse de pie (recompensas pequeñas 0-3) (0-15 episodios)
    NIVEL 2: Balance estable (recompensas medias 0-5)  (15-40 episodios)
    NIVEL 3: Levantar piernas (recompensas altas 0-8) (40+ episodios)
    """
    
    def __init__(self, env):
        self.env=env
        self.frequency_simulation = env.frequency_simulation
        self.switch_interval = env.switch_interval
        self.enable_curriculum = env.enable_curriculum
        self.robot_id = env.robot_id
        self.single_support_ticks = 0

        if self.enable_curriculum==False:
            # MODO SIN CURRICULUM: sistema fijo y permisivo
            self.level = 3  # Siempre nivel 3
            self.level_progression_disabled = True
            both_print(f"🎯 Progressive System: CURRICULUM DISABLED")
            both_print(f"   Mode: Fixed basic balance (Level max only)")
        else:
            # MODO CON CURRICULUM: comportamiento normal
            self.level = 1
            self.level_progression_disabled = False
            both_print(f"🎯 Progressive System: CURRICULUM ENABLED")
            both_print(f"   Mode: Level progression 1→2→3")
        
        # ===== CONFIGURACIÓN SUPER SIMPLE =====
        self.episode_count = 0
        self.recent_episodes = deque(maxlen=5)  # Últimos 5 episodios
        self.success_streak = 0  # Episodios consecutivos exitosos
        
        # Configuración por nivel (muy simple)
        # Configuración más graduada
        self.level_config = {
            1: {
                'description': 'Supervivencia básica',
                'max_reward': 2.0,
                'success_threshold': 1.0,    # Reward mínimo para considerar éxito
                'episodes_needed': 5,       # Episodios mínimos en este nivel
                'success_streak_needed': 3   # Episodios consecutivos exitosos para subir
            },
            2: {
                'description': 'Balance estable',
                'max_reward': 4.0,
                'success_threshold': 2.5,
                'episodes_needed': 15,
                'success_streak_needed': 4
            },
            3: {
                'description': 'Levantar piernas alternando',
                'max_reward': 7.0,
                'success_threshold': 999,    # Nivel final
                'episodes_needed': 999,
                'success_streak_needed': 999
            }
        }

        # Inclinación crítica - MÁS PERMISIVO según nivel
        if self.enable_curriculum==False:
            self.max_tilt_by_level = {
                1: 0.5,  # 
                2: 0.5,  # 
                3: 0.5   # 
            }
            both_print(f"   Max tilt: 70° (permisivo)")
        else:
            self.max_tilt_by_level = {
                1: 0.8,  #  - muy permisivo para aprender básicos
                2: 0.7,  #  - moderadamente permisivo  
                3: 0.5   #  - estricto para habilidades avanzadas
            }
        
        # Para alternancia de piernas (solo nivel 3)
        self.target_leg = 'left'
        self.switch_timer = 0
        # self.leg_switch_bonus = 0.0  # Bonus por cambio exitoso

        # Debug para confirmar configuración
        switch_time_seconds = self.switch_interval / self.frequency_simulation
        both_print(f"🎯 Progressive System initialized:")
        both_print(f"   Switch interval: {self.switch_interval} steps ({switch_time_seconds:.1f}s)")
        both_print(f"   Frequency: {self.frequency_simulation} Hz")
        both_print(f"🎯 Simple Progressive System: Starting at Level {self.level}")
    
    def calculate_reward(self, action, step_count):
        """
        Método principal: calcula reward según el nivel actual
        """

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        if self.level == 1 and self.enable_curriculum:
            reward = self._level_1_reward(pos,euler)      # Solo supervivencia
        elif self.level == 2 and self.enable_curriculum:
            reward = self._level_2_reward(pos, euler)      # + balance estable
        else:  # level == 3
            reward = self._level_3_reward(pos, euler, step_count)  # + levantar piernas
        level = self.level if self.enable_curriculum else 3
        max_reward = self.level_config[level]['max_reward']
        
        # Limitar reward según nivel
        
        
        return max(-2.0, min(reward, max_reward))
    
    def _level_1_reward(self,pos,euler):
        """NIVEL 1: Solo mantenerse de pie (recompensas 0-3)"""
        front_back_positions= self.env.init_pos[0]
        self.dx = float(pos[0] - front_back_positions)
        # Tolerancia sin penalización ±5 cm
        tol = 0.05
        # Penaliza deriva total fuera de tolerancia (suave; tope aprox -2.0)
        drift_pen = - np.clip(abs(self.dx) - tol, 0.0, 0.25) * 8.0
        # Penaliza adicionalmente cuando la deriva es hacia atrás (dx < -tol)
        # tope aprox -1.6
        back_only_pen = - np.clip(-self.dx - tol, 0.0, 0.20) * 8.0

        height = pos[2]
        
        # Recompensa simple por altura
        if height > 0.9:
            height_reward= 1.5  # Buena altura
        elif height > 0.8:
            height_reward= 0.8  # Altura mínima
        else:
            height_reward= -1.0  # Caída

        pitch = euler[1]
        back_pitch_pen = - np.clip(pitch - 0.05, 0.0, 0.30) * 6.0

        return height_reward + drift_pen + back_only_pen +back_pitch_pen
    
    def _level_2_reward(self,pos,euler):
        """NIVEL 2: Balance estable (recompensas 0-5)"""
        
        height_reward=self._level_1_reward(pos, euler)
        
        # + Recompensa por estabilidad (NUEVA)
        tilt = abs(euler[0]) + abs(euler[1])  # roll + pitch
        if tilt < 0.2:
            stability_reward = 1.5  # Muy estable
        elif tilt < 0.4:
            stability_reward = 0.5  # Moderadamente estable
        else:
            stability_reward = -0.5  # Inestable
        
        return height_reward + stability_reward
    
    def _level_3_reward(self,pos,euler, step_count):
        """NIVEL 3: Levantar piernas alternando (recompensas 0-8)"""
        
        # Recompensa base (igual que nivel 2)
        base_reward = self._level_2_reward(pos,euler)
        if base_reward < 0:  # Si se cayó, no calcular más
            return base_reward
        
        # + Recompensa por levantar pierna (NUEVA)
        leg_reward = self._calculate_leg_reward(step_count)
        
        return base_reward + leg_reward
    
    def _calculate_leg_reward(self, step_count):
        """Calcular recompensa por levantar pierna correctamente"""

        left_foot_id=self.env.left_foot_link_id
        right_foot_id=self.env.right_foot_link_id
        F_L = self.env.contact_normal_force(left_foot_id)
        F_R = self.env.contact_normal_force(right_foot_id)
        F_sum = max(F_L + F_R, 1e-6)
        left_hip_id, left_knee_id, right_hip_id, right_knee_id = self.env.joint_indices
        
        # Cambiar pierna cada switch interval
        self.switch_timer += 1
        if self.switch_timer >= self.switch_interval:
            self.target_leg = 'left' if self.target_leg == 'right' else 'right'
            self.switch_timer = 0
            # DEBUG MEJORADO: Mostrar tiempo real además de steps
            seconds_per_switch = self.switch_interval / self.frequency_simulation  
            # Asumiendo 400 Hz
            log_print(f"🔄 Target: Raise {self.target_leg} leg (every {seconds_per_switch:.1f}s)")
        
        # Detectar qué pies están en contacto Ver si seleccionar min_F=22 0 27 0 30
        left_down = self.env.contact_with_force(left_foot_id, min_F=18.0)
        right_down = self.env.contact_with_force(right_foot_id, min_F=18.0)

        target_is_right   = (self.target_leg == 'right')
        target_foot_id    = right_foot_id if target_is_right else left_foot_id
        target_foot_down  = right_down if target_is_right else left_down
        support_foot_down = left_down if target_is_right else right_down

        # if support_foot_down and not target_foot_down:
        #     self.single_support_ticks += 1
        #     single_support_step_reward = 0.05
        #     terminal_single_support_bonus = 0.5 if self.single_support_ticks == 120 else 0.0
        # else:
        #     self.single_support_ticks = 0
        #     single_support_step_reward = 0.0
        #     terminal_single_support_bonus = 0.0

        # ======== SHAPING POR CARGAS (romper toe-touch y cargar el soporte) ========
        # Cargas del pie de SOPORTE (esperado) y del pie OBJETIVO (debe ir al aire)
        F_sup = F_L if target_is_right else F_R   # si objetivo=right, soporte=left (F_L)
        F_tar = F_R if target_is_right else F_L

        # (1) Penaliza doble apoyo fuerte
        both_down_pen = -1.0 if (F_L >= 30.0 and F_R >= 30.0) else 0.0

        # (2) Penaliza toe-touch (1–30 N) del pie objetivo
        toe_touch_pen = -0.6 if (0.0 < F_tar < 30.0) else 0.0

        # (3) Recompensa reparto de carga sano: ≥80% en el pie de soporte
        ratio = F_sup / F_sum
        support_load_reward = np.clip((ratio - 0.80) / 0.20, 0.0, 1.0) * 1.0

        # (4) Bonus por tiempo en apoyo simple sostenido (pie objetivo en aire “limpio”)
        if (F_sup >= 30.0) and (F_tar < 1.0):
            # acumula ticks (~400 Hz ⇒ 0.30 s ≈ 120 ticks)
            self.single_support_ticks += 1
            ss_step = 0.05
            ss_terminal = 0.5 if self.single_support_ticks == int(0.30 * self.frequency_simulation) else 0.0
        else:
            self.single_support_ticks = 0
            ss_step = 0.0
            ss_terminal = 0.0

        # --- Bonuses de forma SOLO si el pie objetivo NO está en contacto ---
        # Clearance
        foot_z = p.getLinkState(self.robot_id, target_foot_id)[0][2]
        clearance_target = 0.08  # 8 cm
        clearance_bonus = 0.0 if target_foot_down else np.clip(foot_z / clearance_target, 0.0, 1.0) * 1.5

        # Rodilla (≈0.6 rad)
        knee_id  = right_knee_id if target_is_right else left_knee_id
        knee_ang = p.getJointState(self.robot_id, knee_id)[0]
        knee_bonus = (1.0 - min(abs(knee_ang - 0.6), 1.0)) * 1.0
        knee_bonus = 0.0 if target_foot_down else knee_bonus

        # Cadera (≈|0.6| rad) — uso el módulo para no depender del signo
        hip_id  = right_hip_id if target_is_right else left_hip_id
        hip_ang = p.getJointState(self.robot_id, hip_id)[0]
        hip_bonus = (1.0 - min(abs(abs(hip_ang) - 0.6), 1.0)) * 0.7
        hip_bonus = 0.0 if target_foot_down else hip_bonus

        # Gating de bonos de forma: solo si has transferido suficiente carga al pie de soporte
        if ratio < 0.70:
            clearance_bonus = 0.0
            knee_bonus = 0.0
            hip_bonus = 0.0

        # Evaluar si está haciendo lo correcto
        if self.target_leg == 'right':
            # Quiero: pie izquierdo abajo, pie derecho arriba
            if left_down and not right_down:
                contacto_reward = 2.0  # ¡Perfecto!
            elif left_down and right_down:
                contacto_reward = 0.5  # Ambos abajo (transición)
            else:
                contacto_reward = -0.5  # Incorrecto
        else:  # target_leg == 'left'
            # Quiero: pie derecho abajo, pie izquierdo arriba
            if right_down and not left_down:
                contacto_reward = 2.0  # ¡Perfecto!
            elif left_down and right_down:
                contacto_reward = 0.5  # Ambos abajo (transición)
            else:
                contacto_reward = -0.5  # Incorrecto

        # Suma total
        shaping = both_down_pen + toe_touch_pen + support_load_reward + ss_step + ss_terminal
        leg_reward = contacto_reward + clearance_bonus + knee_bonus + hip_bonus + shaping
        return leg_reward
    
    def update_after_episode(self, episode_reward, success=None):
        """Actualizar nivel después de cada episodio"""
        
        self.episode_count += 1
        self.recent_episodes.append(episode_reward)
        has_fallen = (self.last_done_reason in ("fall", "tilt", "drift"))
        
        # Mantener solo últimos 5 episodios
        if len(self.recent_episodes) > 5:
            self.recent_episodes.pop(0)

        # Determinar éxito si no te lo pasan explícitamente
        cfg = self.level_config[self.level]

        if success is None:
            # Éxito si supera umbral y no hubo caída
            success = (episode_reward >= cfg['success_threshold']) and (not has_fallen)
        log_print(f"{self.level_progression_disabled=:}, {self.enable_curriculum=:}")
        # Verificar si subir de nivel
        if self.level_progression_disabled is False:  # Necesitamos al menos 5 episodios
            #avg_reward = sum(self.recent_episodes) / len(self.recent_episodes)
            #config = self.level_config[self.level]
            # Actualizar racha
            if len(self.recent_episodes) >= 5:
                self.success_streak = self.success_streak + 1 if success else 0

                # (Opcional) logging
                both_print(f"🏁 Episode {self.episode_count}: "
                        f"reward={episode_reward:.1f} | success={success} | "
                        f"streak={self.success_streak}/{cfg['success_streak_needed']}")
                
                # Promoción de nivel si cumple racha y episodios mínimos
                if (self.success_streak >= cfg['success_streak_needed']
                    and self.episode_count >= cfg['episodes_needed']
                    and self.level < 3):
                    old = self.level
                    self.level += 1
                    self.success_streak = 0
                    both_print(f"🎉 LEVEL UP! {old} → {self.level}")
        else:
            # MODO SIN CURRICULUM: solo logging básico
            both_print(f"🏁 Episode {self.episode_count}: "
                    f"reward={episode_reward:.1f} | success={success} | "
                    f"fixed_level=3")
    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Caída
        if pos[2] <= 0.5:
            self.last_done_reason = "fall"
            log_print("❌ Episode done: Robot fell")
            return True
        
        if abs(self.dx) > 0.35:
            self.last_done_reason = "drift"
            log_print("❌ Episode done: Excessive longitudinal drift")
            return True
        
        max_tilt = self.max_tilt_by_level.get(self.level, 0.5)
        # Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            self.last_done_reason = "tilt"
            log_print("❌ Episode done: Robot tilted too much")
            return True
        
        # Tiempo máximo (crece con nivel)
        max_steps = (200 + ((self.level-1) * 200))*10 if self.enable_curriculum else 6000 # 2000, 4000, 6000 steps
        if step_count >= max_steps:
            self.last_done_reason = "time"
            log_print("⏰ Episode done: Max time reached")
            return True
        
        self.last_done_reason = None
        
        return False
    
    def get_info(self):
        """Info para debugging"""
        avg_reward = sum(self.recent_episodes) / len(self.recent_episodes) if self.recent_episodes else 0
        
        return {
            'level': self.level,
            'episodes': self.episode_count,
            'avg_reward': avg_reward,
            'target_leg': self.target_leg if self.level == 3 else None,
            'max_reward': self.level_config[self.level]['max_reward'],
            'curriculum_enabled': self.enable_curriculum,
            'level_progression_disabled': getattr(self, 'level_progression_disabled', False)
        }
