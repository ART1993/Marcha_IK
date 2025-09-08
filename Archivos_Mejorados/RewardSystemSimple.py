# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

from collections import deque

from Archivos_Apoyo.simple_log_redirect import log_print, both_print

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
        self.expert_help_ratio = 0.85
        self.min_expert_help = 0.2
        
        # ===== NUEVO: CONTROLADOR BASADO EN ÁNGULOS =====
        self.angle_controller = AngleBasedExpertController(env)
        # Musculos PAM McKibben usarlos para generar presiones y angulos
        self.pam_muscle=env.pam_muscles
        
        # ===== ESTADO INTERNO =====
        self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
        self.last_10_rewards = deque(maxlen=10)
        self.time_in_current_stance = 0
        self.target_switch_time = env.switch_interval
        frequency = env.frecuency_simulation
        switch_time_seconds = self.target_switch_time / frequency
        print(f"🤖 Action Selector synchronized:")
        print(f"   Switch interval: {self.target_switch_time} steps ({switch_time_seconds:.1f}s)")
        
        log_print(f"✅ Angle-Based Single Leg Action Selector initialized")
        log_print(f"   Control method: Target angles → PD torques → PAM pressures")
        log_print(f"   Leg raise angle: 40° (0.7 rad)")
    
    def should_use_expert_action(self):
        """Decidir si usar acción experta o del modelo RL"""
        
        # Más ayuda al inicio del episodio (equilibrio en una pierna es difícil)
        if self.env.step_count < 200:
            effective_ratio = min(1.0, self.expert_help_ratio + 0.15)
        else:
            effective_ratio = self.expert_help_ratio
        
        return np.random.random() < effective_ratio
    
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
        
        # PASO 1: Obtener ángulos objetivo
        target_angles = self.angle_controller.get_target_angles_for_task(self.current_action)
        
        # PASO 2: Calcular torques PD
        pd_torques = self.angle_controller.calculate_pd_torques(target_angles)
        
        # PASO 3: Convertir a presiones PAM
        base_pressures = self.angle_controller.torques_to_pam_pressures(pd_torques)
        
        # PASO 4: Añadir correcciones por estabilidad (opcional)
        corrected_pressures = self._add_stability_corrections(base_pressures)
        
        # PASO 5: Variación natural pequeña
        noise = np.random.normal(0, 0.02, size=6)
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
        
        # Corrección sutil por inclinación lateral (roll)
        if abs(roll) > 0.05:  # > 3 grados
            correction_factor = min(0.1, abs(roll) * 0.3)
            
            if roll > 0:  # Inclinado hacia derecha → fortalecer lado izquierdo
                corrected[0] += correction_factor  # left_hip_flexor
                corrected[1] += correction_factor  # left_hip_extensor
            else:  # Inclinado hacia izquierda → fortalecer lado derecho
                corrected[2] += correction_factor  # right_hip_flexor
                corrected[3] += correction_factor  # right_hip_extensor
        
        return corrected
    
    def decide_current_action(self):
        """Decidir qué acción tomar basado en el contexto"""
        
        self.time_in_current_stance += 1
        
        # Si no hay suficiente historial, mantener acción actual
        if len(self.last_10_rewards) < 3:
            return
        
        recent_performance = np.mean(list(self.last_10_rewards)[-3:])
        
        # ===== LÓGICA DE CAMBIO DE PIERNA =====
        
        # Si el rendimiento es bueno y ha pasado suficiente tiempo, cambiar pierna
        if (recent_performance > 4.0 and 
            self.time_in_current_stance > self.target_switch_time):
            
            if self.current_action == SingleLegActionType.BALANCE_LEFT_SUPPORT:
                self.current_action = SingleLegActionType.TRANSITION_TO_RIGHT
            elif self.current_action == SingleLegActionType.BALANCE_RIGHT_SUPPORT:
                self.current_action = SingleLegActionType.TRANSITION_TO_LEFT
            elif self.current_action == SingleLegActionType.TRANSITION_TO_RIGHT:
                self.current_action = SingleLegActionType.BALANCE_RIGHT_SUPPORT
            elif self.current_action == SingleLegActionType.TRANSITION_TO_LEFT:
                self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
            
            self.time_in_current_stance = 0
        
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
        # Curriculum más conservador para tarea difícil
        if total_episode_reward > 80:  # Episodio muy exitoso
            self.expert_help_ratio *= 0.98  # Reducción más gradual
        elif total_episode_reward > 40:  # Episodio moderadamente exitoso
            self.expert_help_ratio *= 0.99
        else:  # Episodio problemático
            self.expert_help_ratio = min(0.95, self.expert_help_ratio * 1.02)
        
        self.expert_help_ratio = max(self.min_expert_help, self.expert_help_ratio)
        
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
        
        # ===== ÁNGULOS OBJETIVO SEGÚN TAREA =====
        self.target_angles = {
            # NIVEL 1: Solo balance básico
            'level_1_balance': {
                'left_hip': 0.0,
                'left_knee': 0.0,
                'right_hip': 0.0,
                'right_knee': 0.0,
                'description': 'Posición erguida básica'
            },
            
            # NIVEL 2: Balance estable con micro-ajustes
            'level_2_balance': {
                'left_hip': 0.05,    # Ligera flexión para estabilidad
                'left_knee': 0.05,
                'right_hip': 0.05,
                'right_knee': 0.05,
                'description': 'Balance estable con micro-flexión'
            },
            
            # NIVEL 3: Equilibrio en una pierna
            'level_3_left_support': {
                'left_hip': 0.0,     # Pierna izq: soporte
                'left_knee': 0.0,
                'right_hip': 0.6,    # Pierna der: levantada 34°
                'right_knee': 0.6,
                'description': 'Pierna derecha levantada'
            },
            
            'level_3_right_support': {
                'left_hip': 0.6,     # Pierna izq: levantada 34°
                'left_knee': 0.6,
                'right_hip': 0.0,    # Pierna der: soporte
                'right_knee': 0.0,
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
            return self.target_angles['level_3_left_support']
        elif current_task == SingleLegActionType.BALANCE_RIGHT_SUPPORT:
            return self.target_angles['level_3_right_support']
        elif current_task in (SingleLegActionType.TRANSITION,
                          SingleLegActionType.TRANSITION_TO_LEFT,
                          SingleLegActionType.TRANSITION_TO_RIGHT):
            # Postura intermedia estable para transiciones
            return self.target_angles['level_2_balance']
        else:  # Transiciones
            return self.target_angles['level_1_balance']
        
    def get_expert_action_for_level(self, reward_system):
        """Obtener acción experta según nivel del curriculum"""
        
        if not reward_system:
            return self._get_basic_balance_action()
        
        curriculum_info = reward_system.get_info()
        current_level = curriculum_info.get('level', 1)
        
        if current_level == 1:
            target_config = self.target_angles['level_1_balance']
        elif current_level == 2:
            target_config = self.target_angles['level_2_balance']
        else:  # level == 3
            target_leg = curriculum_info.get('target_leg', 'right')
            if target_leg == 'right':
                target_config = self.target_angles['level_3_left_support']  # Izq soporte, der levantada
            else:
                target_config = self.target_angles['level_3_right_support']  # Der soporte, izq levantada
        
        # Calcular torques PD hacia ángulos objetivo
        pd_torques = self.calculate_pd_torques(target_config)
        
        # Convertir a presiones PAM
        pam_pressures = self.torques_to_pam_pressures(pd_torques)
        
        return pam_pressures
    
    def _get_basic_balance_action(self):
        """Fallback: acción básica de balance"""
        return np.array([0.4, 0.5, 0.4, 0.5, 0.1, 0.1])
    
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
        env = self.env
        muscle_names=self.env.muscle_names
        pam_muscles = self.env.pam_muscles
        # Obtener estados actuales para cálculos biomecánicos
        joint_states = p.getJointStates(self.robot_id, self.env.joint_indices)
        thetas = [state[0] for state in joint_states] # [left_hip, left_knee, right_hip, right_knee]

        R_min = 1e-3 # Para evitar división por cero
        F_co = 30.0 #Genero rigidez y evito saturación

        def eps_from(theta, R_abs, muscle_name):
            return pam_muscles[muscle_name].epsilon_from_angle(theta, 0.0, max(abs(R_abs), R_min))
        def P_to_u(P,muscle_name):
            return float(np.clip(pam_muscles[muscle_name].normalized_pressure_PAM(P), 0.0, 1.0))
        
        # # Salida
        pam_pressures = np.zeros(6)
        
        # ===== CONVERSIÓN TORQUE → PRESIONES PAM =====
        
        # ------ CADERA IZQUIERDA (antagónica: PAM0 flexor, PAM1 extensor) ------
        flexor_cadera_L, extensor_cadera_L=muscle_names[0], muscle_names[1]
        tau_LH = desired_torques[0]
        thL    = thetas[0]
        RfL = env.hip_flexor_moment_arm(thL)
        ReL = env.hip_extensor_moment_arm(thL)
        eps_flex_L = eps_from(thL, RfL, flexor_cadera_L)
        eps_ext_L  = eps_from(thL, ReL, extensor_cadera_L)
        if tau_LH >= 0.0:  # flexión
            F_main = tau_LH / max(RfL, R_min)
            P0 = pam_muscles[flexor_cadera_L].pressure_from_force_and_contraction(F_co + F_main, eps_flex_L)
            P1 = pam_muscles[extensor_cadera_L].pressure_from_force_and_contraction(max(F_co - 0.5*F_main, 0.0), eps_ext_L)
        else:              # extensión
            F_main = (-tau_LH) / max(ReL, R_min)
            P0 = pam_muscles[flexor_cadera_L].pressure_from_force_and_contraction(max(F_co - 0.5*F_main, 0.0), eps_flex_L)
            P1 = pam_muscles[extensor_cadera_L].pressure_from_force_and_contraction(F_co + F_main, eps_ext_L)
        pam_pressures[0], pam_pressures[1] = P_to_u(P0,flexor_cadera_L), P_to_u(P1, extensor_cadera_L)
        
        # ------ CADERA DERECHA (PAM2 flexor, PAM3 extensor) ------
        flexor_cadera_R, extensor_cadera_R=muscle_names[2], muscle_names[3]
        tau_RH = desired_torques[2]
        thR    = thetas[2]
        RfR = env.hip_flexor_moment_arm(thR)
        ReR = env.hip_extensor_moment_arm(thR)
        eps_flex_R = eps_from(thR, RfR, flexor_cadera_R)
        eps_ext_R  = eps_from(thR, ReR, extensor_cadera_R )
        if tau_RH >= 0.0:
            F_main = tau_RH / max(RfR, R_min)
            P2 = pam_muscles[flexor_cadera_R].pressure_from_force_and_contraction(F_co + F_main, eps_flex_R)
            P3 = pam_muscles[extensor_cadera_R].pressure_from_force_and_contraction(max(F_co - 0.5*F_main, 0.0), eps_ext_R)
        else:
            F_main = (-tau_RH) / max(ReR, R_min)
            P2 = pam_muscles[flexor_cadera_R].pressure_from_force_and_contraction(max(F_co - 0.5*F_main, 0.0), eps_flex_R)
            P3 = pam_muscles[extensor_cadera_R].pressure_from_force_and_contraction(F_co + F_main, eps_ext_R)
        pam_pressures[2], pam_pressures[3] = P_to_u(P2,flexor_cadera_R), P_to_u(P3,extensor_cadera_R)
        
        
        # ------ RODILLA IZQUIERDA (solo flexor activo: PAM4) ------
        flexor_rodilla_L, flexor_rodilla_R=muscle_names[4], muscle_names[5]
        tau_LK = desired_torques[1]
        thKL   = thetas[1]
        RkL = env.knee_flexor_moment_arm(thKL)
        eps_kL = eps_from(thKL, RkL,flexor_rodilla_L)
        if tau_LK > 0.0:  # flexión
            Fk = tau_LK / max(RkL, R_min)
            P4 = pam_muscles[flexor_rodilla_L].pressure_from_force_and_contraction(Fk, eps_kL)
        else:
            P4 = pam_muscles[flexor_rodilla_L].min_pressure
        pam_pressures[4] = P_to_u(P4,flexor_rodilla_L)
        
        # ------ RODILLA DERECHA (solo flexor activo: PAM5) ------
        tau_RK = desired_torques[3]
        thKR   = thetas[3]
        RkR = env.knee_flexor_moment_arm(thKR)
        eps_kR = eps_from(thKR, RkR,flexor_rodilla_R)
        if tau_RK > 0.0:
            Fk = tau_RK / max(RkR, R_min)
            P5 = pam_muscles[flexor_rodilla_R].pressure_from_force_and_contraction(Fk, eps_kR)
        else:
            P5 = pam_muscles[flexor_rodilla_R].min_pressure
        pam_pressures[5] = P_to_u(P5,flexor_rodilla_R)

   
        # Asegurar rango [0, 1]
        pam_pressures = np.clip(pam_pressures, 0.0, 1.0)
        
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
    
    def __init__(self, robot_id, plane_id, frequency_simulation, switch_interval=2000):
        self.robot_id = robot_id
        self.plane_id = plane_id
        self.frequency_simulation = frequency_simulation
        self.switch_interval = switch_interval
        
        # ===== CONFIGURACIÓN SUPER SIMPLE =====
        self.level = 1  # Empezamos en nivel 1
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
        self.max_tilt_by_level = {
            1: 0.8,  # ~85 grados - muy permisivo para aprender básicos
            2: 0.6,  # ~70 grados - moderadamente permisivo  
            3: 0.3   # ~57 grados - estricto para habilidades avanzadas
        }
        
        # Para alternancia de piernas (solo nivel 3)
        self.target_leg = 'left'
        self.switch_timer = 0
        self.leg_switch_bonus = 0.0  # Bonus por cambio exitoso

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
        
        if self.level == 1:
            reward = self._level_1_reward()      # Solo supervivencia
        elif self.level == 2:
            reward = self._level_2_reward()      # + balance estable
        else:  # level == 3
            reward = self._level_3_reward(step_count)  # + levantar piernas
        
        # Limitar reward según nivel
        max_reward = self.level_config[self.level]['max_reward']
        return max(-2.0, min(reward, max_reward))
    
    def _level_1_reward(self):
        """NIVEL 1: Solo mantenerse de pie (recompensas 0-3)"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        height = pos[2]
        
        # Recompensa simple por altura
        if height > 0.9:
            return 1.5  # Buena altura
        elif height > 0.7:
            return 0.8  # Altura mínima
        else:
            return -1.0  # Caída
    
    def _level_2_reward(self):
        """NIVEL 2: Balance estable (recompensas 0-5)"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        height = pos[2]
        
        # Recompensa por altura (igual que nivel 1)
        if height > 0.9:
            height_reward = 1.5
        elif height > 0.7:
            height_reward = 0.8
        else:
            return -1.0  # Caída
        
        # + Recompensa por estabilidad (NUEVA)
        tilt = abs(euler[0]) + abs(euler[1])  # roll + pitch
        if tilt < 0.2:
            stability_reward = 1.5  # Muy estable
        elif tilt < 0.4:
            stability_reward = 0.5  # Moderadamente estable
        else:
            stability_reward = -0.5  # Inestable
        
        return height_reward + stability_reward
    
    def _level_3_reward(self, step_count):
        """NIVEL 3: Levantar piernas alternando (recompensas 0-8)"""
        
        # Recompensa base (igual que nivel 2)
        base_reward = self._level_2_reward()
        if base_reward < 0:  # Si se cayó, no calcular más
            return base_reward
        
        # + Recompensa por levantar pierna (NUEVA)
        leg_reward = self._calculate_leg_reward(step_count)
        
        return base_reward + leg_reward
    
    def _calculate_leg_reward(self, step_count):
        """Calcular recompensa por levantar pierna correctamente"""
        
        # Cambiar pierna objetivo cada 5 segundos (150 steps a 30 FPS)
        self.switch_timer += 1
        if self.switch_timer >= self.switch_interval:
            self.target_leg = 'left' if self.target_leg == 'right' else 'right'
            self.switch_timer = 0

            # DEBUG MEJORADO: Mostrar tiempo real además de steps
            seconds_per_switch = self.switch_interval / self.frequency_simulation  # Asumiendo 400 Hz
            print(f"🔄 Target: Raise {self.target_leg} leg (every {seconds_per_switch:.1f}s)")
        
        # Detectar qué pies están en contacto
        left_contacts = p.getContactPoints(self.robot_id, self.plane_id, 2, -1)
        right_contacts = p.getContactPoints(self.robot_id, self.plane_id, 5, -1)
        
        left_down = len(left_contacts) > 0
        right_down = len(right_contacts) > 0
        
        # Evaluar si está haciendo lo correcto
        if self.target_leg == 'right':
            # Quiero: pie izquierdo abajo, pie derecho arriba
            if left_down and not right_down:
                return 2.0  # ¡Perfecto!
            elif left_down and right_down:
                return 0.5  # Ambos abajo (transición)
            else:
                return -0.5  # Incorrecto
        else:  # target_leg == 'left'
            # Quiero: pie derecho abajo, pie izquierdo arriba
            if right_down and not left_down:
                return 2.0  # ¡Perfecto!
            elif left_down and right_down:
                return 0.5  # Ambos abajo (transición)
            else:
                return -0.5  # Incorrecto
    
    def update_after_episode(self, episode_reward, success=None):
        """Actualizar nivel después de cada episodio"""
        
        self.episode_count += 1
        self.recent_episodes.append(episode_reward)
        has_fallen = (self.last_done_reason in ("fall", "tilt"))
        
        # Mantener solo últimos 5 episodios
        if len(self.recent_episodes) > 5:
            self.recent_episodes.pop(0)

        # Determinar éxito si no te lo pasan explícitamente
        cfg = self.level_config[self.level]

        if success is None:
            # Éxito si supera umbral y no hubo caída
            success = (episode_reward >= cfg['success_threshold']) and (not has_fallen)

        # Actualizar racha
        self.success_streak = self.success_streak + 1 if success else 0

        # (Opcional) logging
        both_print(f"🏁 Episode {self.episode_count}: "
                f"reward={episode_reward:.1f} | success={success} | "
                f"streak={self.success_streak}/{cfg['success_streak_needed']}")
        
        # Verificar si subir de nivel
        if len(self.recent_episodes) >= 5:  # Necesitamos al menos 5 episodios
            #avg_reward = sum(self.recent_episodes) / len(self.recent_episodes)
            #config = self.level_config[self.level]
            
            # Promoción de nivel si cumple racha y episodios mínimos
            if (self.success_streak >= cfg['success_streak_needed']
                and self.episode_count >= cfg['episodes_needed']
                and self.level < 3):
                old = self.level
                self.level += 1
                self.success_streak = 0
                both_print(f"🎉 LEVEL UP! {old} → {self.level}")
    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Caída
        if pos[2] < 0.5:
            self.last_done_reason = "fall"
            log_print("❌ Episode done: Robot fell")
            return True
        
        
        max_tilt = self.max_tilt_by_level.get(self.level, 0.5)
        # Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            self.last_done_reason = "tilt"
            log_print("❌ Episode done: Robot tilted too much")
            return True
        
        # Tiempo máximo (crece con nivel)
        max_steps = (200 + (self.level * 200))*10  # 4000, 6000, 8000 steps
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
            'max_reward': self.level_config[self.level]['max_reward']
        }
