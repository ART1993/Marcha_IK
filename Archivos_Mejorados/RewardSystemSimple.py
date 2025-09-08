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

            
# ===================================================================================================== #
# ================= Clases para selección de accion levantar piernas y recompensas ==================== #
# ===================================================================================================== #

class SingleLegBalanceRewardSystem:
    """
    Sistema de recompensas COMPLETO para equilibrio en una pierna.
    """
    
    def __init__(self, robot_id, plane_id):
        self.robot_id = robot_id
        self.plane_id = plane_id
        
        # ===== PARÁMETROS DE RECOMPENSA =====
        self.base_reward = 1.0
        self.single_leg_bonus = 3.0          # Bonificación por equilibrio en una pierna
        self.knee_height_bonus = 2.0         # Bonificación por altura correcta de rodilla
        self.stability_bonus = 2.0           # Bonificación por estabilidad
        self.energy_penalty_scale = 0.3      # Penalización por uso excesivo de energía
        self.transition_penalty = 1.0        # Penalización durante transiciones
        
        # ===== TRACKING DE ESTADO =====
        self.current_support_leg = None      # 'left' o 'right'
        self.raised_leg = None               # 'left' o 'right'  
        self.single_leg_time = 0             # Tiempo consecutivo en equilibrio
        self.target_knee_height_diff = 0.3   # Diferencia mínima de altura entre rodillas
        
        # ===== HISTORIA PARA ESTABILIDAD =====
        self.height_history = deque(maxlen=20)
        self.orientation_history = deque(maxlen=20)
        
        print(f"✅ Single Leg Balance Reward System initialized")
    
    def calculate_reward(self, action, current_task="single_leg_balance"):
        """
        Cálculo COMPLETO de recompensa para equilibrio en una pierna.
        
        Args:
            action: Array de 6 presiones PAM [0,1]
            current_task: String indicando la tarea actual
            
        Returns:
            float: Recompensa total
        """
        
        # ===== OBTENER ESTADO BÁSICO DEL ROBOT =====
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        height = pos[2]
        roll, pitch = euler[0], euler[1]
        
        # Actualizar historias
        self.height_history.append(height)
        self.orientation_history.append((roll, pitch))
        
        # ===== COMPONENTE 1: RECOMPENSA DE SUPERVIVENCIA =====
        if height > 0.8:  # Altura mínima para equilibrio en una pierna
            survival_reward = self.base_reward
        elif height > 0.6:
            survival_reward = self.base_reward * 0.5  # Parcial si está bajo pero no caído
        else:
            survival_reward = -5.0  # Penalización fuerte por caída
        
        # ===== COMPONENTE 2: RECOMPENSA DE CONTACTO ASIMÉTRICO =====
        contact_reward = self._calculate_contact_reward()
        
        # ===== COMPONENTE 3: RECOMPENSA DE ALTURA DE RODILLA =====
        knee_height_reward = self._calculate_knee_height_reward()
        
        # ===== COMPONENTE 4: RECOMPENSA DE ESTABILIDAD =====
        stability_reward = self._calculate_stability_reward(roll, pitch)
        
        # ===== COMPONENTE 5: RECOMPENSA DE ACCIÓN PAM =====
        action_reward = self._calculate_action_reward(action)
        
        # ===== COMPONENTE 6: RECOMPENSA DE PROGRESO TEMPORAL =====
        temporal_reward = self._calculate_temporal_reward()
        
        # ===== SUMA TOTAL =====
        total_reward = (survival_reward + contact_reward + knee_height_reward + 
                       stability_reward + action_reward + temporal_reward)
        
        # Limitar rango
        total_reward = np.clip(total_reward, -10.0, 15.0)
        
        return total_reward
    
    def _calculate_contact_reward(self):
        """Calcular recompensa basada en patrón de contacto con el suelo"""
        
        # Verificar contactos (usando link IDs, no joint IDs ya que tobillos son fixed)
        left_contacts = p.getContactPoints(self.robot_id, self.plane_id, 2, -1)  # left_foot_link
        right_contacts = p.getContactPoints(self.robot_id, self.plane_id, 5, -1)  # right_foot_link
        
        left_contact = len(left_contacts) > 0
        right_contact = len(right_contacts) > 0
        
        # Actualizar estado de piernas
        if left_contact and not right_contact:
            self.current_support_leg = 'left'
            self.raised_leg = 'right'
            contact_reward = self.single_leg_bonus
            self.single_leg_time += 1
            
        elif right_contact and not left_contact:
            self.current_support_leg = 'right'
            self.raised_leg = 'left'
            contact_reward = self.single_leg_bonus
            self.single_leg_time += 1
            
        elif left_contact and right_contact:
            # Ambas piernas en contacto - puede ser transición o fallo
            self.current_support_leg = None
            self.raised_leg = None
            contact_reward = -self.transition_penalty  # Penalización moderada
            self.single_leg_time = 0
            
        else:
            # Ningún contacto - caída
            self.current_support_leg = None
            self.raised_leg = None
            contact_reward = -5.0  # Penalización fuerte
            self.single_leg_time = 0
        
        return contact_reward
    
    def _calculate_knee_height_reward(self):
        """Calcular recompensa basada en la altura de la rodilla levantada"""
        
        if self.raised_leg is None:
            return 0.0  # No hay pierna levantada
        
        # Obtener estados de rodillas (links, no joints)
        left_knee_state = p.getLinkState(self.robot_id, 1)   # left_knee_link 
        right_knee_state = p.getLinkState(self.robot_id, 4)  # right_knee_link
        
        left_knee_height = left_knee_state[0][2]  # z position
        right_knee_height = right_knee_state[0][2]
        
        # Determinar alturas según pierna levantada
        if self.raised_leg == 'left':
            raised_height = left_knee_height
            support_height = right_knee_height
        else:
            raised_height = right_knee_height
            support_height = left_knee_height
        
        # Calcular diferencia de altura
        height_difference = raised_height - support_height
        
        # Recompensar diferencia apropiada
        if height_difference > self.target_knee_height_diff:
            # Excelente - rodilla bien levantada
            knee_reward = self.knee_height_bonus
        elif height_difference > 0.15:
            # Bueno - rodilla moderadamente levantada
            knee_reward = self.knee_height_bonus * 0.6
        elif height_difference > 0.05:
            # Regular - rodilla ligeramente levantada
            knee_reward = self.knee_height_bonus * 0.2
        else:
            # Malo - rodilla no levantada suficientemente
            knee_reward = -0.5
        
        return knee_reward
    
    def _calculate_stability_reward(self, roll, pitch):
        """Calcular recompensa basada en estabilidad postural"""
        
        # ===== ESTABILIDAD DE ORIENTACIÓN =====
        
        # Tolerancia más estricta para equilibrio en una pierna
        max_tilt = 0.3  # ~17 grados máximo
        
        current_tilt = abs(roll) + abs(pitch)
        if current_tilt < max_tilt * 0.5:
            orientation_reward = self.stability_bonus  # Muy estable
        elif current_tilt < max_tilt:
            orientation_reward = self.stability_bonus * 0.5  # Moderadamente estable
        else:
            orientation_reward = -current_tilt * 2.0  # Penalización proporcional
        
        # ===== ESTABILIDAD TEMPORAL =====
        
        # Recompensar consistencia en altura y orientación
        if len(self.height_history) >= 10:
            height_variance = np.var(list(self.height_history)[-10:])
            consistency_reward = max(0, 1.0 - height_variance * 100)  # Menos varianza = mejor
        else:
            consistency_reward = 0.0
        
        return orientation_reward + consistency_reward
    
    def _calculate_action_reward(self, action):
        """Calcular recompensa basada en las acciones PAM aplicadas"""
        
        # ===== EFICIENCIA ENERGÉTICA =====
        
        total_activation = np.sum(action)
        
        # Para equilibrio en una pierna, esperamos activación moderada-alta
        if 2.0 <= total_activation <= 4.0:
            efficiency_reward = 0.5  # Rango eficiente
        elif total_activation > 4.5:
            efficiency_reward = -(total_activation - 4.5) * self.energy_penalty_scale  # Penalizar exceso
        else:
            efficiency_reward = -0.5  # Penalizar activación muy baja
        
        # ===== PATRONES PAM APROPIADOS =====
        
        pattern_reward = 0.0
        
        if self.raised_leg == 'left':
            # Pierna izquierda levantada: esperamos más activación en flexor izquierdo de rodilla
            if action[4] > 0.3:  # left_knee_flexor activo
                pattern_reward += 0.5
            # Y equilibrio en cadera derecha
            if action[2] > action[3]:  # right_hip_flexor > right_hip_extensor
                pattern_reward += 0.3
                
        elif self.raised_leg == 'right':
            # Pierna derecha levantada: esperamos más activación en flexor derecho de rodilla  
            if action[5] > 0.3:  # right_knee_flexor activo
                pattern_reward += 0.5
            # Y equilibrio en cadera izquierda
            if action[0] > action[1]:  # left_hip_flexor > left_hip_extensor
                pattern_reward += 0.3
        
        return efficiency_reward + pattern_reward
    
    def _calculate_temporal_reward(self):
        """Calcular recompensa basada en tiempo manteniendo equilibrio"""
        
        # Bonificación creciente por mantener equilibrio en una pierna
        if self.single_leg_time > 50:  # ~3+ segundos a 1500Hz
            temporal_bonus = min(2.0, self.single_leg_time / 100)
        elif self.single_leg_time > 20:  # ~1+ segundos
            temporal_bonus = 0.5
        else:
            temporal_bonus = 0.0
        
        return temporal_bonus
    
    def is_episode_done(self, step_count=0, frequency_simulation=1500.0):
        """
        Determinar si el episodio debe terminar.
        Criterios específicos para equilibrio en una pierna.
        """
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # ===== CRITERIOS DE TERMINACIÓN =====
        
        # 1. Caída crítica
        if pos[2] < 0.5:
            return True
        
        # 2. Inclinación crítica (más estricta para equilibrio en una pierna)
        if abs(euler[0]) > 1.0 or abs(euler[1]) > 1.0:  # ~35 grados
            return True
        
        # 3. Tiempo máximo del episodio
        max_episode_time = frequency_simulation * 10  # 10 segundos
        if step_count >= max_episode_time:
            return True
        
        # 4. Éxito completo - equilibrio mantenido por tiempo extendido
        if self.single_leg_time > frequency_simulation * 8:  # 8 segundos consecutivos
            return True  # Terminar por éxito
        
        return False
    
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
    
    def update_after_episode(self, total_episode_reward):
        """Actualizar después de cada episodio"""
        self.episode_count += 1
        
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
            return self.target_angles['balance_left_support']
        elif current_task == SingleLegActionType.BALANCE_RIGHT_SUPPORT:
            return self.target_angles['balance_right_support']
        else:  # Transiciones
            return self.target_angles['transition']
        
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
                'episodes_needed': 10,       # Episodios mínimos en este nivel
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
    
    def update_after_episode(self, episode_reward):
        """Actualizar nivel después de cada episodio"""
        
        self.episode_count += 1
        self.recent_episodes.append(episode_reward)
        
        # Mantener solo últimos 5 episodios
        if len(self.recent_episodes) > 5:
            self.recent_episodes.pop(0)
        
        # Verificar si subir de nivel
        if len(self.recent_episodes) >= 5:  # Necesitamos al menos 5 episodios
            avg_reward = sum(self.recent_episodes) / len(self.recent_episodes)
            config = self.level_config[self.level]
            
            # ¿Debemos subir de nivel?
            if (avg_reward >= config['upgrade_threshold'] and 
                self.episode_count >= config['min_episodes'] and 
                self.level < 3):
                
                old_level = self.level
                self.level += 1
                
                print(f"\n🎉 LEVEL UP! {old_level} → {self.level}")
                print(f"   Average reward: {avg_reward:.1f}")
                print(f"   Episodes completed: {self.episode_count}")
                
                if self.level == 3:
                    print(f"   🦵 Now alternating legs every 5 seconds!")
    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Caída
        if pos[2] < 0.5:
            log_print("❌ Episode done: Robot fell")
            return True
        
        
        max_tilt = self.max_tilt_by_level.get(self.level, 0.5)
        # Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            log_print("❌ Episode done: Robot tilted too much")
            return True
        
        # Tiempo máximo (crece con nivel)
        max_steps = (200 + (self.level * 200))*10  # 2000, 3000, 4000 steps
        if step_count >= max_steps:
            log_print("⏰ Episode done: Max time reached")
            return True
        
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
