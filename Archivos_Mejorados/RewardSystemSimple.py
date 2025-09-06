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
        self.angle_controller = AngleBasedExpertController(env.robot_id)
        
        # ===== ESTADO INTERNO =====
        self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
        self.last_10_rewards = deque(maxlen=10)
        self.time_in_current_stance = 0
        self.target_switch_time = 150
        
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
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # ===== PARÁMETROS DEL CONTROLADOR PD =====
        self.kp = 80.0   # Ganancia proporcional
        self.kd = 12.0   # Ganancia derivativa
        self.max_torque = 50.0  # Torque máximo por articulación
        
        # ===== ÁNGULOS OBJETIVO SEGÚN TAREA =====
        self.target_angles = {
            # Equilibrio en pierna IZQUIERDA (derecha levantada)
            'balance_left_support': {
                'left_hip': 0.0,        # Cadera izq: recta para soporte
                'left_knee': 0.0,       # Rodilla izq: extendida para soporte
                'right_hip': 0.7,       # Cadera der: flexión 40° (0.7 rad ≈ 40°)
                'right_knee': 0.7,      # Rodilla der: flexión 40° para levantar
                'description': 'Pierna derecha levantada 40°'
            },
            
            # Equilibrio en pierna DERECHA (izquierda levantada)
            'balance_right_support': {
                'left_hip': 0.7,        # Cadera izq: flexión 40°
                'left_knee': 0.7,       # Rodilla izq: flexión 40° para levantar
                'right_hip': 0.0,       # Cadera der: recta para soporte
                'right_knee': 0.0,      # Rodilla der: extendida para soporte
                'description': 'Pierna izquierda levantada 40°'
            },
            
            # Transiciones - ángulos intermedios
            'transition': {
                'left_hip': 0.05,        # Ligera flexión bilateral
                'left_knee': 0.05,
                'right_hip': 0.05,
                'right_knee': 0.05,
                'description': 'Posición intermedia para transición'
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
    
    def calculate_pd_torques(self, target_angles_dict):
        """
        Calcular torques usando control PD hacia ángulos objetivo
        
        Args:
            target_angles_dict: Diccionario con ángulos objetivo
            
        Returns:
            numpy.array: Torques para [left_hip, left_knee, right_hip, right_knee]
        """
        
        # Obtener estados actuales de articulaciones
        joint_states = p.getJointStates(self.robot_id, [0, 1, 3, 4])
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
        
        # Obtener estados actuales para cálculos biomecánicos
        joint_states = p.getJointStates(self.robot_id, [0, 1, 3, 4])
        joint_angles = [state[0] for state in joint_states]
        
        # Inicializar presiones PAM
        pam_pressures = np.zeros(6)
        
        # Parámetros biomecánicos (mismos que en _apply_pam_forces)
        moment_arm = 0.05
        max_pressure_normalized = 1.0
        
        # ===== CONVERSIÓN TORQUE → PRESIONES PAM =====
        
        # CADERA IZQUIERDA (PAM 0=flexor, PAM 1=extensor)
        hip_left_torque = desired_torques[0]
        
        if hip_left_torque > 0:  # Flexión necesaria
            # Activar flexor, desactivar extensor
            pam_pressures[0] = min(max_pressure_normalized, 
                                 abs(hip_left_torque) / (moment_arm * 300))  # 300N estimado por PAM
            pam_pressures[1] = 0.1  # Mínimo para el extensor
        else:  # Extensión necesaria
            # Activar extensor, desactivar flexor
            pam_pressures[0] = 0.1  # Mínimo para el flexor
            pam_pressures[1] = min(max_pressure_normalized,
                                 abs(hip_left_torque) / (moment_arm * 300))
        
        # CADERA DERECHA (PAM 2=flexor, PAM 3=extensor)
        hip_right_torque = desired_torques[2]
        
        if hip_right_torque > 0:  # Flexión necesaria
            pam_pressures[2] = min(max_pressure_normalized,
                                 abs(hip_right_torque) / (moment_arm * 300))
            pam_pressures[3] = 0.1
        else:  # Extensión necesaria
            pam_pressures[2] = 0.1
            pam_pressures[3] = min(max_pressure_normalized,
                                 abs(hip_right_torque) / (moment_arm * 300))
        
        # RODILLAS (PAM 4=izq_flexor, PAM 5=der_flexor)
        # Solo flexores, extensión es pasiva por resortes
        
        knee_left_torque = desired_torques[1]
        knee_right_torque = desired_torques[3]
        
        # Rodilla izquierda - solo flexión activa
        if knee_left_torque > 5.0:  # Umbral para activar flexor
            pam_pressures[4] = min(max_pressure_normalized,
                                 knee_left_torque / (moment_arm * 200))  # 200N para rodilla
        else:
            pam_pressures[4] = 0.05  # Muy bajo para permitir extensión pasiva
        
        # Rodilla derecha - solo flexión activa
        if knee_right_torque > 5.0:  # Umbral para activar flexor
            pam_pressures[5] = min(max_pressure_normalized,
                                 knee_right_torque / (moment_arm * 200))
        else:
            pam_pressures[5] = 0.05  # Muy bajo para permitir extensión pasiva
        
        # Asegurar rango [0, 1]
        pam_pressures = np.clip(pam_pressures, 0.0, 1.0)
        
        return pam_pressures
