import numpy as np
import math
from collections import deque
import pybullet as p

class ImprovedRewardSystem:
    """
        Sistema de recompensas mejorado que integra la dinámica PAM
        y controla la velocidad para evitar comportamientos erráticos
    """
    
    def __init__(self, left_foot_id, right_foot_id, num_joints, phase=1):

        self.left_foot_id=left_foot_id
        self.right_foot_id=right_foot_id
        self.num_joints=num_joints
        self.phase= phase

        self.parametros_adicionales

    def redefine_robot(self, robot_id, plane_id):
        self.robot_id=robot_id
        self.plane_id=plane_id

    def _calculate_balanced_reward(self, action, pam_forces):
        """
            Sistema de recompensas balanceado que integra dinámica PAM
        """
        # Estados básicos del robot
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Inicializar componentes de recompensa
        rewards = {}
        total_reward = 0.0
        
        # 1. SUPERVIVENCIA (Recompensa base moderada)
        rewards['survival'] = 1.0  # Reducido de 10.0
        
        # 2. PROGRESO SUAVIZADO
        forward_velocity = lin_vel[0]
        if forward_velocity > 0:
            progress_reward = min(forward_velocity * 10.0, 3.0)
            # Penalización suave por exceso de velocidad
            #if forward_velocity > self.target_forward_velocity:
            #    excess_penalty = (forward_velocity - self.target_forward_velocity) * 0.5
            #    progress_reward = max(0, progress_reward - excess_penalty)
        else:
            progress_reward = forward_velocity * 2.0  # Penalización por ir hacia atrás
        
        rewards['progress'] = progress_reward
        
        # 3. ESTABILIDAD POSTURAL
        # Altura estable
        height_error = abs(pos[2] - self.target_height)
        height_reward = max(0, 2.0 - height_error * 2.0)
        
        # Balance angular
        roll_penalty = abs(euler[0])
        pitch_penalty = abs(euler[1])
        balance_reward = max(0, 0.5 + roll_penalty - pitch_penalty)
        
        rewards['stability'] = (height_reward + balance_reward) / 2.0
        
        # 4. CONTROL DE VELOCIDAD (evitar movimientos erráticos)
        # Penalizar velocidades angulares excesivas
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        if ang_vel_magnitude > self.max_angular_velocity:
            angular_penalty = (ang_vel_magnitude - self.max_angular_velocity) * 2.0
        else:
            angular_penalty = 0.0
        
        # Penalizar aceleración lateral excesiva
        lateral_velocity_penalty = min(abs(lin_vel[1]) * 2.0, 3.0)
        
        rewards['velocity_control'] = max(0, 4.0 - angular_penalty - lateral_velocity_penalty)

        # 5. EFICIENCIA PAM (integración con dinámica real)
        if hasattr(self, 'pam_states'):
            pam_reward = self._calculate_pam_efficiency_reward(action, pam_forces)
        else:
            pam_reward = 0.0
        if self.phase != 1:
            rewards['pam_efficiency'] = pam_reward
        else:
            rewards['pam_efficiency'] = 0.0

        #if hasattr(self, 'previous_action') and self.previous_action is not None:
        #    action_delta = np.linalg.norm(action - self.previous_action)
        #    total_reward -= 0.1 * action_delta  # Ajusta el peso según tus pruebas
        self.previous_action = np.copy(action)
        
        # 6. CALIDAD DE MARCHA
        gait_reward = self._calculate_gait_quality_reward()
        rewards['gait_quality'] = gait_reward
        
        # Combinar recompensas con pesos
        for component, reward in rewards.items():
            total_reward += reward * self.weights[component]
        
        # Aplicar penalizaciones críticas
        total_reward += self._apply_critical_penalties(pos, euler, lin_vel)
        
        # Limitar recompensa total
        total_reward = np.clip(total_reward, -10.0, 20.0)
        
        return total_reward, rewards

    def _calculate_pam_efficiency_reward(self, action, pam_forces):
        """
        Recompensa basada en la eficiencia de los músculos PAM
        """
        if not hasattr(self, 'pam_states'):
            return 0.0
        
        pam_reward = 0.0
        
        # 1. Eficiencia energética: penalizar presiones muy altas sin movimiento útil
        pressures = self.pam_states['pressures']
        forces = self.pam_states['forces']

        # Calcular eficiencia: fuerza útil / presión aplicada
        total_pressure = np.sum(pressures)
        total_force = np.sum(np.abs(forces))
        
        if total_pressure > 0:
            efficiency = total_force / (total_pressure + 1e-6)
            pam_reward += min(efficiency * 0.001, 2.0)  # Normalizar
            energy_penalty= 1e-7 * (total_pressure**2)
            pam_reward -=energy_penalty
        
        # 2. Coordinación muscular: premiar activación alternada
        left_muscles = pressures[:3]  # hip, knee, ankle izquierdo
        right_muscles = pressures[3:6]  # hip, knee, ankle derecho
        
        left_activation = np.mean(left_muscles)
        right_activation = np.mean(right_muscles)
        
        # Premiar diferencia moderada (indicativo de marcha)
        activation_diff = abs(left_activation - right_activation)
        coordination_reward = max(0, 1.5 - abs(activation_diff - self.max_pressure * 0.3) * 0.000005)
        pam_reward += coordination_reward
        
        # 3. Penalizar saturación (presiones máximas constantemente)
        saturated_muscles = np.sum(pressures > self.max_pressure * 0.9)
        if saturated_muscles > 3:  # Más de la mitad de músculos saturados
            pam_reward -= 1.0
        
        # 4. Suavidad en cambios de presión
        if hasattr(self, 'previous_action') and self.previous_action is not None:
            pressure_changes = np.abs(action - self.previous_action)
            smoothness = max(0, 1.0 - np.mean(pressure_changes) * 2.0)
            pam_reward += smoothness
        self.previous_action = np.copy(action)
        
        return max(0, pam_reward)

    def _calculate_gait_quality_reward(self):
        """
        Evalúa la calidad de la marcha basada en contactos y patrones,
        incentivando alternancia de soporte, swing real y penalizando doble apoyo prolongado.
        """
        # 1. Detectar contacto de los pies
        left_contact = len(p.getContactPoints(self.robot_id, 0, self.left_foot_id)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, 0, self.right_foot_id)) > 0

        gait_reward = 0.0

        # 2. Bonus por tener al menos un pie tocando el suelo
        if left_contact or right_contact:
            gait_reward += 1.0

        # 3. Penalizar doble apoyo prolongado
        if left_contact and right_contact:
            self.double_support_steps = getattr(self, "double_support_steps", 0) + 1
        else:
            self.double_support_steps = 0
        if self.double_support_steps > 10:
            gait_reward -= 1.0  # penalización si hay doble apoyo por muchos pasos

        # 4. Incentiva alternancia de apoyo (“single support”)
        if hasattr(self, 'last_single_support') and self.last_single_support is not None:
            prev = self.last_single_support
            current = (left_contact, right_contact)
            if prev != current and (prev in [(True, False), (False, True)]) and (current in [(True, False), (False, True)]):
                gait_reward += 1.5  # Bonus por alternancia de soporte
        self.last_single_support = (left_contact, right_contact)

        # 5. Premia swing real (pie levantado > 2.5 cm)
        left_foot_pos = p.getLinkState(self.robot_id, self.left_foot_id)[0]
        right_foot_pos = p.getLinkState(self.robot_id, self.right_foot_id)[0]
        ground_z = 0.0  # Cambia esto si tu plano base está a otra altura

        if not left_contact and (left_foot_pos[2] - ground_z) > 0.025:
            gait_reward += 0.5
        if not right_contact and (right_foot_pos[2] - ground_z) > 0.025:
            gait_reward += 0.5

        # 6. Penalización fuerte si ambos pies en el aire (posible salto/caída)
        if not left_contact and not right_contact:
            gait_reward -= 3.0

        # 7. Bonus por estabilidad temporal (original)
        contact_stability = 0.5 if (left_contact or right_contact) else 0.0
        gait_reward += contact_stability

        # 8. Actualiza historial
        self.previous_contacts = [left_contact, right_contact]

        return gait_reward

    def _apply_critical_penalties(self, pos, euler, lin_vel):
        """
        Aplica penalizaciones por comportamientos críticos
        """
        penalty = 0.0
        
        # Caída crítica
        if pos[2] < 0.55:
            penalty -= 10.0
        elif pos[2] < 0.8:
            penalty -= 2.0
        elif pos[2] < 1.0:  # Por debajo de altura objetivo
            penalty -= 0.1
        
        #if abs(euler[0]) <math.pi -0.5:
        #    penalty -=1.0
        #elif abs(euler[0]) <math.pi -1.0:
        #    penalty -=10.0
        #elif abs(euler[0]) <math.pi -1.5:
        #    penalty -=20.0
        # Inclinación excesiva
        if abs(euler[1]) > math.pi/3:
            penalty -= 3.0
        
        # Velocidad lateral excesiva
        if abs(lin_vel[1]) > 1.5:
            penalty -= 2.0
        
        # Velocidad hacia atrás
        if lin_vel[0] < -0.5:
            penalty -= 2.0
        
        return penalty

    def setup_random_initial_orientation(self):
        """
        Configura orientación inicial aleatoria para diversificar entrenamiento
        """
        # Rangos de variación para orientación inicial
        roll_range = (-0.1, 0.1)     # ±5.7 grados
        pitch_range = (-0.05, 0.15)  # Ligeramente hacia adelante
        yaw_range = (-0.1, 0.1)      # ±5.7 grados rotación
        
        # Generar ángulos aleatorios
        initial_roll = np.random.uniform(*roll_range)
        initial_pitch = np.random.uniform(*pitch_range)
        initial_yaw = np.random.uniform(*yaw_range)
        
        # Crear quaternion de orientación inicial
        correction_quaternion = p.getQuaternionFromEuler([
            initial_roll,   # Corrección base + variación
            initial_pitch,
            initial_yaw
        ])
        
        return correction_quaternion

##############################################################################################################################
##############################################Atributos adicionales para el entorno###########################################
##############################################################################################################################

    @property
    def parametros_adicionales(self):
        # Parámetros de recompensa calibrados
        self.target_forward_velocity = 1.1  # m/s - velocidad objetivo
        self.max_forward_velocity = 2.0     # m/s - velocidad máxima permitida
        self.target_height = 1.3            # m - altura objetivo
        self.max_angular_velocity = 2.0     # rad/s

        # NUEVO: Parámetros para control de altura de pies
        self.target_foot_height = 0.1  # Altura mínima deseada durante swing
        self.max_foot_height = 0.3     # Altura máxima recomendada
        self.ground_clearance = 0.05    # Clearance mínimo del suelo
        
        # Pesos de recompensas (suman a 1.0 para normalización)
        self.weights = {
            'survival': 0.10,        # Recompensa base por estar vivo
            'progress': 0.40,        # Progreso controlado hacia adelante
            'stability': 0.10,       # Estabilidad postural
            'velocity_control': 0.10, # Control de velocidad
            'pam_efficiency': 0.05,  # Eficiencia de músculos PAM
            'gait_quality': 0.15,     # Calidad de la marcha
            'foot_clearance': 0.10  # NUEVO: Control de altura de pies
        }

        # Variables para suavizado de recompensas
        self.reward_history = deque(maxlen=10)
        self.smoothing_factor = 0.7
        
        # Tracking mejorado
        self.step_count = 0
        self.previous_foot_positions = None
        self.foot_trajectory_history = deque(maxlen=20)
    
# Parámetros adicionales para el entorno
class PAMTrainingConfig:
    """
    Configuración optimizada para entrenamiento PAM
    """
    
    # Límites de velocidad
    TARGET_VELOCITY = 0.8  # m/s
    MAX_VELOCITY = 2.0     # m/s
    
    # Parámetros de recompensa
    REWARD_SCALE = 10.0    # Escala general de recompensas
    
    # Criterios de terminación
    MIN_HEIGHT = 1.6       # m
    MAX_TILT = math.pi/2   # rad
    MAX_EPISODE_STEPS = 2000
    
    # Variación inicial
    ORIENTATION_NOISE = 0.1  # rad
    HEIGHT_NOISE = 0.1       # m
    JOINT_NOISE = 0.02       # rad