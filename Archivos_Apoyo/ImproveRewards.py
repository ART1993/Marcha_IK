import numpy as np
import math
from collections import deque
import pybullet as p

class ImprovedRewardSystem:
    """
    Sistema de recompensas mejorado que integra la dinámica PAM
    y controla la velocidad para evitar comportamientos erráticos
    """
    
    def __init__(self, left_foot_id, right_foot_id, num_joints):
        # Parámetros de recompensa calibrados
        self.target_forward_velocity = 0.8  # m/s - velocidad objetivo
        self.max_forward_velocity = 2.0     # m/s - velocidad máxima permitida
        self.target_height = 1.0            # m - altura objetivo
        self.max_angular_velocity = 2.0     # rad/s

        # NUEVO: Parámetros para control de altura de pies
        self.target_foot_height = 0.05  # Altura mínima deseada durante swing
        self.max_foot_height = 0.15     # Altura máxima recomendada
        self.ground_clearance = 0.02    # Clearance mínimo del suelo
        
        # Pesos de recompensas (suman a 1.0 para normalización)
        self.weights = {
            'survival': 0.10,        # Recompensa base por estar vivo
            'progress': 0.20,        # Progreso controlado hacia adelante
            'stability': 0.15,       # Estabilidad postural
            'velocity_control': 0.10, # Control de velocidad
            'pam_efficiency': 0.10,  # Eficiencia de músculos PAM
            'gait_quality': 0.15,     # Calidad de la marcha
            'foot_clearance': 0.20  # NUEVO: Control de altura de pies
        }

        self.left_foot_id=left_foot_id
        self.right_foot_id=right_foot_id
        self.num_joints=num_joints

        # Variables para suavizado de recompensas
        self.reward_history = deque(maxlen=10)
        self.smoothing_factor = 0.7
        
        # Tracking mejorado
        self.step_count = 0
        self.previous_foot_positions = None
        self.foot_trajectory_history = deque(maxlen=20)

    def reset_tracking(self):
        """Resetear variables de seguimiento"""
        self.step_count = 0
        self.reward_history.clear()
        self.foot_trajectory_history.clear()
        self.previous_foot_positions = None

    def redefine_robot(self, robot_id, plane_id):
        self.robot_id=robot_id
        self.plane_id=plane_id

    def progress_reward_forward_old(self, lin_vel):
        forward_velocity = lin_vel[0]
        # Recompensa por velocidad en rango óptimo
        if 0 < forward_velocity <= self.target_forward_velocity:
            velocity_reward = min((forward_velocity / self.target_forward_velocity) * 2.0, 3)
        elif self.target_forward_velocity < forward_velocity <= self.max_forward_velocity:
            # Penalización suave por exceso de velocidad
            excess_factor = (forward_velocity - self.target_forward_velocity) / (self.max_forward_velocity - self.target_forward_velocity)
            velocity_reward = 5.0 * (1.0 - excess_factor * 0.5)
        else:
            # Penalización por velocidad excesiva o negativa
            velocity_reward = -2.0 if forward_velocity > self.max_forward_velocity else 0.0
        return velocity_reward

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
            progress_reward = min(forward_velocity / self.target_forward_velocity * 2.0, 3.0)
            # Penalización suave por exceso de velocidad
            if forward_velocity > self.target_forward_velocity:
                excess_penalty = (forward_velocity - self.target_forward_velocity) * 0.5
                progress_reward = max(0, progress_reward - excess_penalty)
        else:
            progress_reward = forward_velocity * 2.0  # Penalización por ir hacia atrás
        
        rewards['progress'] = progress_reward
        
        # 3. ESTABILIDAD POSTURAL
        # Altura estable
        height_error = abs(pos[2] - self.target_height)
        height_reward = max(0, 4.0 - height_error * 5.0)
        
        # Balance angular
        roll_penalty = abs(euler[0]) * 2.0
        pitch_penalty = abs(euler[1]) * 2.0
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
        rewards['pam_efficiency'] = pam_reward
        
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
        
        return max(0, pam_reward)

    def _calculate_gait_quality_reward(self):
        """
        Evalúa la calidad de la marcha basada en contactos y patrones
        """
        # Detectar contactos con el suelo
        left_contact = len(p.getContactPoints(self.robot_id, 0, self.left_foot_id)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, 0, self.right_foot_id)) > 0
        
        gait_reward = 0.0
        
        # 1. Contacto con el suelo (al menos un pie)
        if left_contact or right_contact:
            gait_reward += 1.0
        
        # 2. Alternancia de pasos
        if hasattr(self, 'previous_contacts'):
            # Detectar cambio de contacto
            left_changed = left_contact != self.previous_contacts[0]
            right_changed = right_contact != self.previous_contacts[1]
            
            if left_changed or right_changed:
                gait_reward += 2.0  # Bonus por paso
                
            # Penalizar contacto simultáneo prolongado
            if left_contact and right_contact and self.previous_contacts[0] and self.previous_contacts[1]:
                gait_reward -= 0.5
        
        # 3. Estabilidad temporal de contactos
        contact_stability = 0.5 if (left_contact or right_contact) else 0.0
        gait_reward += contact_stability
        
        # Actualizar historial
        self.previous_contacts = [left_contact, right_contact]
        
        return gait_reward

    def _apply_critical_penalties(self, pos, euler, lin_vel):
        """
        Aplica penalizaciones por comportamientos críticos
        """
        penalty = 0.0
        
        # Caída crítica
        if pos[2] < 0.55:
            penalty -= 25.0
        elif pos[2] < 0.8:
            penalty -= 10.0
        elif pos[2] < 1.0:  # Por debajo de altura objetivo
            penalty -= 1.0
        
        if abs(euler[0]) <math.pi -0.5:
            penalty -=1.0
        elif abs(euler[0]) <math.pi -1.0:
            penalty -=10.0
        elif abs(euler[0]) <math.pi -1.5:
            penalty -=20.0
        # Inclinación excesiva
        if abs(euler[1]) > math.pi/6:
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
            math.pi + initial_roll,   # Corrección base + variación
            initial_pitch,
            initial_yaw
        ])
        
        return correction_quaternion

    def reset_with_random_orientation(self, seed=None, options=None):
        """
        Reset mejorado con orientación inicial aleatoria
        """
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Cargar entorno
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Orientación inicial aleatoria
        initial_quaternion = self.setup_random_initial_orientation()
        
        # Altura inicial con variación
        initial_height = np.random.uniform(1.15, 1.25)
        
        self.robot_id = p.loadURDF(
            self.urdf_path,
            [0, 0, initial_height],
            initial_quaternion,
            useFixedBase=False
        )
        
        # Configurar propiedades físicas
        self.setup_physics_properties()
        
        # Posición inicial de articulaciones con variación
        base_positions = [0.01, -0.02, 0.05, -0.01, -0.01, 0.0]
        initial_joint_positions = []
        
        for pos in base_positions:
            variation = np.random.uniform(-0.02, 0.02)  # ±1.15 grados
            initial_joint_positions.append(pos + variation)
        
        for i, pos in enumerate(initial_joint_positions):
            p.resetJointState(self.robot_id, i, pos)
        
        # Velocidad inicial controlada
        initial_forward_vel = np.random.uniform(0.3, 0.7)
        p.resetBaseVelocity(self.robot_id, [initial_forward_vel, 0, 0], [0, 0, 0])
        
        # Resetear sistema de tracking
        self.step_count = 0
        self.total_reward = 0
        self.observation_history.clear()
        self.previous_position = [0, 0, initial_height]
        
        # Variables específicas
        if hasattr(self, 'pam_muscles'):
            self._setup_motors_for_force_control()
            self.pam_states = {
                'pressures': np.zeros(6),
                'contractions': np.zeros(6),
                'forces': np.zeros(6)
            }
        
        self.previous_contacts = [False, False]
        self.previous_action = None
        
        observation = self._stable_observation()
        info = {'episode_reward': 0, 'episode_length': 0}
        
        return observation, info
    
    def reset_tracking_old(self):
        """Resetear variables de seguimiento"""
        # Variables para tracking
        self.step_count = 0
        self.velocity_history = []
        self.position_history = []
    
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