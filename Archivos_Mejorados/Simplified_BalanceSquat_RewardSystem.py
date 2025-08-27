import numpy as np
import pybullet as p
from collections import deque

class Simplified_BalanceSquat_RewardSystem:
    """
    Sistema de recompensas SIMPLIFICADO para balance y sentadillas.
    
    OBJETIVO ESPECÍFICO:
    - Recompensar equilibrio estático (BALANCE_STANDING)
    - Recompensar sentadillas controladas (SQUAT)
    - Eficiencia básica de PAMs
    
    ELIMINADO:
    - Recompensas de marcha complejas (gait_quality, foot_clearance, progress)
    - Múltiples fases de curriculum (0-4+)
    - Métricas biomecánicas avanzadas
    - Análisis temporal de coordinación
    - Sistemas de pesos adaptativos
    """
    
    def __init__(self, robot_id=None, plane_id=None, left_foot_id=2, right_foot_id=5):
        
        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.robot_id = robot_id
        self.plane_id = plane_id  
        self.left_foot_id = left_foot_id
        self.right_foot_id = right_foot_id
        
        # ===== PARÁMETROS DE RECOMPENSA SIMPLIFICADOS =====
        
        # Objetivos para balance
        self.target_height = 1.1  # Altura objetivo del torso
        self.max_roll_pitch = 0.3  # Máxima inclinación permitida (radianes)
        
        # Pesos de recompensa (fijos, no adaptativos)
        self.weights = {
            'survival': 1.0,        # Recompensa base por estar de pie
            'height': 3.0,          # Mantener altura correcta
            'orientation': 4.0,     # Mantener orientación vertical
            'stability_zmp': 2.0,   # Estabilidad ZMP
            'contact': 2.0,         # Contacto con el suelo
            'velocity_penalty': 1.0, # Penalizar movimiento excesivo
            'pam_efficiency': 1.0   # Eficiencia básica PAM
        }
        
        # ===== VARIABLES DE TRACKING BÁSICAS =====
        
        self.step_count = 0
        self.pam_states = None
        self.max_pressure = 5 * 101325  # 5 atm
        
        print(f"🎯 Simplified Balance & Squat Reward System initialized")
        print(f"   Focus: Height + Orientation + Stability")
        print(f"   Weights: {self.weights}")
    
    def redefine_robot(self, robot_id, plane_id):
        """Actualizar IDs del robot"""
        self.robot_id = robot_id
        self.plane_id = plane_id
    
    def calculate_simple_reward(self, action=None, pam_forces=None):
        """
        Calcular recompensa SIMPLIFICADA para balance y sentadillas.
        
        Returns:
            tuple: (total_reward, reward_components)
        """
        
        if self.robot_id is None:
            return 0.0, {}
        
        # Obtener estados básicos del robot
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        left_contact, right_contact=self.get_detect_feet_contact
        contactos_pies=all((left_contact, right_contact))
        rewards = {}
        
        # ===== 1. RECOMPENSA DE SUPERVIVENCIA =====
        rewards['survival'] = 1.0  # Recompensa base por estar activo
        
        # ===== 2. RECOMPENSA POR ALTURA (Mantenerse erguido) =====

        height_error = abs(pos[2] - self.target_height)
        if height_error < 0.3:
            rewards['height'] = 2.0  # Altura perfecta
        elif height_error < 0.5:
            rewards['height'] = 1.0 - height_error  # Degradación gradual
        else:
            rewards['height'] = -2.0  # Muy lejos de altura objetivo
        
        # ===== 3. RECOMPENSA POR ORIENTACIÓN (Vertical) =====
        roll_error = abs(euler[0])  # Inclinación lateral
        pitch_error = abs(euler[1])  # Inclinación adelante/atrás
        
        if roll_error < 0.1 and pitch_error < 0.1:
            rewards['orientation'] = 3.0  # Orientación perfecta
        elif roll_error < self.max_roll_pitch and pitch_error < self.max_roll_pitch:
            # Degradación gradual
            orientation_error = roll_error + pitch_error
            rewards['orientation'] = 2.0 * (1.0 - orientation_error / self.max_roll_pitch)
        else:
            rewards['orientation'] = -5.0  # Inclinación excesiva
        
        # ===== 4. RECOMPENSA POR ESTABILIDAD ZMP =====
        # (Simplificada - verificar contactos de pies)
        
        if contactos_pies:
            rewards['stability_zmp'] = 2.0  # Ambos pies en el suelo = estable
        elif left_contact or right_contact:
            rewards['stability_zmp'] = 0.5  # Un pie en el suelo = moderadamente estable
        else:
            rewards['stability_zmp'] = -10.0  # Sin contacto = inestable
        
        # ===== 5. RECOMPENSA POR CONTACTO =====
        if contactos_pies:
            rewards['contact'] = 1.0
        elif left_contact or right_contact:
            rewards['contact'] = 0.5
        else:
            rewards['contact'] = -3.0
        
        # ===== 6. PENALIZACIÓN POR VELOCIDAD EXCESIVA =====
        # (Para balance, queremos movimiento mínimo)
        velocity_magnitude = np.linalg.norm(lin_vel[2])  # Solo z
        angular_magnitude = np.linalg.norm(ang_vel)
        
        if velocity_magnitude < 0.1 and angular_magnitude < 0.2:
            rewards['velocity_penalty'] = 0.5  # Movimiento mínimo = bueno
        else:
            # Penalizar movimiento excesivo
            velocity_penalty = velocity_magnitude * 2.0 + angular_magnitude * 1.5
            rewards['velocity_penalty'] = -min(velocity_penalty, 3.0)
        
        # ===== 7. EFICIENCIA PAM BÁSICA =====
        if self.pam_states is not None and action is not None:
            rewards['pam_efficiency'] = self._calculate_basic_pam_efficiency(action)
        else:
            rewards['pam_efficiency'] = 0.0
        
        # ===== COMBINACIÓN FINAL =====
        
        total_reward = 0.0
        for component, reward in rewards.items():
            weighted_reward = reward * self.weights[component]
            total_reward += weighted_reward
        
        # ===== PENALIZACIONES CRÍTICAS =====
        
        # Caída (altura muy baja)
        if pos[2] < 0.6:
            total_reward -= 20.0
        
        # Inclinación crítica  
        if roll_error > 0.5 or pitch_error > 0.5:  # ~30 grados
            total_reward -= 15.0
        
        # Limitar recompensa total
        total_reward = np.clip(total_reward, -25.0, 15.0)
        
        return total_reward, rewards
    
    def _calculate_basic_pam_efficiency(self, action):
        """
        Eficiencia PAM BÁSICA - Penalizar uso excesivo de energía.
        
        Para balance, queremos activación mínima pero suficiente.
        """
        
        if self.pam_states is None:
            return 0.0
        
        # Presiones normalizadas [0,1]
        normalized_pressures = np.array(action)
        
        # ===== EFICIENCIA ENERGÉTICA =====
        
        # Penalizar activación total excesiva
        total_activation = np.sum(normalized_pressures)
        if total_activation < 1.0:  # Muy poca activación = inestable
            energy_reward = -1.0
        elif total_activation < 2.5:  # Activación moderada = eficiente
            energy_reward = 1.0
        else:  # Activación excesiva = desperdicio
            energy_reward = -0.5 * (total_activation - 2.5)
        
        # ===== BALANCE ANTAGÓNICO BÁSICO =====
        
        # Para caderas: comparar flexor vs extensor
        antagonistic_reward = 0.0
        
        # Cadera izquierda: PAM 0 (flexor) vs PAM 1 (extensor)
        left_hip_balance = 1.0 - abs(normalized_pressures[0] - normalized_pressures[1])
        antagonistic_reward += left_hip_balance * 0.2
        
        # Cadera derecha: PAM 2 (flexor) vs PAM 3 (extensor)  
        right_hip_balance = 1.0 - abs(normalized_pressures[2] - normalized_pressures[3])
        antagonistic_reward += right_hip_balance * 0.2
        
        # ===== SIMETRÍA BILATERAL =====
        
        # Comparar lado izquierdo vs derecho
        left_side_avg = (normalized_pressures[0] + normalized_pressures[1] + normalized_pressures[4]) / 3
        right_side_avg = (normalized_pressures[2] + normalized_pressures[3] + normalized_pressures[5]) / 3
        
        bilateral_balance = 1.0 - abs(left_side_avg - right_side_avg)
        bilateral_reward = bilateral_balance * 0.3
        
        # Combinar componentes
        total_efficiency = energy_reward + antagonistic_reward + bilateral_reward
        
        return np.clip(total_efficiency, -2.0, 2.0)
    
    def is_episode_done(self):
        """
        Determinar si el episodio debe terminar (condiciones de falla).
        
        SIMPLIFICADO - Solo condiciones críticas.
        """
        
        if self.robot_id is None:
            return False
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Caída (altura muy baja)
        if pos[2] < 0.5:
            return True
        
        # Inclinación crítica (>45 grados)
        if abs(euler[0]) > 0.785 or abs(euler[1]) > 0.785:
            return True
        
        # Desplazamiento lateral excesivo
        if abs(pos[1]) > 1.5:
            return True
        
        return False
    
    def get_reward_summary(self):
        """Resumen de configuración actual"""
        return {
            'system': 'Simplified Balance & Squat Rewards',
            'focus': 'Height + Orientation + Stability',
            'target_height': self.target_height,
            'max_inclination': self.max_roll_pitch,
            'weights': self.weights,
            'pam_efficiency_enabled': self.pam_states is not None
        }
    @property
    def get_detect_feet_contact(self):
        # ✅ VERIFICAR CONTACTO BILATERAL SIMPLE
        left_contacts = len(p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_id, -1))>0
        right_contacts = len(p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_id, -1))>0

        return left_contacts, right_contacts
