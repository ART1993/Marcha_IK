import pybullet as p
import numpy as np
import math
import time

class SimpleWalkingCycle:
    """
    Ciclo de paso sencillo para robot blando con músculos PAM
    Mantiene 4 articulaciones pero con patrones simplificados
    """
    
    def __init__(self, robot_id, step_frequency=2.0, step_length=0.1):
        self.robot_id=robot_id

        self.step_frequency = step_frequency  # Hz
        self.step_length = step_length        # metros
        self.step_height = 0.2  # altura del pie durante swing
        #self.support_knee_drop = 0.35  # Ajusta según lo que te funcione visualmente
        self.base_x_offset = 0.0  # <-- NUEVO: avanza con cada ciclo
        self.prev_phase = 0.0     # Para detectar nuevo ciclo
        self.base_height=1.2
        self.warking_cycle_params
        
    def update_phase(self, dt):
        """Actualiza la fase del ciclo de paso"""
        self.phase += dt * self.step_frequency
        self.phase = self.phase % 1.0  # Mantener en rango [0,1]
    
    def get_simple_walking_actions(self, time_step):
        """
        Genera acciones de paso sencillas basadas en patrones senoidales
        Returns: array de 4 presiones PAM normalizadas [-1, 1]
        """
        self.update_phase(time_step)
        # Patrones básicos usando senos desfasados
        left_leg_phase = self.phase * 2 * math.pi
        right_leg_phase = (self.phase + 0.5) * 2 * math.pi  # Desfase de 180°
        
        actions = []
        
        # PIERNA IZQUIERDA
        # Cadera izquierda: flexión/extensión
        hip_left = self.base_pressures['left_hip'] + \
                  self.modulation_amplitudes['left_hip'] * math.sin(left_leg_phase)
        
        # Rodilla izquierda: flexión durante swing
        knee_left = self.base_pressures['left_knee'] + \
                   self.modulation_amplitudes['left_knee'] * max(0, math.sin(left_leg_phase))
        
        # PIERNA DERECHA (desfasada 180°)
        hip_right = self.base_pressures['right_hip'] + \
                   self.modulation_amplitudes['right_hip'] * math.sin(right_leg_phase)
        
        knee_right = self.base_pressures['right_knee'] + \
                    self.modulation_amplitudes['right_knee'] * max(0, math.sin(right_leg_phase))

        
        # Compilar acciones y normalizar a [-1, 1]
        pressures = [hip_left, knee_left, hip_right, knee_right]
        
        # Convertir de [0,1] a [-1,1] para el entorno
        actions = [2.0 * pressure - 1.0 for pressure in pressures]
        
        # Asegurar límites
        actions = [max(-1.0, min(1.0, action)) for action in actions]
        
        return np.array(actions, dtype=np.float32)
    
    def get_initialization_sequence(self, duration=2.0, dt=0.01):
        """
        Genera secuencia de inicialización para estabilizar el robot
        """
        steps = int(duration / dt)
        sequence = []
        
        for i in range(steps):
            # Ramp-up gradual de presiones
            ramp_factor = min(1.0, i / (steps * 0.3))  # 30% del tiempo para ramp-up
            
            # Presiones simétricas iniciales
            base_pressure = 0.3 * ramp_factor
            
            actions = [
                base_pressure,      # left_hip
                base_pressure * 0.7, # left_knee  
                base_pressure,      # right_hip
                base_pressure * 0.7, # right_knee
            ]
            
            # Normalizar a [-1,1]
            actions = [2.0 * action - 1.0 for action in actions]
            sequence.append(np.array(actions, dtype=np.float32))
        
        return sequence
    
################################################################################################################################
###############################################Parabola simple para trayectorias de pie#########################################
################################################################################################################################
    
    def foot_parabola(self, start_pos, end_pos, height, alpha):
        """
        Trayectoria parabólica simple entre dos puntos en el espacio.
        Args:
            start_pos: posición inicial [x, y, z]
            end_pos: posición final [x, y, z]
            height: altura máxima del pie
            alpha: progreso del swing [0,1]
        Returns:
            posición 3D en [x, y, z]
        """
        x = (1 - alpha) * start_pos[0] + alpha * end_pos[0]
        y = (1 - alpha) * start_pos[1] + alpha * end_pos[1]
        z = (1 - alpha) * start_pos[2] + alpha * end_pos[2] + 4 * height * alpha * (1 - alpha)
        return [x, y, z]
    
    def get_trajectory_walking_actions(self, time_step, left_foot_index, right_foot_index):
        """
        Genera acciones de paso basadas en trayectoria del pie y cinemática inversa.
        Requiere índices de links de pies izquierdo y derecho.
        Returns: array de posiciones articulares
        """
        self.update_phase(time_step)
        alpha = self.phase % 1.0

        # Avanza el offset de la base_link al iniciar un nuevo ciclo
        if alpha < self.prev_phase:
            self.base_x_offset += self.step_length
        self.prev_phase = alpha

        # Target X de la base/cadera en el ciclo actual
        base_target_x = self.base_x_offset + alpha * self.step_length
        base_target_y = 0.0
        base_target_z = self.base_height
        base_target = (base_target_x, base_target_y, base_target_z)

        # Parámetros para pies
        foot_y = 0.10    # Separación lateral de los pies
        swing_z = self.step_height

        # Alternancia de swing entre pies
        if alpha < 0.5:
            # Derecho en swing
            swing_alpha = alpha / 0.5
            right_foot_x = base_target_x + self.step_length / 2
            right_foot_y = -foot_y
            right_foot_z = base_target_z + swing_z * np.sin(np.pi * swing_alpha)
            right_foot_target = (right_foot_x, right_foot_y, right_foot_z)

            left_foot_x = base_target_x - self.step_length / 2
            left_foot_y = foot_y
            left_foot_z = 0.0   # Pie soporte
            left_foot_target = (left_foot_x, left_foot_y, left_foot_z)
        else:
            # Izquierdo en swing
            swing_alpha = (alpha - 0.5) / 0.5
            left_foot_x = base_target_x + self.step_length / 2
            left_foot_y = foot_y
            left_foot_z = base_target_z + swing_z * np.sin(np.pi * swing_alpha)
            left_foot_target = (left_foot_x, left_foot_y, left_foot_z)

            right_foot_x = base_target_x - self.step_length / 2
            right_foot_y = -foot_y
            right_foot_z = 0.0   # Pie soporte
            right_foot_target = (right_foot_x, right_foot_y, right_foot_z)
        
        left_joint_positions = p.calculateInverseKinematics(self.robot_id, left_foot_index, left_foot_target)
        right_joint_positions = p.calculateInverseKinematics(self.robot_id, right_foot_index, right_foot_target)

        # Devuelve todas las articulaciones, por ejemplo concatenando
        # Suponiendo que el orden es [left_hip, left_knee, right_hip, right_knee]
        return np.array(list(left_joint_positions[:2]) + list(right_joint_positions[:2]), dtype=np.float32)  # posición deseada de las articulaciones
    

############################################################################################################################
########################################Agregar al entorno de simulación en futuro##########################################
############################################################################################################################

    @property
    def warking_cycle_params(self):
        self.phase = 0.0                      # Fase del ciclo (0-1)
        # Configuración simplificada del ciclo
        self.cycle_phases = {
            'stance_left': (0.0, 0.5),   # Pie izquierdo en suelo
            'stance_right': (0.5, 1.0),  # Pie derecho en suelo  
        }
        
        # Patrones de presión base para cada articulación (0-1)
        self.base_pressures = {
            'left_hip': 0.3,
            'left_knee': 0.2,
            'right_hip': 0.3,
            'right_knee': 0.2,
        }
        
        # Amplitudes de modulación para cada articulación
        self.modulation_amplitudes = {
            'left_hip': 0.4,
            'left_knee': 0.7,
            'right_hip': 0.4,
            'right_knee': 0.7,
        }

def foot_bezier_parabola(start, end, ctrl1, ctrl2, alpha, height):
    """
        Trayectoria mixta: curva Bézier para (x, y), parábola para z.
        - start, end: posiciones 3D (inicio y fin del swing)
        - ctrl1, ctrl2: puntos de control Bézier (3D)
        - alpha: progreso de swing [0, 1]
        - height: altura máxima de swing (aplica a z)
    """
    # Bézier para X e Y (puedes usarlo también para Z base si quieres)
    def bezier(p0, p1, p2, p3, t):
        return ((1 - t)**3) * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    # Bézier en x, y (z puede quedarse lineal o usar solo inicio y fin)
    x = bezier(start[0], ctrl1[0], ctrl2[0], end[0], alpha)
    y = bezier(start[1], ctrl1[1], ctrl2[1], end[1], alpha)

    # Parábola para z
    z_base = (1 - alpha) * start[2] + alpha * end[2]
    z = z_base + height * 4 * alpha * (1 - alpha)

    return [x, y, z]
