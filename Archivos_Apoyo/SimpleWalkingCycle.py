import pybullet as p
import numpy as np
import math
import time

class SimpleWalkingCycle:
    """
    Ciclo de paso sencillo para robot blando con músculos PAM
    Mantiene 4 articulaciones pero con patrones simplificados
    """
    
    def __init__(self, robot_id, step_frequency=2.0, step_length=0.3):
        self.step_frequency = step_frequency  # Hz
        self.step_length = step_length        # metros
        self.robot_id=robot_id

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
        alpha = self.phase #(self.phase % 1.0)  # [0,1]
        step_length = self.step_length
        step_height = 0.15  # altura del pie durante swing
        print(self.phase)
        # Obtener posiciones actuales de los pies
        left_start = p.getLinkState(self.robot_id, left_foot_index)[0]
        right_start = p.getLinkState(self.robot_id, right_foot_index)[0]

        if alpha < 0.5:
            # Pierna derecha en swing
            swing_alpha = alpha / 0.5
            start=right_start
            end = [right_start[0] + step_length, right_start[1], right_start[2]]
            ctrl1 = [start[0] + 0.1, start[1], start[2] + step_height]
            ctrl2 = [end[0] - 0.1, end[1], end[2] + step_height]
            target_pos = foot_bezier_parabola(
                start=start,end=end,
                ctrl1=ctrl1,ctrl2=ctrl2,
                alpha=swing_alpha,height=step_height
            )
            joint_positions = p.calculateInverseKinematics(self.robot_id, right_foot_index, target_pos)
        else:
            # Pierna izquierda en swing
            swing_alpha = (alpha - 0.5) / 0.5
            start=left_start
            end = [left_start[0] + step_length, left_start[1], left_start[2]]
            ctrl1 = [start[0] + 0.1, start[1], start[2] + step_height]
            ctrl2 = [end[0] - 0.1, end[1], end[2] + step_height]
            target_pos = foot_bezier_parabola(
                start=start,end=end,
                ctrl1=ctrl1,ctrl2=ctrl2,
                alpha=swing_alpha,height=step_height
            )
            joint_positions = p.calculateInverseKinematics(self.robot_id, left_foot_index, target_pos)

        return joint_positions  # posición deseada de las articulaciones
    

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