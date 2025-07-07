import numpy as np
import math


# Alternativa: Versión con 4 articulaciones (sin tobillos)
class FourJointWalkingCycle:
    """
    Versión simplificada con solo 4 articulaciones (sin tobillos)
    """
    
    def __init__(self, step_frequency=1.0):
        self.step_frequency = step_frequency
        self.phase = 0.0
        
        # Solo caderas y rodillas
        self.base_pressures = {
            'left_hip': 0.4,
            'left_knee': 0.3,
            'right_hip': 0.4,
            'right_knee': 0.3
        }
        
        self.modulation_amplitudes = {
            'left_hip': 0.5,
            'left_knee': 0.7,
            'right_hip': 0.5,
            'right_knee': 0.7
        }
    
    def get_four_joint_actions(self, time_step):
        """
        Genera acciones para 4 articulaciones
        """
        self.phase += time_step * self.step_frequency
        self.phase = self.phase % 1.0
        
        left_phase = self.phase * 2 * math.pi
        right_phase = (self.phase + 0.5) * 2 * math.pi
        
        # Solo caderas y rodillas
        hip_left = self.base_pressures['left_hip'] + \
                  self.modulation_amplitudes['left_hip'] * math.sin(left_phase)
        
        knee_left = self.base_pressures['left_knee'] + \
                   self.modulation_amplitudes['left_knee'] * max(0, math.sin(left_phase))
        
        hip_right = self.base_pressures['right_hip'] + \
                   self.modulation_amplitudes['right_hip'] * math.sin(right_phase)
        
        knee_right = self.base_pressures['right_knee'] + \
                    self.modulation_amplitudes['right_knee'] * max(0, math.sin(right_phase))
        
        pressures = [hip_left, knee_left, hip_right, knee_right]
        actions = [2.0 * p - 1.0 for p in pressures]
        
        return np.array(actions, dtype=np.float32)