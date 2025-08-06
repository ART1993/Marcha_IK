from Archivos_Apoyo.SimpleWalkingCycle import SimpleWalkingCycle
import numpy as np


class Enhanced_SimpleWalkingCycle(SimpleWalkingCycle):
    """Ciclo de paso para 6 PAMs antagónicos"""
    
    def get_6pam_walking_actions(self, time_step):
        """Genera patrones expertos para 6 PAMs"""
        self.update_phase(time_step)
        
        # Fases alternadas
        left_phase = self.phase * 2 * np.pi
        right_phase = (self.phase + 0.5) * 2 * np.pi
        
        # Patrones antagónicos para caderas
        left_hip_flexor = 0.3 + 0.4 * max(0, np.sin(left_phase))
        left_hip_extensor = 0.4 + 0.3 * max(0, -np.sin(left_phase))
        
        right_hip_flexor = 0.3 + 0.4 * max(0, np.sin(right_phase))  
        right_hip_extensor = 0.4 + 0.3 * max(0, -np.sin(right_phase))
        
        # Flexores de rodilla (solo durante swing)
        left_knee = 0.2 + 0.6 * max(0, np.sin(left_phase))
        right_knee = 0.2 + 0.6 * max(0, np.sin(right_phase))
        
        # Normalizar a [-1, 1]
        actions = [left_hip_flexor, left_hip_extensor, right_hip_flexor, 
                  right_hip_extensor, left_knee, right_knee]
        actions = [2.0 * p - 1.0 for p in actions]
        actions = [max(-1.0, min(1.0, a)) for a in actions]
        
        return np.array(actions, dtype=np.float32)