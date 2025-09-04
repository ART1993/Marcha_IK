import pybullet as p

class AntiFlexionController:
    """
        Controlador basado en principio de reciprocidad neuronal
        Inhibe flexores cuando extensores necesitan activarse
    """
    
    def __init__(self):
        self.flexion_threshold = 0.05  # rad (3 grados)
        self.reciprocal_inhibition = 0.7  # Factor de inhibición
    
    def apply_reciprocal_inhibition(self, pam_pressures, joint_positions):
        """
        Implementa inhibición recíproca para prevenir flexión excesiva
        """
        corrected_pressures = pam_pressures.copy()
        
        # Para cada rodilla, si está demasiado flexionada, inhibe flexores
        for knee_idx, knee_pos in [(1, joint_positions[1]), (3, joint_positions[3])]:
            if knee_pos > self.flexion_threshold:
                # Calcular factor de inhibición proporcional
                excess_flexion = knee_pos - self.flexion_threshold
                inhibition_factor = min(self.reciprocal_inhibition, 
                                      excess_flexion * 10.0)  # Más flexión = más inhibición
                
                # Inhibir flexores correspondientes
                if knee_idx == 1:  # Rodilla izquierda
                    corrected_pressures[4] *= (1.0 - inhibition_factor)
                else:  # Rodilla derecha  
                    corrected_pressures[5] *= (1.0 - inhibition_factor)
        
        return corrected_pressures