from Archivos_Apoyo.SimplifiedWalkingController import SimplifiedWalkingController
import numpy as np

# ============================================================================
# Modificaciones adicionales para el controlador de paso
# ============================================================================

class Enhanced_SimplifiedWalkingController(SimplifiedWalkingController):
    """
    Controlador de paso adaptado para 6 PAMs antagónicos
    """
    
    def __init__(self, env, mode="pressure", blend_factor=0.0):
        super().__init__(env, mode, blend_factor)
        self.setup_antagonistic_patterns()
    
    def setup_antagonistic_patterns(self):
        """Configurar patrones de activación para pares antagónicos"""
        
        # Patrones base para 6 PAMs (orden: left_hip_flex, left_hip_ext, right_hip_flex, right_hip_ext, left_knee, right_knee)
        self.base_pressures_6pam = {
            'left_hip_flexor': 0.25,
            'left_hip_extensor': 0.35,  # Ligeramente más activo para estabilidad
            'right_hip_flexor': 0.25,
            'right_hip_extensor': 0.35,
            'left_knee_flexor': 0.15,   # Menor activación base
            'right_knee_flexor': 0.15,
        }
        
        # Amplitudes de modulación para cada PAM
        self.modulation_amplitudes_6pam = {
            'left_hip_flexor': 0.4,
            'left_hip_extensor': 0.3,   # Menor modulación en extensores
            'right_hip_flexor': 0.4, 
            'right_hip_extensor': 0.3,
            'left_knee_flexor': 0.6,    # Mayor modulación en rodillas
            'right_knee_flexor': 0.6,
        }
    
    def get_enhanced_walking_actions(self, time_step):
        """
        Genera acciones para 6 PAMs con patrones antagónicos
        """
        self.update_phase(time_step)
        
        # Fases para cada pierna (desfasadas 180°)
        left_leg_phase = self.phase * 2 * np.pi
        right_leg_phase = (self.phase + 0.5) * 2 * np.pi
        
        actions = []
        
        # PIERNA IZQUIERDA
        # Hip flexor: activo durante swing de pierna izquierda
        left_hip_flex = (self.base_pressures_6pam['left_hip_flexor'] + 
                        self.modulation_amplitudes_6pam['left_hip_flexor'] * 
                        max(0, np.sin(left_leg_phase)))
        
        # Hip extensor: activo durante stance de pierna izquierda  
        left_hip_ext = (self.base_pressures_6pam['left_hip_extensor'] + 
                       self.modulation_amplitudes_6pam['left_hip_extensor'] * 
                       max(0, -np.sin(left_leg_phase)))
        
        # PIERNA DERECHA (desfasada)
        right_hip_flex = (self.base_pressures_6pam['right_hip_flexor'] + 
                         self.modulation_amplitudes_6pam['right_hip_flexor'] * 
                         max(0, np.sin(right_leg_phase)))
        
        right_hip_ext = (self.base_pressures_6pam['right_hip_extensor'] + 
                        self.modulation_amplitudes_6pam['right_hip_extensor'] * 
                        max(0, -np.sin(right_leg_phase)))
        
        # RODILLAS (solo flexores)
        left_knee = (self.base_pressures_6pam['left_knee_flexor'] + 
                    self.modulation_amplitudes_6pam['left_knee_flexor'] * 
                    max(0, np.sin(left_leg_phase)))
        
        right_knee = (self.base_pressures_6pam['right_knee_flexor'] + 
                     self.modulation_amplitudes_6pam['right_knee_flexor'] * 
                     max(0, np.sin(right_leg_phase)))
        
        # Compilar acciones en orden correcto
        pressures = [left_hip_flex, left_hip_ext, right_hip_flex, right_hip_ext, left_knee, right_knee]
        
        # Normalizar a [-1, 1] para el entorno
        actions = [2.0 * pressure - 1.0 for pressure in pressures]
        actions = [max(-1.0, min(1.0, action)) for action in actions]
        
        return np.array(actions, dtype=np.float32)