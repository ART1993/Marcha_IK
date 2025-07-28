import numpy as np
import pybullet as p
from dinamica_pam import PAMMcKibben

class AntagonistPAMSystem:
    """
        Sistema antagónico de PAMs (flexor-extensor) para control de articulaciones
    """
    def __init__(self, joint_name, pam_params=None):
        self.joint_name = joint_name
        
        # Parámetros por defecto para PAMs
        if pam_params is None:
            pam_params = {
                'L0': 0.3,     # Longitud inicial (m)
                'r0': 0.02,    # Radio inicial (m) 
                'alpha0': np.pi/4  # Ángulo inicial
            }
        
        # Crear PAMs flexor y extensor (idénticos físicamente)
        self.flexor = PAMMcKibben(**pam_params)
        self.extensor = PAMMcKibben(**pam_params)
        
        # Límites de presión (Pa)
        self.pressure_min = 101325  # 1 atm
        self.pressure_max = 600000  # 6 bar (típico para PAMs)
        
        # Brazo de palanca efectivo (m) - distancia del PAM al eje de rotación
        self.moment_arm = 0.05  # 5cm típico
        
    def torque_to_pressures(self, target_torque, current_angle, angular_velocity=0):
        """
        Convierte torque deseado en presiones para PAMs antagónicos
        
        Args:
            target_torque: Torque objetivo (Nm)
            current_angle: Ángulo actual de la articulación (rad)
            angular_velocity: Velocidad angular actual (rad/s)
            
        Returns:
            tuple: (presión_flexor, presión_extensor) en Pa
        """
        
        # Calcular longitudes actuales de los PAMs basado en ángulo
        # Asumiendo que un PAM se acorta cuando el otro se alarga
        angle_range = np.pi/2  # Rango típico de movimiento articular
        
        # Contracción relativa (0 = longitud inicial, positivo = contracción)
        flexor_contraction = max(0, current_angle / angle_range * self.flexor.epsilon_max)
        extensor_contraction = max(0, -current_angle / angle_range * self.extensor.epsilon_max)
        
        # Limitar contracciones al máximo permitido
        flexor_contraction = min(flexor_contraction, self.flexor.epsilon_max * 0.9)
        extensor_contraction = min(extensor_contraction, self.extensor.epsilon_max * 0.9)
        
        if target_torque > 0:  # Torque positivo -> activar flexor
            # Calcular presión necesaria en flexor
            pressure_flexor = self._calculate_pressure_for_force(
                self.flexor, target_torque / self.moment_arm, flexor_contraction
            )
            # Presión mínima en extensor (co-activación para rigidez)
            pressure_extensor = self.pressure_min + 0.1 * (self.pressure_max - self.pressure_min)
            
        elif target_torque < 0:  # Torque negativo -> activar extensor  
            pressure_flexor = self.pressure_min + 0.1 * (self.pressure_max - self.pressure_min)
            pressure_extensor = self._calculate_pressure_for_force(
                self.extensor, abs(target_torque) / self.moment_arm, extensor_contraction
            )
        else:  # Sin torque -> presiones mínimas
            pressure_flexor = self.pressure_min
            pressure_extensor = self.pressure_min
            
        # Limitar presiones a rangos válidos
        pressure_flexor = np.clip(pressure_flexor, self.pressure_min, self.pressure_max)
        pressure_extensor = np.clip(pressure_extensor, self.pressure_min, self.pressure_max)
        
        return pressure_flexor, pressure_extensor
    
    def _calculate_pressure_for_force(self, pam, target_force, contraction_ratio):
        """
        Calcula la presión necesaria para generar una fuerza específica
        """
        if target_force <= 0:
            return self.pressure_min
            
        # Usar modelo inverso: P = F / (factor_fuerza * área)
        force_factor = ((1 - contraction_ratio**2) * pam.a - pam.b)
        
        if force_factor <= 0:
            return self.pressure_max  # No se puede generar fuerza
            
        required_pressure = target_force / (force_factor * pam.area)
        
        return max(self.pressure_min, required_pressure)
    
    def pressures_to_torque(self, pressure_flexor, pressure_extensor, current_angle):
        """
        Convierte presiones actuales en torque resultante
        """
        # Calcular contracciones actuales
        angle_range = np.pi/2
        flexor_contraction = max(0, current_angle / angle_range * self.flexor.epsilon_max)
        extensor_contraction = max(0, -current_angle / angle_range * self.extensor.epsilon_max)
        
        # Fuerzas generadas por cada PAM
        force_flexor = self.flexor.force_model_new(pressure_flexor, flexor_contraction)
        force_extensor = self.extensor.force_model_new(pressure_extensor, extensor_contraction)
        
        # Torque resultante (flexor positivo, extensor negativo)
        torque = (force_flexor - force_extensor) * self.moment_arm
        
        return torque
    
    def get_pressure_limits(self):
        """
        Retorna límites de presión para usar en RL
        """
        return {
            'min_pressure': self.pressure_min,
            'max_pressure': self.pressure_max,
            'pressure_range': self.pressure_max - self.pressure_min
        }