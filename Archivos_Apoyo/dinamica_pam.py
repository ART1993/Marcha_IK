import numpy as np
import matplotlib.pyplot as plt

# Modelo Actuador Neumatico PAM McKibben
class PAMMcKibben:
    """
        Esta clase modela matemáticamente cómo funcionan 
        los músculos artificiales neumáticos (PAM), 
        calculando fuerza y rigidez según parámetros físicos.
    """
    def __init__(self, L0=0.3, r0=0.02, alpha0=np.pi/4):
        """
            L0: longitud inicial del músculo (m)
            r0: radio inicial (m)
            n_threads: número de hilos de refuerzo
            alpha0: ángulo inicial de los hilos rad
        """
        self.L0 = L0
        self.r0 = r0
        self.area= np.pi * self.r0**2
        self.alpha0 = alpha0
        self.a= 3/(np.tan(alpha0)**2) 
        self.b = 1 / (np.sin(alpha0)**2)  # geometría del hilo
        self.limites_parametros

    @property
    def limites_parametros(self):
        """
            Se usa en  para definir los límites de los parámetros
        """
        # Definimos los límites de los parámetros del PAM
        # Estos valores son ejemplos, deben ajustarse según el diseño real del PAM
        #F_es máxima para epsilon=0
        self.F_max_factor= (self.a -self.b) # Fuerza máxima teórica (Pa) a 5 bar
        self.epsilon_max=1-1/(np.sqrt(3)*np.cos(self.alpha0)) # Epsilon máximo teórico
        self.max_radious = np.sqrt(2/3)*(self.r0/np.sin(self.alpha0)) # Radio máximo teórico
        self.theta_max = np.atan(np.sqrt(2))
    
    def current_radius(self, contraction_ratio):
        """
        Radio actual del músculo durante la contracción
        
        En ε = 0: r = r₀
        En ε = ε_max: r = r_max
        """
        epsilon = contraction_ratio
        if epsilon < 0 or epsilon >= self.epsilon_max:
            return self.r0
        
        # Conservación de volumen en la vejiga
        current_length = self.L0 * (1 - epsilon)
        volume_ratio = self.L0 / current_length
        
        return self.r0 * np.sqrt(volume_ratio)


    def force_model_new(self, pressure, contraction_ratio):
        """
            Modela matemáticamente fuerza genrada
            Modelo de fuerza del actuador PAM
            pressure: presión interna (Pa)
            contraction_ratio: ε = (L0 - L) / L0
        """
        if contraction_ratio >= self.epsilon_max or contraction_ratio < 0:
            return 0.0
            
        epsilon = contraction_ratio

        force_factor = ((1-epsilon**2)*self.a - self.b)
        
        Function = pressure * force_factor * self.area
        # Aseguramos que la fuerza no sea negativa  
        
        return max(0.0, Function)
    
    def stiffness_model(self, pressure, contraction_ratio):
        """Rigidez variable del actuador"""
        base_stiffness = 1000  # N/m base
        pressure_factor = pressure / 100000  # normalizado a 1 bar ver si es mejor usar 5 bar
        contraction_factor = 1 + 2 * contraction_ratio
        
        return base_stiffness * pressure_factor * contraction_factor