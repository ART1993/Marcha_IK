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
        self.pressure=None
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


    def maximum_force(self, pressure):
        """
        Fuerza máxima que puede generar el músculo
        Ocurre en ε = 0 (sin contracción)
        
        F_max = P * π * r₀² * (a - b)
        """
        return pressure * self.area * self.F_max_factor
    
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
    
    def current_angle(self, contraction_ratio):
        """
        Ángulo actual del trenzado durante la contracción
        
        α₀ → α_max conforme ε → ε_max
        """
        epsilon = contraction_ratio
        if epsilon >= self.epsilon_max:
            return self.alpha_max
        
        # Relación geométrica del trenzado
        cos_alpha = np.sqrt(3) * (1 - epsilon) * np.cos(self.alpha0)
        alpha_current = np.arccos(np.clip(cos_alpha, 0, 1))
        
        return min(alpha_current, self.alpha_max)


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
        
    def force_model(self, pressure, contraction_ratio):
        """
        Modela matemáticamente fuerza genrada
        Modelo de fuerza del actuador PAM
        pressure: presión interna (Pa)
        contraction_ratio: ε = (L0 - L) / L0
        """
        if contraction_ratio >= 1.0 or contraction_ratio < 0:
            return 0.0
            
        epsilon = contraction_ratio
        
        # Fuerza teórica del PAM
        cos_alpha = np.sqrt(1 - (self.b * (1 - epsilon) / (np.pi * self.r0))**2)
        
        Function = pressure * (np.pi * self.r0**2) * (
            3 * cos_alpha**2 - 1 - 
            2 * cos_alpha * np.sqrt(1 - cos_alpha**2) * 
            (self.b * (1 - epsilon)) / (np.pi * self.r0)
        )
        
        return max(0.0, Function)
    


    
    def stiffness_model(self, pressure, contraction_ratio):
        """Rigidez variable del actuador"""
        base_stiffness = 1000  # N/m base
        pressure_factor = pressure / 100000  # normalizado a 1 bar ver si es mejor usar 5 bar
        contraction_factor = 1 + 2 * contraction_ratio
        
        return base_stiffness * pressure_factor * contraction_factor
    
    def validate_model_physics(self):
        """
        Valida que el modelo respete principios físicos básicos
        """
        print("\n=== Validación Física del Modelo ===")
        
        # 1. Fuerza máxima debe ocurrir en ε = 0
        epsilon_test = np.linspace(0, self.epsilon_max * 0.9, 50)
        forces = [self.ideal_force(100000, eps) for eps in epsilon_test]
        max_force_idx = np.argmax(forces)
        
        print(f"1. Fuerza máxima en ε = {epsilon_test[max_force_idx]:.3f}")
        print(f"   ✓ Correcto" if max_force_idx == 0 else "   ✗ Error")
        
        # 2. Fuerza debe ser cero en ε_max
        force_at_max = self.ideal_force(100000, self.epsilon_max * 0.999)
        print(f"2. Fuerza cerca de ε_max = {force_at_max:.1f} N")
        print(f"   ✓ Correcto" if force_at_max < 100 else "   ✗ Error")
        
        # 3. Relación cuadrática debe ser evidente
        mid_epsilon = self.epsilon_max / 2
        f_quarter = self.ideal_force(100000, self.epsilon_max / 4)
        f_half = self.ideal_force(100000, mid_epsilon)
        
        print(f"3. Comportamiento cuadrático:")
        print(f"   F(ε/4) = {f_quarter:.0f} N")
        print(f"   F(ε/2) = {f_half:.0f} N")
        print(f"   Ratio = {f_quarter/f_half:.2f} (esperado ≈ 2.25)")

    def plot_force_characteristics(self, max_pressure=500000):
        """
        Grafica las características de fuerza vs contracción
        para diferentes presiones
        """
        epsilon_range = np.linspace(0, self.epsilon_max * 0.95, 100)
        pressures = [100000, 200000, 300000, 400000, max_pressure]  # Pa
        
        plt.figure(figsize=(10, 6))
        
        for p in pressures:
            forces = [self.ideal_force(p, eps) for eps in epsilon_range]
            plt.plot(epsilon_range, np.array(forces)/1000, 
                    label=f'P = {p/1000:.0f} kPa', linewidth=2)
        
        plt.xlabel('Contracción ε', fontsize=12)
        plt.ylabel('Fuerza (kN)', fontsize=12)
        plt.title('Características Fuerza-Contracción del Músculo McKibben\n(Modelo Tondu-Lopez)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(0, self.epsilon_max)
        plt.ylim(0, None)
        
        # Marcar punto de contracción máxima
        plt.axvline(x=self.epsilon_max, color='red', linestyle='--', alpha=0.7, 
                   label=f'ε_max = {self.epsilon_max:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()