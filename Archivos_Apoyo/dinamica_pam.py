import numpy as np
import matplotlib.pyplot as plt

# Modelo Actuador Neumatico PAM McKibben
class PAMMcKibben:
    """
        Esta clase modela matemáticamente cómo funcionan 
        los músculos artificiales neumáticos (PAM), 
        calculando fuerza y rigidez según parámetros físicos.
    """
    def __init__(self, L0=0.3, r0=0.02, alpha0=np.pi/4, min_pressure=101325, max_factor_pressure=5):
        """
            L0: longitud inicial del músculo (m)
            r0: radio inicial (m)
            n_threads: número de hilos de refuerzo
            alpha0: ángulo inicial de los hilos rad
        """
        # Longitud inicial del músculo
        self.L0 = L0
        # radio inicial del músculo
        self.r0 = r0
        # Superficie inicial
        self.area= np.pi * self.r0**2
        # ángulo inicial de los hilos
        self.alpha0 = alpha0
        self.a= 3/(np.tan(alpha0)**2) 
        self.b = 1 / (np.sin(alpha0)**2)  # geometría del hilo
        # asumimos que la contracción es ideal y por tanto es constante
        self.volumen=self.area*self.L0
        self._limites_parametros(min_pressure, max_factor_pressure)

    
    def _limites_parametros(self, min_pressure, max_factor_pressure):
        """
            Se usa en  para definir los límites de los parámetros
        """
        # factor que multiplica la presión interna para obtener la fuerza máxima teórica cuando la contracción es cero (ε = 0)
        self.F_max_factor= (self.a -self.b) 
        # Deformación máxima (contracción relativa). Representa cuánto puede acortarse el músculo respecto a su longitud inicial.
        self.epsilon_max = 1 - 1/(np.sqrt(3)*np.cos(self.alpha0)) 
        # Radio máximo alcanzable debido a la expansión radial.
        self.max_radius = np.sqrt(2/3)*(self.r0/np.sin(self.alpha0)) 
        # Ángulo máximo de apertura de los hilos de refuerzo. Más allá de este ángulo, el modelo deja de ser válido.
        self.theta_max = np.atan(np.sqrt(2))
        self.min_pressure=min_pressure
        self.max_pressure=min_pressure*max_factor_pressure

        # ✅ CORRECCIÓN: Validar que epsilon_max sea positivo
        if self.epsilon_max <= 0:
            print(f"⚠️ Warning: epsilon_max = {self.epsilon_max}, adjusting to 0.3")
            self.epsilon_max = 0.3

    @property
    def L_min(self):
        return self.volumen/self.max_area

    def force_factor(self,alpha):
        # factor de fuerza del PAM. En este caso depende del ángulo de hilo de pam alpha
        return 3.0 * (np.cos(alpha)**2) - 1.0
    
    def contraction_muscle(self,contraction_ratio):
        eps = np.clip(contraction_ratio, 0.0, self.epsilon_max)
        return max(self.L0 * (1.0 - eps), self.L_min)
    
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

            Return F = P * force_factor * área_actual
        """
        if pressure <= 0:
            return 0.0
        
        # ✅ CORRECCIÓN: Validación mejorada de contraction_ratio
        eps = np.clip(contraction_ratio, 0.0, self.epsilon_max)
        alpha = self.braid_angle(eps)
        force_factor = self.force_factor(alpha)
        
        # ✅ CORRECCIÓN: Si force_factor es negativo, usar mínimo
        if force_factor <= 0:
            force_factor = 0.1  # Mínima fuerza positiva
        area = self.current_area(eps)
        F = pressure * force_factor * area
        return max(0.0, F)  # no permitimos tracción negativa
    
    def pressure_from_force_and_contraction(self, target_force, contraction_ratio):
        """
        ✅ MÉTODO INVERSO: Calcular presión necesaria para generar fuerza objetivo
        
        Matemáticamente inverso a force_model_new()
        
        Args:
            target_force: Fuerza objetivo (N)
            contraction_ratio: Contracción deseada [0, epsilon_max)
            
        Returns:
            float: Presión necesaria (Pa)
        """
        if target_force <= 0:
            return self.min_pressure  # Presión atmosférica mínima
        
        # Validar contracción
        eps = np.clip(contraction_ratio, 0.0, self.epsilon_max)
        alpha = self.braid_angle(eps)
        force_factor = self.force_factor(alpha)
        
        area = self.current_area(eps)
        denom = max(force_factor * area, 1e-9)  # evitar división por cero
        
        P = target_force / denom
        
        return np.clip(P, self.min_pressure, self.max_pressure)
    
    def pressure_normalized_from_force_and_contraction(self, target_force, contraction_ratio):
        """
        ✅ MÉTODO INVERSO NORMALIZADO: Para usar directamente en el entorno
        
        Returns:
            float: Presión normalizada [0, 1]
        """
        real_pressure = self.pressure_from_force_and_contraction(target_force, contraction_ratio)
        
        # Normalizar a [0, 1]
        normalized = self.normalized_pressure_PAM(real_pressure) 
        return np.clip(normalized, 0.0, 1.0)
    
    def real_pressure_PAM(self, normalized_pressure):
        # 1. Presión normalizada → Presión real
        return self.min_pressure + normalized_pressure * (self.max_pressure - self.min_pressure)
    
    def normalized_pressure_PAM(self, real_pressure):
        # 1. Presión normalizada → Presión real
        return (real_pressure - self.min_pressure) / (self.max_pressure - self.min_pressure)
    
    def epsilon_from_angle(self, theta, theta0, R):
        dL = R * (theta - theta0)        # signo según tu convención
        eps = dL / self.L0
        return np.clip(eps, 0.0, self.epsilon_max)
    
    def braid_angle(self, contraction_ratio):
        """
            Ángulo dinámico α(ε) a partir de longitud de hilo constante:
            cos α(ε) = (1 - ε) * cos α0
        """
        eps = np.clip(contraction_ratio, 0.0, self.epsilon_max)
        cos_alpha = (1.0 - eps) * np.cos(self.alpha0)
        # Clamp numérico
        cos_alpha = np.clip(cos_alpha, 0.0, 1.0)
        alpha = np.arccos(cos_alpha)
        # Limitar a [0, theta_max]
        return np.clip(alpha, 0.0, self.theta_max)

    def area_from_epsilon(self, eps):
        return self.current_area(eps)
    
    def current_radius(self, contraction_ratio):
        """
            Radio actual del músculo durante la contracción
            
            En ε = 0: r = r₀    En ε = ε_max: r = r_max

            Basado en conservación de volumen de la vejiga interna:
            V = π * r² * L = constante

            Returns:
                float: Radio actual (m)
        """
        
        # Conservación de volumen en la vejiga
        current_length = self.contraction_muscle(contraction_ratio)
        r = self.r0 * np.sqrt(self.L0 / current_length)
        r_max = self.max_radius
        return min(r,r_max)  # Límite físico
    
    def current_area(self, contraction_ratio):
        """
        ✅ ÁREA ACTUAL de la sección transversal
        
        Esta es el área real sobre la que actúa la presión
        """
        current_r = self.current_radius(contraction_ratio)
        return np.pi * current_r**2
    
    def pressure_for_torque(self, tau, theta, theta0, R):
        # 1) ángulo -> contracción
        eps = self.epsilon_from_angle(theta, theta0, R)  # mismo clamp que ya tienes
        # 2) par -> fuerza
        F = tau / max(R, 1e-9)  # evita división por cero
        # 3) inverso común
        return float(self.pressure_from_force_and_contraction(F, eps))
    
    #def pressure_for_torque(self, tau, theta, theta0, R):
    #    eps = self.epsilon_from_angle(theta, theta0, R)
    #    A = self.area_from_epsilon(eps)
    #    alpha = self.braid_angle(eps)
    #    factor = self.force_factor(alpha)
    #    denom = max(R * A * factor, 1e-9)    # a(θ)=R en este caso
    #    P = tau / denom
    #    return float(np.clip(P, self.min_pressure, self.max_pressure))
    
    #def stiffness_model(self, pressure, contraction_ratio):
        """Rigidez variable del actuador"""
    #    base_stiffness = 1000  # N/m base
    #    pressure_factor = pressure / 100000  # normalizado a 1 bar ver si es mejor usar 5 bar
    #    contraction_factor = 1 + 2 * contraction_ratio
        
    #    return base_stiffness * pressure_factor * contraction_factor

    # =========================================================================== #
    # Verificación de la consistencia de transformacion fuerza presión y viceversa
    # =========================================================================== #
    
    def verify_inverse_consistency(self, test_pressure_normalized=0.5, test_contraction=0.15):
        """
        ✅ VERIFICACIÓN: Comprobar que el método inverso es matemáticamente correcto
        
        Test: pressure → force → pressure_back (debería ser igual)
        """
        
        # 1. Presión normalizada → Presión real
        real_pressure = self.real_pressure_PAM(test_pressure_normalized)
        
        # 2. Presión real → Fuerza (método directo)
        force = self.force_model_new(real_pressure, test_contraction)
        
        # 3. Fuerza → Presión (método inverso)
        pressure_back = self.pressure_from_force_and_contraction(force, test_contraction)
        
        # 4. Presión → Normalizada
        pressure_back_normalized = self.normalized_pressure_PAM(pressure_back)
        
        # 5. Verificar consistencia
        error = abs(test_pressure_normalized - pressure_back_normalized)
        
        print(f"🔍 Verificación de consistencia PAM:")
        print(f"   Input normalizado: {test_pressure_normalized:.4f}")
        print(f"   Presión real: {real_pressure:.0f} Pa")
        print(f"   Fuerza generada: {force:.2f} N")
        print(f"   Presión calculada back: {pressure_back:.0f} Pa")
        print(f"   Output normalizado: {pressure_back_normalized:.4f}")
        print(f"   Error: {error:.6f} ({'✅ OK' if error < 0.001 else '❌ PROBLEMA'})")
        
        return error < 0.001
    

# Verificación rápida
if __name__ == "__main__":
    pam = PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4)
    
    # Test múltiples valores
    for pressure_norm in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for contraction in [0.05, 0.15, 0.25]:
            print(f"\n--- Test: pressure_norm={pressure_norm}, contraction={contraction} ---")
            pam.verify_inverse_consistency(pressure_norm, contraction)