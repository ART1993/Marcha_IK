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
        # asumimos que la contracción es ideal y por tanto es constante
        self.volumen=self.area*self.L0
        self._limites_parametros()

    
    def _limites_parametros(self):
        """
            Se usa en  para definir los límites de los parámetros
        """
        # Definimos los límites de los parámetros del PAM
        # Estos valores son ejemplos, deben ajustarse según el diseño real del PAM
        #F_es máxima para epsilon=0
        self.F_max_factor= (self.a -self.b) # Fuerza máxima teórica (Pa) a 5 bar Debería de ser un multiplicador pero no he encontrado donde usarlo
        self.epsilon_max = 1 - 1/(np.sqrt(3)*np.cos(self.alpha0)) # Epsilon máximo teórico
        self.max_radious = np.sqrt(2/3)*(self.r0/np.sin(self.alpha0)) # Radio máximo teórico
        self.max_area=np.pi*self.max_radious**2
        self.theta_max = np.atan(np.sqrt(2))
        self.min_pressure=101325
        self.max_pressure=self.min_pressure*5

        # ✅ CORRECCIÓN: Validar que epsilon_max sea positivo
        if self.epsilon_max <= 0:
            print(f"⚠️ Warning: epsilon_max = {self.epsilon_max}, adjusting to 0.3")
            self.epsilon_max = 0.3
    @property
    def L_min(self):
        return self.volumen/self.max_area
    
    def current_radius(self, contraction_ratio):
        """
        Radio actual del músculo durante la contracción
        
        En ε = 0: r = r₀
        En ε = ε_max: r = r_max

        Basado en conservación de volumen de la vejiga interna:
        V = π * r² * L = constante

        Returns:
            float: Radio actual (m)
        """
        epsilon = contraction_ratio
        if epsilon < 0 or epsilon >= self.epsilon_max:
            return self.r0
        
        # Conservación de volumen en la vejiga
        current_length = self.contaction_muscle(epsilon)
        volume_ratio = self.L0 / current_length
        
        return self.r0 * np.sqrt(volume_ratio)
    
    def current_area(self, contraction_ratio):
        """
        ✅ ÁREA ACTUAL de la sección transversal
        
        Esta es el área real sobre la que actúa la presión
        """
        current_r = self.current_radius(contraction_ratio)
        return np.pi * current_r**2

    
    def contaction_muscle(self,contraction_ratio):
        return self.L0*(1-contraction_ratio)


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
        if contraction_ratio < 0:
            contraction_ratio = 0.0
        elif contraction_ratio >= self.epsilon_max:
            contraction_ratio = self.epsilon_max - 0.01  # Pequeño margen
            
        epsilon = contraction_ratio

        force_factor = ((1-epsilon**2)*self.a - self.b)
        
         # ✅ CORRECCIÓN: Si force_factor es negativo, usar mínimo
        if force_factor <= 0:
            force_factor = 0.1  # Mínima fuerza positiva
        #area=np.pi*(self.current_radius(epsilon))
        Function = pressure * force_factor * self.current_area(epsilon)
        # Aseguramos que la fuerza no sea negativa  
        
        return max(0.0, Function)
    
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
        if contraction_ratio < 0:
            contraction_ratio = 0.0
        elif contraction_ratio >= self.epsilon_max:
            contraction_ratio = self.epsilon_max - 0.01
            
        epsilon = contraction_ratio
        
        # Calcular force_factor (mismo que en force_model_new)
        force_factor = ((1-epsilon**2)*self.a - self.b)
        
        if force_factor <= 0:
            force_factor = 0.1  # Mínimo para evitar división por cero
        #area=np.pi*(self.current_radius(epsilon))
        # DESPEJE MATEMÁTICO: pressure = target_force / (force_factor * area)
        required_pressure = target_force / (force_factor * self.current_area(epsilon))
        
        return np.clip(required_pressure, self.min_pressure, self.max_pressure)
    
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


    def calculate_contraction_from_joint_angle(self, joint_angle, joint_type='hip'):
        """
        ✅ NUEVO: Calcular contracción real desde ángulo articular
        
        Esta función conecta la geometría articular con la física del PAM
        
        Args:
            joint_angle: Ángulo actual de la articulación (rad)
            joint_type: 'hip' o 'knee' para diferentes geometrías
            
        Returns:
            float: Contracción real ε = (L0 - L) / L0
        """
        
        # Parámetros geométricos específicos para tu robot bípedo
        # El cambio debería de realizarse en este caso en el entorno no en el musculo, como esta escrito no tiene sentido
        if joint_type == 'hip':
            # Para caderas: El PAM va desde pelvis hasta muslo
            # Longitud base cuando cadera está en 0°
            
            # El cambio de longitud depende del ángulo de cadera
            # Aproximación trigonométrica para unión músculo-hueso
            length_change = self.L_min * np.sin(joint_angle)  # 10cm max change
            
        elif joint_type == 'knee':
            # Para rodillas: El PAM cruza la articulación de la rodilla
            # Más contracción cuando rodilla se flexiona
            
            # Rodilla flexionada = músculo flexor se contrae más
            length_change = 0.08 * joint_angle  # 8cm change per radian
            
        else:
            # Default: relación lineal simple
            length_change = 0.05 * joint_angle
        
        # Longitud actual del músculo
        current_length = self.L0 - length_change
        current_length = max(current_length, self.L0 * (1 - self.epsilon_max + 0.01))  # Límite físico
        
        # Contracción real
        actual_contraction = (self.L0 - current_length) / self.L0
        
        return np.clip(actual_contraction, 0.0, self.epsilon_max - 0.01)

    # Verificación de la consistencia de transformacion fuerza presión y viceversa
    
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