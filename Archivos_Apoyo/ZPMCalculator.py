import numpy as np
import pybullet as p


class ZMPCalculator:
    """
        Calculador ZMP SIMPLIFICADO para balance y sentadillas.
        
        OBJETIVO ESPECÍFICO:
        - Calcular ZMP básico usando ecuaciones físicas
        - Determinar estabilidad simple (dentro/fuera de polígono de soporte)
        - Calcular margen de estabilidad básico
        
        ELIMINADO:
        - Historia compleja COM para aceleraciones
        - Cálculos avanzados de recompensa ZMP
        - Múltiples parámetros de inicialización
        - Dimensiones complejas del robot
    """
    
    def __init__(self, robot_id, left_foot_id, right_foot_id, robot_data=None):
        
        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.robot_id = robot_id
        self.left_foot_id = left_foot_id  
        self.right_foot_id = right_foot_id
        self.robot_data = robot_data
        
        # ===== PARÁMETROS FÍSICOS SIMPLIFICADOS =====
        
        self.g = 9.81  # Gravedad
        self.l = 1.0   # Altura aproximada COM (simplificada)
        
        # ===== PARÁMETROS DE ESTABILIDAD =====
        
        self.stability_margin = 0.15  # Margen de seguridad (15cm)
        
        # ===== HISTORIA MÍNIMA PARA ACELERACIÓN =====
        
        self.com_history = []
        self.max_history = 3  # Solo 3 puntos para cálculo básico
        
        print(f"🎯 Simplified ZMP Calculator initialized")
        print(f"   Stability margin: {self.stability_margin}m")
        print(f"   COM height estimate: {self.l}m")
    
    def calculate_zmp(self, dt=1.0/1500.0):
        """
        Calcular ZMP usando ecuaciones básicas.
        
        Utiliza las ecuaciones físicas pero con implementación simplificada.
        """
        
        # Obtener posición del centro de masa
        if self.robot_data:
            try:
                com_pos, _ = self.robot_data.get_center_of_mass
                com_pos = np.array(com_pos)
            except:
                # Fallback: usar posición de la base
                com_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                com_pos = np.array(com_pos)
        else:
            com_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            com_pos = np.array(com_pos)
        
        # Actualizar historia para cálculo de aceleración
        self.update_com_history(com_pos)
        
        # Calcular aceleración básica
        com_acceleration = self.calculate_simple_acceleration(dt)
        
        # ===== ECUACIONES ZMP SIMPLIFICADAS =====
        
        # X_zmp = X_com - (l^3/g) * X_com_accel
        # Y_zmp = Y_com - (l^3/g) * Y_com_accel
        
        zmp_x = com_pos[0] - (self.l**3 / self.g) * com_acceleration[0]
        zmp_y = com_pos[1] - (self.l**3 / self.g) * com_acceleration[1]
        
        return np.array([zmp_x, zmp_y])
    
    def update_com_history(self, com_position):
        """Actualizar historia del COM (simplificada)"""
        self.com_history.append(com_position[:3])  # Solo x, y, z
        
        # Mantener solo los últimos puntos necesarios
        if len(self.com_history) > self.max_history:
            self.com_history.pop(0)
    
    def calculate_simple_acceleration(self, dt=1.0/1500.0):
        """
        Calcular aceleración COM usando diferencias finitas SIMPLES.
        
        Si no hay suficiente historia, asumir aceleración cero.
        """
        
        if len(self.com_history) < 3:
            return np.array([0.0, 0.0, 0.0])
        
        # Diferencias finitas de segundo orden: a = (pos[t] - 2*pos[t-1] + pos[t-2]) / dt^2
        pos_current = np.array(self.com_history[-1])
        pos_prev = np.array(self.com_history[-2])
        pos_prev2 = np.array(self.com_history[-3])
        
        acceleration = (pos_current - 2*pos_prev + pos_prev2) / (dt**2)
        
        # Limitar valores extremos (filtrado básico)
        acceleration = np.clip(acceleration, -20.0, 20.0)
        
        return acceleration
    
    def get_support_polygon(self):
        """
        Obtener polígono de soporte SIMPLIFICADO basado en contactos de pies.
        
        Returns:
            np.array: Puntos del polígono de soporte [[x1,y1], [x2,y2], ...]
        """
        
        support_points = []
        
        # Verificar contacto pie izquierdo
        left_contacts = p.getContactPoints(self.robot_id, 0, self.left_foot_id)
        if left_contacts:
            left_pos = p.getLinkState(self.robot_id, self.left_foot_id)[0]
            support_points.append([left_pos[0], left_pos[1]])  # Solo x, y
        
        # Verificar contacto pie derecho  
        right_contacts = p.getContactPoints(self.robot_id, 0, self.right_foot_id)
        if right_contacts:
            right_pos = p.getLinkState(self.robot_id, self.right_foot_id)[0]
            support_points.append([right_pos[0], right_pos[1]])  # Solo x, y
        
        return np.array(support_points) if support_points else np.array([])
    
    def is_stable(self, zmp_point=None):
        """
        Verificar si el ZMP está dentro del polígono de soporte.
        
        SIMPLIFICADO: Para balance de pie, solo verificar que ZMP esté entre los pies.
        """
        
        if zmp_point is None:
            zmp_point = self.calculate_zmp()
        
        support_polygon = self.get_support_polygon()
        
        # Sin contacto = inestable
        if len(support_polygon) == 0:
            return False
        
        # Un solo pie = verificar distancia simple
        if len(support_polygon) == 1:
            distance = np.linalg.norm(zmp_point - support_polygon[0])
            return distance < self.stability_margin
        
        # Dos pies = verificar si ZMP está en el rectángulo entre ellos
        min_x = min(support_polygon[:, 0]) - self.stability_margin
        max_x = max(support_polygon[:, 0]) + self.stability_margin
        min_y = min(support_polygon[:, 1]) - self.stability_margin
        max_y = max(support_polygon[:, 1]) + self.stability_margin
        
        return (min_x <= zmp_point[0] <= max_x and 
                min_y <= zmp_point[1] <= max_y)
    
    def stability_margin_distance(self, zmp_point=None):
        """
        Calcular distancia del ZMP al borde del polígono de soporte.
        
        Returns:
            float: Positivo = dentro (estable), Negativo = fuera (inestable)
        """
        
        if zmp_point is None:
            zmp_point = self.calculate_zmp()
        
        support_polygon = self.get_support_polygon()
        
        # Sin soporte = muy inestable
        if len(support_polygon) == 0:
            return -1.0
        
        # Un pie = distancia al centro del pie
        if len(support_polygon) == 1:
            distance_to_foot = np.linalg.norm(zmp_point - support_polygon[0])
            return self.stability_margin - distance_to_foot
        
        # Dos pies = distancia al borde del rectángulo
        min_x = min(support_polygon[:, 0])
        max_x = max(support_polygon[:, 0])
        min_y = min(support_polygon[:, 1])
        max_y = max(support_polygon[:, 1])
        
        # Distancia al borde más cercano
        dist_x = min(zmp_point[0] - min_x, max_x - zmp_point[0])
        dist_y = min(zmp_point[1] - min_y, max_y - zmp_point[1])
        
        return min(dist_x, dist_y)
    
    def get_stability_info(self):
        """
        Obtener información completa de estabilidad para debugging.
        
        Returns:
            dict: Información de ZMP, estabilidad, margen, etc.
        """
        
        zmp_point = self.calculate_zmp()
        is_stable = self.is_stable(zmp_point)
        margin = self.stability_margin_distance(zmp_point)
        support_polygon = self.get_support_polygon()
        
        return {
            'zmp_position': zmp_point.tolist(),
            'is_stable': is_stable,
            'stability_margin': margin,
            'support_polygon': support_polygon.tolist(),
            'num_contact_points': len(support_polygon),
            'com_history_length': len(self.com_history)
        }
    
    def reset(self):
        """Reset del calculador ZMP"""
        self.com_history.clear()
        print(f"🔄 ZMP Calculator reset")


# ===== FUNCIONES DE UTILIDAD =====

def create_simple_zmp_calculator(robot_id, left_foot_id=2, right_foot_id=5, robot_data=None):
    """Crear calculador ZMP simplificado"""
    
    zmp_calc = ZMPCalculator(
        robot_id=robot_id,
        left_foot_id=left_foot_id,
        right_foot_id=right_foot_id,
        robot_data=robot_data
    )
    
    print(f"✅ Simple ZMP Calculator created")
    print(f"   Focus: Basic stability calculation")
    
    return zmp_calc

def test_zmp_calculator():
    """Test básico del calculador ZMP"""
    
    print("🧪 Testing Simplified ZMP Calculator...")
    
    # Mock para PyBullet (simulación de test)
    class MockPyBullet:
        @staticmethod
        def getBasePositionAndOrientation(robot_id):
            return ([0, 0, 1.1], [0, 0, 0, 1])  # Posición de pie estable
        
        @staticmethod
        def getContactPoints(robot_id, plane_id, foot_id):
            return [{'contact': True}]  # Simular contacto
        
        @staticmethod
        def getLinkState(robot_id, link_id):
            if link_id == 2:  # pie izquierdo
                return ([0, 0.2, 0], None)
            else:  # pie derecho
                return ([0, -0.2, 0], None)
    
    # Reemplazar p temporalmente para test
    import sys
    original_p = sys.modules.get('pybullet')
    sys.modules['pybullet'] = MockPyBullet()
    
    try:
        # Crear calculador de test
        zmp_calc = create_simple_zmp_calculator(robot_id=0)
        
        # Test 1: Cálculo básico de ZMP
        print(f"\n📊 Test 1: Cálculo de ZMP")
        zmp_point = zmp_calc.calculate_zmp()
        print(f"   ZMP position: [{zmp_point[0]:.3f}, {zmp_point[1]:.3f}]")
        
        # Test 2: Verificación de estabilidad
        print(f"\n📊 Test 2: Verificación de estabilidad")
        is_stable = zmp_calc.is_stable(zmp_point)
        print(f"   Is stable: {is_stable}")
        
        # Test 3: Margen de estabilidad
        print(f"\n📊 Test 3: Margen de estabilidad")
        margin = zmp_calc.stability_margin_distance(zmp_point)
        print(f"   Stability margin: {margin:.3f}")
        
        # Test 4: Polígono de soporte
        print(f"\n📊 Test 4: Polígono de soporte")
        support_polygon = zmp_calc.get_support_polygon()
        print(f"   Support points: {len(support_polygon)}")
        if len(support_polygon) > 0:
            print(f"   Points: {support_polygon.tolist()}")
        
        # Test 5: Información completa
        print(f"\n📊 Test 5: Información completa")
        stability_info = zmp_calc.get_stability_info()
        for key, value in stability_info.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎉 ZMP Calculator test completed successfully!")
        
    finally:
        # Restaurar PyBullet original
        if original_p:
            sys.modules['pybullet'] = original_p

def benchmark_zmp_calculation():
    """Benchmark de rendimiento del cálculo ZMP"""
    
    print("⚡ Benchmarking ZMP calculation performance...")
    
    import time
    
    # Simulación de múltiples cálculos
    zmp_calc = create_simple_zmp_calculator(robot_id=0)
    
    # Simular historial COM
    for i in range(5):
        fake_com = [0.1 * i, 0.0, 1.1]
        zmp_calc.update_com_history(fake_com)
    
    # Benchmark
    num_calculations = 1000
    start_time = time.time()
    
    for _ in range(num_calculations):
        zmp_point = zmp_calc.calculate_zmp()
        is_stable = zmp_calc.is_stable(zmp_point)
        margin = zmp_calc.stability_margin_distance(zmp_point)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"   Calculations: {num_calculations}")
    print(f"   Time elapsed: {elapsed:.3f} seconds")
    print(f"   Rate: {num_calculations/elapsed:.1f} calculations/second")
    print(f"   Per calculation: {elapsed/num_calculations*1000:.3f} ms")
    
    # Para robot real a 1500 Hz, necesitamos < 0.67ms por cálculo
    if elapsed/num_calculations < 0.0007:
        print(f"   ✅ Performance suitable for real-time (1500 Hz)")
    else:
        print(f"   ⚠️ May need optimization for real-time")


# ===== EJEMPLO DE INTEGRACIÓN =====

def integrate_with_simplified_env():
    """Ejemplo de integración con entorno simplificado"""
    
    print("🔗 Integration Example: ZMP Calculator + Environment")
    
    try:
        from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
        
        # Crear entorno
        env = create_simple_balance_squat_env(render_mode='direct')
        obs, info = env.reset()
        
        # El entorno ya incluye ZMP calculator, probarlo
        if hasattr(env, 'zmp_calculator'):
            print(f"   ZMP Calculator found in environment")
            
            # Test de integración con el entorno
            for step in range(50):
                action = env.action_space.sample() * 0.5 + 0.3  # Acciones moderadas
                obs, reward, done, truncated, info = env.step(action)
                
                if step % 10 == 0:
                    stability_info = env.zmp_calculator.get_stability_info()
                    print(f"   Step {step}: Stable = {stability_info['is_stable']}, "
                          f"Margin = {stability_info['stability_margin']:.3f}")
                
                if done:
                    print(f"   Episode terminado en step {step}")
                    break
            
            print(f"✅ Integration test completed")
        else:
            print(f"⚠️ ZMP Calculator not found in environment")
        
        env.close()
        
    except ImportError:
        print("⚠️ Entorno simplificado no disponible para test de integración")

if __name__ == "__main__":
    
    print("🎯 SIMPLIFIED ZMP CALCULATOR")
    print("=" * 50)
    print("Calculador ZMP enfocado en balance y estabilidad básica")
    print("Elimina complejidad innecesaria para marcha")
    print("=" * 50)
    
    # Test básico
    test_zmp_calculator()
    
    # Benchmark de rendimiento
    benchmark_zmp_calculation()
    
    # Test de integración
    integrate_with_simplified_env()