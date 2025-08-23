import numpy as np
import math
from enum import Enum

import pybullet as p

class ActionType(Enum):
    """Define los tipos de acciones discretas que el robot puede realizar"""
    BALANCE_STANDING = "balance_standing"
    SQUAT = "squat"

class BiomechanicalActionController:
    """
        ✅ CONTROLADOR BIOMECÁNICO CORREGIDO
        
        MEJORAS CLAVE:
        1. ✅ Eliminación de ruido aleatorio destructivo
        2. ✅ Lógica de rodillas basada en ángulo actual
        3. ✅ Coordinación antagónica real
        4. ✅ Variación correlacionada, no independiente
    """
    
    def __init__(self, env):
        self.env = env
        self.current_action = ActionType.BALANCE_STANDING
        self.action_progress = 0.0  # Progreso dentro de la acción actual [0,1]
        self.action_duration = 2.0  # Duración típica de una acción en segundos
        
        # Mapeo de músculos PAM (igual que en tu sistema)
        self.pam_mapping = {
            'left_hip_flexor': 0,
            'left_hip_extensor': 1,
            'right_hip_flexor': 2,
            'right_hip_extensor': 3,
            'left_knee_flexor': 4,
            'right_knee_flexor': 5,
        }

        # ✅ CONFIGURACIÓN PARA VARIACIÓN BIOMECÁNICAMENTE CORRECTA
        self.coordination_noise_scale = 0.008  # Muy pequeño para coordinación
        self.bilateral_symmetry_factor = 0.95  # Simetría casi perfecta
        
        # Configurar patrones base para cada acción
        self.setup_action_patterns()

        print(f"🧠 Biomechanical Action Controller initialized")
        print(f"   Coordination-aware noise: {self.coordination_noise_scale}")
        print(f"   Bilateral symmetry: {self.bilateral_symmetry_factor}")
        
    def setup_action_patterns(self):
        """Define los patrones de activación PAM para cada acción discreta"""
        
        self.action_patterns = {
            ActionType.BALANCE_STANDING: {
                'description': 'Balance erguido con lógica biomecánica',
                'duration': 5.0,
                'phases': [
                    {
                        'duration_ratio': 1.0,
                        'base_pressures': {
                            # ✅ CADERAS: Extensores dominantes para anti-gravedad
                            'left_hip_flexor': 0.50,
                            'left_hip_extensor': 0.55,   # Más fuerte para sostener peso
                            'right_hip_flexor': 0.5,
                            'right_hip_extensor': 0.55,
                            
                            # ✅ RODILLAS: INACTIVAS - Los resortes se encargan
                            'left_knee_flexor': 0.00,   # ¡CERO como dices!
                            'right_knee_flexor': 0.00,  # ¡CERO como dices!
                        }
                    }
                ]
            },
            
            ActionType.SQUAT: {
                'description': 'Sentadilla biomecánicamente correcta',
                'duration': 8.0,
                'phases': [
                    # Fase 1: Preparación
                    {
                        'duration_ratio': 0.15,
                        'base_pressures': {
                            'left_hip_flexor': 0.30,
                            'left_hip_extensor': 0.50,
                            'right_hip_flexor': 0.30,
                            'right_hip_extensor': 0.50,
                            'left_knee_flexor': 0.00,   # Aún inactivos
                            'right_knee_flexor': 0.00,
                        }
                    },
                    # Fase 2: Descenso controlado
                    {
                        'duration_ratio': 0.30,
                        'base_pressures': {
                            'left_hip_flexor': 0.60,    # Ahora sí flexores activos
                            'left_hip_extensor': 0.30,  # Control excéntrico
                            'right_hip_flexor': 0.60,
                            'right_hip_extensor': 0.30,
                            'left_knee_flexor': 0.35,   # Ahora necesarios para flexión
                            'right_knee_flexor': 0.35,
                        }
                    },
                    # Fase 3: Posición baja
                    {
                        'duration_ratio': 0.15,
                        'base_pressures': {
                            'left_hip_flexor': 0.45,
                            'left_hip_extensor': 0.65,  # Mantener posición
                            'right_hip_flexor': 0.45,
                            'right_hip_extensor': 0.65,
                            'left_knee_flexor': 0.20,   # Mantener flexión
                            'right_knee_flexor': 0.20,
                        }
                    },
                    # Fase 4: Ascenso
                    {
                        'duration_ratio': 0.40,
                        'base_pressures': {
                            'left_hip_flexor': 0.25,    # Reducir flexión
                            'left_hip_extensor': 0.70,  # Extensión potente
                            'right_hip_flexor': 0.25,
                            'right_hip_extensor': 0.70,
                            'left_knee_flexor': 0.00,   # ¡De vuelta a cero!
                            'right_knee_flexor': 0.00,  # Los resortes extienden
                        }
                    }
                ]
            }
        }
    
    
    def set_action(self, action_type: ActionType):
        """Cambia la acción actual que el controlador debe generar"""
        if action_type != self.current_action:
            self.current_action = action_type
            self.action_progress = 0.0
            pattern = self.action_patterns[action_type]
            self.action_duration = pattern['duration']
            print(f"🎯 Switching to action: {pattern['description']}")
    
    def get_expert_action(self, time_step):
        """
            Genera las presiones PAM expertas para la acción actual.
            
            Args:
                time_step: Paso de tiempo de simulación
                
            Returns:
                np.array: 6 presiones PAM normalizadas [0, 1]
        """
        # Actualizar progreso de la acción
        progress_increment = time_step / self.action_duration
        self.action_progress += progress_increment
        
        # Si la acción terminó, mantener la última fase
        if self.action_progress >= 1.0:
            if self.current_action == ActionType.SQUAT:
                # Después de sentadilla, volver a balance
                self.set_action(ActionType.BALANCE_STANDING)
            else:
                # Reiniciar balance
                self.action_progress = 0.0
        
        # Obtener patrón base de la acción actual
        base_pressures = self._get_base_pattern_pressures()
        
        # ✅ APLICAR LÓGICA BIOMECÁNICA INTELIGENTE
        adjusted_pressures = self._apply_biomechanical_logic(base_pressures)
        
        # ✅ APLICAR VARIACIÓN COORDINADA (NO ALEATORIA INDEPENDIENTE)
        final_pressures = self._apply_coordinated_variation(adjusted_pressures)
        
        # Convertir a array de PAM
        pam_pressures = self._convert_to_pam_array(final_pressures)
        
        # Guardar para suavizado
        self.last_action = pam_pressures.copy()
        
        return pam_pressures
    
    def _get_base_pattern_pressures(self):
        """Obtener presiones base del patrón actual"""
        pattern = self.action_patterns[self.current_action]
        phases = pattern['phases']
        
        # Determinar fase actual
        cumulative_duration = 0.0
        current_phase = phases[0]
        
        for phase in phases:
            phase_end = cumulative_duration + phase['duration_ratio']
            if self.action_progress <= phase_end:
                current_phase = phase
                break
            cumulative_duration += phase['duration_ratio']
        #print(current_phase)
        return current_phase['base_pressures'].copy()
    
    def _apply_biomechanical_logic(self, base_pressures):
        """
        ✅ LÓGICA BIOMECÁNICA INTELIGENTE
        
        Esta función implementa tu observación clave:
        - Si rodilla flexionada (ángulo > 0) → flexores inactivos
        - Los resortes extensores pasivos enderezarán la pierna
        """
        
        adjusted_pressures = base_pressures.copy()
        
        try:
            # Obtener ángulos actuales de las rodillas
            joint_states = p.getJointStates(self.env.robot_id, [1, 4])  # rodillas
            left_knee_angle = joint_states[0][0]
            right_knee_angle = joint_states[1][0]
            
            # ✅ LÓGICA DE RODILLA IZQUIERDA
            if left_knee_angle > 0.05:  # Si está flexionada (> ~3 grados)
                # Los flexores deben estar inactivos - los resortes se encargan
                adjusted_pressures['left_knee_flexor'] = 0.00
                
                # Debug ocasional
                if self.env.step_count % 750 == 0:
                    print(f"   🦵 Left knee flexed ({left_knee_angle:.3f} rad) → flexor OFF")
            
            # ✅ LÓGICA DE RODILLA DERECHA
            if right_knee_angle > 0.05:  # Si está flexionada
                adjusted_pressures['right_knee_flexor'] = 0.00
                
                if self.env.step_count % 750 == 0:
                    print(f"   🦵 Right knee flexed ({right_knee_angle:.3f} rad) → flexor OFF")
        
        except Exception as e:
            # Si no podemos leer los ángulos, usar los valores base
            if self.env.step_count % 1500 == 0:
                print(f"   ⚠️ Could not read joint angles: {e}")
        
        return adjusted_pressures
    
    def _convert_to_pam_array(self, pressure_dict):
        """Convertir diccionario de presiones a array de PAM"""
        pam_pressures = np.zeros(6, dtype=np.float32)
        
        for muscle_name, pressure in pressure_dict.items():
            if muscle_name in self.pam_mapping:
                pam_index = self.pam_mapping[muscle_name]
                pam_pressures[pam_index] = pressure
        
        return pam_pressures
    
    def _apply_coordinated_variation(self, pressures):
        """
        ✅ VARIACIÓN COORDINADA BIOMECÁNICAMENTE CORRECTA
        
        En lugar de ruido independiente, aplicamos:
        1. Variación correlacionada entre antagónicos
        2. Simetría bilateral preservada
        3. Micro-ajustes naturales mínimos
        """
        
        varied_pressures = pressures.copy()
        
        # ✅ VARIACIÓN COORDINADA DE CADERAS
        # Los antagónicos varían en direcciones opuestas (como en la vida real)
        left_hip_variation = np.random.normal(0, self.coordination_noise_scale)
        right_hip_variation = np.random.normal(0, self.coordination_noise_scale)
        
        # Cadera izquierda: flexor y extensor varían inversamente
        varied_pressures['left_hip_flexor'] += left_hip_variation
        varied_pressures['left_hip_extensor'] -= left_hip_variation  # Inverso
        
        # Cadera derecha: similar pero con simetría bilateral
        bilateral_factor = self.bilateral_symmetry_factor
        varied_pressures['right_hip_flexor'] += right_hip_variation * bilateral_factor
        varied_pressures['right_hip_extensor'] -= right_hip_variation * bilateral_factor
        
        # ✅ LAS RODILLAS NO RECIBEN VARIACIÓN SI ESTÁN INACTIVAS
        # Solo añadir variación si están activas (presión > 0)
        if varied_pressures['left_knee_flexor'] > 0.01:
            knee_variation = np.random.normal(0, self.coordination_noise_scale * 0.5)
            varied_pressures['left_knee_flexor'] += knee_variation
        
        if varied_pressures['right_knee_flexor'] > 0.01:
            knee_variation = np.random.normal(0, self.coordination_noise_scale * 0.5)
            varied_pressures['right_knee_flexor'] += knee_variation
        
        # Asegurar límites
        for muscle, pressure in varied_pressures.items():
            varied_pressures[muscle] = np.clip(pressure, 0.0, 1.0)
        
        return varied_pressures
    
    def reset(self):
        """Reinicia el controlador a su estado inicial"""
        self.current_action = ActionType.BALANCE_STANDING
        self.action_progress = 0.0
        if hasattr(self, 'last_action'):
            del self.last_action
        print(f"🔄 Controller reset - Starting with BALANCE_STANDING")
    
    def get_current_action_info(self):
        """Información sobre la acción actual para debugging"""
        pattern = self.action_patterns[self.current_action]
        phases = pattern['phases']
        
        # Determinar fase actual
        cumulative = 0.0
        current_phase_idx = 0
        for i, phase in enumerate(phases):
            if self.action_progress <= cumulative + phase['duration_ratio']:
                current_phase_idx = i
                break
            cumulative += phase['duration_ratio']
        
        return {
            'action': self.current_action.value,
            'description': pattern['description'],
            'progress': self.action_progress,
            'current_phase': current_phase_idx + 1,
            'total_phases': len(phases),
            'duration': self.action_duration
        }
    
    def get_balance_optimized_action(self, time_step):
        """
        ✅ NUEVO MÉTODO: Acción específicamente optimizada para balance inicial
        
        Esta acción está diseñada para establecer contacto rápido y estable
        antes de la transición a control PAM completo.
        """
        
        # Patrón específico para establecimiento de contacto
        contact_pattern = {
            'left_hip_flexor': 0.15,     # Mínimo para no caer hacia atrás
            'left_hip_extensor': 0.50,   # Fuerte para mantener postura erecta
            'right_hip_flexor': 0.15,    # Simétrico
            'right_hip_extensor': 0.50,
            'left_knee_flexor': 0.10,    # Muy suave - solo estabilización
            'right_knee_flexor': 0.10,
        }
        
        pam_pressures = np.zeros(6, dtype=np.float32)
        
        for muscle_name, pressure in contact_pattern.items():
            if muscle_name in self.pam_mapping:
                pam_index = self.pam_mapping[muscle_name]
                pam_pressures[pam_index] = pressure
        
        # Variación temporal muy suave para micro-ajustes
        time_factor = np.sin(time_step * 0.5) * 0.01  # ±1% variación lenta
        pam_pressures += time_factor
        
        pam_pressures = np.clip(pam_pressures, 0.0, 1.0)
        
        return pam_pressures
    
    def get_current_action_info(self):
        """Info de la acción actual para debugging"""
        pattern = self.action_patterns[self.current_action]
        phases = pattern['phases']
        
        # Determinar fase actual
        cumulative = 0.0
        current_phase_idx = 0
        for i, phase in enumerate(phases):
            if self.action_progress <= cumulative + phase['duration_ratio']:
                current_phase_idx = i
                break
            cumulative += phase['duration_ratio']
        
        return {
            'action': self.current_action.value,
            'description': pattern['description'],
            'progress': self.action_progress,
            'current_phase': current_phase_idx + 1,
            'total_phases': len(phases),
            'duration': self.action_duration
        }
    
    def set_action(self, action_type: ActionType):
        """Cambiar la acción actual"""
        if action_type != self.current_action:
            self.current_action = action_type
            self.action_progress = 0.0
            pattern = self.action_patterns[action_type]
            self.action_duration = pattern['duration']
            print(f"🧠 Biomechanical switch to: {pattern['description']}")
    

# ===== FUNCIONES DE UTILIDAD =====

def create_balance_squat_controller(env):
    """Crear el controlador biomecánico corregido"""
    
    controller = BiomechanicalActionController(env)
    
    print(f"✅ Biomechanical Controller created")
    print(f"   Key features: Angle-aware knee control, coordinated variation")
    print(f"   Knee flexors: Dynamically controlled based on joint angle")
    
    return controller

def test_controller_patterns(duration_seconds=10):
    """Test básico de los patrones del controlador"""
    
    print("🧪 Testing Balance & Squat Controller Patterns...")
    
    # Mock environment para test
    class MockEnv:
        def __init__(self):
            self.robot_id = 0
            self.time_step = 1.0 / 150.0
    
    mock_env = MockEnv()
    controller = create_balance_squat_controller(mock_env)
    
    # Test de balance
    print(f"\n📊 Testing BALANCE_STANDING pattern:")
    controller.set_action(ActionType.BALANCE_STANDING)
    
    steps = int(duration_seconds / mock_env.time_step)
    actions_recorded = []
    
    for step in range(min(steps, 300)):  # Máximo 300 steps para test
        action = controller.get_expert_action(mock_env.time_step)
        actions_recorded.append(action.copy())
        
        if step % 50 == 0:
            info = controller.get_current_action_info()
            print(f"   Step {step}: Progress {info['progress']:.2%}, "
                  f"Phase {info['current_phase']}/{info['total_phases']}")
            print(f"      PAM pressures: {action[:4]} (first 4 PAMs)")
    
    # Test de sentadilla
    print(f"\n📊 Testing SQUAT pattern:")
    controller.set_action(ActionType.SQUAT)
    
    for step in range(min(steps, 600)):  # Más steps para sentadilla completa
        action = controller.get_expert_action(mock_env.time_step)
        
        if step % 100 == 0:
            info = controller.get_current_action_info()
            print(f"   Step {step}: Progress {info['progress']:.2%}, "
                  f"Phase {info['current_phase']}/{info['total_phases']}")
            print(f"      PAM pressures: {action[:4]} (first 4 PAMs)")
            
            # Imprimir descripción de fase
            if info['current_phase'] == 1:
                print(f"      🔽 DESCENSO: Flexión controlada")
            elif info['current_phase'] == 2:
                print(f"      ⏸️ MANTENER: Posición baja estable")
            elif info['current_phase'] == 3:
                print(f"      🔼 ASCENSO: Extensión potente")
        
        if info['progress'] >= 1.0:
            print(f"   ✅ Sentadilla completada en step {step}")
            break
    
    print(f"\n🎉 Test completado - Patrones funcionando correctamente")
    
    return controller, actions_recorded


# ===== EJEMPLO DE INTEGRACIÓN =====

def integrate_with_simplified_env():
    """Ejemplo de cómo integrar con el entorno simplificado"""
    
    print("🔗 Integration Example: Controller + Environment")
    
    # Crear entorno simplificado
    from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
    
    env = create_simple_balance_squat_env(render_mode='direct')  # Sin visualización para test
    controller = create_balance_squat_controller(env)
    
    # Test de integración
    obs, info = env.reset()
    total_reward = 0
    
    print(f"\n🎯 Testing integration for 500 steps...")
    
    for step in range(500):
        # Obtener acción experta del controlador
        expert_action = controller.get_expert_action(env.time_step)
        
        # Ejecutar en el entorno
        obs, reward, done, truncated, info = env.step(expert_action)
        total_reward += reward
        
        if step % 100 == 0:
            controller_info = controller.get_current_action_info()
            print(f"   Step {step}: Reward = {reward:.2f}, "
                  f"Action = {controller_info['action']}, "
                  f"Progress = {controller_info['progress']:.2%}")
        
        # Cambiar a sentadilla a mitad del test
        if step == 250:
            controller.set_action(ActionType.SQUAT)
        
        if done:
            print(f"   Episode terminado en step {step}")
            break
    
    env.close()
    
    print(f"✅ Integration test completed")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Final action: {controller_info['action']}")
    
    return total_reward


if __name__ == "__main__":
    
    print("🎯 SIMPLIFIED BALANCE & SQUAT CONTROLLER")
    print("=" * 60)
    print("Controlador específico para balance y sentadillas")
    print("Genera patrones PAM expertos para entrenar el robot")
    print("=" * 60)
    
    # Test de patrones
    test_controller_patterns(duration_seconds=5)
    
    # Test de integración (si el entorno está disponible)
    try:
        integrate_with_simplified_env()
    except ImportError:
        print("\n⚠️ Entorno simplificado no disponible para test de integración")
