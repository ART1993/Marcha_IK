import numpy as np
import math
from enum import Enum

class ActionType(Enum):
    """Define los tipos de acciones discretas que el robot puede realizar"""
    BALANCE_STANDING = "balance_standing"
    SQUAT = "squat"

class DiscreteActionController:
    """
        Controlador SIMPLIFICADO para balance y sentadillas con PAMs.
        
        OBJETIVO ESPECÍFICO:
        - BALANCE_STANDING: Mantener equilibrio de pie
        - SQUAT: Realizar sentadillas controladas
        
        ELIMINADO:
        - Acciones de marcha (STEP_LEFT, STEP_RIGHT)
        - Acciones de levantamiento de piernas
        - Mapeo complejo de curriculum phases
        - Múltiples fases complejas por acción
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
        
        # Configurar patrones base para cada acción
        self.setup_action_patterns()

        print(f"🎯 Simplified Balance & Squat Controller initialized")
        print(f"   Available actions: BALANCE_STANDING, SQUAT")
        
    def setup_action_patterns(self):
        """Define los patrones de activación PAM para cada acción discreta"""
        
        self.action_patterns = {
            ActionType.BALANCE_STANDING: {
                'description': 'Mantener postura erguida estable',
                'duration': 5.0,  # Más tiempo para practicar equilibrio
                'phases': [
                    # Fase única: co-activación moderada para estabilidad
                    {
                        'duration_ratio': 1.0,
                        'pressures': {
                            'left_hip_flexor': 0.20,
                            'left_hip_extensor': 0.45,  # Ligeramente más para anti-gravedad
                            'right_hip_flexor': 0.20,
                            'right_hip_extensor': 0.45,
                            # Si se flexiona demasiado entonces pone a cero
                            'left_knee_flexor': 0.12,
                            'right_knee_flexor': 0.12,
                        }
                    }
                ]
            },
            
            ActionType.SQUAT: {
                'description': 'Sentadilla controlada - después de balance estable',
                'duration': 6.0, # Tiempo suficiente para movimiento completo
                'phases': [
                    # Fase 1: Preparación
                    {
                        'duration_ratio': 0.15,
                        'pressures': {
                            'left_hip_flexor': 0.3,
                            'left_hip_extensor': 0.5,  # Control excéntrico
                            'right_hip_flexor': 0.3,
                            'right_hip_extensor': 0.5,
                            'left_knee_flexor': 0.15,   # Flexión controlada
                            'right_knee_flexor': 0.15,
                        }
                    },
                    # Fase 2: Descenso controlado
                    {
                        'duration_ratio': 0.35,
                        'pressures': {
                            'left_hip_flexor': 0.55,
                            'left_hip_extensor': 0.40,  # Co-activación para estabilidad
                            'right_hip_flexor': 0.55,
                            'right_hip_extensor': 0.40,
                            'left_knee_flexor': 0.50,
                            'right_knee_flexor': 0.50,
                        }
                    },
                    # Fase 3: Posición baja, Mantener
                    {
                        'duration_ratio': 0.15,
                        'pressures': {
                            'left_hip_flexor': 0.3,
                            'left_hip_extensor': 0.7,  # Fuerte extensión
                            'right_hip_flexor': 0.3,
                            'right_hip_extensor': 0.7,
                            'left_knee_flexor': 0.2,   # Los resortes ayudan
                            'right_knee_flexor': 0.2,
                        }
                    },
                        # Fase 4: Ascenso - Extensión potente
                    {
                    'duration_ratio': 0.35,  # 35% del tiempo
                    'pressures': {
                        'left_hip_flexor': 0.25,    # Reducir flexión
                        'left_hip_extensor': 0.75,  # Extensión potente
                        'right_hip_flexor': 0.25,
                        'right_hip_extensor': 0.75,
                        'left_knee_flexor': 0.15,   # Los resortes ayudan
                        'right_knee_flexor': 0.15,
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
        
        # Obtener patrón de la acción actual
        pattern = self.action_patterns[self.current_action]
        phases = pattern['phases']
        
        # Determinar en qué fase estamos
        cumulative_duration = 0.0
        current_phase = phases[0]  # Por defecto, primera fase
        phase_local_progress = 0.0
        
        for phase in phases:
            phase_end = cumulative_duration + phase['duration_ratio']
            if self.action_progress <= phase_end:
                # Calcular progreso dentro de la fase
                phase_start = cumulative_duration
                phase_local_progress = (self.action_progress - phase_start) / phase['duration_ratio']
                break
            cumulative_duration += phase['duration_ratio']
        
        # ===== GENERAR PRESIONES PAM =====
    
        pam_pressures = np.zeros(6, dtype=np.float32)
        
        for muscle_name, pressure in current_phase['pressures'].items():
            if muscle_name in self.pam_mapping:
                pam_index = self.pam_mapping[muscle_name]
                pam_pressures[pam_index] = pressure
        
        # Aplicar variación suave para naturalidad
        if hasattr(self, 'last_action'):
            smoothing_factor = 0.15  # 15% de suavizado
            pam_pressures = (1 - smoothing_factor) * pam_pressures + smoothing_factor * self.last_action
        
        # ===== APLICAR VARIACIÓN NATURAL =====
    
        # Añadir pequeña variación para simular control biológico natural
        if self.current_action == ActionType.BALANCE_STANDING:
            # Variación muy pequeña para balance
            noise_amplitude = 0.02
            natural_variation = np.random.normal(0, noise_amplitude, 6)
            pam_pressures += natural_variation
        
        # Asegurar límites
        pam_pressures = np.clip(pam_pressures, 0.0, 1.0)
        
        self.last_action = pam_pressures.copy()
        
        return pam_pressures
    
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
    

# ===== FUNCIONES DE UTILIDAD =====

def create_balance_squat_controller(env):
    """Crear controlador simplificado para balance y sentadillas"""
    
    controller = DiscreteActionController(env)
    
    print(f"✅ Balance & Squat Controller created")
    print(f"   Focus: Static balance + Dynamic squats")
    print(f"   Actions: {len(controller.action_patterns)} patterns defined")
    
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
