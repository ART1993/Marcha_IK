import numpy as np
import math
from enum import Enum

class ActionType(Enum):
    """Define los tipos de acciones discretas que el robot puede realizar"""
    BALANCE_STANDING = "balance_standing"
    SQUAT = "squat"

class DiscreteActionController:
    """
    Controlador para acciones discretas del robot bípedo con PAMs.
    
    Este controlador genera patrones de presión PAM expertos para acciones
    específicas en lugar de ciclos continuos de marcha. Cada acción tiene
    su propio generador de patrones que considera la biomecánica del movimiento.
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
        
    def setup_action_patterns(self):
        """Define los patrones de activación PAM para cada acción discreta"""
        
        self.action_patterns = {
            ActionType.BALANCE_STANDING: {
                'description': 'Mantener postura erguida estable',
                'duration': 3.0,  # Más tiempo para practicar equilibrio
                'phases': [
                    # Fase única: co-activación moderada para estabilidad
                    {
                        'duration_ratio': 1.0,
                        'pressures': {
                            'left_hip_flexor': 0.25,
                            'left_hip_extensor': 0.35,  # Ligeramente más para anti-gravedad
                            'right_hip_flexor': 0.25,
                            'right_hip_extensor': 0.35,
                            'left_knee_flexor': 0.15,
                            'right_knee_flexor': 0.15,
                        }
                    }
                ]
            },
            
            ActionType.SQUAT: {
                'description': 'Sentadilla controlada',
                'duration': 3.0,
                'phases': [
                    # Fase 1: Descenso controlado
                    {
                        'duration_ratio': 0.4,
                        'pressures': {
                            'left_hip_flexor': 0.4,
                            'left_hip_extensor': 0.5,  # Control excéntrico
                            'right_hip_flexor': 0.4,
                            'right_hip_extensor': 0.5,
                            'left_knee_flexor': 0.6,   # Flexión controlada
                            'right_knee_flexor': 0.6,
                        }
                    },
                    # Fase 2: Posición baja - mantener
                    {
                        'duration_ratio': 0.2,
                        'pressures': {
                            'left_hip_flexor': 0.5,
                            'left_hip_extensor': 0.6,  # Co-activación para estabilidad
                            'right_hip_flexor': 0.5,
                            'right_hip_extensor': 0.6,
                            'left_knee_flexor': 0.7,
                            'right_knee_flexor': 0.7,
                        }
                    },
                    # Fase 3: Ascenso - extensión potente
                    {
                        'duration_ratio': 0.4,
                        'pressures': {
                            'left_hip_flexor': 0.3,
                            'left_hip_extensor': 0.8,  # Fuerte extensión
                            'right_hip_flexor': 0.3,
                            'right_hip_extensor': 0.8,
                            'left_knee_flexor': 0.2,   # Los resortes ayudan
                            'right_knee_flexor': 0.2,
                        }
                    }
                ]
            },
            
            ActionType.STEP_LEFT: {
                'description': 'Paso hacia adelante con pierna izquierda',
                'duration': 2.0,
                'phases': [
                    # Fase 1: Preparación y shift de peso
                    {
                        'duration_ratio': 0.25,
                        'pressures': {
                            'left_hip_flexor': 0.35,
                            'left_hip_extensor': 0.3,
                            'right_hip_flexor': 0.3,
                            'right_hip_extensor': 0.5,
                            'left_knee_flexor': 0.3,
                            'right_knee_flexor': 0.25,
                        }
                    },
                    # Fase 2: Swing de pierna izquierda
                    {
                        'duration_ratio': 0.35,
                        'pressures': {
                            'left_hip_flexor': 0.7,    # Flexión para avanzar
                            'left_hip_extensor': 0.2,
                            'right_hip_flexor': 0.25,
                            'right_hip_extensor': 0.6,  # Empuje
                            'left_knee_flexor': 0.7,    # Clearance
                            'right_knee_flexor': 0.2,
                        }
                    },
                    # Fase 3: Contacto y estabilización
                    {
                        'duration_ratio': 0.4,
                        'pressures': {
                            'left_hip_flexor': 0.3,
                            'left_hip_extensor': 0.5,   # Recepción del peso
                            'right_hip_flexor': 0.4,
                            'right_hip_extensor': 0.4,
                            'left_knee_flexor': 0.2,
                            'right_knee_flexor': 0.3,
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
            np.array: 6 presiones PAM normalizadas [-1, 1]
        """
        # Actualizar progreso de la acción
        self.action_progress += time_step / self.action_duration
        
        # Si la acción terminó, mantener la última fase
        if self.action_progress > 1.0:
            self.action_progress = 1.0
        
        # Obtener patrón de la acción actual
        pattern = self.action_patterns[self.current_action]
        phases = pattern['phases']
        
        # Determinar en qué fase estamos
        cumulative_duration = 0.0
        current_phase_data = phases[-1]  # Por defecto, última fase
        phase_local_progress = 1.0
        
        for phase in phases:
            phase_end = cumulative_duration + phase['duration_ratio']
            if self.action_progress <= phase_end:
                current_phase_data = phase
                # Calcular progreso local dentro de esta fase
                phase_start = cumulative_duration
                phase_local_progress = (self.action_progress - phase_start) / phase['duration_ratio']
                phase_local_progress = np.clip(phase_local_progress, 0.0, 1.0)
                break
            cumulative_duration = phase_end
        
        # Obtener presiones de la fase actual
        current_pressures = current_phase_data['pressures']
        
        # Si hay una siguiente fase, interpolar suavemente
        next_phase_idx = phases.index(current_phase_data) + 1
        if next_phase_idx < len(phases) and phase_local_progress > 0.7:
            # Comenzar transición suave en el último 30% de la fase
            next_pressures = phases[next_phase_idx]['pressures']
            blend_factor = (phase_local_progress - 0.7) / 0.3  # 0 a 1 en el último 30%
            
            # Interpolar entre fases
            interpolated_pressures = {}
            for muscle in current_pressures:
                interpolated_pressures[muscle] = (
                    current_pressures[muscle] * (1 - blend_factor) +
                    next_pressures[muscle] * blend_factor
                )
            current_pressures = interpolated_pressures
        
        # Convertir a array de acciones
        actions = np.zeros(6, dtype=np.float32)
        for muscle_name, pressure in current_pressures.items():
            idx = self.pam_mapping[muscle_name]
            # Convertir de [0,1] a [-1,1]
            actions[idx] = 2.0 * pressure - 1.0
        
        # Aplicar suavizado temporal si tenemos historial
        if hasattr(self, 'last_action'):
            # Suavizado para evitar cambios bruscos
            smoothing = 0.9  # Factor de suavizado
            actions = smoothing * self.last_action + (1 - smoothing) * actions
        
        self.last_action = actions.copy()
        
        return actions
    
    def get_action_for_phase(self, curriculum_phase):
        """
        Retorna el tipo de acción apropiado para cada fase del currículo.
        
        Esto mapea las fases del ExpertCurriculumManager a acciones específicas.
        """
        phase_to_action = {
            0: ActionType.BALANCE_STANDING,
            1: ActionType.BALANCE_STANDING,
            2: ActionType.BALANCE_STANDING,
            # Añadir fases para sentadillas si las agregas al currículo
            3: ActionType.SQUAT,
            4: ActionType.SQUAT,
            # Resto de fases como antes
            5: ActionType.LIFT_LEFT_LEG,
            6: ActionType.LIFT_LEFT_LEG,
            7: ActionType.LIFT_RIGHT_LEG,
            8: ActionType.LIFT_RIGHT_LEG,
            9: ActionType.STEP_LEFT,
            10: ActionType.STEP_LEFT,
            11: ActionType.STEP_RIGHT,
            12: ActionType.STEP_RIGHT,
            
        }
        
        return phase_to_action.get(curriculum_phase, ActionType.BALANCE_STANDING)
    
    def reset(self):
        """Reinicia el controlador a su estado inicial"""
        self.current_action = ActionType.BALANCE_STANDING
        self.action_progress = 0.0
        if hasattr(self, 'last_action'):
            del self.last_action
    
    def get_debug_info(self):
        """Retorna información útil para debugging"""
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
            'current_action': self.current_action.value,
            'action_progress': self.action_progress,
            'current_phase': current_phase_idx,
            'total_phases': len(phases),
            'action_duration': self.action_duration,
            'last_action': self.last_action.tolist() if hasattr(self, 'last_action') else None
        }
