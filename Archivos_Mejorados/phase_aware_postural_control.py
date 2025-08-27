# ====================================================================
# SISTEMA DE CONTROL POSTURAL CONSCIENTE DE FASE
# Adaptación dinámica para equilibrio estático y sentadillas dinámicas
# ====================================================================

import numpy as np
from collections import deque
import math
from enum import Enum
import pybullet as p

from Controlador.discrete_action_controller import DiscreteActionController

class MovementPhase(Enum):
    """
    Enumera las diferentes fases del movimiento para control postural adaptativo
    
    Cada fase tiene sus propios objetivos posturales y estrategias de corrección:
    - STATIC_BALANCE: Mantener postura erguida y estable
    - SQUAT_PREPARATION: Inicio controlado de la flexión
    - SQUAT_DESCENT: Descenso controlado hacia posición baja  
    - SQUAT_BOTTOM: Mantener posición baja estable
    - SQUAT_ASCENT: Ascenso controlado hacia posición erguida
    - SQUAT_RECOVERY: Estabilización final después de sentadilla
    """
    STATIC_BALANCE = "static_balance"
    SQUAT_PREPARATION = "squat_preparation" 
    SQUAT_DESCENT = "squat_descent"
    SQUAT_BOTTOM = "squat_bottom"
    SQUAT_ASCENT = "squat_ascent"
    SQUAT_RECOVERY = "squat_recovery"

class PhaseAwarePosturalSystem:
    """
    Sistema de control postural que adapta sus objetivos y estrategias
    según la fase actual del movimiento.
    
    Este sistema imita cómo el sistema nervioso humano ajusta constantemente
    sus estrategias de control motor según la tarea que está ejecutando.
    
    Principios biomecánicos implementados:
    - Control postural específico por fase
    - Transiciones suaves entre objetivos posturales
    - Detección automática de cambios de fase
    - Adaptación de sensibilidad según la fase
    """
    
    def __init__(self):
        # Estado actual del sistema
        self.current_phase = MovementPhase.STATIC_BALANCE
        self.phase_start_time = 0
        self.previous_phase = None
        
        # Definir objetivos posturales para cada fase
        self.phase_targets = self._initialize_phase_targets()
        
        # Parámetros de control adaptativos por fase
        self.phase_control_params = self._initialize_phase_control_params()
        
        # Historia para detección de fases y control suave
        self.hip_angle_history = deque(maxlen=10)
        self.hip_velocity_history = deque(maxlen=5)
        self.phase_history = deque(maxlen=5)
        
        # Métricas de rendimiento por fase
        self.phase_performance = {phase: {'corrections': 0, 'successes': 0} for phase in MovementPhase}
        
        print(f"🧠 Sistema de Control Postural Consciente de Fase inicializado")
        print(f"   Fases soportadas: {len(MovementPhase)} tipos de movimiento")
        print(f"   Fase inicial: {self.current_phase.value}")
    
    def _initialize_phase_targets(self):
        """
        Define los objetivos posturales específicos para cada fase del movimiento
        
        Estos objetivos representan la postura "ideal" que el robot debería
        mantener durante cada fase para maximizar estabilidad y eficiencia
        """
        return {
            MovementPhase.STATIC_BALANCE: {
                'target_hip_angle': 0.0,      # Cadera paralela al suelo
                'tolerance': 0.02,             # ±1.1° de tolerancia
                'target_knee_angle': 0.0,      # Rodillas extendidas
                'stability_priority': 'high'   # Prioridad alta en estabilidad
            },
            
            MovementPhase.SQUAT_PREPARATION: {
                'target_hip_angle': 0.1,      # Inicio suave de flexión (5.7°)
                'tolerance': 0.03,             # Un poco más de tolerancia
                'target_knee_angle': 0.05,     # Inicio ligero de flexión
                'stability_priority': 'high'   # Mantener alta estabilidad
            },
            
            MovementPhase.SQUAT_DESCENT: {
                'target_hip_angle': lambda progress: 0.1 + (1.2 * progress),  # Flexión progresiva
                'tolerance': 0.05,             # Mayor tolerancia durante movimiento
                'target_knee_angle': lambda progress: 0.05 + (1.0 * progress), # Flexión proporcional
                'stability_priority': 'medium' # Permitir cierto movimiento
            },
            
            MovementPhase.SQUAT_BOTTOM: {
                'target_hip_angle': 1.3,      # Flexión profunda (~75°)
                'tolerance': 0.04,             # Tolerancia moderada
                'target_knee_angle': 1.05,     # Flexión profunda de rodillas (~60°)
                'stability_priority': 'high'   # Alta estabilidad en posición vulnerable
            },
            
            MovementPhase.SQUAT_ASCENT: {
                'target_hip_angle': lambda progress: 1.3 - (1.2 * progress),  # Extensión progresiva
                'tolerance': 0.05,             # Mayor tolerancia durante movimiento
                'target_knee_angle': lambda progress: 1.05 - (1.0 * progress), # Extensión proporcional
                'stability_priority': 'medium' # Permitir movimiento controlado
            },
            
            MovementPhase.SQUAT_RECOVERY: {
                'target_hip_angle': 0.05,     # Casi neutral, con micro-flexión
                'tolerance': 0.02,             # Volver a precisión alta
                'target_knee_angle': 0.0,      # Rodillas extendidas
                'stability_priority': 'high'   # Restablecer estabilidad completa
            }
        }
    
    def _initialize_phase_control_params(self):
        """
        Define parámetros de control específicos para cada fase
        
        Diferentes fases requieren diferentes niveles de agresividad y
        sensibilidad en las correcciones posturales
        """
        return {
            MovementPhase.STATIC_BALANCE: {
                'kp_hip': 2.0,        # Control proporcional estándar
                'kd_hip': 0.8,        # Buena amortiguación para estabilidad
                'max_correction': 0.12, # Correcciones moderadas
                'update_frequency': 1    # Correcciones cada step
            },
            
            MovementPhase.SQUAT_PREPARATION: {
                'kp_hip': 1.8,        # Ligeramente menos agresivo
                'kd_hip': 1.0,        # Mayor amortiguación para suavidad
                'max_correction': 0.10, # Correcciones más suaves
                'update_frequency': 1    # Monitoreo constante
            },
            
            MovementPhase.SQUAT_DESCENT: {
                'kp_hip': 1.5,        # Menos agresivo durante movimiento
                'kd_hip': 1.2,        # Alta amortiguación para suavidad
                'max_correction': 0.08, # Correcciones gentiles
                'update_frequency': 2    # Menos frecuente para permitir movimiento
            },
            
            MovementPhase.SQUAT_BOTTOM: {
                'kp_hip': 2.2,        # Más agresivo en posición vulnerable
                'kd_hip': 0.9,        # Amortiguación balanceada
                'max_correction': 0.15, # Correcciones más fuertes si es necesario
                'update_frequency': 1    # Monitoreo constante
            },
            
            MovementPhase.SQUAT_ASCENT: {
                'kp_hip': 1.6,        # Moderado durante ascenso
                'kd_hip': 1.1,        # Buena amortiguación
                'max_correction': 0.09, # Correcciones moderadas
                'update_frequency': 2    # Permitir movimiento natural
            },
            
            MovementPhase.SQUAT_RECOVERY: {
                'kp_hip': 2.3,        # Muy agresivo para estabilización rápida
                'kd_hip': 0.7,        # Menos amortiguación para respuesta rápida
                'max_correction': 0.14, # Correcciones fuertes para recuperar
                'update_frequency': 1    # Monitoreo constante
            }
        }
    
    def detect_current_phase(self, robot_id, controller_action_info):
        """
        Detecta automáticamente en qué fase del movimiento está el robot
        
        Utiliza información del controlador de acciones y estados articulares
        para determinar la fase actual y detectar transiciones
        """
        # Obtener estados articulares actuales
        joint_states = p.getJointStates(robot_id, [0, 1, 3, 4])  # caderas y rodillas
        left_hip_angle = joint_states[0][0]
        right_hip_angle = joint_states[2][0]  # índice correcto para right_hip
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2.0
        
        # Actualizar historia
        self.hip_angle_history.append(avg_hip_angle)
        
        # Calcular velocidad angular promedio de cadera
        if len(self.hip_angle_history) >= 2:
            hip_velocity = (self.hip_angle_history[-1] - self.hip_angle_history[-2]) * 1500
            self.hip_velocity_history.append(hip_velocity)
        else:
            hip_velocity = 0.0
        
        # Obtener información del controlador sobre la acción actual
        current_action = controller_action_info.get('action', 'balance_standing')
        action_progress = controller_action_info.get('progress', 0.0)
        
        # Lógica de detección de fases basada en múltiples factores
        detected_phase = self._analyze_movement_phase(
            avg_hip_angle, hip_velocity, current_action, action_progress
        )
        
        # Manejar transiciones de fase
        if detected_phase != self.current_phase:
            self._handle_phase_transition(detected_phase)
        
        return detected_phase
    
    def _analyze_movement_phase(self, hip_angle, hip_velocity, action, progress):
        """
        Analiza múltiples indicadores para determinar la fase actual
        
        Esta función implementa la lógica de reconocimiento de patrones
        que determina en qué fase del movimiento está el robot
        """
        
        # Si el controlador está explícitamente en modo balance
        if action == 'balance_standing':
            # Determinar si es balance estático o recuperación
            if abs(hip_velocity) < 0.1 and abs(hip_angle) < 0.05:
                return MovementPhase.STATIC_BALANCE
            elif abs(hip_angle) > 0.05 and abs(hip_velocity) > 0.05:
                return MovementPhase.SQUAT_RECOVERY  # Recuperándose de movimiento
            else:
                return MovementPhase.STATIC_BALANCE
        
        # Si el controlador está en modo sentadilla
        elif action == 'squat':
            # Usar el progreso de la acción para determinar sub-fase
            if progress < 0.15:  # Primeros 15% - preparación
                return MovementPhase.SQUAT_PREPARATION
            elif progress < 0.50:  # 15-50% - descenso
                return MovementPhase.SQUAT_DESCENT
            elif progress < 0.65:  # 50-65% - mantener posición baja
                return MovementPhase.SQUAT_BOTTOM
            elif progress < 1.0:   # 65-100% - ascenso
                return MovementPhase.SQUAT_ASCENT
            else:  # Completado - recuperación
                return MovementPhase.SQUAT_RECOVERY
        
        # Fallback - usar ángulo de cadera como indicador principal
        if abs(hip_angle) < 0.05:
            return MovementPhase.STATIC_BALANCE
        elif hip_angle > 0.8:  # Flexión profunda
            if abs(hip_velocity) < 0.2:
                return MovementPhase.SQUAT_BOTTOM
            elif hip_velocity < 0:  # Velocidad negativa = extensión
                return MovementPhase.SQUAT_ASCENT
            else:  # Velocidad positiva = flexión
                return MovementPhase.SQUAT_DESCENT
        else:  # Flexión intermedia
            if hip_velocity > 0.2:  # Flexionando rápido
                return MovementPhase.SQUAT_DESCENT
            elif hip_velocity < -0.2:  # Extendiendo rápido
                return MovementPhase.SQUAT_ASCENT
            else:
                return MovementPhase.SQUAT_PREPARATION
    
    def _handle_phase_transition(self, new_phase):
        """
        Maneja las transiciones entre fases de manera suave y controlada
        
        Las transiciones abruptas pueden causar inestabilidad, por lo que
        implementamos cambios graduales en los parámetros de control
        """
        self.previous_phase = self.current_phase
        self.current_phase = new_phase
        self.phase_start_time = 0  # Reset timer para nueva fase
        
        # Registrar transición para análisis
        self.phase_history.append({
            'from_phase': self.previous_phase.value if self.previous_phase else 'none',
            'to_phase': new_phase.value,
            'transition_time': len(self.phase_history)
        })
        
        # Debug de transiciones importantes
        if self.previous_phase:
            print(f"   🔄 Transición de fase: {self.previous_phase.value} → {new_phase.value}")
    
    def calculate_phase_aware_correction(self, robot_id, controller_info):
        """
        Calcula correcciones posturales adaptadas a la fase actual del movimiento
        
        Esta es la función principal que integra toda la información para
        generar correcciones específicas y apropiadas para cada fase
        """
        # Detectar fase actual
        current_phase = self.detect_current_phase(robot_id, controller_info)
        
        # Obtener objetivos y parámetros para la fase actual
        phase_targets = self.phase_targets[current_phase]
        control_params = self.phase_control_params[current_phase]
        
        # Obtener estados articulares actuales
        joint_states = p.getJointStates(robot_id, [0, 1, 3, 4])
        avg_hip_angle = (joint_states[0][0] + joint_states[2][0]) / 2.0
        
        # Calcular objetivo dinámico si es una fase de movimiento
        if callable(phase_targets['target_hip_angle']):
            # Fases con objetivos que cambian durante el movimiento
            movement_progress = controller_info.get('progress', 0.0)
            target_hip_angle = phase_targets['target_hip_angle'](movement_progress)
        else:
            # Fases con objetivo fijo
            target_hip_angle = phase_targets['target_hip_angle']
        
        # Calcular error postural
        hip_error = avg_hip_angle - target_hip_angle
        
        # Verificar si necesitamos corrección (basado en tolerancia de la fase)
        tolerance = phase_targets['tolerance']
        needs_correction = abs(hip_error) > tolerance
        
        if not needs_correction:
            return {
                'correction_applied': False,
                'phase': current_phase.value,
                'target_angle': target_hip_angle,
                'current_angle': avg_hip_angle,
                'error': hip_error,
                'within_tolerance': True
            }
        
        # Calcular corrección usando parámetros específicos de la fase
        hip_velocity = self.hip_velocity_history[-1] if self.hip_velocity_history else 0.0
        
        # Control PD adaptativo
        proportional_term = control_params['kp_hip'] * hip_error
        derivative_term = control_params['kd_hip'] * hip_velocity
        total_correction = proportional_term + derivative_term
        
        # Aplicar límites específicos de la fase
        max_correction = control_params['max_correction']
        total_correction = np.clip(total_correction, -max_correction, max_correction)
        
        # Generar ajustes PAM específicos según la fase
        pam_adjustments = self._calculate_phase_specific_adjustments(
            current_phase, total_correction, hip_error
        )
        
        # Registrar métricas
        self.phase_performance[current_phase]['corrections'] += 1
        
        return {
            'correction_applied': True,
            'phase': current_phase.value,
            'target_angle': target_hip_angle,
            'current_angle': avg_hip_angle,
            'error': hip_error,
            'total_correction': total_correction,
            'pam_adjustments': pam_adjustments,
            'control_params_used': control_params,
            'within_tolerance': False
        }
    
    def _calculate_phase_specific_adjustments(self, phase, correction, error):
        """
        Calcula ajustes específicos para cada fase del movimiento
        
        Diferentes fases pueden requerir estrategias de corrección diferentes
        Por ejemplo, durante el descenso de sentadilla podríamos querer
        ajustar también las rodillas, no solo las caderas
        """
        base_adjustments = {
            'left_hip_flexor_adj': 0.0,
            'left_hip_extensor_adj': 0.0, 
            'right_hip_flexor_adj': 0.0,
            'right_hip_extensor_adj': 0.0,
            'left_knee_flexor_adj': 0.0,
            'right_knee_flexor_adj': 0.0
        }
        
        # Lógica base de corrección (igual que antes)
        if error > 0:  # Cadera muy flexionada
            flexor_adj = -abs(correction)
            extensor_adj = +abs(correction)
        else:  # Cadera muy extendida
            flexor_adj = +abs(correction)
            extensor_adj = -abs(correction)
        
        # Aplicar ajustes base a caderas
        base_adjustments['left_hip_flexor_adj'] = flexor_adj
        base_adjustments['left_hip_extensor_adj'] = extensor_adj
        base_adjustments['right_hip_flexor_adj'] = flexor_adj
        base_adjustments['right_hip_extensor_adj'] = extensor_adj
        
        # Ajustes específicos por fase
        if phase == MovementPhase.SQUAT_BOTTOM:
            # En posición baja, también ajustar ligeramente las rodillas para estabilidad
            knee_correction = correction * 0.3  # 30% de la corrección de cadera
            base_adjustments['left_knee_flexor_adj'] = knee_correction
            base_adjustments['right_knee_flexor_adj'] = knee_correction
            
        elif phase in [MovementPhase.SQUAT_DESCENT, MovementPhase.SQUAT_ASCENT]:
            # Durante movimiento, correcciones más suaves
            for key in base_adjustments:
                if 'hip' in key:
                    base_adjustments[key] *= 0.7  # Reducir intensidad 30%
                    
        elif phase == MovementPhase.SQUAT_RECOVERY:
            # Durante recuperación, correcciones más agresivas para estabilizar
            for key in base_adjustments:
                if 'hip' in key:
                    base_adjustments[key] *= 1.2  # Aumentar intensidad 20%
        
        return base_adjustments
    
    def get_phase_performance_summary(self):
        """
        Resumen del rendimiento del sistema por fase
        
        Útil para analizar qué fases están funcionando bien y cuáles
        necesitan ajustes en los parámetros
        """
        summary = {}
        
        for phase, performance in self.phase_performance.items():
            corrections = performance['corrections']
            successes = performance['successes']
            
            if corrections > 0:
                success_rate = (successes / corrections) * 100
            else:
                success_rate = 0.0
                
            summary[phase.value] = {
                'corrections_applied': corrections,
                'successes': successes,
                'success_rate_percent': success_rate,
                'needs_tuning': success_rate < 60.0  # Flag fases problemáticas
            }
        
        return summary

# ====================================================================
# INTEGRACIÓN CON EL CONTROLADOR EXISTENTE  
# Modificación del Enhanced Controller para usar el sistema consciente de fase
# ====================================================================

class PhaseAwareEnhancedController(DiscreteActionController):
    """
    Controlador mejorado que integra control postural consciente de fase
    
    Este controlador extiende el sistema anterior para manejar tanto
    equilibrio estático como movimientos dinámicos de sentadilla con
    control postural adaptativo en cada fase
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Integrar sistema postural consciente de fase
        self.postural_system = PhaseAwarePosturalSystem()
        
        # Métricas de rendimiento
        self.total_corrections = 0
        self.phase_specific_corrections = {phase: 0 for phase in MovementPhase}
        
        print(f"🎯 Controlador Consciente de Fase inicializado")
        print(f"   Control postural adaptativo: Activo")
        print(f"   Fases soportadas: Equilibrio + Sentadillas dinámicas")
    
    def get_expert_action(self, time_step):
        """
        Versión mejorada que aplica control postural consciente de fase
        
        El flujo ahora incluye:
        1. Obtener acción base del patrón experto
        2. Determinar fase actual del movimiento  
        3. Aplicar correcciones específicas para esa fase
        4. Devolver acción optimizada para la fase actual
        """
        # PASO 1: Obtener acción base
        base_action = super().get_expert_action(time_step)
        
        # PASO 2: Obtener información sobre la acción actual del controlador
        controller_info = self.get_current_action_info()
        
        # PASO 3: Calcular correcciones conscientes de fase
        phase_correction = self.postural_system.calculate_phase_aware_correction(
            self.env.robot_id, controller_info
        )
        
        # PASO 4: Aplicar correcciones si son necesarias
        if phase_correction['correction_applied']:
            corrected_action = self._apply_phase_aware_corrections(
                base_action, phase_correction['pam_adjustments']
            )
            
            # Registrar métricas
            current_phase = MovementPhase(phase_correction['phase'])
            self.phase_specific_corrections[current_phase] += 1
            self.total_corrections += 1
            
            # Debug cada 50 correcciones para no saturar
            if self.total_corrections % 50 == 0:
                print(f"   🧠 Corrección fase-específica #{self.total_corrections}")
                print(f"      Fase: {phase_correction['phase']}")
                print(f"      Error: {phase_correction['error']:.3f} rad")
                print(f"      Objetivo: {phase_correction['target_angle']:.3f} rad")
        else:
            corrected_action = base_action
        
        return corrected_action
    
    def _apply_phase_aware_corrections(self, base_action, pam_adjustments):
        """
        Aplica correcciones conscientes de fase a la acción base
        
        Similar al método anterior, pero ahora puede incluir ajustes
        tanto de caderas como de rodillas según la fase
        """
        corrected_action = base_action.copy()
        
        # Mapear ajustes a índices PAM (incluyendo rodillas ahora)
        adjustment_mapping = {
            0: pam_adjustments['left_hip_flexor_adj'],     # Left Hip Flexor
            1: pam_adjustments['left_hip_extensor_adj'],   # Left Hip Extensor
            2: pam_adjustments['right_hip_flexor_adj'],    # Right Hip Flexor  
            3: pam_adjustments['right_hip_extensor_adj'],  # Right Hip Extensor
            4: pam_adjustments['left_knee_flexor_adj'],    # Left Knee Flexor
            5: pam_adjustments['right_knee_flexor_adj'],   # Right Knee Flexor
        }
        
        # Aplicar ajustes manteniendo límites
        for pam_idx, adjustment in adjustment_mapping.items():
            corrected_action[pam_idx] += adjustment
            corrected_action[pam_idx] = np.clip(corrected_action[pam_idx], 0.0, 1.0)
        
        return corrected_action
    
    def get_comprehensive_performance_stats(self):
        """
        Estadísticas completas del sistema consciente de fase
        """
        phase_performance = self.postural_system.get_phase_performance_summary()
        
        return {
            'total_corrections': self.total_corrections,
            'corrections_by_phase': {phase.value: count for phase, count in self.phase_specific_corrections.items()},
            'phase_performance_detail': phase_performance,
            'current_phase': self.postural_system.current_phase.value,
            'phases_needing_tuning': [phase for phase, stats in phase_performance.items() if stats['needs_tuning']]
        }