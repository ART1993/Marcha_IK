import numpy as np
import math
import pybullet as p
from Archivos_Apoyo.SimpleWalkingCycle import SimpleWalkingCycle

# ============================================================================
# Modificaciones adicionales para el controlador de paso
# ============================================================================

class Enhanced_SimplifiedWalkingController:
    """
        Controlador de paso avanzado para 6 PAMs antagónicos.
        
        Este controlador implementa patrones biomecánicos realistas donde:
        - Caderas tienen control antagónico completo (flexor + extensor)
        - Rodillas tienen flexores activos + extensores pasivos (resortes)
        - Los patrones siguen principios de marcha humana
    """
    
    def __init__(self, env, mode="pressure", blend_factor=0.0):
        """
            Inicializa el controlador mejorado.
            
            Args:
                env: Entorno Enhanced_PAMIKBipedEnv
                mode: Modo de control ("pressure", "trajectory", "blend")
                blend_factor: Factor de mezcla para modo blend
        """
        self.env = env
        self.mode = mode
        self.blend_factor = blend_factor
        
        # Heredar funcionalidad básica del controlador simple
        self.walking_cycle = SimpleWalkingCycle(robot_id=env.robot_id)
        
        # Estado del controlador
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0
        self.last_valid_action = np.zeros(6, dtype=np.float32)  # 6 PAMs
        self.last_phase = 0.0
        
        # Configurar patrones específicos para 6 PAMs
        self.setup_antagonistic_patterns()
        self.setup_gait_biomechanics()

        print(f"🦴 Enhanced Walking Controller initialized with {len(self.pam_mapping)} PAMs")
        print(f"   Mode: {mode} | Blend factor: {blend_factor}")

    def setup_gait_biomechanics(self):
        """
        Configura parámetros biomecánicos de la marcha humana.
        
        Estos parámetros están basados en estudios de biomecánica que muestran
        cuándo y cómo se activan los diferentes grupos musculares durante la marcha.
        """
        
        # Fases de la marcha (como porcentaje del ciclo completo)
        self.gait_phases = {
            'stance_phase': (0.0, 0.6),      # 60% del ciclo en stance
            'swing_phase': (0.6, 1.0),       # 40% del ciclo en swing
            'double_support': (0.0, 0.1),    # 10% soporte doble inicial
            'single_support': (0.1, 0.5),    # 40% soporte simple
            'pre_swing': (0.5, 0.6),         # 10% preparación swing
        }
        
        # Patrones de activación muscular específicos por fase
        self.muscle_activation_patterns = {
            'hip_flexor_peak': 0.7,      # Máxima activación en 70% del ciclo
            'hip_extensor_peak': 0.1,    # Máxima activación en 10% del ciclo
            'knee_flexor_peak': 0.75,    # Máxima activación en 75% del ciclo
            'co_activation_factor': 0.3, # Factor de co-activación antagónica
        }
        
        # Parámetros de suavizado para transiciones realistas
        self.smoothing_params = {
            'transition_width': 0.1,     # Ancho de transición entre fases
            'activation_rise_time': 0.05, # Tiempo de subida de activación
            'activation_fall_time': 0.08, # Tiempo de bajada de activación
        }
        
        print(f"   ✅ Biomechanical gait patterns configured")
    
    def setup_antagonistic_patterns(self):
        """
        Configura los patrones de activación para pares antagónicos.
        
        Esta función establece las bases fisiológicas del control muscular:
        - Activación base (tono muscular en reposo)
        - Amplitudes de modulación para cada músculo
        - Mapeo claro de músculos a índices de acción
        """
        
        # Mapeo de músculos PAM a índices de acción [0-5]
        self.pam_mapping = {
            'left_hip_flexor': 0,     # Psoas, rectus femoris (anterior)
            'left_hip_extensor': 1,   # Glúteos, hamstrings (posterior)
            'right_hip_flexor': 2,    # Psoas, rectus femoris (anterior)
            'right_hip_extensor': 3,  # Glúteos, hamstrings (posterior)
            'left_knee_flexor': 4,    # Hamstrings, gastrocnemius
            'right_knee_flexor': 5,   # Hamstrings, gastrocnemius
        }
        
        # Presiones base: tono muscular mínimo para estabilidad postural
        # Valores más altos = músculos más activos en reposo
        self.base_pressures_6pam = {
            'left_hip_flexor': 0.2,    # Flexor menos activo en bipedestación
            'left_hip_extensor': 0.4,  # Extensor más activo (anti-gravedad)
            'right_hip_flexor': 0.2,   
            'right_hip_extensor': 0.4, 
            'left_knee_flexor': 0.15,  # Flexores de rodilla mínimos
            'right_knee_flexor': 0.15,
        }
        
        # Amplitudes de modulación: cuánto puede variar cada músculo
        # Valores más altos = mayor rango dinámico de activación
        self.modulation_amplitudes_6pam = {
            'left_hip_flexor': 0.5,    # Alta modulación para swing
            'left_hip_extensor': 0.3,  # Modulación moderada (estabilidad)
            'right_hip_flexor': 0.5,   
            'right_hip_extensor': 0.3,
            'left_knee_flexor': 0.7,   # Alta modulación para clearance
            'right_knee_flexor': 0.7,
        }
        
        print(f"   ✅ Antagonistic patterns configured for {len(self.pam_mapping)} muscles")
    
    def get_enhanced_walking_actions(self, time_step):
        """
            Genera acciones expertas para 6 PAMs siguiendo patrones biomecánicos.
            
            Esta es la función principal que combina todos los patrones y genera
            las presiones PAM que imitan la activación muscular humana durante la marcha.
            
            Args:
                time_step: Paso de tiempo de la simulación
                
            Returns:
                np.array: Array de 6 presiones PAM normalizadas [-1, 1]
        """
        # Actualizar fase del ciclo de marcha
        self.walking_cycle.update_phase(time_step)
        alpha = self.walking_cycle.phase  # Fase actual [0, 1]
        
        # Determinar qué pierna está en swing
        left_leg_phase = alpha * 2 * np.pi
        right_leg_phase = (alpha + 0.5) * 2 * np.pi  # Desfase de 180°
        
        # Calcular activaciones para cada músculo
        activations = self._calculate_muscle_activations(alpha, left_leg_phase, right_leg_phase)
        
        # Aplicar co-activación antagónica para estabilidad
        activations = self._apply_coactivation(activations, alpha)
        
        # Suavizar transiciones entre fases
        activations = self._smooth_transitions(activations, alpha)
        
        # Convertir activaciones a presiones PAM
        pam_pressures = self._activations_to_pressures(activations)
        
        # Normalizar a rango [-1, 1] para el entorno
        normalized_actions = self._normalize_to_action_space(pam_pressures)
        
        # Aplicar factor de seguridad y límites
        safe_actions = self._apply_safety_limits(normalized_actions)
        
        # Actualizar estado interno
        self.last_valid_action = safe_actions
        self.last_phase = alpha
        
        return safe_actions
    
    def _calculate_muscle_activations(self, alpha, left_leg_phase, right_leg_phase):
        """
        Calcula las activaciones musculares basadas en la biomecánica de marcha.
        
        Esta función implementa los patrones específicos de cuándo cada músculo
        debe activarse durante el ciclo de marcha, basado en datos experimentales.
        """
        activations = {}
        
        # === MÚSCULOS DE CADERA IZQUIERDA ===
        
        # Hip flexor izquierdo: activo durante swing y preparación
        # Patrón: pico en 70% del ciclo (mid-swing), mínimo en stance
        if 0.6 <= alpha <= 1.0:  # Swing phase
            swing_progress = (alpha - 0.6) / 0.4
            flexor_activation = 0.8 * np.sin(np.pi * swing_progress) ** 2
        else:  # Stance phase
            flexor_activation = 0.1 * (1 + 0.3 * np.sin(left_leg_phase))
        
        activations['left_hip_flexor'] = flexor_activation
        
        # Hip extensor izquierdo: activo durante stance (empuje)
        # Patrón: pico en 10-30% del ciclo (heel strike a mid-stance)
        if 0.0 <= alpha <= 0.6:  # Stance phase
            stance_progress = alpha / 0.6
            # Doble pico: heel strike y push-off
            extensor_activation = 0.7 * (
                np.exp(-((stance_progress - 0.2) ** 2) / 0.02) +  # Primer pico
                0.6 * np.exp(-((stance_progress - 0.8) ** 2) / 0.03)  # Segundo pico
            )
        else:  # Swing phase
            extensor_activation = 0.2 * (1 + 0.2 * np.sin(left_leg_phase))
        
        activations['left_hip_extensor'] = np.clip(extensor_activation, 0.15, 0.9)
        
        # === MÚSCULOS DE CADERA DERECHA (desfasados 180°) ===
        
        # Calcular activaciones de cadera derecha con desfase
        right_alpha = (alpha + 0.5) % 1.0  # Desfase de 180°
        
        # Hip flexor derecho
        if 0.6 <= right_alpha <= 1.0:
            swing_progress = (right_alpha - 0.6) / 0.4
            right_flexor_activation = 0.8 * np.sin(np.pi * swing_progress) ** 2
        else:
            right_flexor_activation = 0.1 * (1 + 0.3 * np.sin(right_leg_phase))
        
        activations['right_hip_flexor'] = right_flexor_activation
        
        # Hip extensor derecho
        if 0.0 <= right_alpha <= 0.6:
            stance_progress = right_alpha / 0.6
            right_extensor_activation = 0.7 * (
                np.exp(-((stance_progress - 0.2) ** 2) / 0.02) +
                0.6 * np.exp(-((stance_progress - 0.8) ** 2) / 0.03)
            )
        else:
            right_extensor_activation = 0.2 * (1 + 0.2 * np.sin(right_leg_phase))
        
        activations['right_hip_extensor'] = np.clip(right_extensor_activation, 0.15, 0.9)
        
        # === FLEXORES DE RODILLA ===
        
        # Los flexores de rodilla se activan principalmente para toe clearance
        # durante swing y para control de descenso durante stance tardío
        
        # Knee flexor izquierdo
        if 0.65 <= alpha <= 0.85:  # Peak clearance durante swing
            clearance_progress = (alpha - 0.65) / 0.2
            left_knee_activation = 0.9 * np.sin(np.pi * clearance_progress) ** 1.5
        elif 0.4 <= alpha <= 0.6:  # Pre-swing preparation
            prep_progress = (alpha - 0.4) / 0.2
            left_knee_activation = 0.4 * prep_progress
        else:
            left_knee_activation = 0.1 * (1 + 0.2 * np.sin(left_leg_phase))
        
        activations['left_knee_flexor'] = left_knee_activation
        
        # Knee flexor derecho (desfasado)
        right_knee_alpha = (alpha + 0.5) % 1.0
        if 0.65 <= right_knee_alpha <= 0.85:
            clearance_progress = (right_knee_alpha - 0.65) / 0.2
            right_knee_activation = 0.9 * np.sin(np.pi * clearance_progress) ** 1.5
        elif 0.4 <= right_knee_alpha <= 0.6:
            prep_progress = (right_knee_alpha - 0.4) / 0.2
            right_knee_activation = 0.4 * prep_progress
        else:
            right_knee_activation = 0.1 * (1 + 0.2 * np.sin(right_leg_phase))
        
        activations['right_knee_flexor'] = right_knee_activation
        
        return activations
    
    def _apply_coactivation(self, activations, alpha):
        """
        Aplica co-activación antagónica para estabilidad articular.
        
        En biomecánica, los músculos antagónicos nunca se relajan completamente
        cuando su oponente se activa. Siempre mantienen cierto nivel de activación
        para proporcionar estabilidad articular y control fino del movimiento.
        """
        coactivation_factor = self.muscle_activation_patterns['co_activation_factor']
        
        # Co-activación en caderas: cuando un músculo se activa, su antagónico
        # mantiene un mínimo de activación proporcional
        
        # Cadera izquierda
        flexor_activation = activations['left_hip_flexor']
        extensor_activation = activations['left_hip_extensor']
        
        activations['left_hip_flexor'] += coactivation_factor * extensor_activation
        activations['left_hip_extensor'] += coactivation_factor * flexor_activation
        
        # Cadera derecha
        right_flexor_activation = activations['right_hip_flexor']
        right_extensor_activation = activations['right_hip_extensor']
        
        activations['right_hip_flexor'] += coactivation_factor * right_extensor_activation
        activations['right_hip_extensor'] += coactivation_factor * right_flexor_activation
        
        # Las rodillas mantienen activación mínima para estabilidad
        # (los extensores pasivos ya proporcionan parte de esta función)
        min_knee_activation = 0.15
        activations['left_knee_flexor'] = max(activations['left_knee_flexor'], min_knee_activation)
        activations['right_knee_flexor'] = max(activations['right_knee_flexor'], min_knee_activation)
        
        return activations
    
    def _smooth_transitions(self, activations, alpha):
        """
        Suaviza las transiciones entre fases para evitar cambios abruptos.
        
        Los músculos reales no pueden cambiar instantáneamente su activación.
        Esta función simula la inercia neuromuscular aplicando filtrado temporal.
        """
        transition_width = self.smoothing_params['transition_width']
        
        # Identificar zonas de transición críticas
        critical_transitions = [0.0, 0.5, 1.0]  # Heel strikes
        
        smoothing_factor = 1.0
        for transition in critical_transitions:
            distance_to_transition = min(
                abs(alpha - transition),
                abs(alpha - (transition + 1.0) % 1.0),
                abs(alpha - (transition - 1.0) % 1.0)
            )
            
            if distance_to_transition < transition_width:
                # Aplicar suavizado Gaussiano cerca de las transiciones
                smoothing_factor *= 0.8 + 0.2 * np.exp(
                    -(distance_to_transition ** 2) / (2 * (transition_width / 3) ** 2)
                )
        
        # Aplicar suavizado con memoria del estado anterior
        if hasattr(self, '_prev_activations'):
            smoothing_weight = 0.85  # Factor de memoria temporal
            for muscle in activations:
                if muscle in self._prev_activations:
                    activations[muscle] = (
                        smoothing_weight * self._prev_activations[muscle] +
                        (1 - smoothing_weight) * activations[muscle]
                    )
        
        # Guardar para siguiente iteración
        self._prev_activations = activations.copy()
        
        return activations
    
    def _activations_to_pressures(self, activations):
        """
        Convierte activaciones musculares [0,1] a presiones PAM reales.
        
        Esta conversión tiene en cuenta las presiones base y las amplitudes
        de modulación específicas de cada músculo.
        """
        pressures = {}
        
        for muscle, activation in activations.items():
            base_pressure = self.base_pressures_6pam[muscle]
            modulation_amplitude = self.modulation_amplitudes_6pam[muscle]
            
            # Presión = presión_base + activación * amplitud_modulación
            pressure = base_pressure + activation * modulation_amplitude
            
            # Asegurar que la presión esté en rango válido [0, 1]
            pressures[muscle] = np.clip(pressure, 0.0, 1.0)
        
        return pressures
    
    def _normalize_to_action_space(self, pressures):
        """
        Normaliza presiones [0,1] al espacio de acción [-1,1] del entorno.
        """
        normalized_actions = np.zeros(6, dtype=np.float32)
        
        for muscle, pressure in pressures.items():
            index = self.pam_mapping[muscle]
            # Convertir de [0,1] a [-1,1]
            normalized_actions[index] = 2.0 * pressure - 1.0
        
        return normalized_actions
    
    def _apply_safety_limits(self, actions):
        """
        Aplica límites de seguridad para evitar comandos extremos.
        """
        # Limitar a rango válido
        safe_actions = np.clip(actions, -1.0, 1.0)
        
        # Aplicar suavizado adicional si hay cambios muy abruptos
        if hasattr(self, 'last_valid_action'):
            max_change = 0.3  # Máximo cambio permitido por paso
            action_diff = safe_actions - self.last_valid_action
            
            # Limitar cambios abruptos
            limited_diff = np.clip(action_diff, -max_change, max_change)
            safe_actions = self.last_valid_action + limited_diff
            
            # Asegurar que sigamos en rango después del limitado
            safe_actions = np.clip(safe_actions, -1.0, 1.0)
        
        return safe_actions
    
    def get_expert_action(self):
        """Método de compatibilidad para currículo experto"""
        return self.get_enhanced_walking_actions(self.env.time_step)
    
    def get_expert_action_pressures(self):
        """Devuelve las presiones expertas para imitación"""
        return self.get_enhanced_walking_actions(self.env.time_step)
    
    def get_next_action(self):
        """
        Obtiene la siguiente acción según el modo de control configurado.
        """
        if not self.is_initialized:
            return self._get_initialization_action()
        
        if self.mode == "pressure":
            # Usar patrones de presión PAM directos
            return self.get_enhanced_walking_actions(self.env.time_step)
        elif self.mode == "trajectory":
            # Usar trayectorias IK (fallback al método original)
            return self.walking_cycle.get_trajectory_walking_actions(
                self.env.time_step, self.env.left_foot_id, self.env.right_foot_id
            )
        elif self.mode == "blend":
            # Combinar ambos enfoques
            pressure_action = self.get_enhanced_walking_actions(self.env.time_step)
            traj_action = self.walking_cycle.get_trajectory_walking_actions(
                self.env.time_step, self.env.left_foot_id, self.env.right_foot_id
            )
            
            # Mezclar acciones (requiere mapeo cuidadoso de dimensiones)
            if len(pressure_action) == len(traj_action):
                blended = ((1 - self.blend_factor) * pressure_action + 
                          self.blend_factor * traj_action)
                return blended
            else:
                # Si dimensiones no coinciden, usar solo presiones
                return pressure_action
        else:
            raise ValueError(f"Modo no válido: {self.mode}")
        
    def _get_initialization_action(self):
        """Secuencia de inicialización para estabilizar el robot"""
        if self.init_sequence is None:
            self.init_sequence = self._generate_initialization_sequence()
        
        if self.init_step < len(self.init_sequence):
            action = self.init_sequence[self.init_step]
            self.init_step += 1
            return action
        else:
            self.is_initialized = True
            return self.get_next_action()
        
    def _generate_initialization_sequence(self, duration=2.0, dt=0.01):
        """
        Genera secuencia de inicialización gradual para 6 PAMs.
        """
        steps = int(duration / dt)
        sequence = []
        
        for i in range(steps):
            # Ramp-up gradual hacia postura de equilibrio
            ramp_factor = min(1.0, i / (steps * 0.3))
            
            # Activación simétrica inicial con ligero sesgo hacia extensores
            base_activations = {
                'left_hip_flexor': 0.15 * ramp_factor,
                'left_hip_extensor': 0.35 * ramp_factor,
                'right_hip_flexor': 0.15 * ramp_factor,
                'right_hip_extensor': 0.35 * ramp_factor,
                'left_knee_flexor': 0.1 * ramp_factor,
                'right_knee_flexor': 0.1 * ramp_factor,
            }
            
            # Convertir a acciones normalizadas
            normalized_actions = np.zeros(6, dtype=np.float32)
            for muscle, activation in base_activations.items():
                index = self.pam_mapping[muscle]
                normalized_actions[index] = 2.0 * activation - 1.0
            
            sequence.append(normalized_actions)
        
        return sequence
    
    def reset(self):
        """Reinicia el controlador"""
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0
        self.walking_cycle.phase = 0.0
        self.last_valid_action = np.zeros(6, dtype=np.float32)
        self.last_phase = 0.0
        
        # Limpiar estado de suavizado
        if hasattr(self, '_prev_activations'):
            del self._prev_activations
        
        print("🔄 Enhanced Walking Controller reset")

    def set_parameters(self, **kwargs):
        """
        Permite ajustar parámetros del controlador dinámicamente.
        
        Args:
            **kwargs: Parámetros a modificar (step_frequency, step_length, etc.)
        """
        if 'step_frequency' in kwargs:
            self.walking_cycle.step_frequency = kwargs['step_frequency']
            print(f"   Updated step frequency: {kwargs['step_frequency']} Hz")
        
        if 'step_length' in kwargs:
            self.walking_cycle.step_length = kwargs['step_length']
            print(f"   Updated step length: {kwargs['step_length']} m")
        
        if 'blend_factor' in kwargs:
            self.blend_factor = kwargs['blend_factor']
            print(f"   Updated blend factor: {kwargs['blend_factor']}")
        
        if 'mode' in kwargs:
            self.mode = kwargs['mode']
            print(f"   Updated control mode: {kwargs['mode']}")

    def get_debug_info(self):
        """
        Retorna información de debug útil para análisis y ajuste.
        """
        if not hasattr(self, '_prev_activations'):
            return {}
        
        return {
            'current_phase': self.walking_cycle.phase,
            'muscle_activations': self._prev_activations.copy(),
            'last_action': self.last_valid_action.copy(),
            'mode': self.mode,
            'is_initialized': self.is_initialized,
            'gait_phase_info': self._get_current_gait_phase()
        }
    
    def _get_current_gait_phase(self):
        """Identifica la fase actual de marcha para debugging"""
        alpha = self.walking_cycle.phase
        
        if 0.0 <= alpha < 0.1:
            return "double_support_initial"
        elif 0.1 <= alpha < 0.5:
            return "single_support_left"
        elif 0.5 <= alpha < 0.6:
            return "pre_swing_left"
        elif 0.6 <= alpha < 1.0:
            return "swing_left"
        else:
            return "unknown"