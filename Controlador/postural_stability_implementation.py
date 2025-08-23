class PosturalStabilityController:
    """
    🏃‍♂️ SISTEMA DE ESTABILIDAD POSTURAL MULTICAPA
    
    Implementa los tres niveles de control postural biológico:
    1. Reflejos espinales (respuesta rápida < 50ms simulado)
    2. Control automático (respuesta media 100-200ms)  
    3. Control consciente (respuesta lenta > 200ms)
    """
    
    def __init__(self, env):
        self.env = env
        
        # ===== CONFIGURACIÓN DE RETROALIMENTACIÓN SENSORIAL =====
        # Simular los diferentes tipos de receptores biológicos
        self.proprioceptive_history = deque(maxlen=10)  # Historial de posiciones articulares
        self.vestibular_history = deque(maxlen=5)       # Historial de orientación corporal
        self.visual_stability_reference = None          # Referencia visual de verticalidad
        
        # ===== PARÁMETROS DE CONTROL POSTURAL =====
        self.postural_gains = {
            'spinal_reflex_gain': 0.8,      # Qué tan fuerte son los reflejos automáticos
            'automatic_gain': 0.6,          # Qué tan agresivo es el control automático
            'conscious_gain': 0.4,          # Qué tan dominante es el control consciente
            'adaptation_rate': 0.02         # Qué tan rápido se adapta el sistema
        }
        
        # ===== UMBRALES BIOLÓGICOS =====
        self.stability_thresholds = {
            'ankle_strategy_threshold': 0.05,    # Cuándo usar estrategia de tobillo
            'hip_strategy_threshold': 0.15,      # Cuándo cambiar a estrategia de cadera  
            'step_strategy_threshold': 0.30,     # Cuándo necesitar dar un paso
            'emergency_threshold': 0.50          # Cuándo activar respuestas de emergencia
        }
        
        print(f"🧠 Postural Stability Controller initialized")
        print(f"   Multi-layer control: Spinal → Automatic → Conscious")
    
    def update_sensory_feedback(self):
        """
        🔍 ACTUALIZACIÓN DE RETROALIMENTACIÓN SENSORIAL
        
        Simula los sistemas sensoriales que tu cuerpo usa para mantener equilibrio:
        - Propiocepción: posición de articulaciones y músculos
        - Sistema vestibular: orientación del cuerpo en el espacio  
        - Sistema visual: referencias de verticalidad del entorno
        """
        
        try:
            # ===== RETROALIMENTACIÓN PROPIOCEPTIVA =====
            # Leer posiciones actuales de todas las articulaciones críticas
            joint_states = p.getJointStates(self.env.robot_id, [0, 1, 3, 4])  # caderas y rodillas
            joint_positions = [state[0] for state in joint_states]
            joint_velocities = [state[1] for state in joint_states]
            
            proprioceptive_data = {
                'joint_angles': joint_positions,
                'joint_velocities': joint_velocities,
                'timestamp': self.env.step_count * self.env.time_step
            }
            
            self.proprioceptive_history.append(proprioceptive_data)
            
            # ===== RETROALIMENTACIÓN VESTIBULAR =====
            # Leer orientación y velocidad angular del torso
            pos, orn = p.getBasePositionAndOrientation(self.env.robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(self.env.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            vestibular_data = {
                'body_orientation': euler,      # Roll, pitch, yaw
                'angular_velocity': ang_vel,    # Velocidades angulares
                'linear_acceleration': lin_vel, # Aproximación de aceleración
                'height': pos[2]               # Altura del centro de masa
            }
            
            self.vestibular_history.append(vestibular_data)
            
            # ===== RETROALIMENTACIÓN VISUAL (SIMPLIFICADA) =====
            # En un robot real esto sería input de cámara, aquí usamos la referencia de gravedad
            self.visual_stability_reference = {
                'vertical_reference': [0, 0, 1],  # Vector de gravedad
                'horizon_level': 0.0               # Nivel del horizonte
            }
            
        except Exception as e:
            print(f"⚠️ Sensory feedback system malfunction: {e}")
    
    def spinal_reflex_control(self, base_pressures):
        """
        ⚡ NIVEL 1: CONTROL REFLEJO ESPINAL
        
        Los reflejos espinales son las respuestas más rápidas y automáticas.
        Operan sin involucrar el cerebro, directamente a nivel de médula espinal.
        """
        
        reflex_adjustments = base_pressures.copy()
        
        if len(self.proprioceptive_history) < 2:
            return reflex_adjustments  # Necesitamos historial para detectar cambios
        
        current_state = self.proprioceptive_history[-1]
        previous_state = self.proprioceptive_history[-2]
        
        # ===== REFLEJO DE ESTIRAMIENTO MUSCULAR =====
        # Cuando una articulación se mueve más allá de un umbral seguro,
        # los músculos automáticamente se contraen para protegerse
        
        for i, (current_angle, prev_angle) in enumerate(zip(
            current_state['joint_angles'], 
            previous_state['joint_angles']
        )):
            angle_change = abs(current_angle - prev_angle)
            joint_velocity = current_state['joint_velocities'][i]
            
            if angle_change > 0.02:  # Cambio súbito > ~1.1 grados
                # Activar reflejo de protección
                stretch_reflex_strength = min(angle_change * 20, 0.3)  # Máximo 0.3
                
                if i == 0:  # Cadera izquierda
                    if joint_velocity > 0:  # Flexionando rápido
                        reflex_adjustments['left_hip_extensor'] += stretch_reflex_strength
                        print(f"⚡ Stretch reflex: Left hip extensor +{stretch_reflex_strength:.3f}")
                    else:  # Extendiendo rápido
                        reflex_adjustments['left_hip_flexor'] += stretch_reflex_strength
                
                elif i == 1:  # Rodilla izquierda
                    if joint_velocity > 0 and current_angle < 0.05:  # Flexionando pero debería estar recta
                        # Este es el reflejo que implementa tu observación biomecánica
                        reflex_adjustments['left_knee_flexor'] = 0.0  # Inhibición inmediata
                        print(f"⚡ Protective reflex: Left knee flexor INHIBITED")
                
                # Aplicar lógica similar para articulaciones derechas...
        
        # ===== REFLEJO DE ENDEREZAMIENTO =====
        # Si el cuerpo se inclina, activar automáticamente músculos anti-gravedad
        if len(self.vestibular_history) >= 2:
            current_orientation = self.vestibular_history[-1]['body_orientation']
            forward_tilt = current_orientation[1]  # Pitch
            
            if abs(forward_tilt) > 0.08:  # ~4.6 grados
                righting_reflex_strength = min(abs(forward_tilt) * 2, 0.4)
                
                if forward_tilt > 0:  # Inclinado hacia adelante
                    reflex_adjustments['left_hip_extensor'] += righting_reflex_strength
                    reflex_adjustments['right_hip_extensor'] += righting_reflex_strength
                    print(f"⚡ Righting reflex: Hip extensors +{righting_reflex_strength:.3f}")
        
        return reflex_adjustments
    
    def automatic_postural_control(self, reflex_adjusted_pressures):
        """
        🤖 NIVEL 2: CONTROL AUTOMÁTICO POSTURAL
        
        El sistema automático usa estrategias posturales complejas basadas en
        la magnitud y dirección de la perturbación. Implementa las tres estrategias
        principales de control postural humano.
        """
        
        automatic_adjustments = reflex_adjusted_pressures.copy()
        
        if len(self.vestibular_history) < 3:
            return automatic_adjustments
        
        # Calcular el "estado de perturbación" actual
        current_vest = self.vestibular_history[-1]
        perturbation_magnitude = self._calculate_perturbation_magnitude()
        
        # ===== ESTRATEGIA DE TOBILLO (PERTURBACIONES PEQUEÑAS) =====
        if perturbation_magnitude < self.stability_thresholds['ankle_strategy_threshold']:
            # Para perturbaciones pequeñas, usar principalmente los tobillos
            # En nuestro robot, esto significa confiar en los resortes pasivos
            ankle_strategy_gain = 0.8
            
            # Reducir activación de músculos grandes para permitir que los tobillos trabajen
            for muscle in ['left_hip_flexor', 'left_hip_extensor', 'right_hip_flexor', 'right_hip_extensor']:
                automatic_adjustments[muscle] *= ankle_strategy_gain
            
            if self.env.step_count % 1000 == 0:
                print(f"🦶 Ankle strategy active (perturbation: {perturbation_magnitude:.3f})")
        
        # ===== ESTRATEGIA DE CADERA (PERTURBACIONES MEDIANAS) =====
        elif perturbation_magnitude < self.stability_thresholds['hip_strategy_threshold']:
            # Para perturbaciones medianas, usar principalmente las caderas
            forward_tilt = current_vest['body_orientation'][1]
            
            hip_strategy_strength = perturbation_magnitude * 3.0  # Amplificar respuesta
            
            if forward_tilt > 0:  # Inclinado hacia adelante
                automatic_adjustments['left_hip_extensor'] += hip_strategy_strength
                automatic_adjustments['right_hip_extensor'] += hip_strategy_strength
            else:  # Inclinado hacia atrás
                automatic_adjustments['left_hip_flexor'] += hip_strategy_strength
                automatic_adjustments['right_hip_flexor'] += hip_strategy_strength
            
            if self.env.step_count % 1000 == 0:
                print(f"🏃 Hip strategy active (strength: {hip_strategy_strength:.3f})")
        
        # ===== ESTRATEGIA DE PASO (PERTURBACIONES GRANDES) =====
        elif perturbation_magnitude < self.stability_thresholds['step_strategy_threshold']:
            # Para perturbaciones grandes, preparar para dar un paso
            # En nuestro robot estático, esto significa máxima activación defensiva
            step_preparation_strength = 0.6
            
            # Activar fuertemente todos los músculos posturales
            for muscle in automatic_adjustments:
                if 'extensor' in muscle:  # Priorizar músculos anti-gravedad
                    automatic_adjustments[muscle] += step_preparation_strength
            
            if self.env.step_count % 500 == 0:
                print(f"🚨 Step strategy preparation (perturbation: {perturbation_magnitude:.3f})")
        
        # ===== RESPUESTA DE EMERGENCIA (PERTURBACIONES CRÍTICAS) =====
        else:
            # Perturbación demasiado grande - activar respuesta de emergencia
            print(f"🚨 EMERGENCY: Critical instability detected ({perturbation_magnitude:.3f})")
            
            # Activación máxima de todos los músculos anti-gravedad
            automatic_adjustments['left_hip_extensor'] = 1.0
            automatic_adjustments['right_hip_extensor'] = 1.0
            automatic_adjustments['left_hip_flexor'] = 0.1  # Mínimo
            automatic_adjustments['right_hip_flexor'] = 0.1
        
        return automatic_adjustments
    
    def _calculate_perturbation_magnitude(self):
        """
        Calcular la magnitud total de perturbación basada en múltiples señales sensoriales.
        Esta función integra información vestibular, propioceptiva y visual.
        """
        
        perturbation = 0.0
        
        # Contribución vestibular (orientación corporal)
        if len(self.vestibular_history) >= 2:
            current_orient = self.vestibular_history[-1]['body_orientation']
            vestibular_perturbation = abs(current_orient[0]) + abs(current_orient[1])  # Roll + Pitch
            perturbation += vestibular_perturbation
        
        # Contribución propioceptiva (cambios articulares súbitos)
        if len(self.proprioceptive_history) >= 2:
            current_joints = self.proprioceptive_history[-1]['joint_angles']
            prev_joints = self.proprioceptive_history[-2]['joint_angles']
            
            joint_perturbation = sum(abs(curr - prev) for curr, prev in zip(current_joints, prev_joints))
            perturbation += joint_perturbation * 2.0  # Amplificar contribución articular
        
        return perturbation
    
    def conscious_postural_control(self, automatic_adjusted_pressures, desired_action):
        """
        🧠 NIVEL 3: CONTROL CONSCIENTE POSTURAL
        
        El nivel consciente integra las intenciones de movimiento de alto nivel
        con las necesidades de estabilidad postural. Este nivel decide cuándo
        es seguro realizar movimientos voluntarios vs cuándo priorizar estabilidad.
        """
        
        conscious_adjustments = automatic_adjusted_pressures.copy()
        
        # Evaluar si es seguro ejecutar el movimiento deseado
        current_stability = self._assess_current_stability()
        
        if current_stability > 0.7:  # Sistema estable (70%+ de estabilidad)
            # Es seguro ejecutar movimientos voluntarios
            conscious_blend_factor = 0.8  # 80% del movimiento deseado
            stability_blend_factor = 0.2  # 20% de corrección postural
            
            # Combinar el movimiento deseado con las correcciones posturales
            for muscle in desired_action:
                if muscle in conscious_adjustments:
                    desired_activation = desired_action[muscle]
                    postural_activation = conscious_adjustments[muscle]
                    
                    conscious_adjustments[muscle] = (
                        conscious_blend_factor * desired_activation + 
                        stability_blend_factor * postural_activation
                    )
            
            if self.env.step_count % 1500 == 0:
                print(f"🧠 Conscious control: Executing desired movement (stability: {current_stability:.2f})")
        
        else:  # Sistema inestable
            # Priorizar estabilidad sobre movimientos voluntarios
            conscious_blend_factor = 0.2   # Solo 20% del movimiento deseado
            stability_blend_factor = 0.8   # 80% de corrección postural
            
            for muscle in desired_action:
                if muscle in conscious_adjustments:
                    desired_activation = desired_action[muscle]
                    postural_activation = conscious_adjustments[muscle]
                    
                    conscious_adjustments[muscle] = (
                        conscious_blend_factor * desired_activation + 
                        stability_blend_factor * postural_activation
                    )
            
            if self.env.step_count % 750 == 0:
                print(f"🧠 Conscious control: Prioritizing stability (stability: {current_stability:.2f})")
        
        return conscious_adjustments
    
    def _assess_current_stability(self):
        """
        Evaluar la estabilidad postural actual como un porcentaje.
        Integra múltiples indicadores de estabilidad.
        """
        
        stability_score = 1.0  # Comenzar con estabilidad perfecta
        
        # Factor 1: Orientación corporal
        if len(self.vestibular_history) > 0:
            current_orient = self.vestibular_history[-1]['body_orientation']
            orientation_error = abs(current_orient[0]) + abs(current_orient[1])
            stability_score -= orientation_error * 2.0  # Penalizar inclinaciones
        
        # Factor 2: Velocidades articulares
        if len(self.proprioceptive_history) > 0:
            current_velocities = self.proprioceptive_history[-1]['joint_velocities']
            velocity_magnitude = sum(abs(vel) for vel in current_velocities)
            stability_score -= velocity_magnitude * 0.5  # Penalizar movimientos rápidos
        
        # Factor 3: Altura del centro de masa
        if len(self.vestibular_history) > 0:
            current_height = self.vestibular_history[-1]['height']
            target_height = 1.1  # Altura objetivo
            height_error = abs(current_height - target_height)
            stability_score -= height_error * 1.5  # Penalizar desviaciones de altura
        
        return max(0.0, min(1.0, stability_score))  # Mantener entre 0 y 1