def get_integrated_expert_action(self, time_step):
    """
    🌟 PIPELINE DE CONTROL INTEGRADO
    
    Este método implementa la integración completa de reciprocidad neural
    y estabilidad postural, respetando la jerarquía temporal biológica.
    
    ORDEN CRÍTICO DE PROCESAMIENTO:
    1. Reflejos espinales (más rápidos, más primitivos)
    2. Control automático postural (intermedio)  
    3. Reciprocidad neural consciente (coordinación voluntaria)
    4. Integración final con comandos de alto nivel
    """
    
    # ===== PASO 1: OBTENER COMANDO BASE DE ALTO NIVEL =====
    # Este es el equivalente a la "intención consciente" - lo que el robot "quiere" hacer
    base_pattern_pressures = self._get_base_pattern_pressures()
    
    print(f"\n🎯 Integrated Control Pipeline (Step {self.env.step_count}):")
    print(f"   Base intention: {self.current_action.value}")
    
    # ===== PASO 2: ACTUALIZAR RETROALIMENTACIÓN SENSORIAL =====
    # El sistema sensorial debe actualizarse antes que cualquier control
    self.postural_controller.update_sensory_feedback()
    
    # ===== PASO 3: REFLEJOS ESPINALES (PRIORIDAD MÁXIMA) =====
    # Los reflejos tienen prioridad absoluta - pueden anular cualquier comando consciente
    reflex_modulated_pressures = self.postural_controller.spinal_reflex_control(base_pattern_pressures)
    
    # Verificar si los reflejos hicieron cambios significativos
    reflex_changes = sum(abs(reflex_modulated_pressures[muscle] - base_pattern_pressures[muscle]) 
                        for muscle in base_pattern_pressures)
    
    if reflex_changes > 0.1:
        print(f"   ⚡ Spinal reflexes active: {reflex_changes:.3f} total adjustment")
    
    # ===== PASO 4: CONTROL AUTOMÁTICO POSTURAL =====
    # El sistema automático ajusta las salidas reflejas para optimizar estabilidad
    automatic_adjusted_pressures = self.postural_controller.automatic_postural_control(
        reflex_modulated_pressures
    )
    
    # ===== PASO 5: RECIPROCIDAD NEURAL CONSCIENTE =====
    # Aplicar coordinación antagónica consciente que respeta las capas inferiores
    neural_coordinated_pressures = self._apply_conscious_neural_reciprocity(
        automatic_adjusted_pressures
    )
    
    # ===== PASO 6: INTEGRACIÓN FINAL CON CONTROL CONSCIENTE =====
    # El nivel consciente puede modular pero no anular completamente los niveles inferiores
    final_pressures = self.postural_controller.conscious_postural_control(
        neural_coordinated_pressures, 
        base_pattern_pressures  # Referencia de intención original
    )
    
    # ===== PASO 7: APLICAR LÍMITES BIOMECÁNICOS FINALES =====
    biomechanically_limited_pressures = self._apply_biomechanical_constraints(final_pressures)
    
    # ===== PASO 8: CONVERSIÓN A ARRAY PAM =====
    final_pam_array = self._convert_to_pam_array(biomechanically_limited_pressures)
    
    # ===== REGISTRO Y DEBUG =====
    self._log_control_pipeline_activity(
        base_pattern_pressures, 
        reflex_modulated_pressures,
        automatic_adjusted_pressures,
        neural_coordinated_pressures,
        final_pressures,
        final_pam_array
    )
    
    return final_pam_array

def _apply_conscious_neural_reciprocity(self, automatic_pressures):
    """
    🧠 RECIPROCIDAD NEURAL CONSCIENTE
    
    Este nivel implementa la coordinación antagónica voluntaria,
    pero respetando las correcciones ya aplicadas por los niveles inferiores.
    """
    
    conscious_neural_pressures = automatic_pressures.copy()
    
    # ===== PRINCIPIO: LA RECIPROCIDAD CONSCIENTE NO ANULA LOS REFLEJOS =====
    # Si los reflejos han inhibido un músculo por razones de seguridad,
    # el nivel consciente no debe reactivarlo
    
    # Verificar inhibiciones reflejas activas
    active_inhibitions = self._detect_active_reflex_inhibitions()
    
    # ===== APLICAR COORDINACIÓN ANTAGÓNICA CONSCIENTE =====
    # Solo en músculos que no están bajo inhibición refleja
    
    coordination_strength = self.coordination_noise_scale * 5.0  # Amplificar para nivel consciente
    
    # Generar comandos neurales conscientes coordinados
    left_hip_neural_command = np.random.normal(0, coordination_strength)
    right_hip_neural_command = np.random.normal(0, coordination_strength) * self.bilateral_symmetry_factor
    
    # Aplicar solo si no hay inhibiciones activas
    if 'left_hip_flexor' not in active_inhibitions:
        conscious_neural_pressures['left_hip_flexor'] += left_hip_neural_command
        
    if 'left_hip_extensor' not in active_inhibitions:
        conscious_neural_pressures['left_hip_extensor'] -= left_hip_neural_command  # Reciprocidad
    
    if 'right_hip_flexor' not in active_inhibitions:
        conscious_neural_pressures['right_hip_flexor'] += right_hip_neural_command
        
    if 'right_hip_extensor' not in active_inhibitions:
        conscious_neural_pressures['right_hip_extensor'] -= right_hip_neural_command
    
    # ===== RODILLAS: RESPETO ABSOLUTO A LA INHIBICIÓN REFLEJA =====
    # Las rodillas están bajo control principalmente reflejo/automático
    # El nivel consciente puede modular ligeramente, pero no anular
    
    try:
        joint_states = p.getJointStates(self.env.robot_id, [1, 4])
        left_knee_angle = joint_states[0][0]
        right_knee_angle = joint_states[1][0]
        
        # Tu principio biomecánico implementado a nivel consciente
        if left_knee_angle > 0.05:  # Si rodilla flexionada
            # Nivel consciente confirma la inhibición refleja
            conscious_neural_pressures['left_knee_flexor'] = 0.0
            
        if right_knee_angle > 0.05:
            conscious_neural_pressures['right_knee_flexor'] = 0.0
            
    except Exception as e:
        print(f"⚠️ Conscious neural coordination compromised: {e}")
    
    return conscious_neural_pressures

def _apply_biomechanical_constraints(self, pressures):
    """
    🔬 APLICAR LÍMITES BIOMECÁNICOS FINALES
    
    Este es el último filtro que asegura que los comandos finales
    respeten las limitaciones físicas y biomecánicas del sistema.
    """
    
    constrained_pressures = pressures.copy()
    
    # ===== LÍMITE 1: SATURACIÓN FISIOLÓGICA =====
    # Los músculos biológicos no pueden activarse más allá del 100%
    for muscle in constrained_pressures:
        constrained_pressures[muscle] = np.clip(constrained_pressures[muscle], 0.0, 1.0)
    
    # ===== LÍMITE 2: UMBRAL DE ACTIVACIÓN MÍNIMA =====
    # Músculos por debajo de cierto umbral se consideran inactivos
    activation_threshold = 0.02  # 2%
    for muscle in constrained_pressures:
        if constrained_pressures[muscle] < activation_threshold:
            constrained_pressures[muscle] = 0.0
    
    # ===== LÍMITE 3: PROTECCIÓN CONTRA CO-CONTRACCIÓN EXCESIVA =====
    # Prevenir activación simultánea extrema de músculos antagónicos
    antagonistic_pairs = [
        ('left_hip_flexor', 'left_hip_extensor'),
        ('right_hip_flexor', 'right_hip_extensor')
    ]
    
    for flexor, extensor in antagonistic_pairs:
        flexor_activation = constrained_pressures[flexor]
        extensor_activation = constrained_pressures[extensor]
        
        # Si ambos están muy activos, reducir proporcionalmente
        if flexor_activation > 0.8 and extensor_activation > 0.8:
            # Situación biomecánicamente ineficiente
            total_activation = flexor_activation + extensor_activation
            
            # Redistribuir manteniendo la proporción pero reduciendo el total
            max_total_allowed = 1.2  # Máximo 120% de activación total para el par
            
            if total_activation > max_total_allowed:
                scaling_factor = max_total_allowed / total_activation
                constrained_pressures[flexor] *= scaling_factor
                constrained_pressures[extensor] *= scaling_factor
                
                if self.env.step_count % 1000 == 0:
                    print(f"🔬 Biomechanical constraint: {flexor}/{extensor} scaled by {scaling_factor:.2f}")
    
    # ===== LÍMITE 4: PROTECCIÓN DE ARTICULACIONES =====
    # Prevenir comandos que podrían llevar articulaciones a posiciones peligrosas
    try:
        joint_states = p.getJointStates(self.env.robot_id, [0, 1, 3, 4])
        
        for i, joint_angle in enumerate([state[0] for state in joint_states]):
            # Definir límites seguros para cada articulación
            if i in [0, 2]:  # Caderas
                if joint_angle > 1.0:  # Muy flexionada (>57 grados)
                    # Reducir activación de flexores
                    if i == 0:  # Cadera izquierda
                        constrained_pressures['left_hip_flexor'] *= 0.5
                    else:  # Cadera derecha
                        constrained_pressures['right_hip_flexor'] *= 0.5
                        
            elif i in [1, 3]:  # Rodillas
                if joint_angle > 0.3:  # Muy flexionada (>17 grados)
                    # Inhibir completamente flexores (tu principio)
                    if i == 1:  # Rodilla izquierda
                        constrained_pressures['left_knee_flexor'] = 0.0
                    else:  # Rodilla derecha
                        constrained_pressures['right_knee_flexor'] = 0.0
                    
                    print(f"🔬 Joint protection: Knee flexor inhibited (angle: {joint_angle:.3f})")
    
    except Exception as e:
        print(f"⚠️ Joint protection system offline: {e}")
    
    return constrained_pressures

def _log_control_pipeline_activity(self, base, reflex, automatic, neural, final, pam_array):
    """
    📊 LOGGING DEL PIPELINE DE CONTROL
    
    Registra la actividad de cada capa del sistema para análisis y debugging.
    """
    
    if self.env.step_count % 1500 == 0:  # Log cada segundo aprox
        print(f"\n📊 Control Pipeline Analysis:")
        print(f"   Base command → Reflex → Automatic → Neural → Final")
        
        for muscle in base:
            base_val = base[muscle]
            reflex_val = reflex[muscle] 
            auto_val = automatic[muscle]
            neural_val = neural[muscle]
            final_val = final[muscle]
            
            print(f"   {muscle[:15]:<15}: {base_val:.2f} → {reflex_val:.2f} → {auto_val:.2f} → {neural_val:.2f} → {final_val:.2f}")
        
        # Mostrar cambios más significativos
        total_reflex_change = sum(abs(reflex[m] - base[m]) for m in base)
        total_automatic_change = sum(abs(automatic[m] - reflex[m]) for m in base)
        total_neural_change = sum(abs(neural[m] - automatic[m]) for m in base)
        
        print(f"   Total changes: Reflex={total_reflex_change:.2f}, Auto={total_automatic_change:.2f}, Neural={total_neural_change:.2f}")

def _detect_active_reflex_inhibitions(self):
    """
    🔍 DETECTAR INHIBICIONES REFLEJAS ACTIVAS
    
    Identifica qué músculos están bajo inhibición refleja activa
    para evitar que el nivel consciente los reactive inapropiadamente.
    """
    
    active_inhibitions = set()
    
    try:
        # Verificar inhibiciones basadas en ángulos articulares
        joint_states = p.getJointStates(self.env.robot_id, [1, 4])
        left_knee_angle = joint_states[0][0]
        right_knee_angle = joint_states[1][0]
        
        # Tu principio: rodillas flexionadas → flexores inhibidos
        if left_knee_angle > 0.05:
            active_inhibitions.add('left_knee_flexor')
            
        if right_knee_angle > 0.05:
            active_inhibitions.add('right_knee_flexor')
        
        # Verificar inhibiciones basadas en velocidades articulares peligrosas
        joint_velocities = [state[1] for state in joint_states]
        
        for i, velocity in enumerate(joint_velocities):
            if abs(velocity) > 2.0:  # Velocidad peligrosa (>114 grados/segundo)
                if i == 0:  # Rodilla izquierda
                    active_inhibitions.add('left_knee_flexor')
                else:  # Rodilla derecha  
                    active_inhibitions.add('right_knee_flexor')
        
        # Verificar inhibiciones basadas en orientación corporal crítica
        pos, orn = p.getBasePositionAndOrientation(self.env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        if abs(euler[1]) > 0.4:  # Inclinación crítica (>23 grados)
            # En situación crítica, inhibir flexores de cadera también
            active_inhibitions.add('left_hip_flexor')
            active_inhibitions.add('right_hip_flexor')
    
    except Exception as e:
        print(f"⚠️ Reflex inhibition detection failed: {e}")
    
    return active_inhibitions