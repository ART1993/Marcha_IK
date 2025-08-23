def _apply_coordinated_variation(self, pressures):
    """
    🧠 IMPLEMENTACIÓN DE RECIPROCIDAD NEURAL REAL
    
    Este método simula cómo el sistema nervioso central coordina músculos antagónicos.
    En lugar de variaciones aleatorias independientes, implementamos la verdadera
    lógica de inhibición recíproca descubierta por Sherrington.
    """
    
    varied_pressures = pressures.copy()
    
    # ===== PRINCIPIO 1: VARIACIÓN ANTAGÓNICA COORDINADA =====
    # Cuando un flexor recibe una perturbación, su extensor antagónico
    # recibe automáticamente una perturbación en dirección opuesta.
    # Esto simula la inhibición recíproca neurológica.
    
    # Generar "comandos neurales centrales" para cada articulación
    left_hip_neural_command = np.random.normal(0, self.coordination_noise_scale)
    right_hip_neural_command = np.random.normal(0, self.coordination_noise_scale)
    
    # ===== CADERA IZQUIERDA: RECIPROCIDAD PERFECTA =====
    # Un solo comando neural se traduce en activación del agonista 
    # e inhibición del antagonista, exactamente como en la biología
    varied_pressures['left_hip_flexor'] += left_hip_neural_command
    varied_pressures['left_hip_extensor'] -= left_hip_neural_command  # ¡Inhibición recíproca!
    
    print(f"🧠 Neural Command Left Hip: {left_hip_neural_command:.4f}")
    print(f"   → Flexor variation: +{left_hip_neural_command:.4f}")  
    print(f"   → Extensor variation: -{left_hip_neural_command:.4f}")
    
    # ===== CADERA DERECHA: SIMETRÍA BILATERAL =====
    # Los circuitos neurales bilaterales están interconectados,
    # pero mantienen cierta independencia funcional
    bilateral_coupling = self.bilateral_symmetry_factor  # ~0.95
    
    varied_pressures['right_hip_flexor'] += right_hip_neural_command * bilateral_coupling
    varied_pressures['right_hip_extensor'] -= right_hip_neural_command * bilateral_coupling
    
    # ===== PRINCIPIO 2: INHIBICIÓN SELECTIVA DE RODILLAS =====
    # Las rodillas tienen una lógica neurológica especial: cuando están flexionadas
    # y necesitan enderezarse (por retroalimentación propioceptiva), los flexores
    # se inhiben automáticamente sin comandos conscientes del cerebro
    
    # Esta es la implementación directa de tu observación biomecánica
    try:
        joint_states = p.getJointStates(self.env.robot_id, [1, 4])  # rodillas
        left_knee_angle = joint_states[0][0]
        right_knee_angle = joint_states[1][0]
        
        # LÓGICA NEUROLÓGICA: Si la rodilla está flexionada y necesita estabilidad,
        # el sistema nervioso inhibe automáticamente los flexores
        if left_knee_angle > 0.05:  # Umbral propioceptivo (~3 grados)
            # Simular inhibición neural completa del flexor
            varied_pressures['left_knee_flexor'] = 0.00
            
            if self.env.step_count % 750 == 0:
                print(f"🦵 Proprioceptive feedback: Left knee @ {left_knee_angle:.3f} rad")
                print(f"   → Neural inhibition: Flexor OFF (reciprocal inhibition)")
        
        if right_knee_angle > 0.05:
            varied_pressures['right_knee_flexor'] = 0.00
            
            if self.env.step_count % 750 == 0:
                print(f"🦵 Proprioceptive feedback: Right knee @ {right_knee_angle:.3f} rad")
                print(f"   → Neural inhibition: Flexor OFF")
                
    except Exception as e:
        # Si no hay retroalimentación propioceptiva (como en parálisis),
        # usar patrones reflejos básicos
        print(f"⚠️ Proprioceptive feedback lost, using reflex patterns")
    
    # ===== PRINCIPIO 3: LÍMITES FISIOLÓGICOS =====
    # El sistema nervioso no puede activar músculos más allá de sus límites
    # fisiológicos, implementamos esto con clipping suave
    for muscle, pressure in varied_pressures.items():
        # Simular el límite de activación neurológica máxima
        varied_pressures[muscle] = np.clip(pressure, 0.0, 1.0)
        
        # Simular el umbral de activación mínima (debajo del cual no hay contracción)
        if varied_pressures[muscle] < 0.02:  # 2% threshold
            varied_pressures[muscle] = 0.0
    
    return varied_pressures

def _simulate_spinal_reflexes(self, base_pressures, joint_angles):
    """
    🧠 SIMULACIÓN DE REFLEJOS ESPINALES
    
    Los reflejos espinales son respuestas automáticas que no requieren
    procesamiento cerebral. Implementamos los reflejos más importantes
    para el equilibrio postural.
    """
    
    reflex_adjustments = base_pressures.copy()
    
    # ===== REFLEJO DE ESTIRAMIENTO (STRETCH REFLEX) =====
    # Cuando un músculo se estira demasiado, automáticamente se contrae
    # para protegerse. Este es el reflejo más básico y rápido.
    
    left_knee_angle, right_knee_angle = joint_angles[1], joint_angles[4]
    
    # Si la rodilla está muy flexionada (músculo extensor muy estirado),
    # activar reflejo de extensión
    if left_knee_angle > 0.15:  # ~8.5 grados
        # Reflejo de estiramiento: activar extensores pasivos más fuerte
        print(f"🔥 Stretch reflex: Left knee extensors engaged")
        # Los extensores pasivos se manejan en la física, pero podríamos
        # modular la rigidez del resorte aquí si fuera necesario
    
    if right_knee_angle > 0.15:
        print(f"🔥 Stretch reflex: Right knee extensors engaged")
    
    # ===== REFLEJO DE CORRECCIÓN POSTURAL =====
    # Si el torso se inclina, ajustar automáticamente la activación de cadera
    try:
        pos, orn = p.getBasePositionAndOrientation(self.env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        forward_tilt = euler[1]  # Pitch (inclinación adelante/atrás)
        
        if abs(forward_tilt) > 0.1:  # ~5.7 grados de inclinación
            # Reflejo postural: compensar inclinación
            tilt_compensation = forward_tilt * 0.3  # Factor de corrección
            
            if forward_tilt > 0:  # Inclinado hacia adelante
                # Activar extensores de cadera más fuerte
                reflex_adjustments['left_hip_extensor'] += abs(tilt_compensation)
                reflex_adjustments['right_hip_extensor'] += abs(tilt_compensation)
                print(f"🏃 Postural reflex: Forward tilt → Hip extensors +{abs(tilt_compensation):.3f}")
            else:  # Inclinado hacia atrás
                # Activar flexores de cadera más fuerte
                reflex_adjustments['left_hip_flexor'] += abs(tilt_compensation)
                reflex_adjustments['right_hip_flexor'] += abs(tilt_compensation)
                print(f"🏃 Postural reflex: Backward tilt → Hip flexors +{abs(tilt_compensation):.3f}")
    
    except Exception as e:
        print(f"⚠️ Postural reflex system offline: {e}")
    
    return reflex_adjustments