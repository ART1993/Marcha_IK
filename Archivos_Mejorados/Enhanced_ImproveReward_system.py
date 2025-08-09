import numpy as np
import math
from collections import deque
import pybullet as p

class Enhanced_ImproveRewardSystem:
    """
    Sistema de recompensas mejorado específicamente diseñado para 6 PAMs antagónicos.
    
    Este sistema entiende la biomecánica de músculos antagónicos y recompensa:
    - Coordinación eficiente entre flexores y extensores
    - Co-activación apropiada para estabilidad (no excesiva)
    - Patrones de activación que imitan la marcha humana real
    - Eficiencia energética considerando pares musculares
    """
    
    def __init__(self, left_foot_id, right_foot_id, num_joints,
                 pam_states, curriculum_phase=1, num_pams=6):
        
        self.left_foot_id = left_foot_id
        self.right_foot_id = right_foot_id
        self.num_joints = num_joints
        self.pam_states = pam_states
        self.curriculum_phase = curriculum_phase
        self.num_pams = num_pams
        
        # Configurar parámetros específicos para 6 PAMs antagónicos
        self.setup_antagonistic_parameters()
        
        # Configurar nuevas métricas biomecánicas
        self.setup_biomechanical_metrics()
        
        # Parámetros generales del sistema (heredados pero actualizados)
        self.parametros_adicionales
        
        print(f"🧠 Enhanced Reward System initialized for {num_pams} antagonistic PAMs")
        print(f"   Curriculum phase: {curriculum_phase}")

    def setup_antagonistic_parameters(self):
        """
        Configura parámetros específicos para el control de músculos antagónicos.
        
        Esta función establece las métricas que necesitamos para evaluar si los
        músculos están trabajando de manera eficiente y coordinada, no solo fuerte.
        """
        
        # Mapeo de músculos antagónicos - Define qué músculos se oponen entre sí
        self.antagonistic_pairs = {
            'left_hip': {
                'flexor_idx': 0,      # Índice del flexor de cadera izquierda  
                'extensor_idx': 1,    # Índice del extensor de cadera izquierda
                'joint_id': 0,        # ID de la articulación en PyBullet
                'weight': 1.0         # Peso de importancia (cadera es crítica)
            },
            'right_hip': {
                'flexor_idx': 2,      # Índice del flexor de cadera derecha
                'extensor_idx': 3,    # Índice del extensor de cadera derecha  
                'joint_id': 3,        # ID de la articulación en PyBullet
                'weight': 1.0         # Peso de importancia
            },
            'left_knee': {
                'flexor_idx': 4,      # Índice del flexor de rodilla izquierda
                'extensor_idx': None, # Sin extensor activo (solo resorte pasivo)
                'joint_id': 1,        # ID de la articulación en PyBullet
                'weight': 0.8         # Menos crítico que caderas
            },
            'right_knee': {
                'flexor_idx': 5,      # Índice del flexor de rodilla derecha
                'extensor_idx': None, # Sin extensor activo (solo resorte pasivo)
                'joint_id': 4,        # ID de la articulación en PyBullet  
                'weight': 0.8         # Menos crítico que caderas
            }
        }
        
        # Rangos óptimos de co-activación - Basados en estudios biomecánicos
        self.optimal_coactivation = {
            'hip_min': 0.05,    # Mínimo 5% de co-activación para estabilidad
            'hip_max': 0.40,    # Máximo 40% de co-activación (más es desperdicio)
            'knee_min': 0.02,   # Rodillas requieren menos co-activación
            'knee_max': 0.25,   # Porque tienen resortes pasivos
        }
        
        # Patrones esperados de activación durante marcha normal
        self.expected_activation_patterns = {
            'stance_hip_extensor': 0.6,   # Extensores activos durante stance
            'stance_hip_flexor': 0.2,     # Flexores mínimos durante stance
            'swing_hip_flexor': 0.7,      # Flexores activos durante swing
            'swing_hip_extensor': 0.3,    # Extensores moderados durante swing
            'swing_knee_flexor': 0.8,     # Flexores altos para toe clearance
            'stance_knee_flexor': 0.2,    # Flexores bajos durante stance
        }
        
        print(f"   ✅ Antagonistic parameters configured for {len(self.antagonistic_pairs)} pairs")

    def setup_biomechanical_metrics(self):
        """
        Configura nuevas métricas biomecánicas para evaluar calidad de movimiento.
        
        Estas métricas van más allá de simple "fuerza vs presión" y evalúan
        si el robot se mueve de manera natural y eficiente como un humano.
        """
        
        # Métricas de coordinación muscular
        self.coordination_metrics = {
            'reciprocal_inhibition': 0.0,    # Qué tan bien se alternan flexor/extensor
            'coactivation_efficiency': 0.0,   # Co-activación apropiada vs excesiva
            'temporal_coordination': 0.0,     # Timing correcto de activaciones
            'bilateral_symmetry': 0.0,        # Simetría entre piernas izq/der
        }
        
        # Histórico para calcular métricas temporales
        self.muscle_activation_history = deque(maxlen=50)  # ~0.33 segundos a 150Hz
        self.coordination_history = deque(maxlen=20)       # Para suavizado
        
        # Contadores de fases de marcha para evaluación contextual
        self.gait_phase_detector = {
            'current_phase': 'unknown',
            'phase_duration': 0,
            'stance_switches': 0,
            'last_contact_state': [False, False]  # [left_foot, right_foot]
        }
        
        print(f"   ✅ Biomechanical metrics initialized")

    def set_curriculum_phase(self, phase: int):
        """
        Ajusta los pesos de recompensa según la fase del currículo de aprendizaje.
        
        Diferentes fases de entrenamiento requieren diferentes énfasis:
        - Fase 0-1: Enfoque en estabilidad básica y coordinación
        - Fase 2-3: Enfoque en eficiencia y patrones naturales  
        - Fase 4+: Enfoque en optimización avanzada
        """
        self.curriculum_phase = phase
        
        # Ajustar pesos según la fase de aprendizaje
        if phase <= 1:
            # Fases tempranas: priorizar estabilidad y supervivencia
            self.weights = {
                'survival': 0.25,           # Muy importante no caerse
                'progress': 0.20,           # Progreso moderado
                'stability': 0.20,          # Estabilidad crucial
                'velocity_control': 0.10,   # Control básico
                'pam_efficiency': 0.05,     # Eficiencia mínima
                'gait_quality': 0.10,       # Calidad básica
                'foot_clearance': 0.10,     # Clearance básico
            }
        elif phase <= 3:
            # Fases intermedias: balance entre estabilidad y eficiencia
            self.weights = {
                'survival': 0.15,           # Menos peso en supervivencia
                'progress': 0.25,           # Más progreso
                'stability': 0.15,          # Menos peso en estabilidad
                'velocity_control': 0.15,   # Más control
                'pam_efficiency': 0.15,     # Más eficiencia PAM
                'gait_quality': 0.10,       # Calidad moderada
                'foot_clearance': 0.05,     # Clearance estable
            }
        else:
            # Fases avanzadas: optimización y naturalidad
            self.weights = {
                'survival': 0.05,           # Supervivencia asumida
                'progress': 0.30,           # Máximo progreso
                'stability': 0.10,          # Estabilidad asumida
                'velocity_control': 0.15,   # Control refinado
                'pam_efficiency': 0.25,     # Máxima eficiencia PAM
                'gait_quality': 0.10,       # Calidad de marcha
                'foot_clearance': 0.05,     # Clearance refinado
            }
        
        print(f"   📊 Reward weights updated for curriculum phase {phase}")

    def _calculate_balanced_reward(self, action, pam_forces):
        """
        Sistema principal de recompensas balanceado que integra métricas antagónicas.
        
        Este método combina todas las métricas biomecánicas nuevas con las
        recompensas tradicionales para crear una señal de entrenamiento rica
        que guía al robot hacia movimientos naturales y eficientes.
        """
        
        # Manejar fase 0 especial (solo equilibrio)
        if self.curriculum_phase == 0:
            return self._calculate_equilibrium_reward(action, pam_forces)
        
        # Obtener estados básicos del robot
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Inicializar componentes de recompensa
        rewards = {}
        total_reward = 0.0
        
        # === RECOMPENSAS TRADICIONALES (actualizadas) ===
        
        # 1. Supervivencia básica
        rewards['survival'] = 1.0
        
        # 2. Progreso hacia adelante (mejorado)
        rewards['progress'] = self._calculate_progress_reward(lin_vel)
        
        # 3. Estabilidad postural (actualizada)
        rewards['stability'] = self._calculate_stability_reward(pos, euler, ang_vel)
        
        # 4. Control de velocidad (refinado)
        rewards['velocity_control'] = self._calculate_velocity_control_reward(lin_vel, ang_vel)
        
        # 5. Calidad de marcha (mejorada)
        rewards['gait_quality'] = self._calculate_enhanced_gait_quality_reward()
        
        # 6. Clearance de pies (actualizado)
        rewards['foot_clearance'] = self._calculate_foot_clearance_reward()
        
        # === NUEVAS RECOMPENSAS PARA 6 PAMs ANTAGÓNICOS ===
        
        # 7. Eficiencia PAM antagónica (completamente nueva)
        rewards['pam_efficiency'] = self._calculate_antagonistic_pam_efficiency(action, pam_forces)
        
        # === RECOMPENSAS POSTURALES CONTEXTUALES ===
        
        # 8. Postura biomecánica apropiada
        postural_reward = self._calculate_postural_reward(pos, euler)
        total_reward += postural_reward
        
        # === COMBINACIÓN FINAL ===
        
        # Combinar recompensas con pesos del currículo
        for component, reward in rewards.items():
            total_reward += reward * self.weights[component]
        
        # Aplicar penalizaciones críticas
        total_reward += self._apply_critical_penalties(pos, euler, lin_vel)
        
        # Aplicar bonificaciones de coordinación temporal
        coordination_bonus = self._calculate_temporal_coordination_bonus()
        total_reward += coordination_bonus
        
        # Limitar recompensa total para estabilidad del entrenamiento
        total_reward = np.clip(total_reward, -15.0, 25.0)
        
        return total_reward, rewards

    def _calculate_equilibrium_reward(self, action, pam_forces):
        """
        Recompensa especializada para la fase 0: solo equilibrio estático.
        
        En esta fase, el robot debe aprender a mantenerse erguido sin moverse,
        usando co-activación mínima pero efectiva de los músculos antagónicos.
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        reward = 2.0  # Base por sobrevivir sin caer
        
        # Penalización fuerte por inclinación (debe estar perfectamente vertical)
        tilt_penalty = 8.0 * (abs(euler[0]) + abs(euler[1]))
        reward -= tilt_penalty
        
        # Penalización por movimiento (debe quedarse quieto)
        movement_penalty = 3.0 * np.linalg.norm(lin_vel[:2])  # Solo X,Y (Z está bien)
        reward -= movement_penalty
        
        # Penalización por velocidad angular (no debe rotar)
        rotation_penalty = 2.0 * np.linalg.norm(ang_vel)
        reward -= rotation_penalty
        
        # Penalización por caída crítica
        if pos[2] < 0.8:
            reward -= 15.0
        
        # NUEVA: Recompensa por co-activación eficiente en equilibrio
        if hasattr(self, 'pam_states') and self.pam_states is not None:
            equilibrium_efficiency = self._calculate_equilibrium_coactivation_reward()
            reward += equilibrium_efficiency
        
        return np.clip(reward, -25, 8), {
            'survival': 2.0,
            'tilt_penalty': -tilt_penalty,
            'movement_penalty': -movement_penalty,
            'rotation_penalty': -rotation_penalty
        }

    def _calculate_antagonistic_pam_efficiency(self, action, pam_forces):
        """
        Calcula eficiencia específica para músculos antagónicos.
        
        Esta función evalúa si los músculos están trabajando de manera coordinada
        y eficiente, no solo si están generando fuerza bruta.
        """
        if not hasattr(self, 'pam_states') or self.pam_states is None:
            return 0.0
        
        total_efficiency_reward = 0.0
        
        # Actualizar histórico de activaciones musculares
        current_activations = self.pam_states['pressures'] / self.max_pressure
        self.muscle_activation_history.append(current_activations.copy())
        
        # === 1. EFICIENCIA DE CO-ACTIVACIÓN ===
        coactivation_reward = self._evaluate_coactivation_efficiency(current_activations)
        total_efficiency_reward += coactivation_reward
        
        # === 2. INHIBICIÓN RECÍPROCA ===
        reciprocal_reward = self._evaluate_reciprocal_inhibition(current_activations)
        total_efficiency_reward += reciprocal_reward
        
        # === 3. COORDINACIÓN BILATERAL ===
        bilateral_reward = self._evaluate_bilateral_coordination(current_activations)
        total_efficiency_reward += bilateral_reward
        
        # === 4. EFICIENCIA ENERGÉTICA ESPECÍFICA ===
        energy_reward = self._evaluate_antagonistic_energy_efficiency(current_activations, pam_forces)
        total_efficiency_reward += energy_reward
        
        # === 5. COORDINACIÓN TEMPORAL ===
        if len(self.muscle_activation_history) >= 10:
            temporal_reward = self._evaluate_temporal_coordination()
            total_efficiency_reward += temporal_reward
        
        # Actualizar métricas internas para debugging
        self._update_coordination_metrics(current_activations)
        
        return max(0, total_efficiency_reward)

    def _evaluate_coactivation_efficiency(self, activations):
        """
        Evalúa si la co-activación entre músculos antagónicos es apropiada.
        
        Co-activación óptima: suficiente para estabilidad, no tanto que desperdicie energía.
        """
        coactivation_reward = 0.0
        
        # Evaluar cada par antagónico
        for joint_name, pair_info in self.antagonistic_pairs.items():
            if pair_info['extensor_idx'] is None:
                continue  # Skip rodillas (solo tienen flexor activo)
            
            flexor_activation = activations[pair_info['flexor_idx']]
            extensor_activation = activations[pair_info['extensor_idx']]
            
            # Calcular nivel de co-activación
            coactivation_level = min(flexor_activation, extensor_activation)
            max_activation = max(flexor_activation, extensor_activation)
            
            if max_activation > 0.1:  # Solo evaluar si hay activación significativa
                coactivation_ratio = coactivation_level / max_activation
                
                # Determinar rango óptimo según el tipo de articulación
                if 'hip' in joint_name:
                    optimal_min = self.optimal_coactivation['hip_min']
                    optimal_max = self.optimal_coactivation['hip_max']
                else:
                    optimal_min = self.optimal_coactivation['knee_min']
                    optimal_max = self.optimal_coactivation['knee_max']
                
                # Recompensar co-activación en rango óptimo
                if optimal_min <= coactivation_ratio <= optimal_max:
                    coactivation_reward += 1.5 * pair_info['weight']
                elif coactivation_ratio < optimal_min:
                    # Penalizar falta de co-activación (inestabilidad)
                    deficit = optimal_min - coactivation_ratio
                    coactivation_reward -= 0.5 * deficit * pair_info['weight']
                else:
                    # Penalizar exceso de co-activación (desperdicio energético)
                    excess = coactivation_ratio - optimal_max
                    coactivation_reward -= 1.0 * excess * pair_info['weight']
        
        return coactivation_reward

    def _evaluate_reciprocal_inhibition(self, activations):
        """
        Evalúa la inhibición recíproca: cuando un músculo se activa fuertemente,
        su antagónico debería reducir su activación (principio biomecánico).
        """
        reciprocal_reward = 0.0
        
        for joint_name, pair_info in self.antagonistic_pairs.items():
            if pair_info['extensor_idx'] is None:
                continue
            
            flexor_activation = activations[pair_info['flexor_idx']]
            extensor_activation = activations[pair_info['extensor_idx']]
            
            # Cuando un músculo está muy activo (>60%), el otro debería estar menos activo
            if flexor_activation > 0.6:
                # Flexor muy activo, extensor debería estar moderado
                if extensor_activation < 0.4:
                    reciprocal_reward += 0.8 * pair_info['weight']
                else:
                    reciprocal_reward -= 0.5 * pair_info['weight']
            
            if extensor_activation > 0.6:
                # Extensor muy activo, flexor debería estar moderado
                if flexor_activation < 0.4:
                    reciprocal_reward += 0.8 * pair_info['weight']
                else:
                    reciprocal_reward -= 0.5 * pair_info['weight']
        
        return reciprocal_reward

    def _evaluate_bilateral_coordination(self, activations):
        """
        Evalúa la coordinación entre piernas izquierda y derecha.
        
        Durante marcha normal, las piernas deben alternar sus patrones
        de activación de manera coordinada.
        """
        bilateral_reward = 0.0
        
        # Comparar activación entre músculos homólogos (misma función, diferente pierna)
        left_hip_flexor = activations[0]   # left_hip_flexor
        right_hip_flexor = activations[2]  # right_hip_flexor
        
        left_hip_extensor = activations[1]  # left_hip_extensor
        right_hip_extensor = activations[3] # right_hip_extensor
        
        left_knee_flexor = activations[4]   # left_knee_flexor
        right_knee_flexor = activations[5]  # right_knee_flexor
        
        # Durante marcha normal, los patrones deben ser complementarios (no idénticos)
        # Recompensar cuando una pierna está más activa que la otra (alternancia)
        
        # Diferencia en activación entre piernas (queremos alternancia)
        hip_flexor_asymmetry = abs(left_hip_flexor - right_hip_flexor)
        hip_extensor_asymmetry = abs(left_hip_extensor - right_hip_extensor)
        knee_flexor_asymmetry = abs(left_knee_flexor - right_knee_flexor)
        
        # Recompensar alternancia moderada (no extrema)
        optimal_asymmetry = 0.3  # 30% de diferencia es ideal
        
        for asymmetry in [hip_flexor_asymmetry, hip_extensor_asymmetry, knee_flexor_asymmetry]:
            if 0.1 <= asymmetry <= 0.5:  # Alternancia apropiada
                bilateral_reward += 0.5
            elif asymmetry < 0.1:  # Demasiado simétrico (como robot)
                bilateral_reward -= 0.2
            else:  # Demasiado asimétrico (puede indicar problema)
                bilateral_reward -= 0.3
        
        return bilateral_reward

    def _evaluate_antagonistic_energy_efficiency(self, activations, pam_forces):
        """
        Evalúa eficiencia energética específica para músculos antagónicos.
        
        Esta métrica considera que la co-activación cuesta energía pero
        proporciona beneficios de estabilidad y control fino.
        """
        if len(pam_forces) < self.num_pams:
            return 0.0
        
        energy_reward = 0.0
        total_energy_cost = 0.0
        total_useful_work = 0.0
        
        # Calcular costo energético y trabajo útil para cada par antagónico
        for joint_name, pair_info in self.antagonistic_pairs.items():
            flexor_idx = pair_info['flexor_idx']
            flexor_activation = activations[flexor_idx]
            flexor_force = abs(pam_forces[flexor_idx]) if flexor_idx < len(pam_forces) else 0
            
            if pair_info['extensor_idx'] is not None:
                # Par antagónico completo (caderas)
                extensor_idx = pair_info['extensor_idx']
                extensor_activation = activations[extensor_idx]
                extensor_force = abs(pam_forces[extensor_idx]) if extensor_idx < len(pam_forces) else 0
                
                # Costo energético = presión al cuadrado (aproximación fisiológica)
                energy_cost = flexor_activation**2 + extensor_activation**2
                
                # Trabajo útil = fuerza neta efectiva
                net_force = abs(flexor_force - extensor_force)
                useful_work = net_force
                
                # Penalizar trabajo "perdido" por co-activación excesiva
                wasted_work = min(flexor_force, extensor_force)
                useful_work -= 0.3 * wasted_work  # Co-activación tiene algo de valor
                
            else:
                # Solo flexor (rodillas)
                energy_cost = flexor_activation**2
                useful_work = flexor_force
            
            total_energy_cost += energy_cost * pair_info['weight']
            total_useful_work += useful_work * pair_info['weight']
        
        # Calcular eficiencia como work/energy ratio
        if total_energy_cost > 0.01:  # Evitar división por cero
            efficiency = total_useful_work / (total_energy_cost + 0.01)
            energy_reward = min(efficiency * 2.0, 3.0)  # Limitar bonificación máxima
        
        # Penalizar activación excesiva sin propósito
        if total_energy_cost > 2.0:  # Umbral de energía excesiva
            energy_reward -= (total_energy_cost - 2.0) * 0.5
        
        return energy_reward

    def _evaluate_temporal_coordination(self):
        """
        Evalúa la coordinación temporal de activaciones musculares.
        
        Los músculos deben activarse en secuencias temporales apropiadas
        que reflejen patrones de marcha natural.
        """
        if len(self.muscle_activation_history) < 10:
            return 0.0
        
        temporal_reward = 0.0
        
        # Analizar últimas 10 activaciones para detectar patrones temporales
        recent_activations = np.array(list(self.muscle_activation_history)[-10:])
        
        # === 1. SUAVIDAD TEMPORAL ===
        # Los músculos reales no cambian abruptamente su activación
        for muscle_idx in range(self.num_pams):
            muscle_timeline = recent_activations[:, muscle_idx]
            
            # Calcular variabilidad (cambios abruptos son malos)
            temporal_variation = np.std(np.diff(muscle_timeline))
            
            if temporal_variation < 0.1:  # Cambios suaves
                temporal_reward += 0.2
            elif temporal_variation > 0.3:  # Cambios muy abruptos
                temporal_reward -= 0.3
        
        # === 2. ALTERNANCIA TEMPORAL ENTRE PIERNAS ===
        # Las piernas deben alternar sus patrones en el tiempo
        left_hip_timeline = recent_activations[:, 0]   # left_hip_flexor
        right_hip_timeline = recent_activations[:, 2]  # right_hip_flexor
        
        # Calcular correlación cruzada (patrones anti-correlacionados son buenos)
        correlation = np.corrcoef(left_hip_timeline, right_hip_timeline)[0, 1]
        
        if correlation < -0.3:  # Anti-correlación buena (alternancia)
            temporal_reward += 1.0
        elif correlation > 0.5:  # Correlación alta mala (mismos patrones)
            temporal_reward -= 0.8
        
        # === 3. PATRONES CÍCLICOS ===
        # La marcha debe mostrar patrones repetitivos
        try:
            # Detectar periodicidad usando autocorrelación simple
            left_autocorr = np.correlate(left_hip_timeline, left_hip_timeline, mode='full')
            peak_spacing = self._find_peak_spacing(left_autocorr)
            
            # Recompensar periodicidad en rango típico de marcha (5-15 steps por ciclo)
            if 5 <= peak_spacing <= 15:
                temporal_reward += 0.8
                
        except:
            pass  # Si falla el análisis, no penalizar
        
        return temporal_reward

    def _find_peak_spacing(self, autocorr):
        """Función auxiliar para encontrar espaciado entre picos en autocorrelación"""
        mid = len(autocorr) // 2
        right_half = autocorr[mid:]
        
        # Encontrar picos simples
        peaks = []
        for i in range(1, len(right_half) - 1):
            if right_half[i] > right_half[i-1] and right_half[i] > right_half[i+1]:
                peaks.append(i)
        
        return peaks[0] if peaks else 10  # Default a 10 si no encuentra picos

    def _calculate_postural_reward(self, pos, euler):
        """
        Recompensa por mantener postura biomecánicamente apropiada.
        
        Esta función evalúa si el robot mantiene una postura que se asemeje
        a la marcha humana natural, no solo que no se caiga.
        """
        postural_reward = 0.0
        
        # Obtener posiciones de pies para análisis postural
        try:
            left_foot = p.getLinkState(self.robot_id, self.left_foot_id)[0]
            right_foot = p.getLinkState(self.robot_id, self.right_foot_id)[0]
            foot_center_x = (left_foot[0] + right_foot[0]) / 2
            
            # === 1. POSICIÓN DE CADERA RELATIVA A PIES ===
            # La cadera debe estar ligeramente adelantada respecto a los pies (postura activa)
            hip_offset = pos[0] - foot_center_x
            
            if 0.01 <= hip_offset <= 0.08:  # Cadera adelantada 1-8cm
                postural_reward += 1.5
            elif hip_offset < -0.05:  # Cadera muy atrás (postura pasiva mala)
                postural_reward -= 2.0
            
            # === 2. ALTURA APROPIADA DE CADERA ===
            # La cadera debe mantenerse a altura natural, no agachada ni estirada
            target_height = 1.2  # Altura nominal del robot
            height_error = abs(pos[2] - target_height)
            
            if height_error < 0.05:  # Altura muy buena
                postural_reward += 1.0
            elif height_error > 0.15:  # Muy agachado o muy estirado
                postural_reward -= 1.5
            
            # === 3. INCLINACIÓN HACIA ADELANTE APROPIADA ===
            # Ligera inclinación hacia adelante es natural durante marcha
            forward_lean = euler[1]  # Pitch
            
            if 0.02 <= forward_lean <= 0.12:  # 1-7 grados hacia adelante
                postural_reward += 0.8
            elif forward_lean < -0.05:  # Inclinado hacia atrás (antinatural)
                postural_reward -= 1.2
            elif forward_lean > 0.20:  # Demasiado inclinado hacia adelante
                postural_reward -= 1.0
            
            # === 4. ESTABILIDAD LATERAL ===
            # Mínimo balanceo lateral
            lateral_tilt = abs(euler[0])  # Roll
            
            if lateral_tilt < 0.05:  # Muy estable lateralmente
                postural_reward += 0.5
            elif lateral_tilt > 0.15:  # Demasiado balanceo lateral
                postural_reward -= 1.0
                
        except:
            # Si hay error obteniendo posiciones, no penalizar
            pass
        
        return postural_reward

    def _calculate_temporal_coordination_bonus(self):
        """
        Bonificación por coordinación temporal entre músculos y movimiento.
        
        Esta función recompensa cuando las activaciones musculares están
        sincronizadas apropiadamente con las fases de marcha.
        """
        if len(self.muscle_activation_history) < 5:
            return 0.0
        
        coordination_bonus = 0.0
        
        # Detectar fase actual de marcha basada en contactos de pies
        left_contact = len(p.getContactPoints(self.robot_id, 0, self.left_foot_id)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, 0, self.right_foot_id)) > 0
        
        # Determinar fase de marcha
        if left_contact and not right_contact:
            current_phase = 'left_stance'
        elif right_contact and not left_contact:
            current_phase = 'right_stance'
        elif left_contact and right_contact:
            current_phase = 'double_support'
        else:
            current_phase = 'flight'  # Ambos pies en el aire
        
        # Obtener activaciones musculares actuales
        if hasattr(self, 'pam_states') and self.pam_states is not None:
            current_activations = self.pam_states['pressures'] / self.max_pressure
            
            # === RECOMPENSAR ACTIVACIONES APROPIADAS PARA LA FASE ===
            
            if current_phase == 'left_stance':
                # Durante stance izquierdo: extensor izquierdo alto, flexor derecho preparándose
                if current_activations[1] > 0.4:  # left_hip_extensor activo
                    coordination_bonus += 0.8
                if current_activations[2] > 0.3:  # right_hip_flexor preparándose
                    coordination_bonus += 0.5
                    
            elif current_phase == 'right_stance':
                # Durante stance derecho: extensor derecho alto, flexor izquierdo preparándose
                if current_activations[3] > 0.4:  # right_hip_extensor activo
                    coordination_bonus += 0.8
                if current_activations[0] > 0.3:  # left_hip_flexor preparándose
                    coordination_bonus += 0.5
                    
            elif current_phase == 'double_support':
                # Durante doble soporte: ambos extensores moderadamente activos
                if current_activations[1] > 0.3 and current_activations[3] > 0.3:
                    coordination_bonus += 0.6
                    
            elif current_phase == 'flight':
                # Durante vuelo: ambos flexores activos para toe clearance
                if current_activations[4] > 0.5 or current_activations[5] > 0.5:
                    coordination_bonus += 0.4
        
        # Actualizar detector de fase para métricas futuras
        self.gait_phase_detector['current_phase'] = current_phase
        self.gait_phase_detector['last_contact_state'] = [left_contact, right_contact]
        
        return coordination_bonus

    def _update_coordination_metrics(self, activations):
        """Actualizar métricas internas para debugging y análisis"""
        # Calcular métricas actuales para el dashboard de debugging
        
        # Reciprocal inhibition score
        reciprocal_score = 0.0
        for joint_name, pair_info in self.antagonistic_pairs.items():
            if pair_info['extensor_idx'] is not None:
                flexor_act = activations[pair_info['flexor_idx']]
                extensor_act = activations[pair_info['extensor_idx']]
                reciprocal_score += abs(flexor_act - extensor_act) / 2.0
        
        self.coordination_metrics['reciprocal_inhibition'] = reciprocal_score / 2.0  # Normalize
        
        # Bilateral symmetry score
        left_total = sum([activations[0], activations[1], activations[4]])  # Left side
        right_total = sum([activations[2], activations[3], activations[5]])  # Right side
        
        if left_total + right_total > 0:
            symmetry = 1.0 - abs(left_total - right_total) / (left_total + right_total)
            self.coordination_metrics['bilateral_symmetry'] = symmetry
        
        # Actualizar histórico para análisis temporal
        self.coordination_history.append(self.coordination_metrics.copy())

    # === MÉTODOS AUXILIARES ACTUALIZADOS ===

    def _calculate_progress_reward(self, lin_vel):
        """Recompensa por progreso hacia adelante (actualizada)"""
        forward_velocity = lin_vel[0]
        
        if 0 < forward_velocity <= self.target_forward_velocity:
            return min((forward_velocity / self.target_forward_velocity) * 3.0, 4.0)
        elif self.target_forward_velocity < forward_velocity <= self.max_forward_velocity:
            excess_factor = (forward_velocity - self.target_forward_velocity) / (self.max_forward_velocity - self.target_forward_velocity)
            return 4.0 * (1.0 - excess_factor * 0.4)
        else:
            return -1.5 if forward_velocity > self.max_forward_velocity else -0.5

    def _calculate_stability_reward(self, pos, euler, ang_vel):
        """Recompensa por estabilidad (actualizada)"""
        height_error = abs(pos[2] - self.target_height)
        height_reward = max(0, 2.5 - height_error * 3.0)
        
        roll_penalty = abs(euler[0]) * 2.0
        pitch_penalty = abs(euler[1]) * 1.5
        balance_reward = max(0, 2.0 - roll_penalty - pitch_penalty)
        
        return (height_reward + balance_reward) / 2.0

    def _calculate_velocity_control_reward(self, lin_vel, ang_vel):
        """Recompensa por control de velocidad (actualizada)"""
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        angular_penalty = max(0, (ang_vel_magnitude - self.max_angular_velocity) * 2.0)
        
        lateral_velocity_penalty = min(abs(lin_vel[1]) * 1.5, 2.5)
        
        return max(0, 3.5 - angular_penalty - lateral_velocity_penalty)

    def _calculate_enhanced_gait_quality_reward(self):
        """Calidad de marcha mejorada para 6 PAMs"""
        return self._calculate_gait_quality_reward()  # Usar implementación existente por ahora

    def _calculate_foot_clearance_reward(self):
        """Recompensa por clearance de pies (mantenida)"""
        left_foot_pos = p.getLinkState(self.robot_id, self.left_foot_id)[0]
        right_foot_pos = p.getLinkState(self.robot_id, self.right_foot_id)[0]
        ground_z = 0.0
        
        left_clearance = left_foot_pos[2] - ground_z
        right_clearance = right_foot_pos[2] - ground_z
        
        foot_clearance_reward = 0.0
        
        # Recompensar alternancia de pies
        if (left_clearance > 0.07 and right_clearance < 0.04) or (right_clearance > 0.07 and left_clearance < 0.04):
            foot_clearance_reward += 1.2
        
        # Penalizar ambos pies muy bajos o muy altos
        if left_clearance < 0.015 and right_clearance < 0.015:
            foot_clearance_reward -= 0.6
        if left_clearance > 0.06 and right_clearance > 0.06:
            foot_clearance_reward -= 2.0
        
        return foot_clearance_reward

    def _calculate_equilibrium_coactivation_reward(self):
        """Recompensa por co-activación eficiente durante equilibrio estático"""
        if not hasattr(self, 'pam_states') or self.pam_states is None:
            return 0.0
        
        current_activations = self.pam_states['pressures'] / self.max_pressure
        equilibrium_reward = 0.0
        
        # Durante equilibrio, queremos co-activación mínima pero presente
        for joint_name, pair_info in self.antagonistic_pairs.items():
            if pair_info['extensor_idx'] is None:
                continue
            
            flexor_act = current_activations[pair_info['flexor_idx']]
            extensor_act = current_activations[pair_info['extensor_idx']]
            
            # Recompensar activación balanceada y moderada
            balance = 1.0 - abs(flexor_act - extensor_act)
            avg_activation = (flexor_act + extensor_act) / 2.0
            
            if 0.2 <= avg_activation <= 0.5:  # Activación moderada
                equilibrium_reward += balance * 0.8
            elif avg_activation < 0.1:  # Muy poca activación (inestable)
                equilibrium_reward -= 0.5
            elif avg_activation > 0.7:  # Demasiada activación (desperdicio)
                equilibrium_reward -= 0.3
        
        return equilibrium_reward

    # === MÉTODOS HEREDADOS SIN CAMBIOS ===
    
    def _calculate_gait_quality_reward(self):
        """Heredado de la implementación original"""
        left_contact = len(p.getContactPoints(self.robot_id, 0, self.left_foot_id)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, 0, self.right_foot_id)) > 0

        gait_reward = 0.0

        if left_contact or right_contact:
            gait_reward += 1.0

        if left_contact and right_contact:
            self.double_support_steps = getattr(self, "double_support_steps", 0) + 1
        else:
            self.double_support_steps = 0
        if self.double_support_steps > 10:
            gait_reward -= 1.0

        if hasattr(self, 'last_single_support') and self.last_single_support is not None:
            prev = self.last_single_support
            current = (left_contact, right_contact)
            if prev != current and (prev in [(True, False), (False, True)]) and (current in [(True, False), (False, True)]):
                gait_reward += 1.5
        self.last_single_support = (left_contact, right_contact)

        left_foot_pos = p.getLinkState(self.robot_id, self.left_foot_id)[0]
        right_foot_pos = p.getLinkState(self.robot_id, self.right_foot_id)[0]
        ground_z = 0.0

        if not left_contact and (left_foot_pos[2] - ground_z) > 0.025:
            gait_reward += 0.5
        if not right_contact and (right_foot_pos[2] - ground_z) > 0.025:
            gait_reward += 0.5

        if not left_contact and not right_contact:
            gait_reward -= 3.0

        contact_stability = 0.5 if (left_contact or right_contact) else 0.0
        gait_reward += contact_stability

        self.previous_contacts = [left_contact, right_contact]
        return gait_reward

    def _apply_critical_penalties(self, pos, euler, lin_vel):
        """Penalizaciones críticas (heredadas)"""
        penalty = 0.0
        
        if pos[2] < 0.55:
            penalty -= 12.0
        if abs(euler[1]) > math.pi/2:
            penalty -= 4.0
        if abs(lin_vel[1]) > 1.0:
            penalty -= 2.5
        if lin_vel[0] < -0.2:
            penalty -= 3.0
        
        return penalty

    def reset_tracking(self):
        """Resetear variables de seguimiento"""
        self.step_count = 0
        self.reward_history.clear()
        self.foot_trajectory_history.clear()
        self.muscle_activation_history.clear()
        self.coordination_history.clear()
        self.previous_position = None

    def redefine_robot(self, robot_id, plane_id):
        """Redefinir IDs del robot"""
        self.robot_id = robot_id
        self.plane_id = plane_id

    @property
    def parametros_adicionales(self):
        """Parámetros adicionales del sistema (heredados y actualizados)"""
        self.target_forward_velocity = 1.0
        self.max_forward_velocity = 2.0
        self.target_height = 1.2
        self.max_angular_velocity = 2.0
        self.max_pressure = 601325  # Actualizar según Enhanced_PAMIKBipedEnv

        # Parámetros de clearance de pies
        self.target_foot_height = 0.1
        self.max_foot_height = 0.3
        self.ground_clearance = 0.05
        
        # Pesos por defecto (se actualizan con set_curriculum_phase)
        self.weights = {
            'survival': 0.15,
            'progress': 0.25,
            'stability': 0.15,
            'velocity_control': 0.15,
            'pam_efficiency': 0.15,
            'gait_quality': 0.10,
            'foot_clearance': 0.05
        }

        # Variables para suavizado
        self.reward_history = deque(maxlen=10)
        self.smoothing_factor = 0.7
        
        # Tracking
        self.step_count = 0
        self.previous_position = None
        self.previous_foot_positions = None
        self.foot_trajectory_history = deque(maxlen=20)


def test_enhanced_reward_system():
    """Función de prueba para el sistema de recompensas mejorado"""
    print("🧪 Testing Enhanced Reward System for 6 PAMs...")
    
    # Crear sistema de prueba
    reward_system = Enhanced_ImproveRewardSystem(
        left_foot_id=2,
        right_foot_id=5,
        num_joints=4,
        pam_states=None,
        curriculum_phase=1,
        num_pams=6
    )
    
    print(f"✅ Reward system created successfully")
    print(f"   Antagonistic pairs: {len(reward_system.antagonistic_pairs)}")
    print(f"   Curriculum phase: {reward_system.curriculum_phase}")
    print(f"   Weights: {reward_system.weights}")
    
    # Simular estados PAM
    reward_system.pam_states = {
        'pressures': np.array([0.3, 0.4, 0.2, 0.5, 0.1, 0.15]) * 601325,
        'contractions': np.array([0.2, 0.3, 0.1, 0.4, 0.05, 0.08]),
        'forces': np.array([150, 200, 100, 250, 50, 75])
    }
    reward_system.max_pressure = 601325
    
    # Probar evaluación de co-activación
    activations = reward_system.pam_states['pressures'] / reward_system.max_pressure
    coactivation_reward = reward_system._evaluate_coactivation_efficiency(activations)
    print(f"   Coactivation reward: {coactivation_reward:.3f}")
    
    reciprocal_reward = reward_system._evaluate_reciprocal_inhibition(activations)
    print(f"   Reciprocal inhibition reward: {reciprocal_reward:.3f}")
    
    bilateral_reward = reward_system._evaluate_bilateral_coordination(activations)
    print(f"   Bilateral coordination reward: {bilateral_reward:.3f}")
    
    print("🎉 Enhanced reward system test completed!")

if __name__ == "__main__":
    test_enhanced_reward_system()
