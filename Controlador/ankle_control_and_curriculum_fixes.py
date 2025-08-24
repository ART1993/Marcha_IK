# ====================================================================
# SISTEMA HÍBRIDO DE CONTROL DE TOBILLOS
# Resortes inteligentes + control adaptativo para balance estable
# ====================================================================

import numpy as np
import math
from collections import deque
import pybullet as p

class IntelligentAnkleControl:
    """
    Sistema híbrido de control de tobillos que combina:
    1. Resortes pasivos para soporte estructural básico
    2. Control activo adaptativo para balance fino
    3. Detección de perturbaciones para respuesta rápida
    
    Los tobillos humanos proporcionan ~70% del control postural en posición erguida.
    Este sistema imita esa funcionalidad crítica sin necesidad de PAMs adicionales.
    """
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # Parámetros de resortes base (más fuertes que rodillas)
        self.base_spring_stiffness = 120.0  # Más rígido que rodillas (eran 150)
        self.base_damping = 8.0  # Amortiguación moderada
        
        # Parámetros de control activo
        self.balance_gain = 0.8  # Ganancia para corrección de balance
        self.max_active_torque = 25.0  # Máximo torque activo adicional
        
        # Sistema de detección de perturbaciones
        self.com_history = deque(maxlen=5)
        self.ankle_angle_history = deque(maxlen=5)
        
        # Umbral para activación de control activo
        self.activation_threshold = 0.02  # rad (~1.1 grados)
        
        print(f"🦶 Sistema Inteligente de Control de Tobillos inicializado")
        print(f"   Rigidez base: {self.base_spring_stiffness} N⋅m/rad")
        print(f"   Control activo: Habilitado para balance fino")
    
    def calculate_ankle_torques(self, robot_data, zmp_calculator=None):
        """
        Calcula torques de tobillos usando sistema híbrido:
        1. Resortes pasivos para soporte base
        2. Control activo para correcciones de balance
        
        Args:
            robot_data: Datos del robot (PyBullet_Robot_Data)
            zmp_calculator: Calculador ZMP para información de estabilidad
            
        Returns:
            tuple: (left_ankle_torque, right_ankle_torque)
        """
        
        # Obtener estados de tobillos
        ankle_states = p.getJointStates(self.robot_id, [2, 5])  # left_ankle, right_ankle
        left_ankle_pos = ankle_states[0][0]
        right_ankle_pos = ankle_states[1][0]
        left_ankle_vel = ankle_states[0][1]
        right_ankle_vel = ankle_states[1][1]
        
        # ===== COMPONENTE 1: RESORTES PASIVOS BÁSICOS =====
        
        # Torques de resorte hacia posición neutral (0.0)
        left_spring_torque = -self.base_spring_stiffness * left_ankle_pos
        right_spring_torque = -self.base_spring_stiffness * right_ankle_pos
        
        # Amortiguación proporcional a velocidad
        left_damping_torque = -self.base_damping * left_ankle_vel
        right_damping_torque = -self.base_damping * right_ankle_vel
        
        # Torques pasivos totales
        left_passive_torque = left_spring_torque + left_damping_torque
        right_passive_torque = right_spring_torque + right_damping_torque
        
        # ===== COMPONENTE 2: CONTROL ACTIVO PARA BALANCE =====
        
        # Obtener información de balance si está disponible
        if zmp_calculator and robot_data:
            try:
                # Detectar si necesitamos control activo
                balance_correction = self._calculate_balance_correction(
                    robot_data, zmp_calculator, left_ankle_pos, right_ankle_pos
                )
                
                # Añadir corrección activa si es necesaria
                left_active_torque = balance_correction['left_correction']
                right_active_torque = balance_correction['right_correction']
                
            except Exception as e:
                # Fallback si hay problemas con cálculos avanzados
                left_active_torque = 0.0
                right_active_torque = 0.0
        else:
            # Sin información de balance, solo usar control postural básico
            left_active_torque = self._basic_postural_correction(left_ankle_pos)
            right_active_torque = self._basic_postural_correction(right_ankle_pos)
        
        # ===== COMBINACIÓN FINAL =====
        
        total_left_torque = left_passive_torque + left_active_torque
        total_right_torque = right_passive_torque + right_active_torque
        
        # Limitar torques para seguridad
        max_total_torque = 80.0  # N⋅m
        total_left_torque = np.clip(total_left_torque, -max_total_torque, max_total_torque)
        total_right_torque = np.clip(total_right_torque, -max_total_torque, max_total_torque)
        
        return total_left_torque, total_right_torque
    
    def _calculate_balance_correction(self, robot_data, zmp_calculator, left_ankle_pos, right_ankle_pos):
        """
        Calcula correcciones activas basadas en información de balance y ZMP
        
        Esta función implementa control postural específicamente diseñado
        para tobillos, usando información del centro de masas y ZMP
        """
        
        # Obtener centro de masas
        try:
            com_position, total_mass = robot_data.get_center_of_mass
            self.com_history.append(com_position)
        except:
            com_position = np.array([0, 0, 1.1])
            self.com_history.append(com_position)
        
        # Calcular movimiento del COM
        if len(self.com_history) >= 2:
            com_velocity = self.com_history[-1] - self.com_history[-2]
            com_velocity_magnitude = np.linalg.norm(com_velocity[:2])  # Solo x,y
        else:
            com_velocity_magnitude = 0.0
        
        # Obtener información ZMP
        try:
            zmp_point = zmp_calculator.calculate_zmp()
            is_stable = zmp_calculator.is_stable(zmp_point)
            stability_margin = zmp_calculator.stability_margin_distance(zmp_point)
        except:
            zmp_point = np.array([0.0, 0.0])
            is_stable = True
            stability_margin = 0.1
        
        # ===== LÓGICA DE CORRECCIÓN INTELIGENTE =====
        
        correction_factor = 0.0
        
        # Si el robot está perdiendo estabilidad, activar control
        if not is_stable or stability_margin < 0.05:
            # Corrección basada en posición del ZMP
            zmp_error_x = zmp_point[0]  # Desviación lateral del ZMP
            correction_factor = self.balance_gain * zmp_error_x
            
        # Si el COM se está moviendo rápido, anticipar corrección
        elif com_velocity_magnitude > 0.05:  # m/s
            com_error_x = com_position[0]  # Desviación del COM
            correction_factor = self.balance_gain * 0.5 * com_error_x
        
        # Aplicar corrección de manera diferencial (más en el tobillo que necesita trabajar más)
        if abs(correction_factor) > 0.01:
            # Tobillo que está más comprometido trabaja más
            left_load_factor = 1.0 + abs(left_ankle_pos)
            right_load_factor = 1.0 + abs(right_ankle_pos)
            total_load = left_load_factor + right_load_factor
            
            left_correction = correction_factor * (left_load_factor / total_load)
            right_correction = correction_factor * (right_load_factor / total_load)
        else:
            left_correction = 0.0
            right_correction = 0.0
        
        # Limitar correcciones activas
        left_correction = np.clip(left_correction, -self.max_active_torque, self.max_active_torque)
        right_correction = np.clip(right_correction, -self.max_active_torque, self.max_active_torque)
        
        return {
            'left_correction': left_correction,
            'right_correction': right_correction,
            'zmp_stable': is_stable,
            'correction_magnitude': abs(correction_factor)
        }
    
    def _basic_postural_correction(self, ankle_angle):
        """
        Corrección postural básica cuando no hay información avanzada disponible
        
        Usa solo el ángulo del tobillo para determinar si necesita corrección
        """
        
        if abs(ankle_angle) > self.activation_threshold:
            # Corrección proporcional simple
            correction = -self.balance_gain * ankle_angle
            return np.clip(correction, -self.max_active_torque, self.max_active_torque)
        
        return 0.0
    
    def apply_ankle_control(self):
        """
        Aplica el control calculado a los tobillos en PyBullet
        
        Este método debe ser llamado durante el step() del entorno
        """
        
        # Calcular torques (necesita acceso a robot_data y zmp_calculator)
        # Esto se implementará en la integración con el entorno
        pass

# ====================================================================
# AJUSTES ESPECÍFICOS DEL CURRICULUM PARA CONVERGENCIA RÁPIDA
# Optimización para reducir episodios hasta balance estable
# ====================================================================

class OptimizedCurriculumSelector:
    """
    Versión optimizada del curriculum que converge más rápido al balance estable.
    
    Cambios clave respecto al original:
    1. Inicio con mayor dependencia del experto (90% vs 70%)
    2. Reducción más gradual de la ayuda experta
    3. Umbrales de éxito más conservadores
    4. Buffer de estabilización antes de permitir sentadillas
    """
    
    def __init__(self):
        # Parámetros de curriculum OPTIMIZADOS para convergencia rápida
        self.episode_count = 0
        self.recent_rewards = deque(maxlen=50)  # Ventana más pequeña para respuesta rápida
        self.expert_ratio = 0.95  # CAMBIO: Empezar con 95% experto (vs 100% original)
        
        # Umbrales más conservadores para transiciones
        self.balance_reward_threshold = 3.0     # Reducido de 5.0 a 3.0
        self.squat_reward_threshold = 6.0       # Reducido de 8.0 a 6.0
        self.min_expert_ratio = 0.15            # Aumentado de 0.1 a 0.15 (más ayuda siempre)
        
        # Estados del curriculum OPTIMIZADOS
        self.current_phase = "BALANCE_LEARNING"
        self.phase_episodes = 0
        self.balance_stable_episodes = 0  # NUEVO: Contador de episodios estables
        
        # NUEVO: Buffer de episodios exitosos antes de permitir sentadillas
        self.required_stable_episodes = 25  # Necesitar 25 episodios estables antes de sentadillas
        
        print(f"🎓 Curriculum OPTIMIZADO inicializado para convergencia rápida")
        print(f"   Expert ratio inicial: {self.expert_ratio:.1%}")
        print(f"   Balance threshold: {self.balance_reward_threshold}")
        print(f"   Stable episodes required: {self.required_stable_episodes}")
    
    def should_use_expert_action(self, step_count_in_episode):
        """
        Lógica optimizada para uso de acciones expertas.
        
        CAMBIOS CLAVE:
        1. Más ayuda experta al inicio de cada episodio
        2. Reducción más gradual del expert_ratio
        3. Nunca bajar de 15% de ayuda experta
        """
        
        # Factor aleatorio base
        random_factor = np.random.random()
        
        # CAMBIO: Más ayuda al inicio de episodio (primeros 100 steps)
        if step_count_in_episode < 100:
            early_bonus = 0.3  # Aumentado de 0.2 a 0.3
        elif step_count_in_episode < 200:
            early_bonus = 0.15  # NUEVO: Ayuda intermedia
        else:
            early_bonus = 0.0
        
        effective_expert_ratio = min(1.0, self.expert_ratio + early_bonus)
        
        return random_factor < effective_expert_ratio
    
    def should_transition_to_squat(self):
        """
        Lógica MUCHO más conservadora para transición a sentadillas.
        
        CAMBIO CLAVE: Requiere un buffer de episodios estables antes de permitir sentadillas
        """
        
        if self.current_phase == "BALANCE_LEARNING":
            # Verificar umbral de reward
            recent_avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else -10
            
            # NUEVO: Contar episodios estables consecutivos
            if recent_avg_reward >= self.balance_reward_threshold:
                return (self.balance_stable_episodes >= self.required_stable_episodes and 
                        self.phase_episodes >= 50)  # Mínimo 50 episodios total
            else:
                self.balance_stable_episodes = 0  # Reset contador si no es estable
                return False
                
        elif self.current_phase in ["SQUAT_LEARNING", "ADVANCED"]:
            # En fases avanzadas, usar lógica original pero más conservadora
            if len(self.recent_rewards) >= 10:
                last_10_avg = np.mean(list(self.recent_rewards)[-10:])
                return last_10_avg >= self.squat_reward_threshold
        
        return False
    
    def update_after_episode(self, total_episode_reward, episode_length):
        """
        Update optimizado con tracking de estabilidad mejorado
        """
        
        self.episode_count += 1
        self.phase_episodes += 1
        self.recent_rewards.append(total_episode_reward)
        
        # Calcular métricas
        recent_avg = np.mean(self.recent_rewards) if len(self.recent_rewards) >= 5 else total_episode_reward
        
        # NUEVO: Tracking de episodios estables
        if recent_avg >= self.balance_reward_threshold:
            self.balance_stable_episodes += 1
        else:
            self.balance_stable_episodes = max(0, self.balance_stable_episodes - 1)  # Decaimiento gradual
        
        # ===== ACTUALIZAR EXPERT RATIO (más conservador) =====
        
        old_ratio = self.expert_ratio
        
        if recent_avg >= self.balance_reward_threshold:
            # Rendimiento bueno → Reducir dependencia MUY GRADUALMENTE
            self.expert_ratio *= 0.998  # Reducción MUCHO más gradual (0.2% por episodio)
        else:
            # Rendimiento malo → Aumentar ayuda del experto MÁS AGRESIVAMENTE
            self.expert_ratio = min(0.95, self.expert_ratio * 1.02)  # Aumento 2%
        
        # Mantener límites más conservadores
        self.expert_ratio = max(self.min_expert_ratio, min(0.95, self.expert_ratio))
        
        # ===== ACTUALIZAR FASE DEL CURRICULUM (más conservador) =====
        
        old_phase = self.current_phase
        
        if self.current_phase == "BALANCE_LEARNING":
            # CAMBIO: Requerir más episodios y mejor rendimiento
            if (recent_avg >= self.balance_reward_threshold and 
                self.phase_episodes >= 100 and  # Aumentado de 150 a 100
                self.expert_ratio < 0.4 and     # Reducido de 0.5 a 0.4
                self.balance_stable_episodes >= self.required_stable_episodes):
                
                self.current_phase = "SQUAT_LEARNING"
                self.phase_episodes = 0
                self.balance_stable_episodes = 0
                
        elif self.current_phase == "SQUAT_LEARNING":
            if (recent_avg >= self.squat_reward_threshold and 
                self.phase_episodes >= 150 and  # Más episodios para sentadillas
                self.expert_ratio < 0.25):       # Más independencia requerida
                
                self.current_phase = "ADVANCED"
                self.phase_episodes = 0
        
        # ===== LOGGING OPTIMIZADO =====
        
        if self.episode_count % 25 == 0 or old_phase != self.current_phase:  # Más frecuente
            print(f"\n📚 Curriculum Update (Episode {self.episode_count}):")
            print(f"   Phase: {self.current_phase} (episode {self.phase_episodes})")
            print(f"   Expert ratio: {old_ratio:.1%} → {self.expert_ratio:.1%}")
            print(f"   Recent avg reward: {recent_avg:.2f}")
            print(f"   Balance stable episodes: {self.balance_stable_episodes}/{self.required_stable_episodes}")
            
            if old_phase != self.current_phase:
                print(f"   🎉 PHASE TRANSITION: {old_phase} → {self.current_phase}")

# ====================================================================
# INTEGRACIÓN CON EL ENTORNO - CAMBIOS ESPECÍFICOS
# ====================================================================

def integrate_ankle_control_in_environment():
    """
    Integración específica del sistema de control de tobillos
    
    CAMBIOS NECESARIOS EN Simple_BalanceSquat_BipedEnv.py:
    """
    
    integration_code = '''
    # 1. En el método __init__, añadir después de crear otros sistemas:
    
    def __init__(self, render_mode='human', action_space="pam", enable_curriculum=True):
        # ... código existente ...
        
        # NUEVO: Sistema de control inteligente de tobillos
        self.ankle_control = None  # Se inicializa en reset()
        
        # CAMBIO: Usar curriculum optimizado
        if enable_curriculum:
            self.curriculum = OptimizedCurriculumSelector()  # En lugar de CurriculumActionSelector()
        else:
            self.curriculum = None
    
    # 2. En el método reset(), añadir después de crear otros sistemas:
    
    def reset(self, seed=None, options=None):
        # ... código existente hasta crear robot ...
        
        # NUEVO: Inicializar sistema de control de tobillos
        self.ankle_control = IntelligentAnkleControl(self.robot_id)
        
        # ... resto del código existente ...
    
    # 3. En el método step(), modificar la aplicación de torques:
    
    def step(self, action):
        # ... código existente hasta calcular joint_torques ...
        
        # CAMBIO: Aplicar torques con control inteligente de tobillos
        
        # Torques de caderas y rodillas (como antes)
        torque_mapping = [
            (0, joint_torques[0]),  # left_hip_joint
            (1, joint_torques[1]),  # left_knee_joint  
            (3, joint_torques[2]),  # right_hip_joint
            (4, joint_torques[3])   # right_knee_joint
        ]

        for joint_id, torque in torque_mapping:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )
        
        # NUEVO: Control inteligente de tobillos
        if self.ankle_control:
            left_ankle_torque, right_ankle_torque = self.ankle_control.calculate_ankle_torques(
                self.robot_data, 
                self.zmp_calculator
            )
            
            # Aplicar torques de tobillos
            p.setJointMotorControl2(
                self.robot_id, 2,  # left_ankle
                p.TORQUE_CONTROL,
                force=left_ankle_torque
            )
            p.setJointMotorControl2(
                self.robot_id, 5,  # right_ankle  
                p.TORQUE_CONTROL,
                force=right_ankle_torque
            )
        
        # ... resto del código de step() ...
    '''
    
    return integration_code

# ====================================================================
# MÉTRICAS DE DIAGNÓSTICO PARA MONITOREAR CONVERGENCIA
# ====================================================================

class ConvergenceMonitor:
    """
    Sistema de monitoreo para trackear qué tan rápido está convergiendo
    el robot hacia balance estable
    """
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.balance_time_history = []
        self.fall_episodes = []
        
    def record_episode(self, reward, length, balance_time, fell):
        """Registrar resultados de un episodio"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.balance_time_history.append(balance_time)
        
        if fell:
            self.fall_episodes.append(len(self.episode_rewards))
    
    def get_convergence_stats(self, window=25):
        """Obtener estadísticas de convergencia recientes"""
        if len(self.episode_rewards) < window:
            return None
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_balance_times = self.balance_time_history[-window:]
        
        # Contar caídas recientes
        recent_falls = sum(1 for fall_ep in self.fall_episodes 
                          if fall_ep > len(self.episode_rewards) - window)
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'avg_episode_length': np.mean(recent_lengths),
            'avg_balance_time': np.mean(recent_balance_times),
            'fall_rate': recent_falls / window,
            'is_converging': np.mean(recent_rewards) > 2.0 and recent_falls < 3
        }

# ====================================================================
# PARÁMETROS RECOMENDADOS PARA CONVERGENCIA RÁPIDA
# ====================================================================

OPTIMIZED_TRAINING_PARAMS = {
    'curriculum_settings': {
        'initial_expert_ratio': 0.95,
        'min_expert_ratio': 0.15,
        'balance_threshold': 3.0,
        'stable_episodes_required': 25,
        'reduction_rate': 0.998  # Muy gradual
    },
    
    'ankle_control_settings': {
        'base_spring_stiffness': 120.0,
        'base_damping': 8.0,
        'balance_gain': 0.8,
        'max_active_torque': 25.0,
        'activation_threshold': 0.02
    },
    
    'training_expectations': {
        'episodes_to_first_balance': 15,
        'episodes_to_stable_balance': 75,
        'episodes_to_squat_ready': 150,
        'max_acceptable_falls_per_25_episodes': 3
    }
}
