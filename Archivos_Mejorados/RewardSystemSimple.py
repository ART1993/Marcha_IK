# Modificaciones SIMPLIFICADAS para Simple_BalanceSquat_BipedEnv
# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

# ================================================================
# NUEVAS CLASES SIMPLIFICADAS (INSERTAR AL INICIO DEL ARCHIVO)
# Estas reemplazan múltiples archivos complejos anteriores
# ================================================================

class UltraSimpleRewardSystem:
    """
    REEMPLAZA COMPLETAMENTE:
    - Simplified_BalanceSquat_RewardSystem
    - _calculate_context_aware_reward()
    - _calculate_balance_reward() 
    - _calculate_squat_reward()
    - _get_consecutive_balance_time()
    """
    
    def __init__(self, robot_id, plane_id):
        self.robot_id = robot_id
        self.plane_id = plane_id
        
        # Solo 3 parámetros principales (vs 7+ anteriores)
        self.base_reward = 1.0          # Recompensa por estar vivo y de pie
        self.stability_bonus = 2.0       # Bonificación por estar estable
        self.energy_penalty_scale = 0.5  # Escala de penalización energética
        
        print(f"✅ Ultra-Simple Reward System initialized")
    
    def calculate_reward(self, action, current_task="balance"):
        """
        FUNCIÓN ÚNICA que reemplaza TODOS los métodos de recompensa anteriores.
        
        Args:
            action: Array de 6 presiones PAM [0,1]
            current_task: "balance" o "squat" (simple string)
            
        Returns:
            float: Recompensa total (-10 a +10 típicamente)
        """
        
        # ===== OBTENER ESTADO BÁSICO DEL ROBOT =====
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        height = pos[2]
        roll, pitch = euler[0], euler[1]
        
        # ===== COMPONENTE 1: RECOMPENSA BASE (¿Está de pie?) =====
        if height > 0.7:  # Altura mínima razonable
            standing_reward = self.base_reward
        else:
            standing_reward = -5.0  # Penalización fuerte por caída
        
        # ===== COMPONENTE 2: ESTABILIDAD (¿Está equilibrado?) =====
        # Simple: verificar inclinación y contacto con pies
        tilt_penalty = (abs(roll) + abs(pitch)) * 3.0  # Penalizar inclinación
        
        # Verificar contacto con el suelo (ambos pies)
        left_contact = len(p.getContactPoints(self.robot_id, self.plane_id, 2, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.plane_id, 5, -1)) > 0
        
        if left_contact and right_contact:
            contact_bonus = self.stability_bonus  # Ambos pies → estable
        elif left_contact or right_contact:
            contact_bonus = 0.5  # Un pie → parcialmente estable
        else:
            contact_bonus = -3.0  # Sin contacto → inestable
        
        stability_reward = contact_bonus - tilt_penalty
        
        # ===== COMPONENTE 3: PROGRESO DE TAREA (¿Está haciendo lo correcto?) =====
        if current_task == "balance":
            # Para balance: recompensar quietud (velocidades bajas)
            lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
            movement_magnitude = np.linalg.norm(lin_vel) + np.linalg.norm(ang_vel)
            task_reward = max(0, 1.0 - movement_magnitude)  # Menos movimiento = mejor
            
        elif current_task == "squat":
            # Para sentadilla: recompensar flexión controlada de caderas
            joint_states = p.getJointStates(self.robot_id, [0, 3])  # caderas
            hip_flexion = (joint_states[0][0] + joint_states[1][0]) / 2.0
            
            if 0.2 < hip_flexion < 1.0:  # Rango de flexión bueno para sentadilla
                task_reward = 1.5  # Bonificación por estar en posición de sentadilla
            else:
                task_reward = 0.0  # Neutral si no está en posición correcta
        else:
            task_reward = 0.0
        
        # ===== COMPONENTE 4: EFICIENCIA ENERGÉTICA (¿Está desperdiciando energía?) =====
        # Simple: penalizar activación PAM total excesiva
        total_activation = np.sum(action)
        if total_activation > 3.0:  # Más de la mitad de los PAMs muy activos
            energy_penalty = (total_activation - 3.0) * self.energy_penalty_scale
        else:
            energy_penalty = 0.0  # No penalizar activación razonable
        
        # ===== SUMA FINAL =====
        total_reward = standing_reward + stability_reward + task_reward - energy_penalty
        
        # Limitar rango para predictibilidad
        total_reward = np.clip(total_reward, -10.0, 10.0)
        
        return total_reward
    
    def is_episode_done(self, step_count=0, frecuency_simulation=1500.0):
        """
        REEMPLAZA COMPLETAMENTE:
        - _is_done()
        - _is_done_with_context()
        - Y todos los otros métodos de terminación complejos
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Caída crítica (altura muy baja)
        if pos[2] < 0.4:
            return True
            
        # Inclinación crítica (>60 grados ≈ 1 radián)
        if abs(euler[0]) > 1.0 or abs(euler[1]) > 1.0:
            return True
        
        if step_count> frecuency_simulation*8:
            return True
        
        return False


class SimpleActionType(Enum):
    """
    REEMPLAZA: ActionType del DiscreteActionController
    Pero mucho más simple - solo las acciones esenciales
    """
    BALANCE = "balance"
    SQUAT = "squat"


class UltraSimpleActionSelector:
    """
    REEMPLAZA COMPLETAMENTE:
    - OptimizedCurriculumSelector
    - PhaseAwareEnhancedController  
    - DiscreteActionController (en gran medida)
    - Todo el sistema de fases y correcciones posturales
    """
    
    def __init__(self, env):
        self.env = env
        self.episode_count = 0
        
        # ===== CONFIGURACIÓN SIMPLE DEL CURRICULUM =====
        # Solo 2 parámetros que importan realmente
        self.expert_help_ratio = 0.8  # Comenzar con 80% ayuda experta
        self.min_expert_help = 0.1    # Nunca bajar de 10% ayuda
        
        # ===== PATRONES DE ACCIÓN SIMPLIFICADOS =====
        # En lugar de fases complejas, solo patrones base simples
        self.action_patterns = {
            SimpleActionType.BALANCE: {
                # Presiones PAM para balance estático
                'pam_pressures': [0.4, 0.5, 0.4, 0.5, 0.1, 0.1],  # [hip_fl, hip_ex, hip_fl, hip_ex, knee_fl, knee_fl]
                'description': 'Equilibrio estático de pie'
            },
            SimpleActionType.SQUAT: {
                # Presiones PAM que promueven sentadilla
                'pam_pressures': [0.6, 0.3, 0.6, 0.3, 0.3, 0.3],  # Más flexión, menos extensión
                'description': 'Posición de sentadilla'
            }
        }
        
        # ===== ESTADO INTERNO MÍNIMO =====
        self.current_action = SimpleActionType.BALANCE  # Empezar siempre con balance
        self.last_10_rewards = []  # Solo tracking básico de progreso
        
        print(f"✅ Ultra-Simple Action Selector initialized")
        print(f"   Starting expert help: {self.expert_help_ratio:.0%}")
        print(f"   Available actions: {[action.value for action in SimpleActionType]}")
    
    def should_use_expert_action(self):
        """
        REEMPLAZA COMPLETAMENTE:
        - curriculum.should_use_expert_action(step_count_in_episode)
        - Toda la lógica compleja con early_bonus, phase_episodes, etc.
        """
        # Decisión basada solo en probabilidad actual + un poco de ruido para exploración
        random_roll = np.random.random()
        
        # Pequeña bonificación de ayuda al inicio del episodio (primeros 100 steps)
        if self.env.step_count < 100:
            effective_ratio = min(1.0, self.expert_help_ratio + 0.2)
        else:
            effective_ratio = self.expert_help_ratio
        
        return random_roll < effective_ratio
    
    def get_expert_action(self):
        """
        REEMPLAZA COMPLETAMENTE:
        - controller.get_expert_action(time_step)
        - Todo el sistema de PhaseAwareEnhancedController
        - Los patrones complejos por fase
        - Las correcciones posturales dinámicas
        """
        # Obtener patrón base para la acción actual
        pattern = self.action_patterns[self.current_action]
        base_pressures = np.array(pattern['pam_pressures'])
        
        # Añadir pequeña variación natural (±5%)
        noise = np.random.normal(0, 0.05, size=6)
        varied_pressures = base_pressures + noise
        
        # Asegurar límites [0, 1]
        final_pressures = np.clip(varied_pressures, 0.0, 1.0)
        
        return final_pressures
    
    def decide_current_action(self):
        """
        REEMPLAZA COMPLETAMENTE:
        - curriculum.should_transition_to_squat()
        - detect_current_phase()
        - Toda la lógica de transición de fases
        """
        # Si no tenemos suficiente historial, mantener balance
        if len(self.last_10_rewards) < 5:
            self.current_action = SimpleActionType.BALANCE
            return
        
        # Lógica simple: si está yendo bien con balance, probar sentadilla ocasionalmente
        recent_avg = np.mean(self.last_10_rewards[-5:])
        
        if recent_avg > 2.0:  # Performance decente
            # 20% chance de intentar sentadilla
            if np.random.random() < 0.2:
                self.current_action = SimpleActionType.SQUAT
            else:
                self.current_action = SimpleActionType.BALANCE
        else:
            # Performance malo, focus en balance
            self.current_action = SimpleActionType.BALANCE
    
    def update_after_step(self, reward):
        """
        REEMPLAZA COMPLETAMENTE:
        - Todo el tracking complejo de fases y métricas múltiples
        - phase_performance updates
        - postural_system updates
        """
        # Solo tracking básico que importa
        self.last_10_rewards.append(reward)
        if len(self.last_10_rewards) > 10:
            self.last_10_rewards.pop(0)
        
        # Decidir acción para próximo step
        self.decide_current_action()
    
    def update_after_episode(self, total_episode_reward):
        """
        REEMPLAZA COMPLETAMENTE:
        - curriculum.update_after_episode(reward, episode_length)
        - Toda la lógica compleja del curriculum
        - balance_stable_episodes tracking
        - phase_episodes tracking
        """
        self.episode_count += 1
        
        # Lógica de curriculum ULTRA-SIMPLE
        if total_episode_reward > 50:  # Episodio exitoso
            # Reducir ayuda experta gradualmente
            self.expert_help_ratio *= 0.95  # 5% reducción
        else:  # Episodio problemático
            # Aumentar ayuda experta
            self.expert_help_ratio = min(0.9, self.expert_help_ratio * 1.05)
        
        # Mantener límites
        self.expert_help_ratio = max(self.min_expert_help, self.expert_help_ratio)
        
        # Reset para nuevo episodio
        self.current_action = SimpleActionType.BALANCE
        
        # Log cada 25 episodios
        if self.episode_count % 25 == 0:
            avg_reward = np.mean(self.last_10_rewards) if self.last_10_rewards else 0
            print(f"Episode {self.episode_count}: Expert help {self.expert_help_ratio:.1%}, "
                  f"Avg reward: {avg_reward:.1f}")