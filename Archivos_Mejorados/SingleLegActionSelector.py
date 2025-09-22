import pybullet as p
import numpy as np

from collections import deque

from Archivos_Apoyo.simple_log_redirect import log_print, both_print
from Archivos_Mejorados.AngleBasedExpertController import AngleBasedExpertController
from Archivos_Mejorados.RewardSystemSimple import SingleLegActionType



class SingleLegActionSelector:
    """
        Selector de acciones expertas
        PAM 0: left_hip_flexor
        PAM 1: left_hip_extensor
        PAM 2: right_hip_flexor
        PAM 3: right_hip_extensor
        PAM 4: left_knee_flexor
        PAM 5: right_knee_flexor

    """
    
    def __init__(self, env):
        self.env = env
        self.episode_count = 0
        
        # ===== PARÁMETROS DE CURRICULUM =====
        if hasattr(env, 'enable_curriculum') and not env.enable_curriculum:
            # MODO SIN CURRICULUM: RL puro
            self.expert_help_ratio = 0.0  # ⭐ CLAVE: Sin ayuda experta
            self.min_expert_help = 0.0
            self.curriculum_enabled = False
            log_print(f"🎯 Action Selector: CURRICULUM DISABLED - Pure RL mode")
        else:
            # MODO CON CURRICULUM: comportamiento normal
            self.expert_help_ratio = 0.85
            self.min_expert_help = 0.0
            self.curriculum_enabled = True
            log_print(f"🎯 Action Selector: CURRICULUM ENABLED - Expert help starts at {self.expert_help_ratio:.1%}")

        # Solo crear si realmente necesitamos acciones expertas
        # ===== NUEVO: CONTROLADOR BASADO EN ÁNGULOS =====
        if self.expert_help_ratio > 0.0:
            self.angle_controller = AngleBasedExpertController(env)
            log_print(f"   Control method: Target angles → PD torques → PAM pressures")
        else:
            self.angle_controller = None
            log_print(f"   Control method: Pure RL (no expert controller needed)")
        
        
        # Musculos PAM McKibben usarlos para generar presiones y angulos
        self.pam_muscle=env.pam_muscles
        
        # ===== ESTADO INTERNO =====
        self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
        self.last_10_rewards = deque(maxlen=10)
        self.time_in_current_stance = 0
        self.target_switch_time = env.switch_interval
        frequency = env.frequency_simulation
        switch_time_seconds = self.target_switch_time / frequency
        print(f"🤖 Action Selector synchronized:")
        print(f"   Switch interval: {self.target_switch_time} steps ({switch_time_seconds:.1f}s)")
        
        log_print(f"✅ Angle-Based Single Leg Action Selector initialized")
        log_print(f"   Control method: Target angles → PD torques → PAM pressures")
        log_print(f"   Leg raise angle: 40° (0.7 rad)")
    
    def get_expert_action(self):
        """
        NUEVO: Obtener acción experta usando control basado en ángulos objetivo
        
        Proceso:
        1. Definir ángulos objetivo según tarea actual
        2. Calcular torques PD necesarios para alcanzar esos ángulos
        3. Convertir torques a presiones PAM equivalentes
        4. Añadir pequeñas correcciones por estabilidad
        
        Returns:
            numpy.array: Presiones PAM normalizadas [0,1]
        """
        level = self.env.simple_reward_system.level
        if level == 1:
            target_angles = self.angle_controller.target_angles['level_1_balance']
        elif level == 2:
            target_angles = self.angle_controller.target_angles['level_2_balance']
        else:
            # 🔗 Sincroniza con la pierna objetivo del sistema de recompensas
            if hasattr(self.env, "simple_reward_system"):
                self.current_action = (SingleLegActionType.BALANCE_LEFT_SUPPORT
                                    if self.env.simple_reward_system.target_leg == 'right'
                                else SingleLegActionType.BALANCE_RIGHT_SUPPORT)
            # PASO 1: Obtener ángulos objetivo
            target_angles = self.angle_controller.get_target_angles_for_task(self.current_action)
        # PASO 1: Obtener ángulos objetivo
        #target_angles = self.angle_controller.get_target_angles_for_task(self.current_action)
        
        # PASO 2: Calcular torques PD
        pd_torques = self.angle_controller.calculate_pd_torques(target_angles)
        
        # PASO 3: Convertir a presiones PAM
        base_pressures = self.angle_controller.torques_to_pam_pressures(pd_torques, target_angles)
        
        # PASO 4: Añadir correcciones por estabilidad (opcional)
        corrected_pressures = self._add_stability_corrections(base_pressures)
        
        # PASO 5: Variación natural pequeña
        noise = np.random.normal(0, 0.02, size=self.env.num_active_pams)
        final_pressures = corrected_pressures + noise
        
        return np.clip(final_pressures, 0.0, 1.0)
    
    def _add_stability_corrections(self, base_pressures):
        """
        Añadir correcciones pequeñas por inclinación para mayor estabilidad
        """
        
        # Obtener orientación actual
        pos, orn = p.getBasePositionAndOrientation(self.env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch = euler[0], euler[1]
        
        corrected = base_pressures.copy()

        # Ganancias muy pequeñas para no dominar la política
        k_pitch = 0.10   # corrige inclinación hacia delante/atrás
        k_roll  = 0.08   # corrige caída lateral

        if self.current_action == SingleLegActionType.BALANCE_LEFT_SUPPORT:
            corrected[1] += k_pitch * max(pitch, 0.0)  # L_hip_ext ↑
            corrected[0] -= k_pitch * max(pitch, 0.0)  # L_hip_flex ↓
        elif self.current_action == SingleLegActionType.BALANCE_RIGHT_SUPPORT:
            corrected[3] += k_pitch * max(pitch, 0.0)  # R_hip_ext ↑
            corrected[2] -= k_pitch * max(pitch, 0.0)  # R_hip_flex ↓

        # Roll: empuja a “plantar” más la cadera del lado que sube
        corrected[0] += k_roll * max(-roll, 0.0)  # roll<0 → flexión cadera izq
        corrected[2] += k_roll * max( roll, 0.0)  # roll>0 → flexión cadera dcha
        
        return np.clip(corrected, 0.0, 1.0)
    
    def decide_current_action(self):
        """Decidir qué acción tomar basado en el contexto"""
        
        self.time_in_current_stance += 1
        
        # Si no hay suficiente historial, mantener acción actual
        if len(self.last_10_rewards) < 5:
            return
        
        recent_performance = np.mean(list(self.last_10_rewards)[-3:])

        if self.time_in_current_stance >= self.target_switch_time:
            if self.current_action == SingleLegActionType.BALANCE_LEFT_SUPPORT:
                self.current_action = SingleLegActionType.BALANCE_RIGHT_SUPPORT
            else:
                self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
            self.time_in_current_stance = 0
        
        # ===== LÓGICA DE CAMBIO DE PIERNA =====
        
        # Si el rendimiento es malo, volver a posición estable
        elif recent_performance < 1.0:
            if self.current_action not in [SingleLegActionType.BALANCE_LEFT_SUPPORT, 
                                          SingleLegActionType.BALANCE_RIGHT_SUPPORT]:
                self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
                self.time_in_current_stance = 0
    
    def update_after_step(self, reward):
        """Actualizar después de cada step"""
        self.last_10_rewards.append(reward)
        self.decide_current_action()
    
    def on_episode_end(self, total_episode_reward):
        """Actualizar después de cada episodio"""
        self.episode_count += 1
        log_print(f"📊 Episode {self.episode_count} ended with total reward: {total_episode_reward:.1f}")
        if self.curriculum_enabled:
            # Curriculum más conservador para tarea difícil
            if total_episode_reward > 80:  # Episodio muy exitoso
                self.expert_help_ratio *= 0.90  # Reducción más gradual
            elif total_episode_reward > 40:  # Episodio moderadamente exitoso
                self.expert_help_ratio *= 0.95
            else:  # Episodio problemático
                self.expert_help_ratio = min(0.97, self.expert_help_ratio * 1.02)
            
            self.expert_help_ratio = max(self.min_expert_help, self.expert_help_ratio)
            log_print(f"   Expert help ratio updated to: {self.expert_help_ratio:.1%}")
        else:
            log_print(f"   Pure RL mode - no expert help adjustment")
        
        # Reset para nuevo episodio
        self.current_action = SingleLegActionType.BALANCE_LEFT_SUPPORT
        self.time_in_current_stance = 0