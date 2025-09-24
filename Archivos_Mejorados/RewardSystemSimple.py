# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

from collections import deque

from Archivos_Apoyo.simple_log_redirect import log_print, both_print
from Archivos_Apoyo.Configuraciones_adicionales import split_cocontraction_torque_neutral

class SingleLegActionType(Enum):
    """Acciones para equilibrio en una pierna"""
    BALANCE_LEFT_SUPPORT = "balance_left_support"    # Equilibrio con pie izquierdo
    BALANCE_RIGHT_SUPPORT = "balance_right_support"  # Equilibrio con pie derecho
    # Lo dejo en caso de que encuentre una forma más explicita de realizar la transición
    TRANSITION = "transition"              # Transición entre piernas
    TRANSITION_TO_LEFT = "transition_to_left"        # Transición hacia apoyo izquierdo
    TRANSITION_TO_RIGHT = "transition_to_right"      # Transición hacia apoyo derecho
    

# =============================================================================
# SISTEMA DE RECOMPENSAS PROGRESIVO SIMPLE
# Solo 3 niveles, fácil de entender y modificar
# =============================================================================
    
class SimpleProgressiveReward:
    """
    Sistema súper simple: 3 niveles que van aumentando la dificultad y las recompensas
    
    NIVEL 1: Solo mantenerse de pie (recompensas pequeñas 0-3) (0-15 episodios)
    NIVEL 2: Balance estable (recompensas medias 0-5)  (15-40 episodios)
    NIVEL 3: Levantar piernas (recompensas altas 0-8) (40+ episodios)
    """
    
    def __init__(self, env):
        self.env=env
        self.frequency_simulation = env.frequency_simulation
        self.switch_interval = env.switch_interval
        self.enable_curriculum = env.enable_curriculum
        self.robot_id = env.robot_id
        self.single_support_ticks = 0

        if self.enable_curriculum==False:
            # MODO SIN CURRICULUM: sistema fijo y permisivo
            self.level = 3  # Siempre nivel 3
            self.level_progression_disabled = True
            both_print(f"🎯 Progressive System: CURRICULUM DISABLED")
            both_print(f"   Mode: Fixed basic balance (Level max only)")
        else:
            # MODO CON CURRICULUM: comportamiento normal
            self.level = 1
            self.level_progression_disabled = False
            both_print(f"🎯 Progressive System: CURRICULUM ENABLED")
            both_print(f"   Mode: Level progression 1→2→3")
        
        # ===== CONFIGURACIÓN SUPER SIMPLE =====
        self.episode_count = 0
        self.recent_episodes = deque(maxlen=5)  # Últimos 5 episodios
        self.success_streak = 0  # Episodios consecutivos exitosos
        self._no_contact_steps = 0
        
        # Configuración por nivel (muy simple)
        # Configuración más graduada
        self.level_config = {
            1: {
                'description': 'Supervivencia básica',
                'max_reward': 2.0,
                'success_threshold': 1.0,    # Reward mínimo para considerar éxito
                'episodes_needed': 5,       # Episodios mínimos en este nivel
                'success_streak_needed': 3   # Episodios consecutivos exitosos para subir
            },
            2: {
                'description': 'Balance estable',
                'max_reward': 4.0,
                'success_threshold': 2.5,
                'episodes_needed': 15,
                'success_streak_needed': 4
            },
            3: {
                'description': 'Levantar piernas alternando',
                'max_reward': 7.0,
                'success_threshold': 999,    # Nivel final
                'episodes_needed': 999,
                'success_streak_needed': 999
            }
        }

        # ===== ÁNGULOS OBJETIVO SEGÚN TAREA =====
        self.target_angles = {
            # NIVEL 1: Solo balance básico
            "level_3_left_support":  {"left_hip_roll":  0.0,"left_hip_pitch":0.0, "left_knee": 0.0, "right_hip_roll":  0.6, "right_hip_pitch":0.0,"right_knee": 0.6},
            "level_3_right_support": {"left_hip_roll": 0.6, "left_hip_pitch":0.0, "left_knee": 0.6, "right_hip_roll": 0.0, "right_hip_pitch":0.0,"right_knee": 0.0},
        }

        # Inclinación crítica - MÁS PERMISIVO según nivel
        if self.enable_curriculum==False:
            self.max_tilt_by_level = {
                1: 0.8,  # 
                2: 0.8,  # 
                3: 0.8   # 
            }
            both_print(f"   Max tilt: {np.degrees(self.max_tilt_by_level[3]):.1f}° (permisivo)")
        else:
            self.max_tilt_by_level = {
                1: 0.8,  #  - muy permisivo para aprender básicos
                2: 0.7,  #  - moderadamente permisivo  
                3: 0.5   #  - estricto para habilidades avanzadas
            }
        
        # Para alternancia de piernas (solo nivel 3)
        self.target_leg = 'left'
        self.switch_timer = 0
        # self.leg_switch_bonus = 0.0  # Bonus por cambio exitoso

        # Debug para confirmar configuración
        switch_time_seconds = self.switch_interval / self.frequency_simulation
        both_print(f"🎯 Progressive System initialized:")
        both_print(f"   Switch interval: {self.switch_interval} steps ({switch_time_seconds:.1f}s)")
        both_print(f"   Frequency: {self.frequency_simulation} Hz")
        both_print(f"🎯 Simple Progressive System: Starting at Level {self.level}")
    
    def calculate_reward(self, action, step_count):
        """
        Método principal: calcula reward según el nivel actual
        """

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        if self.level == 1 and self.enable_curriculum:
            reward = self._level_1_reward(pos,euler)      # Solo supervivencia
        elif self.level == 2 and self.enable_curriculum:
            reward = self._level_2_reward(pos, euler)      # + balance estable
            #stability, r_score, p_score = _stability_terms_from_euler(euler)
            #reward += 1.2 * stability
        else:  # level == 3
            reward = self._level_3_reward(pos, euler, step_count)  # + levantar piernas
            #stability, r_score, p_score = _stability_terms_from_euler(euler)
            #reward += 1.6 * stability
        level = self.level if self.enable_curriculum else 3
        max_reward = self.level_config[level]['max_reward']
        
        # Limitar reward según nivel
        
        
        return max(-2.0, min(reward, max_reward))
    
    def _level_1_reward(self,pos,euler):
        """NIVEL 1: Solo mantenerse de pie (recompensas 0-3)"""
        self.dx = float(pos[0] - self.env.init_pos[0])
        # Tolerancia sin penalización ±5 cm
        tol = 0.05
        # Penaliza deriva total fuera de tolerancia (suave; tope aprox -2.0)
        drift_pen = - np.clip(abs(self.dx) - tol, 0.0, 0.25) * 8.0
        # Penaliza adicionalmente cuando la deriva es hacia atrás (dx < -tol)
        # tope aprox -1.6
        back_only_pen = - np.clip(-self.dx - tol, 0.0, 0.20) * 8.0

        height = pos[2]
        
        # Recompensa simple por altura
        if height > 0.9:
            height_reward= 1.5  # Buena altura
        elif height > 0.8:
            height_reward= 0.8  # Altura mínima
        else:
            height_reward= -1.0  # Caída

        pitch = euler[1]
        back_pitch_pen = - np.clip(pitch - 0.05, 0.0, 0.30) * 6.0

        return height_reward + drift_pen + back_only_pen +back_pitch_pen
    
    def _level_2_reward(self,pos,euler):
        """NIVEL 2: Balance estable (recompensas 0-5)"""
        
        height_reward=self._level_1_reward(pos, euler)
        
        # + Recompensa por estabilidad (NUEVA)
        # tilt = abs(euler[0]) + abs(euler[1])  # roll + pitch
        # if tilt < 0.2:
        #     stability_reward = 1.5  # Muy estable
        # elif tilt < 0.4:
        #     stability_reward = 0.5  # Moderadamente estable
        # else:
        #     stability_reward = -0.5  # Inestable
        
        # Pitch (erguido)
        
        pitch_pen = - (abs(euler[1]) / np.deg2rad(30)) * (0.5 / self.frequency_simulation)

        # Roll con zona muerta
        roll_soft = np.deg2rad(8)
        roll_excess = max(0.0, abs(euler[0]) - roll_soft)
        roll_pen = - (roll_excess / np.deg2rad(20)) * (0.5 / self.frequency_simulation)
        
        return height_reward +  pitch_pen + roll_pen
    
    def _level_3_reward(self,pos,euler, step_count):
        """NIVEL 3: Levantar piernas alternando (recompensas 0-8)"""
        
        # Recompensa base (igual que nivel 2)
        base_reward = self._level_2_reward(pos,euler)
        if base_reward < 0:  # Si se cayó, no calcular más
            return base_reward
        
        # + Recompensa por levantar pierna (NUEVA)
        leg_reward = self._calculate_leg_reward(step_count)
        
        return base_reward + leg_reward
    
    def _calculate_leg_reward(self, step_count):
        """Calcular recompensa por levantar pierna correctamente"""

        left_foot_id=self.env.left_foot_link_id
        right_foot_id=self.env.right_foot_link_id
        F_L = self.env.contact_normal_force(left_foot_id)
        F_R = self.env.contact_normal_force(right_foot_id)
        F_sum = max(F_L + F_R, 1e-6)
        left_hip_roll_id, left_hip_pitch_id, left_knee_id, right_hip_roll_id, right_hip_pitch_id, right_knee_id = self.env.joint_indices
        min_F=20
        # Cambiar pierna cada switch interval
        self.switch_timer += 1
        if self.switch_timer >= self.switch_interval:
            self.target_leg = 'left' if self.target_leg == 'right' else 'right'
            self.switch_timer = 0
            # DEBUG MEJORADO: Mostrar tiempo real además de steps
            seconds_per_switch = self.switch_interval / self.frequency_simulation  
            # Asumiendo 400 Hz
            log_print(f"🔄 Target: Raise {self.target_leg} leg (every {seconds_per_switch:.1f}s)")
        
        # Detectar qué pies están en contacto Ver si seleccionar min_F=20 0 27 0 30
        left_down = self.env.contact_with_force(left_foot_id, min_F=min_F)
        right_down = self.env.contact_with_force(right_foot_id, min_F=min_F)

        # === Bonus por 'usar' roll para recentrar COM sobre el soporte ===
        support_sign = +1.0 if (left_down and not right_down) else (-1.0 if (right_down and not left_down) else 0.0)
        torso_roll = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])[0]
        hiproll_align_bonus = 0.0
        if support_sign != 0.0:
            hiproll_align_bonus = np.clip(support_sign * torso_roll / np.deg2rad(10), -1.0, 1.0) * (0.3 / self.frequency_simulation)

        target_is_right   = (self.target_leg == 'right')
        target_foot_id    = right_foot_id if target_is_right else left_foot_id
        target_foot_down  = right_down if target_is_right else left_down
        support_foot_down = left_down if target_is_right else right_down


        # ======== SHAPING POR CARGAS (romper toe-touch y cargar el soporte) ========
        # Cargas del pie de SOPORTE (esperado) y del pie OBJETIVO (debe ir al aire)
        F_sup = F_L if target_is_right else F_R   # si objetivo=right, soporte=left (F_L)
        F_tar = F_R if target_is_right else F_L

        lpos = p.getLinkState(self.robot_id, left_foot_id)[0]
        rpos = p.getLinkState(self.robot_id, right_foot_id)[0]
        ml = abs((rpos[1] - lpos[1]) if target_is_right else (lpos[1] - rpos[1]))  # eje Y
        ml_min = 0.08  # 8 cm de separación deseable
        support_foot_down = left_down if target_is_right else right_down
        midline_pen = 0.0 if (not support_foot_down) else -np.clip((ml_min - ml)/ml_min, 0.0, 1.0) * 0.6

        # (1) Penaliza doble apoyo fuerte
        both_down_pen = -1.0 if (F_L >= min_F and F_R >= min_F) else 0.0

        # (2) Penaliza toe-touch (1–30 N) del pie objetivo
        toe_touch_pen = -0.6 if (0.0 < F_tar < min_F) else 0.0

        # (3) Recompensa reparto de carga sano: ≥80% en el pie de soporte
        ratio = F_sup / F_sum
        support_load_reward = np.clip((ratio - 0.80) / 0.20, 0.0, 1.0) * 1.0

        # (4) Bonus por tiempo en apoyo simple sostenido (pie objetivo en aire “limpio”)
        if (F_sup >= min_F) and (F_tar < 1.0):
            # acumula ticks (~400 Hz ⇒ 0.30 s ≈ 120 ticks)
            self.single_support_ticks += 1
            ss_step = 0.05/ self.frequency_simulation
            ss_terminal = 0.5 if self.single_support_ticks == int(0.30 * self.frequency_simulation) else 0.0
        else:
            self.single_support_ticks = 0
            ss_step = 0.0
            ss_terminal = 0.0

        # --- Bonuses de forma SOLO si el pie objetivo NO está en contacto ---
        # Clearance
        foot_z = p.getLinkState(self.robot_id, target_foot_id)[0][2]
        clearance_target = 0.09  # 9 cm
        clearance_bonus = 0.0 if target_foot_down else np.clip(foot_z / clearance_target, 0.0, 1.0) * 1.5

        # Rodilla (≈0.6 rad)
        knee_id  = right_knee_id if target_is_right else left_knee_id
        knee_ang = p.getJointState(self.robot_id, knee_id)[0]
        knee_bonus = (1.0 - min(abs(knee_ang - 0.6), 1.0)) * 1.0
        knee_bonus = 0.0 if target_foot_down else knee_bonus

        # Cadera (≈|0.6| rad) — uso el módulo para no depender del signo
        hip_id  = right_hip_roll_id if target_is_right else left_hip_roll_id
        hip_ang = p.getJointState(self.robot_id, hip_id)[0]
        hip_bonus = (1.0 - min(abs(abs(hip_ang) - 0.6), 1.0)) * 0.7
        hip_bonus = 0.0 if target_foot_down else hip_bonus

        # Gating de bonos de forma: solo si has transferido suficiente carga al pie de soporte
        if ratio < 0.70:
            clearance_bonus = 0.0
            knee_bonus = 0.0
            hip_bonus = 0.0

        # Evaluar si está haciendo lo correcto
        if self.target_leg == 'right':
            # Quiero: pie izquierdo abajo, pie derecho arriba
            if left_down and not right_down:
                contacto_reward = 2.0  # ¡Perfecto!
            elif left_down and right_down:
                contacto_reward = 0.5  # Ambos abajo (transición)
            else:
                contacto_reward = -0.5  # Incorrecto
        else:  # target_leg == 'left'
            # Quiero: pie derecho abajo, pie izquierdo arriba
            if right_down and not left_down:
                contacto_reward = 2.0  # ¡Perfecto!
            elif left_down and right_down:
                contacto_reward = 0.5  # Ambos abajo (transición)
            else:
                contacto_reward = -0.5  # Incorrecto

        # Suma total
        shaping = both_down_pen + toe_touch_pen + support_load_reward + ss_step + ss_terminal
        leg_reward = contacto_reward + clearance_bonus + knee_bonus + hip_bonus + shaping + hiproll_align_bonus + midline_pen
        return leg_reward
    
    def update_after_episode(self, episode_reward, success=None):
        """Actualizar nivel después de cada episodio"""
        
        self.episode_count += 1
        self.recent_episodes.append(episode_reward)
        has_fallen = (self.last_done_reason in ("fall", "tilt", "drift"))
        
        # Mantener solo últimos 5 episodios
        if len(self.recent_episodes) > 5:
            self.recent_episodes.pop(0)

        # Determinar éxito si no te lo pasan explícitamente
        cfg = self.level_config[self.level]

        if success is None:
            # Éxito si supera umbral y no hubo caída
            success = (episode_reward >= cfg['success_threshold']) and (not has_fallen)
        log_print(f"{self.level_progression_disabled=:}, {self.enable_curriculum=:}")
        # Verificar si subir de nivel
        if self.level_progression_disabled is False:  # Necesitamos al menos 5 episodios
            #avg_reward = sum(self.recent_episodes) / len(self.recent_episodes)
            #config = self.level_config[self.level]
            # Actualizar racha
            if len(self.recent_episodes) >= 5:
                self.success_streak = self.success_streak + 1 if success else 0

                # (Opcional) logging
                both_print(f"🏁 Episode {self.episode_count}: "
                        f"reward={episode_reward:.1f} | success={success} | "
                        f"streak={self.success_streak}/{cfg['success_streak_needed']}")
                
                # Promoción de nivel si cumple racha y episodios mínimos
                if (self.success_streak >= cfg['success_streak_needed']
                    and self.episode_count >= cfg['episodes_needed']
                    and self.level < 3):
                    old = self.level
                    self.level += 1
                    self.success_streak = 0
                    both_print(f"🎉 LEVEL UP! {old} → {self.level}")
        else:
            # MODO SIN CURRICULUM: solo logging básico
            both_print(f"🏁 Episode {self.episode_count}: "
                    f"reward={episode_reward:.1f} | success={success} | "
                    f"fixed_level=3")
    
    def is_episode_done(self, step_count, testeo_movimiento):
        """Criterios simples de terminación"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Caída
        #if testeo_movimiento==False:
        if pos[2] <= 0.5:
            self.last_done_reason = "fall"
            log_print("❌ Episode done: Robot fell")
            return True
        
        if abs(self.dx) > 0.35:
            self.last_done_reason = "drift"
            log_print("❌ Episode done: Excessive longitudinal drift")
            return True
        
        max_tilt = self.max_tilt_by_level.get(self.level, 0.5)
        # Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            self.last_done_reason = "tilt"
            log_print("❌ Episode done: Robot tilted too much")
            return True
        
        # dentro de is_episode_done(...) tras calcular contactos:
        left_on, right_on = self.env.contacto_pies   # si ya tienes util; si no, usa getContactPoints
        if not (left_on or right_on):
            self._no_contact_steps += 1
        else:
            self._no_contact_steps = 0
        if self._no_contact_steps >= int(0.20 * self.frequency_simulation):  # 0.2 s
            self.last_done_reason = "no_support"
            log_print("❌ Episode done: No foot support for too long")
            return True
        
        # Tiempo máximo (crece con nivel)
        max_steps = (200 + ((self.level-1) * 200))*10 if self.enable_curriculum else 6000 # 2000, 4000, 6000 steps
        if step_count >= max_steps:
            self.last_done_reason = "time"
            log_print("⏰ Episode done: Max time reached")
            return True
        
        self.last_done_reason = None
        
        return False
    
    def get_info(self):
        """Info para debugging"""
        avg_reward = sum(self.recent_episodes) / len(self.recent_episodes) if self.recent_episodes else 0
        
        return {
            'level': self.level,
            'episodes': self.episode_count,
            'avg_reward': avg_reward,
            'target_leg': self.target_leg if self.level == 3 else None,
            'max_reward': self.level_config[self.level]['max_reward'],
            'curriculum_enabled': self.enable_curriculum,
            'level_progression_disabled': getattr(self, 'level_progression_disabled', False)
        }
    
    # ======================================================================================================================================================================= #
    # ================================================Recompensas adicionales================================================================================================ #
    # ======================================================================================================================================================================= #

    # --- helpers de estabilidad postural ---
def _band_reward(x, tol_ok, tol_max):
    """
    Devuelve 1.0 si |x|<=tol_ok, cae linealmente hasta 0 en |x|=tol_max
    y <0 (penaliza) de forma suave más allá (hasta -1 con cola cuadrática).
    """
    ax = abs(x)
    if ax <= tol_ok:
        return 1.0
    if ax <= tol_max:
        return 1.0 - (ax - tol_ok) / (tol_max - tol_ok)
    # cola suave: cuadrática, acotada a [-1,0)
    over = (ax - tol_max) / max(1e-6, tol_max)
    return -min(1.0, over * over)

def _stability_terms_from_euler(euler):
    roll, pitch, _ = euler
    # tolerancias en radianes (≈5° ok, 10° margen):
    tol_ok   = np.deg2rad(5.0)
    tol_max  = np.deg2rad(10.0)
    r_score  = _band_reward(roll,  tol_ok, tol_max)  # [-1..1]
    p_score  = _band_reward(pitch, tol_ok, tol_max)  # [-1..1]
    # peso ligeramente mayor a roll para single-support
    stability = 0.6*r_score + 0.4*p_score
    return stability, r_score, p_score
