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
        # --- Parámetros configurables (puedes sobreescribirlos desde env) ---
        # Cadera de la pierna en el aire (roll absoluto). Recomendado 0.3–0.5
        self.swing_hip_target  = float(getattr(env, "swing_hip_target", 0.10))
        # Ventana suave para cadera y rodilla (ancho de tolerancia)
        self.swing_hip_tol     = float(getattr(env, "swing_hip_tol",  0.10))  # ±0.10 rad
        # Rodilla en el aire: rango recomendado 0.45–0.75
        self.swing_knee_lo     = float(getattr(env, "swing_knee_lo",  0.45))
        self.swing_knee_hi     = float(getattr(env, "swing_knee_hi",  0.75)) 
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
        self.contact_both = 0
        
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
        # Objetivos "blandos" (solo orientativos para control experto/PD)
        # ===== ÁNGULOS OBJETIVO SEGÚN TAREA =====
        self.target_angles = {
            "level_3_left_support":  {
                "left_hip_roll":  0.0, "left_hip_pitch":0.0, "left_knee": 0.0,
                "right_hip_roll":  self.swing_hip_target, "right_hip_pitch":0.0, "right_knee": (self.swing_knee_lo + self.swing_knee_hi)/2
            },
            "level_3_right_support": {
                "left_hip_roll":  self.swing_hip_target, "left_hip_pitch":0.0, "left_knee": (self.swing_knee_lo + self.swing_knee_hi)/2,
                "right_hip_roll": 0.0, "right_hip_pitch":0.0, "right_knee": 0.0
            },
        }

        # Presets por nivel (ajustables)
        self.level_ranges = {
            1: {"swing_hip_target": 0.35, "swing_hip_tol": 0.08, "knee_lo": 0.45, "knee_hi": 0.70},
            2: {"swing_hip_target": 0.35, "swing_hip_tol": 0.10, "knee_lo": 0.45, "knee_hi": 0.75},
            3: {"swing_hip_target": 0.40, "swing_hip_tol": 0.10, "knee_lo": 0.40, "knee_hi": 0.85},
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
        self.target_leg = getattr(env, "fixed_target_leg", None)  # 'left'|'right'|None self.fixed_target_leg
        self.fixed_target_leg=self.target_leg
        self.switch_timer = 0
        # self.leg_switch_bonus = 0.0  # Bonus por cambio exitoso
        self.bad_ending=("fall", "tilt", "drift", "no_support")
        # Debug para confirmar configuración
        switch_time_seconds = self.switch_interval / self.frequency_simulation
        self.min_F=30
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
        # self.
        if self.level == 1 and self.enable_curriculum:
            reward = self._level_1_reward(pos,euler)      # Solo supervivencia
        elif self.level == 2 and self.enable_curriculum:
            reward = self._level_2_reward(pos, euler)      # + balance estable
        else:  # level == 3
            reward = self._level_3_reward(pos, euler, step_count)  # + levantar piernas
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
            height_reward= 1.0  # Buena altura
        elif height > 0.8:
            height_reward= 0.8  # Altura mínima
        elif height <= 0.8:
            height_reward= -1.0  # Caída
        elif height<= 0.7:       # and self.last_done_reason == self.bad_ending[0]:
            height_reward= -10

        pitch = euler[1]
        back_pitch_pen = - np.clip(pitch - 0.05, 0.0, 0.30) * 6.0

        pie_izquierdo_contacto, pie_derecho_contacto = self.env.contacto_pies
        if pie_izquierdo_contacto is False and pie_derecho_contacto:
            contact_reward= 1.0
        elif pie_izquierdo_contacto and pie_derecho_contacto:
            contact_reward= 0.1
        else:
            contact_reward= -1.0

        # === Bonus por 'usar' roll para recentrar COM sobre el soporte ===
        support_sign = +1.0 if (pie_izquierdo_contacto and not pie_derecho_contacto) else (-1.0 if (pie_derecho_contacto and not pie_izquierdo_contacto) else 0.0)
        torso_roll = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])[0]
        hiproll_align_bonus = 0.0
        if support_sign != 0.0:
            hiproll_align_bonus = np.clip(support_sign * torso_roll / np.deg2rad(10), -1.0, 1.0) * (0.3 / self.frequency_simulation)

        

        return height_reward + drift_pen + back_only_pen +back_pitch_pen  + hiproll_align_bonus + contact_reward
    
    def _level_2_reward(self,pos,euler):
        """NIVEL 2: Balance estable (recompensas 0-5)"""
        
        height_reward=self._level_1_reward(pos, euler)
        
        # Versión antigua
        roll,pitch = abs(euler[0]) , abs(euler[1])  # roll + pitch
        if pitch < 0.2:
            stability_reward = 1.5  # Muy estable
        elif pitch < 0.4:
            stability_reward = 0.5  # Moderadamente estable
        elif pitch < self.max_tilt_by_level[self.level]:
            stability_reward = -0.5  # Inestable
        elif pitch >= self.max_tilt_by_level[self.level]:# self.last_done_reason == self.bad_ending[1]:
            stability_reward = -5.0  # Inestable
    
    # Guardarraíl adicional de roll (torso) para evitar extremos
        torso_roll = euler[0]
        guard_pen = self._roll_guardrail_pen(
            torso_roll,
            level_soft=0.20,
            level_hard=self.max_tilt_by_level[self.level]
        )
        return height_reward + stability_reward + guard_pen
    
    def _level_3_reward(self,pos,euler, step_count):
        """NIVEL 3: Levantar piernas alternando (recompensas 0-8)"""
        
        # Recompensa base (igual que nivel 2)
        base_reward = self._level_2_reward(pos,euler)
        if base_reward < 0:  # Si se cayó, no calcular más
            return base_reward
        
        # + Recompensa por levantar pierna (NUEVA)
        leg_reward = self._calculate_leg_reward(step_count)

        zmp_reward = self.zmp_and_smooth_reward()

        # Guardarraíl de tobillos (frena 'zarpazo de tobillo')
        _,_,_, left_anckle_id, _,_,_, right_anckle_id = self.env.joint_indices
        ankle_pen = self._ankle_guardrail_pen(left_anckle_id, right_anckle_id)
        
        return base_reward + leg_reward + zmp_reward + ankle_pen
    
    def _calculate_leg_reward(self, step_count):
        """Calcular recompensa por levantar pierna correctamente"""


        left_hip_roll_id, left_hip_pitch_id, left_knee_id, left_anckle_id, right_hip_roll_id, right_hip_pitch_id, right_knee_id,right_anckle_id = self.env.joint_indices
        contact_l,F_L = self.env.contact_normal_force(left_anckle_id)
        contact_r, F_R = self.env.contact_normal_force(right_anckle_id)
        F_sum = max(F_L + F_R, 1e-6)

        # Target/support segun objetivo fijo (sigues con 'left' como pierna objetivo)
        target_is_right = False if (self.fixed_target_leg == 'left') else True
        target_id = right_anckle_id if target_is_right else left_anckle_id
        stance_id = left_anckle_id  if target_is_right else right_anckle_id
        F_sup = F_R if (stance_id == right_anckle_id) else F_L
        F_tar = F_R if (target_id == right_anckle_id) else F_L

        stance_ok, stance_q = self._stance_quality(stance_id, F_sum)  # 0..1 aprox
        
        
        # Detectar qué pies están en contacto Ver si seleccionar min_F=20 0 27 0 30
        left_down = (F_L>self.min_F) and contact_l>2 #self.env.contact_with_force(left_foot_id, min_F=self.min_F) Contacto bueno con suelo
        right_down = (F_R>self.min_F) and contact_r>0 #self.env.contact_with_force(right_foot_id, min_F=self.min_F)

        

        
        # NUEVO: si estamos en left-only, el soporte debe ser el pie derecho
        if self.fixed_target_leg == 'left':
            # fuerza la semántica: soporte=right, objetivo=left
            target_is_right = False
            support_foot_down = right_down
            target_foot_down  = left_down
        # target_is_right   = (self.target_leg == 'right') # Sera siempre false
        # target_foot_id    = right_foot_id if target_is_right else left_foot_id
        # target_foot_down  = right_down if target_is_right else left_down
        # support_foot_down = left_down if target_is_right else right_down


        # ======== SHAPING POR CARGAS (romper toe-touch y cargar el soporte) ========
        # Cargas del pie de SOPORTE (esperado) y del pie OBJETIVO (debe ir al aire)
        # F_sup = F_L if target_is_right else F_R   # si objetivo=right, soporte=left (F_L)
        # F_tar = F_R if target_is_right else F_L
        if self.fixed_target_leg == 'left':
            F_sup = F_R      # soporte = pie derecho
            F_tar = F_L      # objetivo = pie izquierdo
            # Carga mínima en soporte (80%) y toe-touch estricto del objetivo
            support_load_reward = np.clip((F_sup / F_sum - 0.80) / 0.20, 0.0, 1.0) * 1.2
            toe_touch_pen = -0.8 if (0.0 < F_tar < self.min_F) else 0.0
       
        #self.extra_reward_1


        #lpos = p.getLinkState(self.robot_id, left_foot_id)[0]
        #rpos = p.getLinkState(self.robot_id, right_foot_id)[0]
        #ml = abs((rpos[1] - lpos[1]) if target_is_right else (lpos[1] - rpos[1]))  # eje Y
        #ml_min = 0.12  # 12 cm de separación deseable
        #support_foot_down = left_down if target_is_right else right_down
        #midline_pen = 0.0 if (not support_foot_down) else -np.clip((ml_min - ml)/ml_min, 0.0, 1.0) * 1.2 # Antes 0.6

        # (1) Penaliza doble apoyo fuerte
        both_down_pen = -1.0 if (F_L >= self.min_F and F_R >= self.min_F) else 0.0

        # (2) Penaliza toe-touch (1–30 N) del pie objetivo
        toe_touch_pen = -0.6 if (0.0 < F_tar < self.min_F) else 0.0

        # (3) Recompensa reparto de carga sano: ≥80% en el pie de soporte
        ratio = F_sup / F_sum
        support_load_reward = np.clip((ratio - 0.80) / 0.20, 0.0, 1.0) * 1.0

        # (2.5) Penalización por casi-cruce entre el pie en swing y el pie de soporte
        if (F_sup >= self.min_F) and (F_tar <= 0):
            swing_id  = left_anckle_id if (not target_is_right) else right_anckle_id
            stance_id = right_anckle_id if (not target_is_right) else left_anckle_id
            dmin = 0.04  # 4 cm
            close_pen = 0.0
            cps = p.getClosestPoints(self.env.robot_id, self.env.robot_id, dmin, swing_id, stance_id)
            if cps:  # hay algún punto más cerca que dmin
                worst = min(cp[8] for cp in cps)  # cp[8] = distance
                close_pen += -1.0 * max(0.0, (dmin - worst) / dmin)
        else:
            close_pen = 0.0

        # (4) BONUS por tiempo en apoyo simple con saturación (sin toe-touch)
        ss_reward = self._single_support_dwell_reward(F_sup, F_tar, self.frequency_simulation)

        # --- Bonuses de forma SOLO si el pie objetivo NO está en contacto ---
        # Clearance
        # self.fixed_target_leg porque target foot es siempre left
        foot_z = p.getLinkState(self.robot_id, left_anckle_id)[0][2]
        clearance_target = 0.09  # 9 cm
        clearance_bonus = 0.0 if target_foot_down else np.clip(foot_z / clearance_target, 0.0, 1.0) * 0.5

        # === Helpers de rango suave ===
        def soft_range_bonus(x, lo, hi, slope=0.15):
            """
            Devuelve 1.0 dentro de [lo,hi]. Fuera, cae linealmente con pendiente 'slope' (rad^-1), acotado a [0,1].
            """
            if x < lo:
                return max(0.0, 1.0 - (lo - x)/slope)
            if x > hi:
                return max(0.0, 1.0 - (x - hi)/slope)
            return 1.0

        def soft_center_bonus(x, center, tol, slope=0.15):
            """Meseta alrededor de 'center' ± tol. Cae fuera con pendiente 'slope' (rad^-1)."""
            return soft_range_bonus(x, center - tol, center + tol, slope=slope)
        # Rodilla (rango recomendado 0.45–0.75 rad)
        knee_id  = right_knee_id if target_is_right else left_knee_id
        knee_ang = p.getJointState(self.robot_id, knee_id)[0]
        knee_bonus = soft_range_bonus(knee_ang, self.swing_knee_lo, self.swing_knee_hi, slope=0.20) * 1.0
        knee_bonus = 0.0 if target_foot_down else knee_bonus

        # Cadera (pitch) del swing — objetivo configurable, con meseta ± self.swing_hip_tol
        hip_id  = right_hip_pitch_id if target_is_right else left_hip_pitch_id
        hip_ang = p.getJointState(self.robot_id, hip_id)[0]
        #hip_bonus = soft_center_bonus(abs(hip_ang), self.swing_hip_target, self.swing_hip_tol, slope=0.20) * 0.7
        #hip_bonus = 0.0 if target_foot_down else hip_bonus
        # después (direccional):
        desired_sign = -1.0  # pon -1.0 si en tu robot la flexión hacia delante es negativa
        hip_bonus_dir = soft_center_bonus(desired_sign * hip_ang,
                                        self.swing_hip_target, self.swing_hip_tol,
                                        slope=0.20) * 0.7
        hip_bonus = 0.0 if target_foot_down else hip_bonus_dir

        # ✅ Bono de hip ROLL del swing (abducción cómoda)
        swing_roll_jid = (left_hip_roll_id if (not target_is_right) else right_hip_roll_id)
        q_roll_swing = p.getJointState(self.robot_id, swing_roll_jid)[0]
        roll_abd_center = 0.15  # ~8–10°
        roll_abd_tol    = 0.08
        roll_swing_bonus = soft_center_bonus(q_roll_swing, roll_abd_center, roll_abd_tol, slope=0.20) * 0.8
        if target_foot_down or ratio < 0.70:
            roll_swing_bonus = 0.0

        # ✅ Penalización por velocidad articular excesiva en la cadera del swing
        hip_vel = p.getJointState(self.robot_id, hip_id)[1]
        v_thresh = 0.8   # rad/s umbral de "demasiado rápido"
        kv = 0.15        # ganancia de penalización
        speed_pen = -kv * max(0.0, abs(hip_vel) - v_thresh)

        if ratio < 0.70:
            clearance_bonus = 0.0
            knee_bonus = 0.0
            hip_bonus = 0.0
            # roll_swing_bonus ya se pone a 0 arriba con ratio < 0.70

        # Suma total
        support_load_reward = min(support_load_reward, 0.8)
        shaping = both_down_pen + toe_touch_pen + support_load_reward +ss_reward
        leg_reward = (clearance_bonus + knee_bonus + hip_bonus
                      + roll_swing_bonus  + close_pen
                      + shaping + speed_pen) # contacto_reward
        # Recompensa por pie de soporte 'plano' (planta paralela al suelo)
        if F_sup >= self.min_F:
            stance_foot_id = (right_anckle_id if (F_R >= F_L) else left_anckle_id)
            flat_reward = self._foot_flat_reward(stance_foot_id, only_if_contact=True)
            leg_reward += flat_reward
        return leg_reward
    
    def zmp_and_smooth_reward(self):
        zmp_term = 0.0
        try:
            if hasattr(self.env, "zmp") and hasattr(self.env.zmp, "stability_margin_distance"):
                margin = float(self.env.zmp.stability_margin_distance())  # metros (+ estable si >0)
                # Escala: +0.5 en margen >= 5 cm; -0.5 si -5 cm
                zmp_term = 0.5 * np.clip(margin / 0.05, -1.0, 1.0)
                # Exporta KPI opcional
                if hasattr(self.env, "info"):
                    self.env.info["kpi"]["zmp_margin_m"] = margin
        except Exception:
            pass

        # --- Smoothness term (penaliza vibraciones de presiones/pares) ---
        smooth_pen = 0.0
        try:
            ps_prev = getattr(self.env, "pam_states_prev", {}).get("pressures", None)
            ps_curr = getattr(self.env, "pam_states", {}).get("pressures", None)
            if ps_prev is not None and ps_curr is not None:
                du = float(np.linalg.norm(np.asarray(ps_curr) - np.asarray(ps_prev)))
                smooth_pen = -0.01 * du
                if hasattr(self.env, "info"):
                    self.env.info["kpi"]["dP_norm"] = du
            else:
                # fallback a torques si no hay presiones
                tq_prev = getattr(self.env, "pam_states_prev", {}).get("joint_torques", None)
                tq_curr = getattr(self.env, "pam_states", {}).get("joint_torques", None)
                if tq_prev is not None and tq_curr is not None:
                    dtau = float(np.linalg.norm(np.asarray(tq_curr) - np.asarray(tq_prev)))
                    smooth_pen = -0.005 * dtau
                    if hasattr(self.env, "info"):
                        self.env.info["kpi"]["dTau_norm"] = dtau
        except Exception:
            pass
        return zmp_term + smooth_pen
    
    def _foot_contact_ok(self, foot_link_id, nmin=2):
        """Pie en buen apoyo: suficiente fuerza y >= nmin puntos."""
        n, F = self.env.contact_normal_force(foot_link_id)
        return (F >= self.min_F) and (n >= nmin), n, F
    
    def _stance_quality(self, stance_id, F_sum, n_min=2):
        """Calidad de apoyo del pie soporte = carga + 'planitud'."""
        ok, n, F = self._foot_contact_ok(stance_id, nmin=n_min)
        # ratio de carga sobre total (suavizado)
        ratio = F / max(F_sum, 1e-6)
        load = np.clip((ratio - 0.80) / 0.20, 0.0, 1.0)   # >=80% ⇒ 1.0
        flat = self._foot_flat_reward(stance_id, only_if_contact=True)  # 0..~0.8
        # Combino 70% carga + 30% planitud, y escalo a ~[0..1.0]
        return ok, (0.7 * load + 0.3 * (flat / 0.8))
    
    def hip_reward(self):
        left_hip_roll=self.env.left_hip_roll_angle
        left_hip_pitch=self.env.left_hip_pitch_angle
        right_hip_roll=self.env.right_hip_roll_angle
        right_hip_pitch=self.env.right_hip_pitch_angle
        if -0.5<left_hip_pitch<-0.2:
            reward_hip_left=2
            is_ok=True
        elif -0.2 < left_hip_pitch < 0.2:
            reward_hip_left=0
            is_ok=True
        else:
            is_ok=False
            reward_hip_left=-2

        if 0.2<abs(right_hip_pitch) and is_ok:
            reward_hip_right=2
        else:
            reward_hip_right=-2

        if left_hip_roll<-0.2:
            reward_roll=-5
        else:
            reward_roll=0
        
        return reward_hip_left+ reward_hip_right +reward_roll
    
    def knee_reward(self):
        left_knee=self.env.left_knee_angle
        right_knee=self.env.right_knee_angle
        if left_knee<0.2:
            reward_knee_left=0.5
        elif 0.2<= left_knee < 0.4:
            reward_knee_left=2
        else:
            reward_knee_left=-2

        if 0.2<right_knee:
            reward_knee_right=2
        else:
            reward_knee_right=-2
        
        return reward_knee_right+ reward_knee_left
    
    def anckle_reward(self):
        left_anckle=self.env.left_anckle_angle
        right_anckle=self.env.right_anckle_angle
        if left_anckle<0.2:
            reward_knee_left=0.5
        elif 0.2<= left_anckle < 0.4:
            reward_knee_left=2
        else:
            reward_knee_left=-2

        if 0.2<right_anckle:
            reward_knee_right=2
        else:
            reward_knee_right=-2
        
        return reward_knee_right+ reward_knee_left
    
    def update_after_episode(self, episode_reward, success=None):
        """Actualizar nivel después de cada episodio"""
        
        self.episode_count += 1
        self.recent_episodes.append(episode_reward)
        has_fallen = (self.last_done_reason in self.bad_ending)
        
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
                    self._apply_level_ranges()
        else:
            # MODO SIN CURRICULUM: solo logging básico
            both_print(f"🏁 Episode {self.episode_count}: "
                    f"reward={episode_reward:.1f} | success={success} | "
                    f"fixed_level=3")
    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        self.dx = float(pos[0] - self.env.init_pos[0])
        # Caída
        if pos[2] <= 0.7:
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
        pie_izquierdo_contacto, pie_derecho_contacto = self.env.contacto_pies   # si ya tienes util; si no, usa getContactPoints
        if not (pie_izquierdo_contacto or pie_derecho_contacto):
            self._no_contact_steps += 1
        else:
            self._no_contact_steps = 0
        if self._no_contact_steps >= int(0.20 * self.frequency_simulation):  # 0.2 s
            self.last_done_reason = "no_support"
            log_print("❌ Episode done: No foot support for too long")
            return True
        if (pie_izquierdo_contacto and pie_derecho_contacto):
            self.contact_both += 1
            if self.contact_both >int(0.80 * self.frequency_simulation):
                self.last_done_reason = "excessive support"
                log_print("❌ Episode done: No foot support for too long")
                return True
        else:
            self.contact_both=0
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
    
    def _apply_level_ranges(self):
        cfg = self.level_ranges.get(self.level, {})
        if cfg:
            self.swing_hip_target = cfg["swing_hip_target"]
            self.swing_hip_tol    = cfg["swing_hip_tol"]
            self.swing_knee_lo    = cfg["knee_lo"]
            self.swing_knee_hi    = cfg["knee_hi"]
            # actualizar “blandos” orientativos para el experto/PD
            self.target_angles["level_3_left_support"]["right_hip_roll"] = self.swing_hip_target
            self.target_angles["level_3_left_support"]["right_knee"]     = (self.swing_knee_lo + self.swing_knee_hi)/2
            self.target_angles["level_3_right_support"]["left_hip_roll"] = self.swing_hip_target
            self.target_angles["level_3_right_support"]["left_knee"]     = (self.swing_knee_lo + self.swing_knee_hi)/2

    # =========================
    # NUEVOS HELPERS DE SHAPING
    # =========================
    def _soft_quadratic_penalty(self, x: float, lim: float, gain: float) -> float:
        """
        Penaliza 0 dentro de [-lim, lim] y cuadrática fuera (suave).
        """
        over = max(0.0, abs(float(x)) - float(lim))
        return -float(gain) * (over ** 2)

    def _roll_guardrail_pen(self, torso_roll: float, level_soft: float = 0.20, level_hard: float = 0.35) -> float:
        """
        Guardarraíl de roll (torso): 
        - |roll| <= level_soft  -> 0
        - level_soft..level_hard -> penalización cuadrática suave
        - |roll| > level_hard   -> golpe extra
        """
        pen = self._soft_quadratic_penalty(torso_roll, lim=level_soft, gain=6.0)
        if abs(torso_roll) > level_hard:
            pen += -3.0
        return pen

    def _ankle_guardrail_pen(self, left_ankle_id: int, right_ankle_id: int) -> float:
        """
        Penaliza ángulos excesivos de tobillo (pitch) en ambos pies para
        evitar la estrategia de 'zarpazo de tobillo'.
        """
        qL = p.getJointState(self.env.robot_id, left_ankle_id)[0]
        qR = p.getJointState(self.env.robot_id, right_ankle_id)[0]
        penL = self._soft_quadratic_penalty(qL, lim=0.22, gain=4.0)  # ~12.6°
        penR = self._soft_quadratic_penalty(qR, lim=0.22, gain=4.0)
        return penL + penR

    def _single_support_dwell_reward(self, F_sup: float, F_tar: float, freq: float) -> float:
        """
        Recompensa por sostener apoyo simple sin 'toe-touch'.
        Crece con el tiempo y satura para no incentivar posturas extremas.
        """
        if (F_sup >= self.min_F) and (F_tar < 1.0):
            self.single_support_ticks += 1
            t = self.single_support_ticks / float(freq)  # segundos
            # Curva suave: ~+0.6 a los 0.6–0.8 s, luego satura
            return 0.6 * (1.0 - np.exp(-2.2 * t))
        else:
            self.single_support_ticks = 0
            return 0.0

    def _foot_flat_reward(self, foot_link_id: int, only_if_contact: bool = True, target_roll: float = 0.0, target_pitch: float = 0.0) -> float:
        """
        Recompensa por pie 'plano' (link roll/pitch ~ 0) — se aplica típicamente al pie de soporte.
        """
        if only_if_contact:
            if not self.env.contact_with_force(foot_link_id, min_F=self.min_F):
                return 0.0
        try:
            _, orn, _, _, _, _ = p.getLinkState(self.env.robot_id, foot_link_id, computeForwardKinematics=True)
            r, p, _ = self.env.p.getEulerFromQuaternion(orn)
        except Exception:
            return 0.0
        def soft_center(x, c, tol=0.05, slope=0.10):
            if x < c - tol: return max(0.0, 1.0 - (c - tol - x)/slope)
            if x > c + tol: return max(0.0, 1.0 - (x - (c + tol))/slope)
            return 1.0
        r_bonus = soft_center(r, target_roll)
        p_bonus = soft_center(p, target_pitch)
        return 0.8 * 0.5 * (r_bonus + p_bonus)  # máx ≈ +0.8
    


    # ============================================================================================================================================== #
    # ============================================Recompensas ofrecidas por Modelo en 2D=========================================================== #
    # ============================================================================================================================================== #

    def calculate_reward_2d_walking(self, action, step_count):
        env = self.env
        # --- Estado base ---
        (x, z), orn = (env.pos[0], env.pos[2]), p.getBasePositionAndOrientation(env.robot_id)[1]
        euler = p.getEulerFromQuaternion(orn)
        pitch = euler[1]
        lin_vel, ang_vel = p.getBaseVelocity(env.robot_id)
        vx, vz = lin_vel[0], lin_vel[2]

        # --- Contactos y fuerzas ---
        left_down, right_down = env.contacto_pies
        # Tu función devuelve (n_points, total_force) cuando hay contactos; si no, 0.0
        F_L = env.contact_normal_force(env.left_foot_link_id)
        F_R = env.contact_normal_force(env.right_foot_link_id)
        Ftot_L = F_L[1] if isinstance(F_L, tuple) else 0.0
        Ftot_R = F_R[1] if isinstance(F_R, tuple) else 0.0

        # ===== 1) Progreso en +X =====
        v_max = 1.0
        r_vel = max(0.0, min(vx / v_max, 1.0))

        # ===== 2) Postura (tronco erguido) =====
        pitch_ref = 0.25
        r_post = np.exp(-abs(pitch)/pitch_ref)

        # ===== 3) Patrones de paso =====
        # bonus por alternancia de pie de soporte
        support_now = 'L' if Ftot_L > Ftot_R else 'R'
        if step_count == 1:
            self._last_support = support_now
        switched = (support_now != getattr(self, "_last_support", support_now))
        self._last_support = support_now

        b_switch = 0.15 if switched else 0.0
        # bonus por apoyo “sano” (fuerza suficiente y el otro pie ligero)
        Fmin = 60.0
        if support_now == 'L':
            b_contact = 0.10 if (Ftot_L > Fmin and Ftot_R < 0.5*Fmin) else 0.0
        else:
            b_contact = 0.10 if (Ftot_R > Fmin and Ftot_L < 0.5*Fmin) else 0.0
        r_foot = b_switch + b_contact

        # ===== 4) Coste de energía / suavidad =====
        # usa la media de presiones y la variación respecto a la anterior
        u = np.clip(action, 0.0, 1.0)
        if not hasattr(self, "_prev_u"):
            self._prev_u = np.zeros_like(u)
        energy = float(np.mean(u))
        delta_u = float(np.mean(np.abs(u - self._prev_u)))
        self._prev_u = u

        # ===== 5) Resbalón e impacto =====
        # Simplificado: si hay apoyo y la velocidad tangencial del pie es grande -> penaliza
        slip_pen = 0.0
        impact_pen = 0.0
        for foot_id, Ftot in [(env.left_foot_link_id, Ftot_L), (env.right_foot_link_id, Ftot_R)]:
            if Ftot > Fmin:
                ls = p.getLinkState(env.robot_id, foot_id, computeLinkVelocity=1)
                vfx, vfy, vfz = ls[6]
                v_tan = np.hypot(vfx, vfz)  # sagital
                if v_tan > 0.6:
                    slip_pen += (v_tan - 0.6)
                # impacto (crudo): penaliza grandes fuerzas instantáneas
                impact_pen += max(0.0, (Ftot - 400.0)/400.0)  # normaliza un poco

        # ===== 6) Caída / terminación =====
        fall = 0.0
        z_base = z
        if (z_base < 0.75) or (abs(pitch) > 0.6):
            fall = 1.0
            self.last_done_reason = "fall_2D"
        
        # ===== Pésos =====
        w_v, w_post, w_foot = 2.0, 0.5, 0.4
        w_en, w_slip, w_impact, w_fall = 0.05, 0.2, 0.1, 5.0

        reward = (w_v*r_vel + w_post*r_post + w_foot*r_foot
                - w_en*(energy + 0.5*delta_u) - w_slip*slip_pen
                - w_impact*impact_pen - w_fall*fall)

        # done por caída
        if fall > 0.5:
            self._episode_done = True

        return float(reward)

