# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

from collections import deque

from Archivos_Apoyo.simple_log_redirect import log_print, both_print
from Archivos_Apoyo.Configuraciones_adicionales import split_cocontraction_torque_neutral

class RewardMode(Enum):
    PROGRESSIVE = "progressive"      # curriculum por niveles (modo actual por defecto)
    WALK3D = "walk3d"                # caminar en 3D (avance +X)
    LIFT_LEG = "lift_leg"            # levantar una pierna estable
    MARCH_IN_PLACE = "march_in_place" # alternar piernas en el sitio (ambos pies en el aire permitido)
    

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
        self.robot_id = env.robot_id
        self.single_support_ticks = 0
        # === Modo de recompensa seleccionable desde el env (si no existe, progressive) ===
        mode_str = getattr(env, "simple_reward_mode", "progressive")
        try:
            self.mode = RewardMode(mode_str)
        except Exception:
            self.mode = RewardMode.PROGRESSIVE
        # Parametrización útil para modos nuevos
        self.vx_target = float(getattr(env, "vx_target", 0.6))
        self.allow_hops = bool(getattr(env, "allow_hops", False))

        # --- Parámetros configurables (puedes sobreescribirlos desde env) ---
        # Cadera de la pierna en el aire (roll absoluto). Recomendado 0.3–0.5
        self.swing_hip_target  = float(getattr(env, "swing_hip_target", 0.10))
        # Ventana suave para cadera y rodilla (ancho de tolerancia)
        self.swing_hip_tol     = float(getattr(env, "swing_hip_tol",  0.10))  # ±0.10 rad
        # Rodilla en el aire: rango recomendado 0.45–0.75
        self.swing_knee_lo     = float(getattr(env, "swing_knee_lo",  0.45))
        self.swing_knee_hi     = float(getattr(env, "swing_knee_hi",  0.75)) 
        # MODO SIN CURRICULUM: sistema fijo y permisivo
        self.level = 3  # Siempre nivel 3
        self.level_progression_disabled = True
        if self.env.logger:
            self.env.logger.log("main",f"🎯 Progressive System: CURRICULUM DISABLED")
            self.env.logger.log("main",f"   Mode: Fixed basic balance (Level max only)")
        
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
        self.max_tilt_by_level = {
            1: 0.8,  # 
            2: 0.7,  # 
            3: 0.7   # 
        }
        if self.env.logger:
            self.env.logger.log("main",f"   Max tilt: {np.degrees(self.max_tilt_by_level[3]):.1f}° (permisivo)")
        
        # Para alternancia de piernas (solo nivel 3)
        self.target_leg = getattr(env, "fixed_target_leg", None)  # 'left'|'right'|None self.fixed_target_leg
        self.fixed_target_leg=self.target_leg
        self.switch_timer = 0
        # self.leg_switch_bonus = 0.0  # Bonus por cambio exitoso
        self.bad_ending=("fall", "tilt", "drift", "no_support", "excessive support")
        # Debug para confirmar configuración
        switch_time_seconds = self.switch_interval / self.frequency_simulation
        self.min_F=20
        self.reawrd_step=self.env.reawrd_step
        if self.env.logger:
            self.env.logger.log("main",f"🎯 Progressive System initialized:")
            self.env.logger.log("main",f"   Switch interval: {self.switch_interval} steps ({switch_time_seconds:.1f}s)")
            self.env.logger.log("main",f"   Frequency: {self.frequency_simulation} Hz")
            self.env.logger.log("main",f"🎯 Simple Progressive System: Starting at Level {self.level}")
    
    def calculate_reward(self, action, step_count):
        """
        Método principal: calcula reward según el nivel actual
        """

        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        # Decido usar este método para crear varias opciones de creación de recompensas. Else, curriculo clásico
        if getattr(self, "mode", RewardMode.PROGRESSIVE) == RewardMode.WALK3D:
            return self.calculate_reward_walk3d(action, step_count)
        if getattr(self, "mode", RewardMode.PROGRESSIVE) == RewardMode.LIFT_LEG:
            return self.calculate_reward_lift_leg(action, step_count)
        if getattr(self, "mode", RewardMode.PROGRESSIVE) == RewardMode.MARCH_IN_PLACE:
            return self.calculate_reward_march_in_place(action, step_count)
        else:
            reward = self._level_3_reward(pos, euler, step_count)  # + levantar piernas
            level =  3
            max_reward = self.level_config[level]['max_reward']
            
            # Limitar reward según nivel
            
            
            return max(-2.0, min(reward, max_reward))
    
    def _level_1_reward(self,pos,euler):
        """NIVEL 1: Solo mantenerse de pie (recompensas 0-3)"""
        self.dx = float(pos[0] - self.env.init_pos[0])
        self.dy = float(pos[1] - self.env.init_pos[1])
        # Tolerancia sin penalización ±5 cm
        tol = 0.05
        # Penaliza deriva total fuera de tolerancia (suave; tope aprox -2.0)
        drift_pen = - np.clip(abs(self.dx) - tol, 0.0, 0.25) * 8.0
        lateral_pen = - np.clip(abs(self.dy) - 0.03, 0.0, 0.25) * 10.0
        # Penaliza adicionalmente cuando la deriva es hacia atrás (dx < -tol)
        # tope aprox -1.6
        back_only_pen = - np.clip(-self.dx - tol, 0.0, 0.20) * 8.0

        height = pos[2]
        
        # Recompensa simple por altura
        if height > 0.8:
            height_reward= 1.0  # Buena altura
        elif height > 0.7:
            height_reward= 0.8  # Altura mínima
        elif height <= 0.7:
            height_reward= -1.0  # Caída
        elif height<= 0.5:       # and self.last_done_reason == self.bad_ending[0]:
            height_reward= -10

        pitch = euler[1]
        back_pitch_pen = - np.clip(pitch - 0.05, 0.0, 0.30) * 6.0

        pie_izquierdo_contacto, pie_derecho_contacto = self.env.contacto_pies
        if pie_izquierdo_contacto is False and pie_derecho_contacto:
            contact_reward= 2.0
        elif pie_izquierdo_contacto and pie_derecho_contacto:
            contact_reward= 0.1
        else:
            contact_reward= -2.0

        knee_reward = self.knee_reward(self.env.left_knee_angle, self.env.right_knee_angle)

        # === Bonus por 'usar' roll para recentrar COM sobre el soporte ===
        # Afecta al roll de align bonus
        support_sign = -1.0 if (pie_izquierdo_contacto and not pie_derecho_contacto) else (-1.0 if (pie_derecho_contacto and not pie_izquierdo_contacto) else 0.0)
        torso_roll = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])[0]
        hiproll_align_bonus = 0.0
        if support_sign != 0.0:
            hiproll_align_bonus = np.clip(support_sign * torso_roll / np.deg2rad(10), -1.0, 1.0) * (0.2 / self.frequency_simulation)

        self.reawrd_step['height_reward'] = height_reward
        self.reawrd_step['drift_pen'] = drift_pen
        self.reawrd_step['back_only_pen'] = back_only_pen
        self.reawrd_step['back_pitch_pen'] = back_pitch_pen
        self.reawrd_step['hiproll_align_bonus'] = hiproll_align_bonus
        self.reawrd_step['contact_reward']=contact_reward
        self.reawrd_step['lateral_pen']=lateral_pen

        return height_reward + drift_pen + back_only_pen +back_pitch_pen  + hiproll_align_bonus + contact_reward + lateral_pen
    
    def _level_2_reward(self,pos,euler):
        """NIVEL 2: Balance estable (recompensas 0-5)"""
        
        height_reward=self._level_1_reward(pos, euler)
        
        # Versión antigua
        roll,pitch = euler[0] , abs(euler[1])  # roll + pitch
        if pitch < 0.2:
            stability_reward = 2.5  # Muy estable
        elif pitch < 0.4:
            stability_reward = 0.5  # Moderadamente estable
        elif pitch < self.max_tilt_by_level[self.level]:
            stability_reward = -2.0  # Inestable
        elif pitch >= self.max_tilt_by_level[self.level]:# self.last_done_reason == self.bad_ending[1]:
            stability_reward = -2.5  # Inestable
    
    # Guardarraíl adicional de roll (torso) para evitar extremos
        guard_pen = self._roll_guardrail_pen(
            roll,
            level_soft=0.15,
            level_hard=self.max_tilt_by_level[self.level]
        )
        # Activar términos que ya tenías calculados
        comz     = self._com_zmp_stability_reward()   # estabilidad COM/ZMP
        comproj  = self._com_projection_reward()      # proyección COM dentro del soporte

        # Log por-step (si lo consumes en CSV)
        self.reawrd_step["guard_pen"]      = guard_pen
        self.reawrd_step["stability_reward"]= locals().get("stability_reward", 0.0)
        #self.reawrd_step["decouple_pen"]   = decouple
        self.reawrd_step["comz_term"]      = comz
        self.reawrd_step["comproj_term"]   = comproj
        #self.reawrd_step["anti_sway_bonus"]= anti_sway

        return (
            height_reward
            + stability_reward
            + guard_pen
            + 0.40 * comz          # pesos prudentes
            + 0.30 * comproj
            #+ anti_sway
            #- 0.30 * decouple
        )
    
    def _level_3_reward(self,pos,euler, step_count):
        """NIVEL 3: Levantar piernas alternando (recompensas 0-8)"""
        # Recompensa base (igual que nivel 2)
        base_reward = self._level_2_reward(pos,euler)
        if base_reward < 0:  # Si se cayó, no calcular más
            return base_reward
        
        # + Recompensa por levantar pierna (NUEVA)
        leg_reward = self._calculate_leg_reward(step_count)
        _,_,_, left_ankle_id, _,_,_, right_ankle_id = self.env.joint_indices
        ankle_pen = self._ankle_guardrail_pen(left_ankle_id, right_ankle_id)
        zmp_reward = self.zmp_and_smooth_reward()

        # Guardarraíl de tobillos (frena 'zarpazo de tobillo')
        # NEW: COM lateral en apoyo simple
        com_lat_pen = self._com_lateral_pen_single_support(k=3.0)
        self.reawrd_step["com_lat_pen"] = com_lat_pen
        pam_pairs = [(self.env.muscle_names[2*i], self.env.muscle_names[2*i+1]) for i in range(len(self.env.control_joint_names))]
        cocontr_pen = - self._cocontraction_pen(pam_pairs, lam=0.01)
        self.reawrd_step["cocontr_pen"] = cocontr_pen
        
        self.reawrd_step["ankle_pen"]=ankle_pen
        return base_reward + leg_reward + zmp_reward + ankle_pen + com_lat_pen + cocontr_pen
    
    def _calculate_leg_reward(self, step_count):
        """Calcular recompensa por levantar pierna correctamente"""

        
        left_hip_roll_id, left_hip_pitch_id, left_knee_id, left_ankle_id, right_hip_roll_id, right_hip_pitch_id, right_knee_id,right_ankle_id = self.env.joint_indices
        
        n_l,F_L = self.env.contact_normal_force(left_ankle_id)
        n_r,F_R = self.env.contact_normal_force(right_ankle_id)
        F_sum = max(F_L + F_R, 1e-6)
        
        # NUEVO: si estamos en left-only, el soporte debe ser el pie derecho
        if self.fixed_target_leg == 'left':
            # fuerza la semántica: soporte=right, objetivo=left
            target_is_right = False
            target_is_left = not target_is_right
            support_foot_down = self.env.contact_with_force(right_ankle_id, stable_foot=(not target_is_right), min_F=self.min_F)
            target_foot_down  = self.env.contact_with_force(left_ankle_id, stable_foot= (not target_is_left), min_F=self.min_F)
            F_sup = F_R      # soporte = pie derecho
            F_tar = F_L      # objetivo = pie izquierdo
        else:
            target_is_right = True
            target_is_left = not target_is_right
            support_foot_down = self.env.contact_with_force(left_ankle_id, stable_foot= (not target_is_left), min_F=self.min_F)
            target_foot_down  = self.env.contact_with_force(right_ankle_id, stable_foot=(not target_is_right), min_F=self.min_F)
            F_sup = F_L      # soporte = pie derecho
            F_tar = F_R      # objetivo = pie izquierdo
    
        if self.fixed_target_leg == 'left':
            
            # Carga mínima en soporte (80%) y toe-touch estricto del objetivo
            support_load_reward = np.clip((F_sup / F_sum - 0.80) / 0.20, 0.0, 1.0) * 1.2
            toe_touch_pen = -0.8 if (0.0 < F_tar < self.min_F) else 0.0

        # (1) Penaliza doble apoyo fuerte
        both_down_pen = -1.0 if (F_L >= self.min_F and F_R >= self.min_F) else 0.0

        # (2) Penaliza toe-touch (1–30 N) del pie objetivo
        toe_touch_pen = -0.6 if (0.0 < F_tar < self.min_F) else 0.0

        # (3) Recompensa reparto de carga sano: ≥80% en el pie de soporte
        ratio = F_sup / F_sum
        support_load_reward = np.clip((ratio - 0.80) / 0.20, 0.0, 1.0) * 1.0

        # (2.5) Penalización por casi-cruce entre el pie en swing y el pie de soporte
        if (F_sup >= self.min_F) and (F_tar < self.min_F):
            swing_id  = left_ankle_id if (not target_is_right) else right_ankle_id
            stance_id = right_ankle_id if (not target_is_right) else left_ankle_id
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
        foot_z = p.getLinkState(self.robot_id, left_ankle_id)[0][2]
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
            

        # ✅ Penalización por velocidad articular excesiva en la cadera del swing
        hip_vel = p.getJointState(self.robot_id, hip_id)[1]
        v_thresh = 0.8   # rad/s umbral de "demasiado rápido"
        kv = 0.15        # ganancia de penalización
        speed_pen = -kv * max(0.0, abs(hip_vel) - v_thresh)

        if ratio < 0.60:
            clearance_bonus = 0.0
            knee_bonus = 0.0
            hip_bonus = 0.0
            roll_swing_bonus = 0.0
            # roll_swing_bonus ya se pone a 0 arriba con ratio < 0.70

        # Suma total
        support_load_reward = min(support_load_reward, 0.8)
        self.reawrd_step['both_down_pen']=both_down_pen
        self.reawrd_step['toe_touch_pen']=toe_touch_pen
        self.reawrd_step['support_load_reward']=support_load_reward
        self.reawrd_step['ss_reward']=ss_reward
        shaping = both_down_pen + toe_touch_pen + support_load_reward +ss_reward
        self.reawrd_step['clearance_bonus']=clearance_bonus
        self.reawrd_step['knee_bonus']=knee_bonus
        self.reawrd_step['hip_bonus']=hip_bonus
        self.reawrd_step['speed_pen']=speed_pen
        leg_reward = (clearance_bonus + knee_bonus + hip_bonus
                      + roll_swing_bonus  + close_pen
                      + shaping + speed_pen) # contacto_reward
        # Recompensa por pie de soporte 'plano' (planta paralela al suelo)
        if F_sup >= self.min_F and support_foot_down:
            stance_foot_id = (right_ankle_id if (F_R >= F_L) else left_ankle_id)
            flat_reward = self._foot_flat_reward(stance_foot_id, only_if_contact=True)
            leg_reward += flat_reward
        self.reawrd_step['flat_reward']=flat_reward if support_foot_down else 0.0
        return leg_reward
    
    def zmp_and_smooth_reward(self):
        zmp_term = 0.0
        try:
            if hasattr(self.env, "zmp_calculator") and hasattr(self.env.zmp_calculator, "stability_margin_distance"):
                margin = float(self.env.zmp_calculator.stability_margin_distance()) # Se medirá en metros
                zmp_term = 0.7 * np.clip(margin / 0.05, -1.0, 1.0)
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
        self.reawrd_step['zmp_term'] =  zmp_term
        self.reawrd_step['smooth_pen'] = smooth_pen
        return zmp_term + smooth_pen
    
    def hip_reward(self,left_hip_roll,left_hip_pitch, right_hip_roll, right_hip_pitch):
        
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
    
    def knee_reward(self, left_knee, right_knee):
        
        if 0.1<left_knee<0.2:
            reward_knee_left=1
        elif 0.2<= left_knee < 0.4:
            reward_knee_left=2
        else:
            reward_knee_left=-2

        if 0.1<right_knee<0.2:
            reward_knee_right=1
        elif 0<=right_knee<=0.1:
            reward_knee_right=0.5
        else:
            reward_knee_right=-2

        self.reawrd_step['reward_knee_right'] =  reward_knee_right
        self.reawrd_step['reward_knee_left'] =  reward_knee_left
        
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
        if self.env.logger:
            self.env.logger.log("main",f"{self.level_progression_disabled=:}")
        # Verificar si subir de nivel
        if self.level_progression_disabled is False:  # Necesitamos al menos 5 episodios
            #avg_reward = sum(self.recent_episodes) / len(self.recent_episodes)
            #config = self.level_config[self.level]
            # Actualizar racha
            if len(self.recent_episodes) >= 5:
                self.success_streak = self.success_streak + 1 if success else 0

                # (Opcional) logging
                if self.env.logger:
                    self.env.logger.log("main",f"🏁 Episode {self.episode_count}: "
                        f"reward={episode_reward:.1f} | success={success} | "
                        f"streak={self.success_streak}/{cfg['success_streak_needed']}")
                
                # Promoción de nivel si cumple racha y episodios mínimos
                if (self.success_streak >= cfg['success_streak_needed']
                    and self.episode_count >= cfg['episodes_needed']
                    and self.level < 3):
                    old = self.level
                    self.level += 1
                    self.success_streak = 0
                    if self.env.logger:
                        self.env.logger.log("main",f"🎉 LEVEL UP! {old} → {self.level}")
                    self._apply_level_ranges()
        else:
            # MODO SIN CURRICULUM: solo logging básico
            if self.env.logger:
                self.env.logger.log("main",f"🏁 Episode {self.episode_count}: "
                    f"reward={episode_reward:.1f} | success={success} | "
                    f"fixed_level=3")
    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        # Penalizo deriva frente y lateral
        self.dx = float(pos[0] - self.env.init_pos[0])
        self.dy = float(pos[1] - self.env.init_pos[1])
        # Caída
        if pos[2] <= 0.5:
            self.last_done_reason = "fall"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Robot fell")
            return True
        
        if abs(self.dx) > 0.35:
            self.last_done_reason = "drift"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Excessive longitudinal drift")
            return True
        
        max_tilt = self.max_tilt_by_level.get(self.level, 0.5)
        # Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            self.last_done_reason = "tilt"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Robot tilted too much")
            return True
        
        # dentro de is_episode_done(...) tras calcular contactos:
        pie_izquierdo_contacto, pie_derecho_contacto = self.env.contacto_pies   # si ya tienes util; si no, usa getContactPoints
        if not (pie_izquierdo_contacto or pie_derecho_contacto):
            self._no_contact_steps += 1
        else:
            self._no_contact_steps = 0
        if self._no_contact_steps >= int(0.50 * self.frequency_simulation):  # 0.2 s
            self.last_done_reason = "no_support"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: No foot support for too long")
            return True
        if (pie_izquierdo_contacto and pie_derecho_contacto):
            self.contact_both += 1
            if self.contact_both >int(0.80 * self.frequency_simulation):
                self.last_done_reason = "excessive support"
                if self.env.logger:
                    self.env.logger.log("main","❌ Episode done: excessive support")
                return True
        else:
            self.contact_both=0
        # Tiempo máximo (crece con nivel)
        max_steps =  6000 # 2000, 4000, 6000 steps
        if step_count >= max_steps:
            self.last_done_reason = "time"
            if self.env.logger:
                self.env.logger.log("main","⏰ Episode done: Max time reached")
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
            'curriculum_enabled': False,
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
            pen -= 9.0
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
            # Pie de soporte "estable": F > F_min y múltiples puntos de contacto
            if not self.env.contact_with_force(foot_link_id, stable_foot=True, min_F=self.min_F):
                return 0.0
        try:
            _, orn, _, _, _, _ = p.getLinkState(self.env.robot_id, foot_link_id, computeForwardKinematics=True)
            r, pch, _ = p.getEulerFromQuaternion(orn)
        except Exception:
            return 0.0
        def soft_center(x, c, tol=0.05, slope=0.10):
            if x < c - tol: return max(0.0, 1.0 - (c - tol - x)/slope)
            if x > c + tol: return max(0.0, 1.0 - (x - (c + tol))/slope)
            return 1.0
        r_bonus = soft_center(r, target_roll)
        p_bonus = soft_center(pch, target_pitch)
        return 0.8 * 0.5 * (r_bonus + p_bonus)  # máx ≈ +0.8
    
    
    
    # NEW: penalización de COM lateral durante apoyo simple (COM cerca del pie de soporte)
    def _com_lateral_pen_single_support(self, k: float = 3.0):
        kpi = self.env.info.get("kpi", {})
        left_down  = bool(kpi.get("left_down", 0))
        right_down = bool(kpi.get("right_down", 0))
        if left_down ^ right_down:
            # usa COM_y del KPI y la posición Y del pie de soporte
            com_y = float(kpi.get("com_y", 0.0))
            # estima pie de soporte por fuerza vertical promedio (o flags)
            F_L = float(kpi.get("F_L", 0.0)); F_R = float(kpi.get("F_R", 0.0))
            support = "left" if (F_L >= F_R) else "right"
            foot_id = self.env.left_foot_link_id if support == "left" else self.env.right_foot_link_id
            foot_pos = self.env.get_link_world_position(foot_id)  # asume helper; equiv. a p.getLinkState(...)[0]
            foot_y = float(foot_pos[1])
            return -k * abs(com_y - foot_y)
        return 0.0
    
    # NEW: penalización de co-contracción PAM (producto flex*ext)
    # Ver si hay que eliminar o no
    def _cocontraction_pen(self, pairs, lam: float = 0.01):
        ps = getattr(self.env, "pam_states", {}).get("pressures_by_name", {})
        pen = 0.0
        for flex, ext in pairs:
            uf = float(ps.get(flex, 0.0)); ue = float(ps.get(ext, 0.0))
            pen += uf * ue
        return lam * pen
    
    def _com_zmp_stability_reward(self):
        z = getattr(self.env, "zmp_calculator", None)
        if z:
            margin = float(z.stability_margin_distance())  # m
            # +0.7 si margen >= 5 cm; -0.7 si <= -5 cm
            term = 0.7 * np.clip(margin/0.05, -5.0, 5.0)
            # Exporta KPI
            self.env.info["kpi"]["zmp_margin_m"] = margin
        else:
            # Fallback: usa COM estático
            try:
                com, _ = self.env.robot_data.get_center_of_mass
                # Simple: dentro de la caja entre pies ± margen
                term = 0.3 if self.env.zmp_calculator.is_stable(np.array(com[:2])) else -0.3
            except Exception:
                term = 0.0
        return term
        

    def _com_projection_reward(self):
        """
        Empuja la proyección del COM hacia el pie de soporte.
        """
        try:
            env = self.env
            kpi = env.info.get("kpi", {})
            # fuerzas en pies para decidir soporte
            F_L = float(kpi.get("F_L", 0.0)); F_R = float(kpi.get("F_R", 0.0))
            left_id, right_id = env.left_foot_link_id, env.right_foot_link_id
            support_id = left_id if (F_L >= F_R) else right_id
            foot_xy = p.getLinkState(env.robot_id, support_id)[0][:2]
            com_xy  = (float(kpi.get("com_x", 0.0)), float(kpi.get("com_y", 0.0)))
            dx = float(com_xy[0] - foot_xy[0]); dy = float(com_xy[1] - foot_xy[1])
            r = np.hypot(dx, dy)
            r0 = 0.08  # 8 cm
            self.env.info["kpi"]["com_dist_to_support"] = float(r)      # distancia XY del COM al pie soporte (m)
            self.env.info["kpi"]["com_stable_flag"]     = int(r < r0) # “cerca” si < 8 cm
            return 0.5 * np.exp(- (r / r0)**2 )
        except Exception:
            return 0.0
        
    # ============================================================================================================================================= #
    # ================================================= Nuevos metodos de recompensa para nuevas acciones ========================================= #
    # ============================================================================================================================================= #

    # ===================== NUEVO: Caminar 3D =====================
    def calculate_reward_walk3d(self, action, step_count):
        env = self.env
        pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        lin_vel, ang_vel = p.getBaseVelocity(env.robot_id)
        vx = lin_vel[0]

        # Contactos y fuerzas
        F_L = env.contact_normal_force(env.left_foot_link_id); Ftot_L = F_L[1] if isinstance(F_L, tuple) else 0.0
        F_R = env.contact_normal_force(env.right_foot_link_id); Ftot_R = F_R[1] if isinstance(F_R, tuple) else 0.0
        Fmin = 40.0

        # 1) Progreso hacia +X con cap
        v_max = max(0.3, float(self.vx_target))
        r_vel = max(0.0, min(vx / v_max, 1.0))

        # 2) Postura
        r_post = np.exp(-abs(pitch)/0.25) * np.exp(-abs(roll)/0.20)

        # 3) Estabilidad (ZMP + suavidad interna ya implementada)
        r_stab = self.zmp_and_smooth_reward(pos, euler, step_count)

        # 4) Patrón de paso ligero: pie soporte claro + alternancia
        support_now = 'L' if Ftot_L > Ftot_R else 'R'
        if step_count == 1: self._last_support3d = support_now
        switched = (support_now != getattr(self, "_last_support3d", support_now))
        self._last_support3d = support_now
        b_switch = 0.10 if switched else 0.0
        b_contact = 0.10 if ((support_now=='L' and Ftot_L>Fmin and Ftot_R<0.5*Fmin) or
                             (support_now=='R' and Ftot_R>Fmin and Ftot_L<0.5*Fmin)) else 0.0
        r_foot = b_switch + b_contact

        # 5) Suavidad/energía
        u = np.clip(action, 0.0, 1.0)
        if not hasattr(self, "_prev_u3d"): self._prev_u3d = np.zeros_like(u)
        energy = float(np.mean(u)); delta_u = float(np.mean(np.abs(u - self._prev_u3d))); self._prev_u3d = u

        # 6) Guardarraíles
        pen_roll = self._roll_guardrail_pen(roll)
        pen_ankle = self._ankle_guardrail_pen()

        # 7) Caída / vuelos (si no se permiten)
        fall = 0.0
        z_base = pos[2]
        if (z_base < 0.75) or (abs(pitch) > 0.7) or (abs(roll) > 0.7):
            fall = 1.0; self.last_done_reason = "fall_3D"; self._episode_done = True
        if not self.allow_hops and (Ftot_L < 10.0 and Ftot_R < 10.0):
            fall += 0.3

        # Pesos
        w_v, w_post, w_stab, w_foot = 2.2, 0.6, 0.4, 0.3
        w_en, w_du, w_guard, w_fall = 0.04, 0.08, 0.3, 5.0
        reward = (w_v*r_vel + w_post*r_post + w_stab*r_stab + w_foot*r_foot
                  - w_en*energy - w_du*delta_u - w_guard*(pen_roll + pen_ankle) - w_fall*fall)
        return float(reward)
    
    # ===================== NUEVO: Levantar una pierna (simple) =====================
    def calculate_reward_lift_leg(self, action, step_count):
        env = self.env
        pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler

        # Pierna de soporte deseada (si el env lo indica); si no, inferimos
        target_action = getattr(env, "single_leg_action", None)
        if str(target_action).endswith("BALANCE_LEFT_SUPPORT"):
            support_desired = 'L'
        elif str(target_action).endswith("BALANCE_RIGHT_SUPPORT"):
            support_desired = 'R'
        else:
            F_L = env.contact_normal_force(env.left_foot_link_id); Ftot_L = F_L[1] if isinstance(F_L, tuple) else 0.0
            F_R = env.contact_normal_force(env.right_foot_link_id); Ftot_R = F_R[1] if isinstance(F_R, tuple) else 0.0
            support_desired = 'L' if Ftot_L >= Ftot_R else 'R'

        # Fuerzas
        F_L = env.contact_normal_force(env.left_foot_link_id); Ftot_L = F_L[1] if isinstance(F_L, tuple) else 0.0
        F_R = env.contact_normal_force(env.right_foot_link_id); Ftot_R = F_R[1] if isinstance(F_R, tuple) else 0.0
        Fmin = 45.0

        # 1) Estabilidad + soporte claro (una pierna)
        r_stab = self.zmp_and_smooth_reward(pos, euler, step_count)
        if support_desired == 'L':
            r_support = 1.0 if (Ftot_L > Fmin and Ftot_R < 0.4*Fmin) else 0.0
            swing_foot_id = env.right_foot_link_id
        else:
            r_support = 1.0 if (Ftot_R > Fmin and Ftot_L < 0.4*Fmin) else 0.0
            swing_foot_id = env.left_foot_link_id

        # 2) Clearance del pie en el aire (~10–15 cm)
        ls = p.getLinkState(env.robot_id, swing_foot_id, computeForwardKinematics=1)
        foot_z = ls[0][2]; base_z = pos[2]
        clearance = max(0.0, min((foot_z - (base_z - 0.20)) / 0.15, 1.0))

        # 3) Postura y pie soporte plano
        r_post = np.exp(-abs(pitch)/0.25) * np.exp(-abs(roll)/0.20)
        r_flat = self._foot_flat_reward()

        # 4) Suavidad/energía
        u = np.clip(action, 0.0, 1.0)
        if not hasattr(self, "_prev_ulift"): self._prev_ulift = np.zeros_like(u)
        energy = float(np.mean(u)); delta_u = float(np.mean(np.abs(u - self._prev_ulift))); self._prev_ulift = u

        # 5) Caída
        fall = 0.0
        if (base_z < 0.75) or (abs(pitch) > 0.7) or (abs(roll) > 0.7):
            fall = 1.0; self.last_done_reason = "fall_lift"; self._episode_done = True

        # Pesos
        w_stab, w_sup, w_clear, w_post, w_flat = 0.6, 0.8, 0.8, 0.4, 0.2
        w_en, w_du, w_fall = 0.04, 0.08, 5.0
        reward = (w_stab*r_stab + w_sup*r_support + w_clear*clearance + w_post*r_post + w_flat*r_flat
                  - w_en*energy - w_du*delta_u - w_fall*fall)
        return float(reward)
    
    # ===================== NUEVO: Marcha en el sitio (alternar piernas; se permiten “vuelos”) =====================
    def calculate_reward_march_in_place(self, action, step_count):
        env = self.env
        pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        lin_vel, _ = p.getBaseVelocity(env.robot_id)
        vx, vy = lin_vel[0], lin_vel[1]

        # Ancla de referencia para “no moverse del sitio”
        if step_count == 1 or not hasattr(self, "_anchor_xy"):
            self._anchor_xy = (pos[0], pos[1])
            self._last_swing = None
        ax, ay = self._anchor_xy
        dx, dy = pos[0]-ax, pos[1]-ay

        # Fuerzas (pueden ser ~0 si ambos pies están en el aire; aquí NO lo penalizamos)
        F_L = env.contact_normal_force(env.left_foot_link_id); Ftot_L = F_L[1] if isinstance(F_L, tuple) else 0.0
        F_R = env.contact_normal_force(env.right_foot_link_id); Ftot_R = F_R[1] if isinstance(F_R, tuple) else 0.0

        # 1) Alternancia: premiar cambio de pie en swing (detectar pie más “ligero”)
        swing_now = 'L' if Ftot_L < Ftot_R else 'R'
        switched = (self._last_swing is not None and swing_now != self._last_swing)
        self._last_swing = swing_now
        r_alt = 1.0 if switched else 0.0

        # 2) Clearance del pie en swing (~10–15 cm sobre base)
        foot_id = env.left_foot_link_id if swing_now=='L' else env.right_foot_link_id
        ls = p.getLinkState(env.robot_id, foot_id, computeForwardKinematics=1)
        foot_z = ls[0][2]; base_z = pos[2]
        clearance = max(0.0, min((foot_z - (base_z - 0.20)) / 0.15, 1.0))

        # 3) Quedarse “en el sitio”: baja velocidad y baja deriva XY
        r_stay = np.exp(- (abs(vx)+abs(vy)) / 0.3) * np.exp(- (abs(dx)+abs(dy)) / 0.25)

        # 4) Postura (roll/pitch bajos). ZMP sólo tiene sentido con contacto; si ambos en el aire, lo omitimos
        r_post = np.exp(-abs(pitch)/0.25) * np.exp(-abs(roll)/0.20)
        r_stab = 0.0
        if (Ftot_L > 10.0 or Ftot_R > 10.0):
            r_stab = self.zmp_and_smooth_reward(pos, euler, step_count)

        # 5) Suavidad/energía
        u = np.clip(action, 0.0, 1.0)
        if not hasattr(self, "_prev_umin"): self._prev_umin = np.zeros_like(u)
        energy = float(np.mean(u)); delta_u = float(np.mean(np.abs(u - self._prev_umin))); self._prev_umin = u

        # 6) Caída (permitimos “vuelo” —ambos pies en el aire— pero no caídas reales)
        fall = 0.0
        if (pos[2] < 0.75) or (abs(pitch) > 0.7) or (abs(roll) > 0.7):
            fall = 1.0; self.last_done_reason = "fall_march"; self._episode_done = True

        # Pesos
        w_alt, w_clear, w_stay, w_post, w_stab = 0.9, 0.7, 0.9, 0.3, 0.3
        w_en, w_du, w_fall = 0.04, 0.08, 5.0
        reward = (w_alt*r_alt + w_clear*clearance + w_stay*r_stay + w_post*r_post + w_stab*r_stab
                  - w_en*energy - w_du*delta_u - w_fall*fall)
        return float(reward)