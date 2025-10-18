# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

from collections import deque

from Archivos_Recompensas.MetodosRecompensas import (
    height_reward_method, contacto_pies_reward, knee_reward_method, hip_roll_align_reward,
    pitch_stability_rewards, com_zmp_stability_reward, com_projection_reward,
    seleccion_fuerzas, proximity_legs_penalization, soft_center_bonus, soft_range_bonus,
    EffortWeightScheduler, effort_cost_proxy   # <-- NUEVO
)

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
            1: 1.2,  # 
            2: 1.1,  # 
            3: 1.0   # 
        }
        if self.env.logger:
            self.env.logger.log("main",f"   Max tilt: {np.degrees(self.max_tilt_by_level[3]):.1f}° (permisivo)")
        
        # Para alternancia de piernas (solo nivel 3)
        self.target_leg = getattr(env, "fixed_target_leg", None)  # 'left'|'right'|None self.fixed_target_leg
        self.fixed_target_leg=self.target_leg
        self.switch_timer = 0
        # self.leg_switch_bonus = 0.0  # Bonus por cambio exitoso
        self.bad_ending=("fall", "tilt", "drift")
        # Debug para confirmar configuración
        self.min_F=20
        self.reawrd_step=self.env.reawrd_step
        # --- Effort weight scheduler ---
        self.effort = EffortWeightScheduler() #q=0.6, b=0.95, delta_a=1e-3, lam=0.5, a_min=0.0, a_max=0.5
        self._task_accum = 0.0
        self._task_N = 0
        self._task_score_sum = 0.0
        self._task_score_n = 0
        self._prev_u_for_effort = None  # para suavidad (du^2)
        if self.env.logger:
            self.env.logger.log("main",f"🎯 Progressive System initialized:")
            self.env.logger.log("main",f"   Frequency: {self.frequency_simulation} Hz")
            self.env.logger.log("main",f"🎯 Simple Progressive System: Starting at Level {self.level}")

    def _accumulate_task_term(self, r_task: float):
        try:
            self._task_accum += float(r_task)
            self._task_N += 1
        except Exception:
            pass

    def end_of_episode_hook(self):
        """
        Llamar al finalizar un episodio (env.step detecta done=True).
        Actualiza el peso de esfuerzo usando el retorno medio de TAREA.
        """
        if self._task_N > 0:
            task_return = self._task_accum / self._task_N
            new_a = self.effort.update_after_episode(task_return)
            if self.env and self.env.logger:
                self.env.logger.log("main", f"🟣 Effort weight updated: a_t={new_a:.4f} (r_task_mean={task_return:.3f})")
        # reset acumuladores
        self._task_accum = 0.0
        self._task_N = 0
        self._prev_u_for_effort = None

    def pop_episode_task_score(self) -> float:
        """Score medio de tarea acumulado en este episodio (0..1), y resetea acumuladores."""
        if self._task_score_n <= 0:
            return 0.0
        s = self._task_score_sum / self._task_score_n
        self._task_score_sum = 0.0
        self._task_score_n = 0
        return float(np.clip(s, 0.0, 1.0))
    
    def calculate_reward(self, action, step_count):
        """
        Método principal: calcula reward según el nivel actual
        """
        self.com_x,self.com_y,self.com_z=self.env.com_x,self.env.com_y,self.env.com_z
        self.zmp_x, self.zmp_y=self.env.zmp_x, self.env.zmp_x
        self.vel_COM=self.env.vel_COM
        # Decido usar este método para crear varias opciones de creación de recompensas. Else, curriculo clásico
        if getattr(self, "mode", RewardMode.PROGRESSIVE).value == RewardMode.WALK3D.value:
            return self.calculate_reward_walk3d(action, step_count)
        if getattr(self, "mode", RewardMode.PROGRESSIVE).value == RewardMode.LIFT_LEG.value:
            return self.calculate_reward_lift_leg(action, step_count)
        if getattr(self, "mode", RewardMode.PROGRESSIVE).value == RewardMode.MARCH_IN_PLACE.value:
            return self.calculate_reward_march_in_place(action, step_count)
        else:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
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

        height_reward= height_reward_method(pos[2])

        torso_pitch = euler[1]
        torso_roll = euler[0]
        back_pitch_pen = - np.clip(torso_pitch - 0.05, 0.0, 0.30) * 6.0

        pie_izquierdo_contacto, pie_derecho_contacto = self.env.contacto_pies
        #contacto_reward= contacto_pies_reward(pie_izquierdo_contacto, pie_derecho_contacto)

        knee_reward = knee_reward_method(self, self.env.left_knee_angle, self.env.right_knee_angle)

        # === Bonus por 'usar' roll para recentrar COM sobre el soporte ===
        # Afecta al roll de align bonus
        hiproll_align_bonus = hip_roll_align_reward(pie_izquierdo_contacto, pie_derecho_contacto, torso_roll, self.frequency_simulation)

        self.reawrd_step['height_reward'] = height_reward
        self.reawrd_step['drift_pen'] = drift_pen
        self.reawrd_step['back_only_pen'] = back_only_pen
        self.reawrd_step['back_pitch_pen'] = back_pitch_pen
        self.reawrd_step['hiproll_align_bonus'] = hiproll_align_bonus
        #self.reawrd_step['contact_reward']=contacto_reward
        self.reawrd_step['lateral_pen']=lateral_pen

        return height_reward + drift_pen + back_only_pen +back_pitch_pen  + hiproll_align_bonus + lateral_pen + knee_reward
    
    def _level_2_reward(self,pos,euler):
        """NIVEL 2: Balance estable (recompensas 0-5)"""
        
        height_reward=self._level_1_reward(pos, euler)
        

        roll,pitch = euler[0] , abs(euler[1])  # roll + pitch
        stability_reward = pitch_stability_rewards(self, pitch)
    
    # Guardarraíl adicional de roll (torso) para evitar extremos
        guard_pen = self._roll_guardrail_pen(
            roll,
            level_soft=0.15,
            level_hard=self.max_tilt_by_level[self.level]
        )
        # Activar términos que ya tenías calculados
        comz     = com_zmp_stability_reward(self)   # estabilidad COM/ZMP
        comproj  = com_projection_reward(self)      # proyección COM dentro del soporte

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
        left_hip_roll_id, left_hip_pitch_id, left_knee_id, left_ankle_id, \
        right_hip_roll_id, right_hip_pitch_id, right_knee_id,right_ankle_id = self.env.joint_indices
        
        state_L, nL, FL = self.env.foot_contact_state(self.env.left_foot_link_id,  f_min=self.min_F)
        state_R, nR, FR = self.env.foot_contact_state(self.env.right_foot_link_id, f_min=self.min_F)
        F_sum = max(FL + FR, 1e-6)
        
        F_sup, F_tar, support_is_left,\
              support_state, target_state = seleccion_fuerzas(state_L,state_R, self.fixed_target_leg,FL, FR)

        # (1) Penaliza doble apoyo fuerte
        both_down_pen = -1.0 if (state_L == self.env.footcontact_state.PLANTED.value and state_R == self.env.footcontact_state.PLANTED.value) else 0.0

        # (2) Toe-touch sólo si el pie objetivo es TOUCH (no PLANTED)
        toe_touch_pen = -0.6 if (target_state == self.env.footcontact_state.TOUCH.value) else 0.0

        # (3) Recompensa reparto de carga sano: ≥80% en el pie de soporte
        ratio = F_sup / F_sum
        support_load_reward = np.clip((ratio - 0.80) / 0.20, 0.0, 1.0) * 1.0 if (support_state == self.env.footcontact_state.PLANTED.value) else 0.0

        # (2.5) Penalización por casi-cruce entre el pie en swing y el pie de soporte
        close_pen = proximity_legs_penalization(self, F_sup, F_tar, self.env.left_foot_link_id, self.env.right_foot_link_id, support_is_left)
        # (4) Tiempo en apoyo simple SÓLO si soporte=PLANTED y objetivo no PLANTED
        ss_reward = self._single_support_dwell_reward(FL if support_is_left else FR,
                                              FR if support_is_left else FL,
                                              self.frequency_simulation) if (support_state == self.env.footcontact_state.PLANTED.value
                                                                              and target_state != self.env.footcontact_state.PLANTED.value) else 0.0
        
        # (5) Recompensa de pie soporte "plano" solo si PLANTED
        if support_state == self.env.footcontact_state.PLANTED.value:
            stance_foot_id = self.env.left_foot_link_id if support_is_left else self.env.right_foot_link_id
            flat_reward = self._foot_flat_reward(stance_foot_id, only_if_contact=True)
        else:
            flat_reward = 0.0

        # --- Bonuses de forma SOLO si el pie objetivo NO está en contacto ---
        # Clearance
        swing_foot_id = (self.env.right_foot_link_id if support_is_left else self.env.left_foot_link_id)
        ls = p.getLinkState(self.robot_id, swing_foot_id, computeForwardKinematics=1)
        foot_z = ls[0][2]
        clearance_target = 0.09  # 9 cm
        clearance_bonus = (np.clip(foot_z / clearance_target, 0.0, 1.0) * 0.5
                           if target_state == self.env.footcontact_state.NONE.value else 0.0)

        
        # Rodilla (rango recomendado 0.45–0.75 rad)
        knee_id  = right_knee_id if support_is_left else left_knee_id
        knee_ang = p.getJointState(self.robot_id, knee_id)[0]
        knee_bonus = soft_range_bonus(knee_ang, self.swing_knee_lo, self.swing_knee_hi, slope=0.20) * 1.0
        knee_bonus = 0.0 if target_state else knee_bonus

        # Cadera (pitch) del swing — objetivo configurable, con meseta ± self.swing_hip_tol
        hip_id  = right_hip_pitch_id if support_is_left else left_hip_pitch_id
        hip_ang = p.getJointState(self.robot_id, hip_id)[0]
        
        # después (direccional):
        desired_sign = -1.0  # pon -1.0 si en tu robot la flexión hacia delante es negativa
        hip_bonus_dir = soft_center_bonus(desired_sign * hip_ang,
                                        self.swing_hip_target, self.swing_hip_tol,
                                        slope=0.20) * 0.7
        hip_bonus = hip_bonus_dir if target_state == self.env.footcontact_state.NONE.value else 0.0

        # ✅ Bono de hip ROLL del swing (abducción cómoda)
        swing_roll_jid = (left_hip_roll_id if support_is_left else right_hip_roll_id)
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
        self.reawrd_step['flat_reward']=flat_reward
        self.reawrd_step['close_pen'] = close_pen
        leg_reward = (clearance_bonus + knee_bonus + hip_bonus
                      + roll_swing_bonus + flat_reward
                      + shaping + speed_pen+ close_pen) # contacto_reward
        # Recompensa por pie de soporte 'plano' (planta paralela al suelo)
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

    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        # Penalizo deriva frente y lateral
        self.dx = float(pos[0])
        self.dy = float(pos[1])
        # Caída
        if pos[2] <= 0.5:
            self.last_done_reason = "fall"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Robot fell")
            return True
        
        if self.dx < -1.0 or abs(self.dy)>1.0:
            self.last_done_reason = "drift"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Excessive longitudinal drift")
            return True
        
        max_tilt = self.max_tilt_by_level.get(self.level, 1.0)
        # Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            self.last_done_reason = "tilt"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Robot tilted too much")
            return True
        
        # dentro de is_episode_done(...) tras calcular contactos:
        # if self.mode == RewardMode.MARCH_IN_PLACE.value:
        #     fin_simulacion_no_contacto = self.frequency_simulation
        #     fin_simulacion_exceso_contacto = self.frequency_simulation//2  
        # else:
        #     fin_simulacion_no_contacto = 0.5 * self.frequency_simulation  # 3 segundos de gracia
        #     fin_simulacion_exceso_contacto = 0.8 * self.frequency_simulation
        # pie_izquierdo_contacto, pie_derecho_contacto = self.env.contacto_pies   # si ya tienes util; si no, usa getContactPoints
        # if not (pie_izquierdo_contacto or pie_derecho_contacto):
        #     self._no_contact_steps += 1
        # else:
        #     self._no_contact_steps = 0
        # if self._no_contact_steps >= fin_simulacion_no_contacto:  # 0.5 s
        #     self.last_done_reason = "no_support"
        #     if self.env.logger:
        #         self.env.logger.log("main","❌ Episode done: No foot support for too long")
        #     return True
        # if (pie_izquierdo_contacto and pie_derecho_contacto):
        #     self.contact_both += 1
        #     if self.contact_both >fin_simulacion_exceso_contacto:# 0.8 s
        #         self.last_done_reason = "excessive support"
        #         if self.env.logger:
        #             self.env.logger.log("main","❌ Episode done: excessive support")
        #         return True
        # else:
        #     self.contact_both=0
        # Tiempo máximo (crece con nivel)
        max_steps =  6000 # 6000 steps
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
        
    # ============================================================================================================================================= #
    # ================================================= Nuevos metodos de recompensa para nuevas acciones ========================================= #
    # ============================================================================================================================================= #

    # ===================== NUEVO: Caminar 3D =====================
    def calculate_reward_walk3d(self, action, step_count):
        env = self.env
        pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        # Da la velocidad lineal y angular de la pelvis
        #lin_vel, ang_vel = p.getBaseVelocity(env.robot_id)
        vx = float(self.vel_COM[0])

        # Contactos y fuerzas
        (state_L, n_l, FL) = env.foot_contact_state(env.left_foot_link_id, f_min=self.min_F)
        (state_R, n_r, FR) = env.foot_contact_state(env.right_foot_link_id, f_min=self.min_F)

        L_on = (state_L == self.env.footcontact_state.PLANTED.value)
        R_on = (state_R == self.env.footcontact_state.PLANTED.value)

        # Contacto de ritmo de paso, claridad y alternancia
        # 3) Patrón de paso ligero: pie soporte claro + alternancia (con histéresis)
        # le doy un peso de 0.4  # Evita "chatter" cuando FL≈FR usando un umbral de cambio.
        deltaF = FL - FR
        hysteresis = getattr(self, "_support_hyst", 10.0)  # N, ajusta según tu escala de fuerzas
        prev_support = getattr(self, "_last_support3d", 'L' if FL >= FR else 'R')
        if deltaF > +hysteresis:
            support_now = 'L'
        elif deltaF < -hysteresis:
            support_now = 'R'
        else:
            support_now = prev_support
        b_clear = 0.10 if ((support_now=='L' and L_on and not R_on) or
                   (support_now=='R' and R_on and not L_on)) else 0.0
        if step_count == 1: 
            self._last_support3d = support_now
        switched = (support_now != getattr(self, "_last_support3d", support_now))
        self._last_support3d = support_now
        # bonus pequeño por cambio de soporte y por "soporte claro"
        b_switch = 0.10 if switched else 0.0
        r_contact = b_switch + b_clear

        # --- Posiciones de pies y base ---
        Lpos = p.getLinkState(env.robot_id, env.left_foot_link_id, computeForwardKinematics=1)[0]
        Rpos = p.getLinkState(env.robot_id, env.right_foot_link_id, computeForwardKinematics=1)[0]
        com_x = float(self.com_x) # Pos x del COM

        # --- 1) Velocidad objetivo (campana) ---
        # Si se quiere "sigma" estadístico, usa -0.5*((...)/sigma)^2; si prefieres más estrecha, quita el 0.5.
        v_tgt   = float(getattr(self, "vx_target", 0.25))
        sigma_v = 0.20
        r_vel = np.exp(-0.5*((vx - v_tgt)/sigma_v)**2) # Si no aumenta el valor para dif bajas derucir constante multiplicacion
        self._accumulate_task_term(r_vel)
        self.reawrd_step['r_vel_raw'] = r_vel

        
        alive = 0.2
        w_v, w_post = 2.4, 0.6
        w_cont, w_step, w_ahead = 0.4, 0.9, 0.3
        # --- Pesos (usar effort weight adaptativo) ---
        a_eff = float(getattr(self.env, "effort_weight", 0.02))  # NEW
        w_en, w_du = max(0.0, a_eff), max(0.0, 3.0*a_eff)        # NEW: E y derivada (suavidad)
        w_stuck, w_fall = 1.5, 5.0

        # --- 2) Mantenerse erguido simple (con leve "forward lean" positivo) ---
        # Para tu morfología sin tronco, un leve pitch negativo puede ser preferible
        pitch_tgt = -0.05  # ~3°
        r_post = np.exp(- ((roll/0.25)**2) - (((pitch - pitch_tgt)/0.30)**2))

        # --- 1.b) Estabilidad vertical (COM/altura) y rampas de currículo ---
        # Objetivo: mantener base_z cerca de su valor de referencia antes de priorizar avance
        base_z = pos[2]
        if step_count == 1:
            # referencia razonable: altura inicial de la base
            self._base_z_ref = float(getattr(self.env, "init_pos", (0,0,base_z))[2]) if hasattr(self.env, "init_pos") else float(base_z)
            self._prev_vx_for_impulse = float(vx)
        z_ref = float(getattr(self, "base_z_target", getattr(self, "_base_z_ref", base_z)))
        sigma_z = float(getattr(self, "base_z_sigma", 0.03))  # ~3 cm
        r_vert = float(np.exp(- ((base_z - z_ref)/max(1e-6, sigma_z))**2))
        # rampas: reduce temporalmente el peso de avance al principio del episodio y de cada paso
        warmup_steps = int(getattr(self, "warmup_steps_walk", 2*self.frequency_simulation))  # ~2 s
        step_ramp = min(1.0, max(0.0, float(step_count)/max(1, warmup_steps)))
        # también escala por episodios para un currículo temprano de equilibrio
        episodes_ramp = int(getattr(self, "warmup_episodes_walk", 10))
        epi_factor = min(1.0, max(0.0, float(getattr(self, "episode_count", 0))/max(1, episodes_ramp)))
        ramp_factor = step_ramp * epi_factor  # 0→1
        # (opcional) pequeña penalización de 'aceleración' para amortiguar impulsos iniciales
        dv = float(abs(vx - getattr(self, "_prev_vx_for_impulse", vx)))
        self._prev_vx_for_impulse = float(vx)

        

        # --- 4) Step length al IMPACTO: pie nuevo por delante del COM ---
        # detecta nuevo impacto (transición off->on del pie que NO estaba soportando)
         # Inicializa flags antes de usarlos y usa getattr para evitar accesos no definidos en step 1
        if step_count == 1:
            self._L_on_prev = L_on
            self._R_on_prev = R_on
            self._last_step_len = 0.0
        # Detecta impacto por transición off->on, sin exigir soporte_now igual (evita perder impactos en doble apoyo)
        impact_L = (L_on and not getattr(self, "_L_on_prev", False))
        impact_R = (R_on and not getattr(self, "_R_on_prev", False))
        
        #peso aportado: 0.9
        step_len_tgt = 0.18  # 18 cm
        step_len_tol = 0.06  # meseta ±6 cm
        r_step = 0.0
        if impact_L:
            step_len = float(Lpos[0] - com_x)  # por delante del COM
            self._last_step_len = step_len
        elif impact_R:
            step_len = float(Rpos[0] - com_x)
            self._last_step_len = step_len
        else:
            step_len = getattr(self, "_last_step_len", 0.0)

        # "soft center" alrededor del target
        def soft_center(x, c, tol, slope=0.10):
            if x < c - tol: return max(0.0, 1.0 - (c - tol - x)/slope)
            if x > c + tol: return max(0.0, 1.0 - (x - (c + tol))/slope)
            return 1.0
        r_step = soft_center(step_len, step_len_tgt, step_len_tol)

        # --- 5) Swing ADELANTE del COM (shaping en el aire, pequeño) ---
        # Si el pie está en el aire, anímalo a ir por delante del COM
        ahead_margin = 0.06  # 6 cm
        r_ahead = 0.0
        if not L_on:
            r_ahead += 1.0 if (Lpos[0] > com_x + ahead_margin) else 0.0
        if not R_on:
            r_ahead += 1.0 if (Rpos[0] > com_x + ahead_margin) else 0.0
        r_ahead *= 0.5  # normaliza a [0,1]

        # --- 6) Suavidad / energía ---
        u = np.clip(action, 0.0, 1.0)
        if not hasattr(self, "_prev_u3d"):
            self._prev_u3d = np.zeros_like(u)

         # --- 7) Caída y "no avanzar" ---
        fall = 0.0
        if self.last_done_reason in self.bad_ending:
            fall = 1.0

        # penaliza "atasco" si tras 1.0 s el promedio de vx < 0.03 m/s
        if step_count == 1:
            self._vx_accum = 0.0; self._vx_n = 0
        self._vx_accum += vx; self._vx_n += 1
        r_stuck = 0.0
        if self._vx_n >= int(self.frequency_simulation):  # ~1 s
            if (self._vx_accum / self._vx_n) < 0.03:
                r_stuck = 0.4  # penaliza atascos
            self._vx_accum = 0.0; self._vx_n = 0

         # --- 6b) Coste de esfuerzo con descomposición energía/suavidad/esparsidad ---
        eff_base, eff_dbg = effort_cost_proxy(u, getattr(self, "_prev_u3d", None),
                                      activity_pow=3.0, w_smooth=0.7, w_sparse=0.1,
                                      act_threshold=0.15)
        # Usa w_en (energía) y w_du (suavidad) aquí; mantenemos esparsidad ligada a a_eff (más suave).
         # Si tu proxy devuelve componentes en eff_dbg, combínalas explícitamente:
        eff_energy = float(eff_dbg.get('activity', 0.0))   # "energía"/actividad
        eff_smooth = float(eff_dbg.get('smooth', 0.0))     # derivada / smoothness
        eff_sparse = float(eff_dbg.get('sparse', 0.0))     # sparsidad
        
        eff_cost = (w_en * eff_energy) + (w_du * eff_smooth) + (0.5 * a_eff * eff_sparse)
        # # Si prefieres, aún puedes escalar todo por self.effort.a_t:
        # eff_cost *= float(self.effort.a_t) # Anteriormente era esto eff_cost = a_eff * eff_base
        #eff_cost = a_eff * eff_base
        self._prev_u3d = u

        # Ajuste de pesos con rampas (avance reducido al principio)
        w_v_eff = w_v * float(ramp_factor)
        # El término vertical pesa más al principio y se apaga al avanzar el step
        w_vert = float(getattr(self, "w_vert", 0.8)) * (1.0 - float(step_ramp))
        # Penalización suave de 'aceleración' (cambios bruscos de velocidad del cuerpo)
        w_acc  = float(getattr(self, "w_accel_pen", 0.05))

        reward = (alive
                + w_v_eff*r_vel
                + w_post*r_post
                + w_cont*r_contact
                + w_step*r_step
                + w_ahead*r_ahead
                + w_vert*r_vert
                + self.dx*2
                - abs(self.dy)
                - w_acc*dv
                - w_stuck*r_stuck
                - w_fall*fall
                -eff_cost)
        
        # logging
        self.reawrd_step['alive'] = alive
        self.reawrd_step['r_vel'] = w_v*r_vel
        self.reawrd_step['r_post'] = w_post*r_post
        self.reawrd_step['r_contact'] = w_cont*r_contact
        self.reawrd_step['r_step'] = w_step*r_step
        self.reawrd_step['r_ahead'] = w_ahead*r_ahead
        self.reawrd_step['r_vert']  = w_vert*r_vert
        self.reawrd_step['accel_pen'] = - w_acc*dv
        self.reawrd_step['stuck_pen'] = - w_stuck*r_stuck
        self.reawrd_step['fall_pen'] = - w_fall*fall
        self.reawrd_step['effort_weight'] = float(getattr(self.env, "effort_weight", 0.0))
        self.reawrd_step['effort_cost'] = - float(eff_cost)
        self.reawrd_step['effort_activity'] = eff_dbg.get('activity', 0.0)
        self.reawrd_step['effort_smooth'] = eff_dbg.get('smooth', 0.0)
        self.reawrd_step['effort_sparse'] = eff_dbg.get('sparse', 0.0)

        # actualizar flags de contacto para siguiente paso
        self._L_on_prev, self._R_on_prev = L_on, R_on
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
            state_L, nL, FL = self.env.foot_contact_state(self.env.left_foot_link_id,  f_min=self.min_F)
            state_R, nR, FR = self.env.foot_contact_state(self.env.right_foot_link_id, f_min=self.min_F)
            support_desired = 'L' if FL >= FR else 'R'

        # Fuerzas
        state_L, nL, FL = self.env.foot_contact_state(self.env.left_foot_link_id,  f_min=self.min_F)
        state_R, nR, FR = self.env.foot_contact_state(self.env.right_foot_link_id, f_min=self.min_F)
        Fmin = 30.0

        # 1) Estabilidad + soporte claro (una pierna)
        r_stab = self.zmp_and_smooth_reward()
        if support_desired == 'L':
            r_support = 1.0 if (FL > Fmin and FR < 0.4*Fmin) else 0.0
            swing_foot_id = env.right_foot_link_id
        else:
            r_support = 1.0 if (FR > Fmin and FL < 0.4*Fmin) else 0.0
            swing_foot_id = env.left_foot_link_id

        # 2) Clearance del pie en el aire (~10–15 cm)
        ls = p.getLinkState(env.robot_id, swing_foot_id, computeForwardKinematics=1)
        # foot_z es el comz del robot
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
        if (base_z < 0.5) or (abs(pitch) > 0.7) or (abs(roll) > 0.7):
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
        state_L, nL, FL = self.env.foot_contact_state(self.env.left_foot_link_id,  f_min=self.min_F)
        state_R, nR, FR = self.env.foot_contact_state(self.env.right_foot_link_id, f_min=self.min_F)

        # 1) Alternancia: premiar cambio de pie en swing (detectar pie más “ligero”)
        swing_now = 'L' if FL < FR else 'R'
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
        if (FL > 10.0 or FR > 10.0):
            r_stab = self.zmp_and_smooth_reward() #porque se puso pos, euler, step_count

        # 5) Suavidad/energía 
        u = np.clip(action, 0.0, 1.0)
        if not hasattr(self, "_prev_umin"): self._prev_umin = np.zeros_like(u)
        energy = float(np.mean(u)); delta_u = float(np.mean(np.abs(u - self._prev_umin))); self._prev_umin = u

        # 6) Caída (permitimos “vuelo” —ambos pies en el aire— pero no caídas reales)
        fall = 0.0
        if self.last_done_reason in self.bad_ending:
            fall = 1.0 
            self._episode_done = True

        # Pesos
        w_alt, w_clear, w_stay, w_post, w_stab = 0.9, 0.7, 0.9, 0.3, 0.3
        w_en, w_du, w_fall = 0.04, 0.08, 5.0
        reward = (w_alt*r_alt + w_clear*clearance + w_stay*r_stay + w_post*r_post + w_stab*r_stab
                  - w_en*energy - w_du*delta_u - w_fall*fall)
        
        self.reawrd_step['r_alt'] = w_alt*r_alt
        self.reawrd_step['clearance'] = w_clear*clearance
        self.reawrd_step['r_stay'] = w_stay*r_stay
        self.reawrd_step['r_post'] = w_post*r_post
        self.reawrd_step['r_stab'] = w_stab*r_stab
        self.reawrd_step['energy_pen'] = - w_en*energy
        self.reawrd_step['delta_u_pen'] = - w_du*delta_u
        self.reawrd_step['fall_pen'] = - w_fall*fall
        return float(reward)
    