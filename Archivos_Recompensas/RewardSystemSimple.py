# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

from collections import deque

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
        self.foot_links=(self.env.left_foot_link_id, self.env.right_foot_link_id)
        # ===== CONFIGURACIÓN SUPER SIMPLE =====
        self.episode_count = 0
        self.recent_episodes = deque(maxlen=5)  # Últimos 5 episodios
        self.success_streak = 0  # Episodios consecutivos exitosos
        self._no_contact_steps = 0
        self.contact_both = 0
        
        self.target_leg = getattr(env, "fixed_target_leg", None)  # 'left'|'right'|None self.fixed_target_leg
        self.fixed_target_leg=self.target_leg
        self.switch_timer = 0
        # self.leg_switch_bonus = 0.0  # Bonus por cambio exitoso
        self.bad_ending=("fall", "tilt", "drift")
        # Debug para confirmar configuración
        self.min_F=20
        self.reawrd_step=self.env.reawrd_step
        # --- Effort weight scheduler ---
        self._task_accum = 0.0
        self._task_N = 0
        self._task_score_sum = 0.0
        self._task_score_n = 0
        self.action_previous = None  # para suavidad (du^2)
        # mean reward running, switching mean value and
        self.r_mean=0
        self.s_mean=0
        self.alpha_t=0
        self.smoothing=0.99
        self.threshold = 0.4
        self.change_adaptation_rate = 1e-3
        self.decay_term=0.1
        self._vx_target=self.env.vx_target
        self.com_z_star = 0.69

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

    def reset(self):
        self.action_previous = None
        self.r_mean=0
        self._task_accum = 0.0
        self._task_N = 0
        self.s_mean=0

    def calculate_s_mean(self):
        s_target=1 if self.r_mean>self.threshold else 0
        self.s_mean=self.s_mean*self.smoothing + (1-self.smoothing)*s_target
        self.s_mean=np.clip(self.s_mean,0.0,1.0)

    def calculate_reward(self, action, torque_mapping, step_count):
        """
        Método principal: calcula reward según el nivel actual
        """
        self.com_x,self.com_y,self.com_z=self.env.com_x,self.env.com_y,self.env.com_z
        self.zmp_x, self.zmp_y=self.env.zmp_x, self.env.zmp_y
        self.vel_COM=self.env.vel_COM
        # Decido usar este método para crear varias opciones de creación de recompensas. Else, curriculo clásico
        if getattr(self, "mode", RewardMode.PROGRESSIVE).value == RewardMode.WALK3D.value:
            return self.calculate_reward_walk3d(action, torque_mapping, step_count)
        else:
            raise Exception ("Solo se acepta caminar ahora")

    
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
        
        max_tilt = 0.8
        #Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            self.last_done_reason = "tilt"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Robot tilted too much")
            return True

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
            'curriculum_enabled': False,
            'level_progression_disabled': getattr(self, 'level_progression_disabled', False)
        }
    
    # ============================================================================================================================================= #
    # ================================================= Nuevos metodos de recompensa para nuevas acciones ========================================= #
    # ============================================================================================================================================= #

    # ===================== NUEVO: Caminar 3D =====================
    def calculate_reward_walk3d(self, action, torque_mapping:dict, step_count):
        env = self.env
        pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        # Da la velocidad lineal y angular de la pelvis
        #lin_vel, ang_vel = p.getBaseVelocity(env.robot_id)
        vx = float(self.vel_COM[0])
        vy = float(self.vel_COM[1])
        #r_pitch=0.2 * (roll**2 + pitch**2) Versión anterior tardaba demasiado tiempo

        # --- NORMALIZATION: tolerances and half-life mapping ---
        d_theta = np.deg2rad(5.0)   # 5 degrees tolerance for roll/pitch
        dz_band = 0.02              # 2 cm deadband for CoM height
        d_s     = 0.04              # 4 cm CoM->support (si lo usas)
        dv_foot = 0.05              # 5 cm/s no-slip
        dv_cmd  = 0.20              # 0.20 m/s vel tracking (x)
        dy_pos  = 0.08              # 8 cm tolerancia lateral (y)
        dvy     = 0.10              # 0.10 m/s vel lateral
        d_back  = 0.05              # 0.05 m/s tolerancia hacia atrás (más severo)

        r_post = exp_term(roll, d_theta) * exp_term(pitch, d_theta)

        # Altura del CoM (absoluta) alrededor de z*
        z_star = getattr(self, "com_z_star", 0.69)
        e_z = band_error(self.env.com_z, z_star, dz_band)
        r_z  = exp_term(e_z, dz_band, r_at_tol=0.6)

        # Velocidad del CoM: |vx - vcmd|
        vcmd = float(getattr(self, "_vx_target",1.2))
        v_err = abs(vx - vcmd)
        if vx >= vcmd:
            r_vel = 1.0
        elif vx<0:
            r_vel=0
        else:
            r_vel = exp_term(v_err, dv_cmd, r_at_tol=0.5)

        # Lateral: posición y velocidad (objetivo y*=0, vy*=0)
        y =self.dy
        r_lat_pos = exp_term(abs(y),  dy_pos, r_at_tol=0.5)
        r_lat_vel = exp_term(abs(vy), dvy,    r_at_tol=0.5)
        r_lat = r_lat_pos * r_lat_vel

        # Suavidad en presiones (acciones en [0,1])
        accion_previa = self.action_previous if self.action_previous is not None else np.zeros_like(action)
        delta_p = np.asarray(action) - np.asarray(accion_previa)
        r_dp = exp_term(np.linalg.norm(delta_p), 0.05*np.sqrt(len(action)), r_at_tol=0.6)
        
        alive = 0.6
        # Pesos de términos normalizados (ajústalos con tus logs)
        w_v, w_lat, w_post, w_z, w_dp = 0.30, 0.20, 0.15, 0.10, 0.10
        w_tau = 0.10
        r_tau=self.torque_pain_reduction(torque_mapping=torque_mapping)
        #r_GRF=_grf_reward(self.foot_links,env.contact_normal_force, masa_robot=self.env.mass, bw_mult=1.2)

        self._accumulate_task_term(r_vel)
        self.reawrd_step['reward_speed'] = w_v * r_vel
        self.reawrd_step['reward_posture'] = w_post* r_post
        self.reawrd_step['reward_height'] = w_z * r_z
        self.reawrd_step['reward_pressure'] = w_dp * r_dp
        self.reawrd_step['reward_lateral'] = w_lat*r_lat
        
        
        reward = (
                alive
                + w_v   * r_vel
                + w_lat*r_lat
                + w_post* r_post
                + w_z   * r_z
                + w_dp  * r_dp
                + w_tau * r_tau
        )
        # Se guarda la acción previa
        #self.parametro_pesado_acciones()
        self.action_previous=np.array(action)
        # actualizar flags de contacto para siguiente paso
        return float(reward)
    
    # Requisitos minimos threshold, smoothing, change_in_adaptation_rate
    # parametros de entrenamiento: r_mean, temporal adaptation_sate, s_mean
    def parametro_pesado_acciones(self):
        self.r_mean=self.smoothing*self.r_mean +(1-self.smoothing)*self.env.episode_reward

        if self.r_mean >self.threshold and self.s_mean < 0.5:
            self.change_adaptation_rate=self.decay_term*self.change_adaptation_rate
        elif self.r_mean > self.threshold and self.s_mean > 0.5:
            # se genera alpha_t para el next step
            self.alpha_t=self.alpha_t+self.change_adaptation_rate
        else:
            self.alpha_t=self.alpha_t-self.change_adaptation_rate
        #self.alpha_t=np.clip(self.alpha_t,0,0.5)
        self.calculate_s_mean()

    def torque_pain_reduction(self, torque_mapping):
        """
            Recompensa de “bajo dolor” basada en utilización de par por-junta
            usando límites dependientes del ángulo:
                tau ∈ [-tau_max_ext(theta), +tau_max_flex(theta)]
            - Devuelve un valor en [0,1] (1 = nada de dolor).
            - Fallback a clip global si no hay mapas.
        """
        env = self.env
        # 1) ¿Tenemos mapas de límite por ángulo?
        has_maps = hasattr(env, "tau_limit_interp") and isinstance(env.tau_limit_interp, dict) and len(env.tau_limit_interp) > 0
        # 2) Estados actuales de las articulaciones (para θ)
        try:
            joint_states = p.getJointStates(env.robot_id, env.joint_indices)
            joint_positions = [float(s[0]) for s in joint_states]
        except Exception:
            joint_positions = None
            has_maps = False

        tau_utils = []
        if has_maps and joint_positions is not None:
            # === Utilización con límites asimétricos dependientes de θ
            for i, jid in enumerate(env.joint_indices):
                tau_cmd = float(torque_mapping.get(jid, 0.0))
                th_i = joint_positions[i]
                lims = env.tau_limit_interp.get(jid, None)
                if lims is None:
                    continue
                # límites positivos/negativos en el ángulo actual
                tau_flex_max = max(0.0, float(lims["flex"](th_i)))  # + (flex)
                tau_ext_max  = max(0.0, float(lims["ext"](th_i)))   # - (ext)
                denom = tau_flex_max if tau_cmd >= 0.0 else tau_ext_max
                denom = max(denom, 1e-6)  # seguridad
                tau_utils.append(abs(tau_cmd) / denom)
        else:
            # === Fallback: escalado global previo
            joint_tau_scale = getattr(env, "joint_tau_scale", None)
            max_reasonable = float(getattr(env, "MAX_REASONABLE_TORQUE", 240.0))
            for jid, tau_cmd in torque_mapping.items():
                scale = max_reasonable
                if isinstance(joint_tau_scale, dict):
                    scale = float(joint_tau_scale.get(jid, max_reasonable))
                tau_utils.append(abs(float(tau_cmd)) / max(scale, 1e-6))

        if not tau_utils:
            return 1.0  # sin info → sin dolor

        # 3) Agregación tipo RMS de utilización
        u_rms = float(np.sqrt(np.mean(np.square(tau_utils))))

        # 4) Sólo “duele” por encima de la tolerancia
        #    (p.ej., hasta el 60% de utilización promedio no penaliza)
        u_tol = 0.60
        e_tau = max(0.0, u_rms - u_tol)

        # 5) Mapear exceso a recompensa [0,1] (alto => poco dolor)
        #    tol exceso 0.20: a u_rms≈0.80 => r_tau≈0.6
        return exp_term(e_tau, tol=0.20, r_at_tol=0.6)
    
    def _grf_reward(self, foot_links, metodo_fuerzas_pies, masa_robot, bw_mult=1.2,
                    mode="gauss", sigma_bw=0.15):
        """
        Devuelve recompensa en [0,1] a partir del exceso en BW.
        - mode="gauss": r = exp(-0.5 * (exceso_bw / sigma_bw)^2)
        - mode="linear": r = 1 - clip(exceso_bw, 0, 1)
        """
        exceso_bw = _grf_excess_cost_bw(foot_links, masa_robot, metodo_fuerzas_pies, bw_mult)
        if mode == "linear":
            return float(1.0 - np.clip(exceso_bw, 0.0, 1.0))
        return float(np.exp(-0.5 * (exceso_bw / max(sigma_bw, 1e-6))**2))



def exp_term(error, tol, r_at_tol=0.5):
    error = float(error)
    tol = max(float(tol), 1e-9)
    alpha = -np.log(r_at_tol)
    return np.exp(-alpha * (error / tol)**2)

def band_error(x, x_star, deadband):
    return max(0.0, abs(float(x) - float(x_star)) - float(deadband))


def _grf_excess_cost(foot_links, metodo_fuerzas_pies, masa_robot, bw_mult=1.2):
    """
    Suma fuerzas normales en los 'pies' y penaliza solo el exceso sobre 1.2x BW.
    Devuelve N (no negativo).
    """
    # Suma normalForce solo para contactos de cada pie con cualquier objeto (suelo)
    Fz_total = 0.0
    for link in foot_links:
        _, _, F_foot= metodo_fuerzas_pies(link_id=link)
        Fz_total += float(F_foot)

    BW = masa_robot*9.81  # N
    thr = bw_mult * BW
    return float(max(0.0, Fz_total - thr))

def _grf_excess_cost_bw(foot_links, metodo_fuerzas_pies, masa_robot, bw_mult=1.2):
    """
    Suma fuerzas normales en los 'pies' y penaliza solo el exceso sobre 1.2x BW.
    Devuelve N (no negativo).
    """
    # Suma normalForce solo para contactos de cada pie con cualquier objeto (suelo)
    Fz_total = 0.0
    for link in foot_links:
        _, _, F_foot= metodo_fuerzas_pies(link_id=link)
        Fz_total += float(F_foot)

    BW = masa_robot*9.81  # N
    bw_sum = Fz_total / BW
    return float(max(0.0, bw_sum - float(bw_mult)))
    
    
    