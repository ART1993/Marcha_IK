# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum
import json, os
from collections import deque

class RewardMode(Enum):
    PROGRESSIVE = "progressive"      # curriculum por niveles (modo actual por defecto)
    WALK3D = "walk3d"                # caminar en 3D (avance +X)
    

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

        # MODO SIN CURRICULUM: sistema fijo y permisivo
        self.level = 3  # Siempre nivel 3
        self.level_progression_disabled = True
        self.foot_links=(self.env.left_foot_link_id, self.env.right_foot_link_id)
        # ===== CONFIGURACIÓN SUPER SIMPLE =====
        self.episode_count = 0
        #self.recent_episodes = deque(maxlen=5)  # Últimos 5 episodios
        self._no_contact_steps = 0
        
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
        #self.com_z_star = self.env.init_com_z

        # --- Opciones de guardado (puedes cambiarlas desde fuera) ---
        self.checkpoint_dir = "checkpoints"
        self.autosave_every = 100000//8   # ej.: 5000 para guardar cada 5k pasos; None = desactivado
        self.autoload_scheduler = True # intentar cargar el último estado al iniciar
        # Auto-carga del estado del scheduler si existe alguno
        # if self.autoload_scheduler:
        #     try:
        #         latest = self._find_latest_scheduler_state(self.checkpoint_dir)
        #         if latest is not None:
        #             self.cargar_params_checkpoint(latest)
        #             if self.env.logger:
        #                 self.env.logger.log("main", f"🔁 Scheduler state loaded from: {latest}")
        #     except Exception as _e:
        #         if self.env.logger:
        #             self.env.logger.log("main", f"⚠️ Could not autoload scheduler state: {_e}")

        if self.env.logger:
            self.env.logger.log("main",f"🎯 Progressive System initialized:")
            self.env.logger.log("main",f"   Frequency: {self.frequency_simulation} Hz")
            self.env.logger.log("main",f"🎯 Simple Progressive System: Starting at Level {self.level}")

    def _find_latest_scheduler_state(self, ckpt_dir:str):
        """
        Busca el último archivo scheduler_state_XXXXXXXXX.json en ckpt_dir.
        Devuelve ruta o None si no hay.
        """
        try:
            if not os.path.isdir(ckpt_dir):
                return None
            files = [f for f in os.listdir(ckpt_dir) if f.startswith("scheduler_state_") and f.endswith(".json")]
            if not files:
                return None
            files.sort()  # lexicográfico; con ceros a la izquierda queda por step
            return os.path.join(ckpt_dir, files[-1])
        except Exception:
            return None

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
        #self.alpha_t = 0.0

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
        
        if self.dx < -0.6 or abs(self.dy)>1.0:
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
        
        return {
            'level': self.level,
            'episodes': self.episode_count,
            'target_leg': self.target_leg if self.level == 3 else None,
            'curriculum_enabled': False,
            'level_progression_disabled': getattr(self, 'level_progression_disabled', False)
        }
    
    def guardar_params_checkpoint(self, path, step_count):
        if self.env.step_total % self.autosave_every ==0 and self.env.step_total>0:
            sched_state = {
                            "alpha_t": self.alpha_t,
                            "r_mean": self.r_mean,
                            "s_mean": self.s_mean,
                            "change_adaptation_rate": self.change_adaptation_rate,
                            "smoothing": self.smoothing,
                            "threshold": self.threshold,
                            "decay_term": self.decay_term,
                            "step_total": int(self.env.step_total if hasattr(self.env, "step_total") else self.env.step_total),
                            "action_previous": (self.action_previous.tolist() if self.action_previous is not None else None),
                            "_task_accum": float(self._task_accum),
                            "_task_N": int(self._task_N),
                            
                        }

            os.makedirs("checkpoints", exist_ok=True)
            with open(path, "w") as f:
                json.dump(sched_state, f)

    def cargar_params_checkpoint(self, path):
        # "checkpoints/scheduler_state.json"
        with open(path, "r") as f:
            s = json.load(f)
            self.alpha_t = float(s.get("alpha_t", 0.0))
            self.r_mean = float(s.get("r_mean", 0.0))
            self.s_mean = float(s.get("s_mean", 0.0))
            self.change_adaptation_rate = float(s.get("change_adaptation_rate", 1e-3))
            self.smoothing = float(s.get("smoothing", 0.99))
            self.threshold = float(s.get("threshold", 0.4))
            self.decay_term = float(s.get("decay_term", 0.1))
            self.env.step_total = int(s.get("step_total", 0))
            ap = s.get("action_previous", None)
            self.action_previous = (np.array(ap, dtype=float) if ap is not None else None)
            self._task_accum = float(s.get("_task_accum", 0.0))
            self._task_N = int(s.get("_task_N", 0))
    
    # ============================================================================================================================================= #
    # ================================================= Nuevos metodos de recompensa para nuevas acciones ========================================= #
    # ============================================================================================================================================= #

    # ===================== NUEVO: Caminar 3D =====================
    def calculate_reward_walk3d(self, action, torque_mapping:dict, step_count):
        #env = self.env
        #pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        #euler = p.getEulerFromQuaternion(orn)
        #roll, pitch, yaw = euler
        #num_acciones=len(action)
        vx = float(self.vel_COM[0])
        vy = float(self.vel_COM[1])
        z_star = getattr(self, "init_com_z", 0.89)
        vcmd = float(getattr(self, "_vx_target",0.6))
        w_velocidad=0.6
        w_altura=0.3
        # Este para las acciones de y
        w_lateral=0.1
        w_smooth=0.2
        w_activos = 0.2
        # Para indicar al modelo que más tiempo igual a más recompensa
        supervivencia=0.8
        # Recompensa velocidad
        if 0<=vx<vcmd:
            reward_speed= np.exp(-(vx-vcmd)**2)
        elif vx<0:
            reward_speed=0
        else:
            reward_speed = 1
        #SI com_z esta fuera de la altura objetivo
        castigo_altura = ((self.com_z-z_star)/0.1)**2
        castigo_posicion = (self.com_y/0.1)**2
        castigo_velocidad_lateral=(vy)**2

        castigo_esfuerzo = self.castigo_effort(action, w_smooth, w_activos)
        
        reward= supervivencia + w_velocidad*reward_speed -(w_altura*castigo_altura+ w_lateral*castigo_posicion+ 
                                                           w_lateral*castigo_velocidad_lateral+ castigo_esfuerzo)
        self.action_previous=action
        return float(reward)
    
    def castigo_effort(self,action, w_smooth, w_activos):
        # Suavidad en presiones (acciones en [0,1])
        accion_previa = self.action_previous if self.action_previous is not None else np.zeros_like(action)
        delta_p = np.asarray(action) - np.asarray(accion_previa)
        # a = actividad ≈ acción en [0,1]; términos: (1/M)∑a^3, (1/M)∑(Δu)^2, fracción activa
        #actividad=np.asarray(action)
        #actividad_efectiva=float(np.mean(actividad**3))
        smooth_efectivo=float(np.mean(delta_p**2))
        n_activos=float(np.mean(np.asarray(action) > 0.15))
        return w_smooth*smooth_efectivo + n_activos*w_activos
    
    def reward_for_knees(self, torque_mapping,contact_feets):
        # --- Bonus pro-rodilla (solo si hay margen de estabilidad) ---
        # Contacto de pies
        left_state = contact_feets[0]
        right_state = contact_feets[1]
        NONE, PLANTED = 0, 2
        is_swing_L = (left_state == NONE)
        is_swing_R = (right_state == NONE)
        # Gate por margen ZMP (usa tu ZMP calculator)
        zmp_margin = self.env.zmp_calculator.stability_margin_distance() if self.env.zmp_calculator else 0.0
        gate = 1.0 if zmp_margin > 0.02 else 0.0  # ~2 cm de margen
        # Velocidades articulares
        qd = [s[1] for s in self.env.joint_states_properties]
        knee_L = self.env.dict_joints["left_knee_joint"]
        knee_R = self.env.dict_joints["right_knee_joint"]
        i_kL = self.env.joint_indices.index(knee_L)
        i_kR = self.env.joint_indices.index(knee_R)
        knee_bonus = 0.02 * gate * (
            (abs(qd[i_kL]) if is_swing_L else 0.0) + (abs(qd[i_kR]) if is_swing_R else 0.0)
        )
        reward = knee_bonus

        # --- Tobillo “caro” durante STANCE (evita sostenerse a base de tobillo) ---
        ankle_L = self.env.dict_joints["left_ankle_pitch_joint"]
        ankle_R = self.env.dict_joints["right_ankle_pitch_joint"]
        tau_map = torque_mapping  # ya lo calculas antes
        tau_aL = abs(float(tau_map.get(ankle_L, 0.0)))
        tau_aR = abs(float(tau_map.get(ankle_R, 0.0)))
        ankle_pen = 0.002 * (
            (tau_aL if left_state  == PLANTED else 0.0) +
            (tau_aR if right_state == PLANTED else 0.0)
        )
        reward -= ankle_pen

        return reward

    
    # Requisitos minimos threshold, smoothing, change_in_adaptation_rate
    # parametros de entrenamiento: r_mean, temporal adaptation_sate, s_mean
    def parametro_pesado_acciones(self):
        r_inst = (self._task_accum / max(self._task_N, 1)) if self._task_N > 0 else 0.0
        self.r_mean=self.smoothing*self.r_mean +(1-self.smoothing)*r_inst
        self._task_N=0
        self._task_accum=0
        if self.r_mean >self.threshold and self.s_mean < 0.5:
            self.change_adaptation_rate=self.decay_term*self.change_adaptation_rate
        elif self.r_mean > self.threshold and self.s_mean > 0.5:
            # se genera alpha_t para el next step
            self.alpha_t=self.alpha_t+self.change_adaptation_rate
        else:
            self.alpha_t=self.alpha_t-self.change_adaptation_rate
        self.alpha_t=np.clip(self.alpha_t,0,0.5)
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
    
    def _grf_reward(self, foot_links, metodo_fuerzas_pies,masa_robot, bw_min=0.7, bw_max=1.2,
                    sigma_low=0.10, sigma_high=0.15,# suavidad (Gauss) para defecto/exceso
                    check_split=True,                # activar reparto por pie (recomendado)
                    split_hi=0.8, split_lo=0.1,      # límites por pie (en ×BW) durante doble apoyo
                    split_gain=2.0):                  # “dureza” del reparto por pie
        """
        Devuelve recompensa en [0,1] a partir del exceso en BW.
        - mode="gauss": r = exp(-0.5 * (exceso_bw / sigma_bw)^2)
        - mode="linear": r = 1 - clip(exceso_bw, 0, 1)
        """
        BW,n_contact_feet,Fz,feet_state,deficit, exceso = _grf_excess_cost_bw(foot_links, metodo_fuerzas_pies, masa_robot, bw_min, bw_max)
        if deficit==0 and exceso==0:
            return 1.0,n_contact_feet,Fz,feet_state
        elif n_contact_feet[0]==0 and n_contact_feet[1]==0:
            return 0.0,n_contact_feet,Fz,feet_state
        Fz_L, Fz_R=Fz
        r_band_low  = np.exp(-0.5 * (deficit / max(sigma_low, 1e-6))**2)
        r_band_high = np.exp(-0.5 * (exceso  / max(sigma_high,1e-6))**2)
        r_band = min(r_band_low, r_band_high)
        # Opcional: reparto por pie durante doble apoyo
        if check_split:
            bw_L = Fz_L / max(BW, 1e-6)
            bw_R = Fz_R / max(BW, 1e-6)
            # si ambos pies están al menos TOUCH, aplicamos chequeos
            in_double_support = (feet_state[0] != 0) and (len(feet_state) > 1 and feet_state[1] != 0)

            if in_double_support:
                # Evita que un pie cargue > split_hi BW y que el otro esté casi vacío < split_lo BW
                excess_one = max(0.0, max(bw_L, bw_R) - split_hi)
                lack_other = max(0.0, split_lo - min(bw_L, bw_R))
                r_split = np.exp(-split_gain * (excess_one + lack_other))
            else:
                # En apoyo simple, empuja a ~1.0 BW con tolerancia suave
                bw_active = bw_L if feet_state[0] != 0 else bw_R
                r_split = np.exp(-0.5 * ((bw_active - 1.0) / 0.2)**2)  # tolerancia ~±0.2 BW
        else:
            r_split = 1.0
    
        # Combina (ajusta pesos si quieres)
        return float(0.8 * r_band + 0.2 * r_split), n_contact_feet,Fz,feet_state
    
    # =========================
    # ZMP / Soporte
    # =========================
    def _foot_world_centers(self):
        """
        Centros (x,y) de cada pie en coordenadas mundo.
        Devuelve: [(xL,yL), (xR,yR)]
        """
        env = self.env
        centers = []
        for link in self.foot_links:
            ws = p.getLinkState(env.robot_id, link, computeForwardKinematics=True)
            (x, y, _) = ws[0]
            centers.append((float(x), float(y)))
        return centers

    def _get_Fz_pair_and_states(self, f_min=20.0):
        """
        Cargas verticales y estado de contacto por pie usando el método del env.
        Devuelve: ( (FzL, FzR), (stateL, stateR) )
        """
        env = self.env
        (stateL, n_l, FzL) = env.foot_contact_state(env.left_foot_link_id,  f_min=f_min)
        (stateR, n_r, FzR) = env.foot_contact_state(env.right_foot_link_id, f_min=f_min)
        return (float(FzL), float(FzR)), (stateL, stateR), (n_l,n_r)

    def _center_of_support(self, Fz_pair):
        """
        Centro de soporte ponderado por carga: p_csp = sum_i w_i * c_i / sum_i w_i,
        con w_i = max(Fz_i, 0). Si no hay apoyo suficiente, devuelve None.
        """
        (cL, cR) = self._foot_world_centers()
        FzL, FzR = max(0.0, Fz_pair[0]), max(0.0, Fz_pair[1])
        wsum = FzL + FzR
        if wsum <= 1e-9:
            return None
        x = (FzL * cL[0] + FzR * cR[0]) / wsum
        y = (FzL * cL[1] + FzR * cR[1]) / wsum
        return (x, y)

    def _r_zmp_to_csp(self, Fz_pair, tol_xy=0.06, r_at_tol=0.6):
        """
        Recompensa por llevar ZMP cerca del centro de soporte ponderado (paper-like).
        tol_xy ~ 6 cm en tu escala. Si no hay apoyo, devuelve 0.
        """
        csp = self._center_of_support(Fz_pair)
        if csp is None:
            return 0.0
        zx, zy = float(self.zmp_x), float(self.zmp_y)
        dx = zx - csp[0]
        dy = zy - csp[1]
        d = (dx*dx + dy*dy) ** 0.5
        return exp_term(d, tol_xy, r_at_tol=r_at_tol)

    def _r_zmp_margin(self, tol=0.02, r_at_tol=0.6):
        """
        Recompensa suave del margen ZMP→polígono. Si el margen >= tol, recompensa ~1.
        Si no hay zmp_calculator, devuelve 0.
        """
        zcalc = getattr(self.env, "zmp_calculator", None)
        if zcalc is None:
            return 0.0
        m = float(zcalc.stability_margin_distance())  # puede ser <0 si ZMP fuera
        m = max(0.0, m)
        e = max(0.0, (tol - m))  # “error” a 0 cuando el margen ya supera tol
        return exp_term(e, tol, r_at_tol=r_at_tol)
    
    def _grf_reward_old(self, foot_links, metodo_fuerzas_pies, masa_robot, bw_mult=1.2,
                    mode="gauss", sigma_bw=0.15):
        """
        Devuelve recompensa en [0,1] a partir del exceso en BW.
        - mode="gauss": r = exp(-0.5 * (exceso_bw / sigma_bw)^2)
        - mode="linear": r = 1 - clip(exceso_bw, 0, 1)
        """
        exceso_bw, n_contacts_feet, states = _grf_excess_cost_bw_old(foot_links, metodo_fuerzas_pies,masa_robot, bw_mult)
        if mode == "linear":
            return float(1.0 - np.clip(exceso_bw, 0.0, 1.0)), n_contacts_feet, states
        return float(np.exp(-0.5 * (exceso_bw / max(sigma_bw, 1e-6))**2)), n_contacts_feet, states
    
    def calculate_reward_walk3d_old(self, action, torque_mapping:dict, step_count):
        env = self.env
        pos, orn = p.getBasePositionAndOrientation(env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        num_acciones=len(action)
        # --- NORMALIZATION: tolerances and half-life mapping ---
        d_theta = np.deg2rad(5.0)   # 5 degrees tolerance for roll/pitch
        dz_band = 0.02              # 2 cm deadband for CoM height
        #d_s     = 0.04              # 4 cm CoM->support (si lo usas)
        dv_foot = 0.05              # 5 cm/s no-slip
        dv_cmd  = 0.10              # 0.20 m/s vel tracking (x)
        dy_pos  = 0.08              # 8 cm tolerancia lateral (y)
        dvy     = 0.10              # 0.10 m/s vel lateral
        #d_back  = 0.05              # 0.05 m/s tolerancia hacia atrás (más severo)
        # Da la velocidad lineal y angular de la pelvis
        #lin_vel, ang_vel = p.getBaseVelocity(env.robot_id)
        vx = float(self.vel_COM[0])
        vy = float(self.vel_COM[1])
        # Velocidad del CoM: |vx - vcmd|
        vcmd = float(getattr(self, "_vx_target",0.6))
        # Lateral: posición y velocidad (objetivo y*=0, vy*=0)
        y =self.dy
        r_lat_pos = exp_term(abs(y),  dy_pos, r_at_tol=0.5)
        r_lat_vel = exp_term(abs(vy), dvy,    r_at_tol=0.5)
        # coste suave lateral
        r_lat = r_lat_pos * r_lat_vel
        
        

        # Pesos de recompensa (que debería de recompensar)
        alive = 0.5
        # Pesos de términos normalizados (ajústalos con tus logs)
        w_v, w_post, w_z = 0.40, 0.05, 0.10
        w_tau, w_GRF = 0.10, 0.06
        w_csp, w_marg = 0.0, 0.12
        #w_knees=0.05
        w_activos=0.05
        w_smooth =0.05
        w_lat = 0.02   # <= pon 0.0 si quieres desactivar el término lateral clásico

        r_post = exp_term(abs(roll), d_theta) * exp_term(abs(pitch), d_theta)

        # Altura del CoM (absoluta) alrededor de z*
        z_star = getattr(self, "init_com_z", 0.89)
        e_z = band_error(self.env.com_z, z_star, dz_band)
        r_z  = exp_term(e_z, dz_band, r_at_tol=0.6)

        
        v_err = abs(vx - vcmd)
        if vx>vcmd:
            r_vel=1.0
        elif vx<=0:
            r_vel=0
        else:
            r_vel = exp_term(v_err, dv_cmd, r_at_tol=0.5)

        

        
        #r_dp = exp_term(np.linalg.norm(delta_p), 0.05*np.sqrt(len(action)), r_at_tol=0.6)
        r_tau=self.torque_pain_reduction(torque_mapping=torque_mapping)

        
        # 1) GRF band (exceso de cargas en pies) – en [0,1]
        r_GRF, n_contacts_feet, states = self._grf_reward_old(self.foot_links, env.foot_contact_state, masa_robot=env.mass)
        # 1) Cargas verticales por pie (para ponderar centro de soporte)
        # Si el modelo esta mal al final lo quito
        #r_marg = self._r_zmp_margin(tol=0.02, r_at_tol=0.6)
        #reward_knees=self.reward_for_knees(torque_mapping=torque_mapping, contact_feets=feet_state)
        # Trato de maximizar número de pies en contacto
        #reward_contact_feet=max(n_contacts_feet)/4
        #recompensa_pisada=0.05*reward_contact_feet
        c_tau = 1-r_tau
        c_grf = (1 - r_GRF)
        castigo_effort= self.castigo_effort(action=action, w_activos=w_activos, w_smooth=w_smooth)

        #self._accumulate_task_term(r_vel)
        self.reawrd_step['reward_speed']   = w_v   * r_vel
        self.reawrd_step['reward_posture'] = w_post* r_post
        self.reawrd_step['reward_height']  = w_z   * r_z
        self.reawrd_step['castigo_tau'] = w_tau*c_tau
        #self.reawrd_step['reward_pressure']= w_dp  * r_dp
        #self.reawrd_step['reward_lateral'] = w_lat * r_lat
        self.reawrd_step['castigo_grf']     = w_GRF * c_grf
        #self.reawrd_step['reward_csp']     = w_csp * r_csp
        #self.reawrd_step['reward_margin']  = w_marg* r_marg
        self.reawrd_step['castigo']  = castigo_effort
        # self.reawrd_step['reward_knees'] = w_knees *reward_knees
        #tau y grf_excess_only son castigo de pain
        castigo_pain=w_GRF * c_grf+w_tau*c_tau
        reward = (
            alive
            + w_v   * r_vel
            + w_lat * r_lat
            + w_post* r_post
            + w_z   * r_z
            #+ w_marg* r_marg
            - castigo_pain
            -castigo_effort 
        )
        self._accumulate_task_term(r_vel)
        
        # Se guarda la acción previa
        self.parametro_pesado_acciones()
        # --- Guardado automático del estado del scheduler (opcional) ---
        # if getattr(self, "autosave_every", None):
        #     try:
        #         if step_count % int(self.autosave_every) == 0 and step_count > 0:
        #             os.makedirs(self.checkpoint_dir, exist_ok=True)
        #             path = os.path.join(self.checkpoint_dir, f"scheduler_state_{step_count:09d}.json")
        #             self.guardar_params_checkpoint(path, step_count)
        #     except Exception as _e:
        #         # No interrumpir el entrenamiento por fallos de IO
        #         pass
        self.action_previous=np.array(action)
        # actualizar flags de contacto para siguiente paso
        return float(reward)



def exp_term(error, tol, r_at_tol=0.5):
    error = float(error)
    tol = max(float(tol), 1e-9)
    alpha = -np.log(r_at_tol)
    return np.exp(-alpha * (error / tol)**2)

def band_error(x, x_star, deadband):
    return max(0.0, abs(float(x) - float(x_star)) - float(deadband))


def _grf_excess_cost_bw(foot_links, metodo_fuerzas_pies, masa_robot, bw_min=0.7,bw_max=1.2):
    """
    Suma fuerzas normales en los 'pies' y penaliza solo el exceso sobre 1.2x BW.
    Devuelve N (no negativo).
    """
    # Suma normalForce solo para contactos de cada pie con cualquier objeto (suelo)
    Fz_total = 0.0
    feet_state=[]
    n_contact_feet=[]
    Fz=[]
    for link in foot_links:
        foot_state, n_foot, F_foot= metodo_fuerzas_pies(link_id=link)
        Fz_total += float(F_foot)
        Fz.append(max(0.0, float(F_foot)))
        feet_state.append(foot_state)
        n_contact_feet.append(n_foot)
    BW = masa_robot*9.81  # N

    #BW = masa_robot*9.81  # N
    bw_sum = Fz_total / BW
    deficit = max(0.0, bw_min - bw_sum)
    exceso  = max(0.0, bw_sum - bw_max)
    return BW,n_contact_feet,Fz,feet_state, deficit, exceso

def _grf_excess_cost_bw_old(foot_links, metodo_fuerzas_pies, masa_robot, bw_mult=1.2):
    """
    Suma fuerzas normales en los 'pies' y penaliza solo el exceso sobre 1.2x BW.
    Devuelve N (no negativo).
    """
    # Suma normalForce solo para contactos de cada pie con cualquier objeto (suelo)
    Fz_total = 0.0
    states=[]
    n_contacts_feet=[]
    for link in foot_links:
        state, n_feet, F_foot= metodo_fuerzas_pies(link_id=link)
        Fz_total += float(F_foot)
        n_contacts_feet.append(n_feet)
        states.append(state)

    BW = masa_robot*9.81  # N
    bw_sum = Fz_total / BW
    return float(max(0.0, bw_sum - float(bw_mult))), n_contacts_feet, states


    
    
    