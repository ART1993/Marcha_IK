import numpy as np
import pybullet as p

from dataclasses import dataclass

"""
    Aquí se crearán los métodos por los cuales se producen las distintas recompensas en RewardSystemSimple
    El objetivo es que se puedan reutilizar para distintos movimientos y así no tener que reescribir todo el código
    cada vez que se quiera crear una nueva recompensa.
"""

def height_reward_method(height):
    """
        Recompensa por altura ganada
    """
    if 1.3> height > 0.7:
        return height-1.2  # Buena altura
    elif 0.7>= height > 0.5:
        return -1.0  # Caída
    else:       # and self.last_done_reason == self.bad_ending[0]:
        return -10
    

def contacto_pies_reward(pie_izquierdo_contacto, pie_derecho_contacto):
    """
        Recompensa por contacto de los pies con el suelo
    """
    
    if pie_izquierdo_contacto is False and pie_derecho_contacto:
        return 2.0
    elif pie_izquierdo_contacto and pie_derecho_contacto:
        return 0.1
    else:
        return -2.0
    
def knee_reward_method(self, target_knee, support_knee):
        
        """
            Metodo que recompensa tener una rodilla doblada u otra rigida
            target_knee: pierna que se doblara.
            support_knee: pierna que se mantendra como apoyo.
        """
        
        if 0.1<target_knee<0.2:
            reward_knee_left=1
        elif 0.2<= target_knee < 0.4:
            reward_knee_left=2
        else:
            reward_knee_left=-2

        if 0.1<support_knee<0.2:
            reward_knee_right=1
        elif 0<=support_knee<=0.1:
            reward_knee_right=0.5
        else:
            reward_knee_right=-2

        self.reawrd_step['reward_knee_right'] =  reward_knee_right
        self.reawrd_step['reward_knee_left'] =  reward_knee_left
        
        return reward_knee_right+ reward_knee_left
    

def hip_reward_method(self,left_hip_roll,left_hip_pitch, right_hip_roll, right_hip_pitch):
        
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

    self.reawrd_step['reward_hip_left'] =  reward_hip_left
    self.reawrd_step['reward_hip_right'] =  reward_hip_right
    self.reawrd_step['reward_roll'] =  reward_roll
    
    return reward_hip_left+ reward_hip_right +reward_roll


def hip_roll_align_reward(pie_izquierdo_contacto, pie_derecho_contacto, torso_roll, frequency_simulation):
    support_sign = -1.0 if (pie_izquierdo_contacto and not pie_derecho_contacto) else (-1.0 if (pie_derecho_contacto and not pie_izquierdo_contacto) else 0.0)
        
    hiproll_align_bonus = 0.0
    if support_sign != 0.0:
        hiproll_align_bonus = np.clip(support_sign * torso_roll / np.deg2rad(10), -1.0, 1.0) * (0.2 / frequency_simulation)
    return hiproll_align_bonus

def pitch_stability_rewards(pitch):
    if pitch < 0.2:
        return 1.0  # Muy estable
    elif pitch <= 0.6:
        return 0.5  # Moderadamente estable
    elif pitch > 0.6:
        return -1.0  # Inestable
    elif pitch >= 1.0:# self.last_done_reason == self.bad_ending[1]:
        return  -2  # Inestable
    
def com_zmp_stability_reward(self):
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
            com, _ = self.env.robot_data.get_center_of_mass()
            # Simple: dentro de la caja entre pies ± margen
            term = 0.3 if self.env.zmp_calculator.is_stable(np.array(com[:2])) else -0.3
        except Exception:
            term = 0.0
    return term

def com_projection_reward(self):
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
    

def seleccion_fuerzas(state_L,state_R,fixed_target_leg, FL, FR):
    # NUEVO: si estamos en left-only, el soporte debe ser el pie derecho
    """
     Decide pie de soporte y pie objetivo con semántica consistente.
     Devuelve: F_sup, F_tar, support_is_left(bool), support_state(int), target_state(int)
     Nota: state_* deben ser valores 0/1/2 del clasificador de contacto.
    """
    if fixed_target_leg == 'left':
        # fuerza la semántica: soporte=right, objetivo=left
        support_is_left = False
        support_state = state_R
        target_state  = state_L
        F_sup = FR      # soporte = pie derecho
        F_tar = FL      # objetivo = pie izquierdo
    elif fixed_target_leg =='right':
        support_is_left = True
        support_state = state_L
        target_state  = state_R
        F_sup = FL      # soporte = pie derecho
        F_tar = FR      # objetivo = pie izquierdo
    else:
        support_is_left = (FL >= FR)
        support_state = state_L if support_is_left else state_R
        F_sup = FL if support_is_left else FR
        target_state  = state_R if support_is_left else state_L
        F_sup = FR if support_is_left else FL

    return F_sup, F_tar, support_is_left, support_state, target_state


def proximity_legs_penalization(self, F_sup, F_tar, left_ankle_id, right_ankle_id, target_is_right):
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
    return close_pen

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


# ==== Effort weight adaptation (paper) =======================================
@dataclass
class EffortWeightScheduler:
    q: float = 0.6          # umbral de desempeño (tu r_vel normalizado en [0,1])
    b: float = 0.95         # suavizado de medias móviles
    delta_a: float = 1e-3   # tasa de adaptación inicial (Δa)
    lam: float = 0.5        # decaimiento al entrar por primera vez en zona "buena"
    a_t: float = 0.0        # peso actual del esfuerzo
    s_mean: float = 0.0     # media móvil del 'switch' (0..1)
    r_mean: float = 0.0     # media móvil del retorno de TAREA
    a_min: float = 0.0
    a_max: float = 0.4

    def update_after_episode(self, task_return: float) -> float:
        """
        Llamar al FINAL de cada episodio con el retorno de TAREA (p.ej., r_vel medio en [0,1]).
        """
        # 1) media móvil del retorno de tarea
        self.r_mean = self.b * self.r_mean + (1.0 - self.b) * float(task_return)
        # 2) objetivo binario vs umbral
        s_target = 1.0 if self.r_mean > self.q else 0.0
        # 3) media móvil del 'switch' de estabilidad
        self.s_mean = self.b * self.s_mean + (1.0 - self.b) * s_target
        # 4) actualizar tasa y peso
        if self.r_mean > self.q and self.s_mean < 0.5:
            # acabas de entrar en ‘zona buena’: sube con cautela
            self.delta_a *= self.lam
        elif self.r_mean > self.q and self.s_mean >= 0.5:
            # buen desempeño sostenido → subir peso del esfuerzo
            self.a_t += self.delta_a
        else:
            # desempeño por debajo del umbral → bajar peso del esfuerzo
            self.a_t -= self.delta_a
        # 5) cotas
        self.a_t = max(self.a_min, min(self.a_t, self.a_max))
        return self.a_t


def effort_cost_proxy(u_now: np.ndarray,
                      u_prev: np.ndarray | None = None,
                      activity_power: float = 3.0,
                      w_smooth: float = 0.5,
                      w_sparse: float = 0.1,
                      act_threshold: float = 0.2) -> tuple[float, dict]:
    """
    Aproxima c_effort del paper con términos:
      - actividad: mean(|u|^p)   (p~3)
      - suavidad:  mean((u - u_prev)^2)
      - esparsidad: mean(1[u>thr])
    Retorna (coste, breakdown_dict).
    """
    u = np.clip(np.asarray(u_now, dtype=float), 0.0, 1.0)
    # actividad a
    activity = float(np.mean(np.power(np.abs(u), activity_power)))
    if u_prev is None:
        smooth = 0.0
    else:
        du = np.asarray(u_now, dtype=float) - np.asarray(u_prev, dtype=float)
        smooth = float(np.mean(du * du))
    sparse = float(np.mean((u > act_threshold).astype(float)))
    cost = activity + w_smooth * smooth + w_sparse * sparse
    return cost, {"activity": activity, "smooth": smooth, "sparse": sparse}

def effort_penalty(u_now, u_prev, a_t, w_smooth=0.04, sparsity_thr=0.15, w_sparse=0.02):
    """
    Coste de esfuerzo inspirado en el paper:
      - Actividad cúbica (aprox energía muscular) escalada por a_t
      - Suavidad (∆u)^2
      - Esparsidad (#activaciones > umbral)
    u_* en [0,1] (acciones PAM normalizadas)
    """
    c, dbg = effort_cost_proxy(u_now, u_prev, activity_pow=3.0,
                               w_smooth=w_smooth, w_sparse=w_sparse,
                               act_threshold=sparsity_thr)
    return a_t * c

# Ignorar por el momento
def calculate_reward_walk3d_slim(self, obs, info):
        """
            Recompensa compacta:
            r = r_vel + w_c * r_foot - a(t) * c_effort - w_p * c_pain
            Usa vel_COM, flags de contacto y GRF del entorno.
        """

        vx = float(self.env.vel_COM[0]) if hasattr(self.env, "vel_COM") else 0.0
        v_star, sigma = 1.2, 0.5
        if vx < v_star:
            r_vel = np.exp(-((vx - v_star) ** 2) / (sigma ** 2))
        else:
            r_vel = 1.0

        # -------- Contacto de pies (ligero anti-vuelo) --------
        left_down = int(getattr(self.env, "left_down", 0))
        right_down = int(getattr(self.env, "right_down", 0))
        support = left_down + right_down    # 0,1,2
        # Recompensa 1 si hay apoyo (1 ó 2), -1 si está en vuelo (0)
        r_foot = (1 if support in (1, 2) else 0) - (1 if support == 0 else 0)

        # -------- Esfuerzo (elige una de las dos variantes) --------
        # a) si trabajas con torques normalizados en info["joint_torques_norm"] -> lista de (tau, tau_max)
        c_effort = None
        if "joint_torques_norm" in info and info["joint_torques_norm"]:
            vals = []
            for tau, tau_max in info["joint_torques_norm"]:
                if tau_max is None or tau_max == 0:
                    continue
                vals.append((tau / tau_max) ** 2)
            c_effort = float(np.mean(vals)) if len(vals) else 0.0
        # b) si controlas PAM/activaciones en info["muscle_activations"] (0..1)
        if c_effort is None and "muscle_activations" in info and info["muscle_activations"]:
            acts = np.array(info["muscle_activations"], dtype=float)
            c_effort = float(np.mean(acts ** 3))
        if c_effort is None:
            c_effort = 0.0

        # -------- “Dolor”: límites + GRF excesivo --------
        # Límites
        c_pain_lim = 0.0
        q_dict = info.get("q", {})  # mapa joint_name -> posición
        for j in self.joints_monitored:
            if j not in self.joint_limits or j not in q_dict:
                continue
            q = float(q_dict[j])
            qmin, qmax = self.joint_limits[j]
            over = max(0.0, q - qmax)
            under = max(0.0, qmin - q)
            c_pain_lim += over ** 2 + under ** 2
        # GRF
        F_L = float(getattr(self.env, "F_L", info.get("F_L", 0.0)))
        F_R = float(getattr(self.env, "F_R", info.get("F_R", 0.0)))
        BW = float(self.robot_weight_N)
        pain_grf = max(0.0, (F_L + F_R) / BW - 1.2)
        c_pain = c_pain_lim + pain_grf

        # -------- Pesos --------
        w_c, w_p = 0.05, 0.5
        a_eff = float(self.effort_adapter.a)

        reward = r_vel + w_c * r_foot - a_eff * c_effort - w_p * c_pain

        # Penalización terminal (si existe en tu flujo)
        if info.get("fell", False):
            reward += -5.0

        # ---- Logging y acumuladores ----
        self._ep_r_vel += r_vel
        self._ep_steps += 1
        logs = {
            "r_vel": float(r_vel),
            "r_foot": float(r_foot),
            "effort_weight": float(a_eff),
            "effort_cost": float(c_effort),
            "pain": float(c_pain),
            "pain_lim": float(c_pain_lim),
            "pain_grf": float(pain_grf)
        }
        return float(reward), logs


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
        v_tgt   = float(getattr(self, "vx_target", 1.2))
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
        # probar con min self.dx, 0 para que solo castique posiciones negativas
        reward = (alive
                + w_v*r_vel
                + w_post*r_post
                + w_cont*r_contact
                + w_step*r_step
                + w_ahead*r_ahead
                + w_vert*r_vert
                + min(self.dx*10,0.0)
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