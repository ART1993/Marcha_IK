import pybullet as p
import numpy as np

def foot_slip_penalty(self):
        left_foot_id, right_foot_id = self.foot_links
        def slip_for(foot_id, in_contact):
            if not in_contact: return 0.0
            ls = p.getLinkState(self.robot_id, foot_id, computeLinkVelocity=1)
            v = np.array(ls[6]); w = np.array(ls[7])
            return float(np.dot(v[:2], v[:2]) + 0.1*np.dot(w[:2], w[:2]))
        L_in = self.env.L_in
        R_in = self.env.R_in
        return slip_for(left_foot_id, L_in) + slip_for(right_foot_id, R_in)
    
def feet_phase_reward(self, foot_link, timer, sigma_z=0.01):
    z_real = p.getLinkState(self.robot_id, foot_link, computeForwardKinematics=True)[0][2]
    if timer.is_contact: return 0.0
    z_err = z_real - timer.z_ref()
    return float(np.exp(- (z_err*z_err) / (2*sigma_z*sigma_z)))

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
    
    # 1) ¿Tenemos mapas de límite por ángulo?
    tau_utils = self.env.torque_max_generation(torque_mapping=torque_mapping)

    if len(tau_utils)==0:
        return 1.0  # sin info → sin dolor

    # 3) Agregación tipo RMS de utilización
    u_rms = float(np.sqrt(np.mean(np.square(tau_utils))))

    # 4) Sólo “duele” por encima de la tolerancia
    #    (p.ej., hasta el 60% de utilización promedio no penaliza)
    u_tol = 0.80
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


def penalty_joint_speed_guard(self, qd_vec=None, qd_max_map=None, deadband=0.10, r_at_tol=0.6):
    """
    Penaliza |qd| cerca/sobre su límite. Útil en TORQUE_MODE como “cinturón de seguridad”.
    - qd_max_map: dict {joint_id: vmax_pos} (usa |vmax| simétrica).
    - deadband: fracción del límite sin penalización (10% por defecto).
    Devuelve c in [0,1] con 0=sin coste, 1=exceso fuerte.
    """
    import numpy as np
    env = self.env
    if qd_vec is None:
        # usa tu buffer de estados si lo expones; si no, lee de PyBullet
        try:
            qd_vec = [s[1] for s in env.joint_states_properties]
        except Exception:
            qd_vec = [s[1] for s in p.getJointStates(env.robot_id, env.joint_indices)]
    if qd_max_map is None:
        qd_max_map = {}
        for jid in env.joint_indices:
            info = p.getJointInfo(env.robot_id, jid)
            vmax = float(info[11]) if info[11] is not None else 0.0  # maxVelocity
            qd_max_map[jid] = max(1e-6, abs(vmax))  # fallback seguro

    errs = []
    for i, jid in enumerate(env.joint_indices):
        vmax = max(1e-6, float(qd_max_map.get(jid, 1.0)))
        qd = abs(float(qd_vec[i]))
        tol = deadband * vmax
        e = max(0.0, qd - (vmax - tol))  # solo “dueLE” cerca del tope
        # Mapear e∈[0,tol] -> c∈[0,1] con tu exp_term
        errs.append(exp_term(e, tol=max(tol,1e-6), r_at_tol=r_at_tol))
    # Queremos un castigo: 1-mean(recompensa_suavizada)
    return 1.0 - float(np.mean(errs))

# 2) "Headroom" de presión/acción (evita saturaciones 0 ó 1, favorece controlador neumático estable)
def reward_pressure_headroom(self, action, mid=0.5, band=0.25, r_at_tol=0.6):
    """
    Recompensa alta cuando la acción está en banda útil [mid-band, mid+band].
    Penaliza saturaciones (0 ó 1) que en PAMs dificultan control fino y generan golpes.
    """
    import numpy as np
    a = np.asarray(action, dtype=float)
    # error = distancia a la banda
    low, high = mid - band, mid + band
    # e=0 dentro de la banda, positivo fuera
    e = np.maximum.reduce([low - a, a - high, np.zeros_like(a)])
    e = np.maximum(e, 0.0)
    r = exp_term(np.linalg.norm(e) / max(np.sqrt(len(a)),1e-6), tol=0.10, r_at_tol=r_at_tol)
    return float(r)

def penalty_cocontraction(self, action, antagonists, v_thresh=0.2, contact_gate=True, k=1.0):
    """
    antagonists: dict { joint_name_or_id: (idx_flex, idx_ext) } índices en 'action'
    Penaliza a_flex * a_ext cuando |qdot| es baja (no hay razón para "tirar de ambos").
    Durante touchdown (si contact_gate) suaviza penalización (rigidez útil en impacto).
    """
    import numpy as np
    qd = [s[1] for s in getattr(self.env, "joint_states_properties", p.getJointStates(self.env.robot_id, self.env.joint_indices))]
    qd = np.asarray(qd, dtype=float)
    a = np.asarray(action, dtype=float)

    # Gate por contacto (menos castigo cuando hay impacto)
    relax = 1.0
    if contact_gate:
        # Si alguno de los pies acaba de entrar en apoyo, relajamos (mitigación impactos)
        L = self.env.foot_contact_state(self.env.left_foot_link_id)[0]
        R = self.env.foot_contact_state(self.env.right_foot_link_id)[0]
        TOUCH, PLANTED = 1, 2
        recent_touch = (L == TOUCH) or (R == TOUCH)
        relax = 0.5 if recent_touch else 1.0

    cc = []
    for key, (i_flex, i_ext) in antagonists.items():
        prod = float(a[i_flex]) * float(a[i_ext])
        # si apenas se mueve la articulación, pagar más por co-contracción
        jidx = self.env.joint_indices.index(key) if key in self.env.joint_indices else None
        scale = 1.0
        if jidx is not None and abs(qd[jidx]) < v_thresh:
            scale = 1.0
        else:
            scale = 0.5  # si está moviendo rápido, menos penalización
        cc.append(scale * prod)

    return float(k * relax * np.mean(cc) if cc else 0.0)

# 4) Penalización de potencia mecánica (|τ·qdot|) normalizada
def penalty_mech_power(self, torque_mapping, p_norm=None):
    """
    Suma |tau*qdot| y la normaliza por un 'p_norm' razonable (por-junta o global).
    En PAMs equivale a “gasto neumático correlacionado” (proxy de energía).
    """
    import numpy as np
    qd = [s[1] for s in getattr(self.env, "joint_states_properties", p.getJointStates(self.env.robot_id, self.env.joint_indices))]
    qd = np.asarray(qd, dtype=float)
    taus = np.zeros_like(qd)
    for i, jid in enumerate(self.env.joint_indices):
        taus[i] = float(torque_mapping.get(jid, 0.0))
    power = np.sum(np.abs(taus * qd))
    if p_norm is None:
        # Normalización simple: (tau_lim_rms * qd_lim_rms) * N
        # Si tienes 'joint_tau_scale' y 'maxVelocity', úsalos:
        tau_lim = []
        qd_lim = []
        for jid in self.env.joint_indices:
            tl = getattr(self.env, "joint_tau_scale", {}).get(jid, getattr(self.env, "MAX_REASONABLE_TORQUE", 240.0))
            qi = p.getJointInfo(self.env.robot_id, jid)
            vmax = float(qi[11]) if qi[11] else 5.0
            tau_lim.append(abs(float(tl)))
            qd_lim.append(abs(vmax))
        p_norm = np.sqrt(np.mean(np.square(tau_lim))) * np.sqrt(np.mean(np.square(qd_lim))) * len(self.env.joint_indices)
    return float(np.clip(power / max(p_norm,1e-6), 0.0, 1.0))

# 5) Penalización de “golpe vertical” (derivada de Fz) para amortiguar impactos
def penalty_fz_jerk(self, beta=0.001):
    """
    Penaliza cambios bruscos de cargas verticales en pies: |ΔFz_L|+|ΔFz_R|.
    Útil con PAMs: favorece usos que “tomen contacto” de forma amortiguada.
    Requiere estado previo: self._prev_Fz (inicialízala en reset()).
    """
    _, _, Fz_pair, _ = _grf_excess_cost_bw(self.foot_links, self.env.foot_contact_state, self.env.mass)
    FzL, FzR = [float(F) for F in Fz_pair]
    if not hasattr(self, "_prev_Fz"):
        self._prev_Fz = (FzL, FzR)
        return 0.0
    d = abs(FzL - self._prev_Fz[0]) + abs(FzR - self._prev_Fz[1])
    self._prev_Fz = (FzL, FzR)
    # Mapear a [0,1] con una escala suave
    return float(1.0 - np.exp(-beta * d))

def reward_swing_pressure_profile(self, action, swing_foot, muscle_map, prefer_mid=0.5, band=0.3):
    """
    Premia que los músculos agonistas del pie en swing operen en banda media (más control).
    muscle_map: { 'left': [idx_musculos_implicados], 'right': [...] }
    swing_foot: 'left' | 'right'
    """
    import numpy as np
    idxs = muscle_map.get(swing_foot, [])
    if not idxs: return 0.0
    a = np.asarray(action)[np.asarray(idxs, dtype=int)]
    low, high = prefer_mid - band, prefer_mid + band
    e = np.maximum.reduce([low - a, a - high, np.zeros_like(a)])
    e = np.maximum(e, 0.0)
    return float(exp_term(np.linalg.norm(e)/max(np.sqrt(len(a)),1e-6), tol=0.1, r_at_tol=0.7))

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