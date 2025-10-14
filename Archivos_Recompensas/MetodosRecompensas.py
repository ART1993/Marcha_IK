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
    if height > 0.8:
        return 1.0  # Buena altura
    elif height > 0.7:
        return 0.8  # Altura mínima
    elif height > 0.5:
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

def pitch_stability_rewards(self, pitch):
    if pitch < 0.2:
        return 2.5  # Muy estable
    elif pitch < 0.4:
        return 0.5  # Moderadamente estable
    elif pitch < self.max_tilt_by_level[self.level]:
        return -2.0  # Inestable
    elif pitch >= self.max_tilt_by_level[self.level]:# self.last_done_reason == self.bad_ending[1]:
        return  -25  # Inestable
    
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
                      activity_pow: float = 3.0,
                      w_smooth: float = 0.5,
                      w_sparse: float = 0.1,
                      act_threshold: float = 0.15) -> tuple[float, dict]:
    """
    Aproxima c_effort del paper con términos:
      - actividad: mean(|u|^p)   (p~3)
      - suavidad:  mean((u - u_prev)^2)
      - esparsidad: mean(1[u>thr])
    Retorna (coste, breakdown_dict).
    """
    u = np.clip(np.asarray(u_now, dtype=float), 0.0, 1.0)
    activity = float(np.mean(np.power(np.abs(u), activity_pow)))
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