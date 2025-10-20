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
        r_pitch=pitch_stability_rewards(pitch)
        
        w_ac=0.02
        w_pressure=0.5
        w_active=1.0
        w_en, w_du = max(0.0, w_ac), max(0.0, 3.0*w_ac)        # NEW: E y derivada (suavidad)
        alive = 0.6
        w_v, w_height =3.0, 5
        w_dev=2
        

        # --- 1) Velocidad objetivo (campana) ---
        # Si se quiere "sigma" estadístico, usa -0.5*((...)/sigma)^2; si prefieres más estrecha, quita el 0.5.
        v_tgt   = float(getattr(self, "vx_target", 1.2))
        sigma_v = 0.20
        # Recompensa de velocidad
        if 0<vx<v_tgt:
            r_vel = np.exp(-0.5*((vx - v_tgt)/sigma_v)**2) # Si no aumenta el valor para dif bajas derucir constante multiplicacion
        elif vx<=0:
            r_vel=vx
        else:
            r_vel= 1.0
        # Recompensa de mantener altura
        r_height= height_reward_method(pos[2])
        accion_previa= self.action_previous if self.action_previous is not None else np.zeros_like(action)
        diferencia_presiones=np.mean(np.array(action)-accion_previa)**2
        n_active=len([accion_singular for accion_singular in action if accion_singular>0.2])
        # Minimizar coste de actuadores
        suma_actuadores_normalizados=sum(action)
        eficiencia_acciones=self.alpha_t*suma_actuadores_normalizados + w_pressure*diferencia_presiones +w_active*n_active
        
        
        # ===================== NUEVO: "Dolor" (límites + GRF) =====================
        # 1) Término de límite articular (sumatorio sobre juntas controladas)
        #    Usa torques del paso actual si los guardas en el env; si no, usa proxy de signo.

        tlim_sum = 0.0
        for jid, tau_cmd in torque_mapping.items():
            tau_cmd = float(tau_cmd)
            tlim_sum += _tlim_cost(env.robot_id, jid, tau_cmd)

        # 2) Término por exceso de GRF (pies)
        grf_excess = _grf_excess_cost(env.robot_id, self.foot_links, bw_mult=1.2)

        # 3) Castigo total "dolor"
        w3 = float(getattr(self, "peso_3", 0.3))
        w4 = float(getattr(self, "peso_4", 0.3))
        C_pain = w3 * tlim_sum + w4 * grf_excess



        self._accumulate_task_term(r_vel)
        self.reawrd_step['r_vel_raw'] = w_v*r_vel
        self.reawrd_step['r_dev'] = self.dy**2
        #self.reawrd_step['r_pitch'] = r_pitch
        self.reawrd_step['r_height'] = w_height*r_height**2
        self.reawrd_step['r_act']=eficiencia_acciones
        self.reawrd_step['C_pain']=C_pain
        self.action_previous=action
       
        
        
        reward = (alive
                + w_v*r_vel
                -w_height*r_height**2
                #+w_height*r_pitch
                - w_dev*self.dy**2
                -eficiencia_acciones
                -C_pain
                )
        # Se guarda la accioón previa
        self.parametro_pesado_acciones()
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
    

    # ---- Helpers de límites articulares ----
def _get_joint_limits(robot_id, jid):
    ji = p.getJointInfo(robot_id, jid)
    lower, upper = float(ji[8]), float(ji[9])
    axis = np.array(ji[13], dtype=float)
    return lower, upper, axis

def _tlim_cost(robot_id, jid, tau_cmd, margin=1e-2):
    """
    Coste instantáneo por 'dolor' de límite articular:
    - Si está cerca del límite y el par empuja hacia fuera -> coste = |tau_cmd|
    - Si ya está pegando tope -> usa el par de reacción proyectado en el eje
    Devuelve float (N·m, no negativo).
    """
    q, dq, reaction6, _ = p.getJointState(robot_id, jid)
    lower, upper, axis = _get_joint_limits(robot_id, jid)

    near_lower = (q <= lower + margin)
    near_upper = (q >= upper - margin)

    pushing_lower = near_lower and (tau_cmd < 0.0)
    pushing_upper = near_upper and (tau_cmd > 0.0)

    cost_push = abs(tau_cmd) if (pushing_lower or pushing_upper) else 0.0

    at_lower = q < lower + 1e-3
    at_upper = q > upper - 1e-3
    cost_react = 0.0
    if (at_lower or at_upper):
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        M = np.array(reaction6[3:6], dtype=float)     # momento de reacción [Mx,My,Mz]
        tau_stop = float(np.dot(M, axis_n))           # par de tope sobre el eje
        cost_react = abs(tau_stop)

    return float(max(cost_push, cost_react))

# ---- Helpers de GRF ----
def _robot_weight_N(robot_id, gravity=9.81):
    # masa del base (-1) + todos los links
    num_joints = p.getNumJoints(robot_id)
    m = p.getDynamicsInfo(robot_id, -1)[0]
    for i in range(num_joints):
        m += p.getDynamicsInfo(robot_id, i)[0]
    return float(m * gravity)

def _grf_excess_cost(robot_id, foot_links, world_or_ground_id=None, bw_mult=1.2):
    """
    Suma fuerzas normales en los 'pies' y penaliza solo el exceso sobre 1.2x BW.
    Devuelve N (no negativo).
    """
    # Suma normalForce solo para contactos de cada pie con cualquier objeto (suelo)
    Fz_total = 0.0
    for link in foot_links:
        cps = p.getContactPoints(bodyA=robot_id, linkIndexA=link)
        for cp in cps:
            # cp[9] = normalForce (N). Si quieres asegurarte de contar solo suelo, filtra por bodyB/world_or_ground_id.
            Fz_total += float(cp[9])

    BW = _robot_weight_N(robot_id)  # N
    thr = bw_mult * BW
    return float(max(0.0, Fz_total - thr))

# ---- Calculador de 'dolor' cpain = w3*sum(tlim) + w4*sum(GRF_exceso) ----
def compute_pain_cost(robot_id, joint_ids, tau_cmds_by_joint, foot_links, w3, w4):
    """
    - joint_ids: lista de IDs de junta.
    - tau_cmds_by_joint: dict {jid: tau_cmd} o lista con el mismo orden que joint_ids.
    - foot_links: lista de link indices de los pies (izq, der).
    - w3, w4: pesos de tu función de castigo.
    """
    # Asegurar acceso uniforme a tau_cmd por jid
    if not isinstance(tau_cmds_by_joint, dict):
        tau_cmds_by_joint = {jid: float(tau_cmds_by_joint[k]) for k, jid in enumerate(joint_ids)}

    # Suma tlim por junta
    tlim_sum = 0.0
    for jid in joint_ids:
        tau_cmd = float(tau_cmds_by_joint.get(jid, 0.0))
        tlim_sum += _tlim_cost(robot_id, jid, tau_cmd)

    # Exceso de GRF sobre 1.2xBW
    grf_excess = _grf_excess_cost(robot_id, foot_links)

    # cpain (positivo): si tu reward resta costes, usarás -cpain
    return w3 * tlim_sum + w4 * grf_excess


def exp_term(error, tol, r_at_tol=0.5):
    error = float(error)
    tol = max(float(tol), 1e-9)
    alpha = -np.log(r_at_tol)
    return np.exp(-alpha * (error / tol)**2)

def band_error(x, x_star, deadband):
    return max(0.0, abs(float(x) - float(x_star)) - float(deadband))
    
    
    