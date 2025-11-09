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
        self.joint_indices=self.env.joint_indices
        self.limit_upper_lower_angles=self.env.limit_upper_lower_angles
        self.joint_tau_max_force=self.env.joint_tau_max_force
        self.joint_max_angular_speed=self.env.joint_max_angular_speed
        self.prev_euler=None
        #self.com_z_star = self.env.init_com_z

        # --- Opciones de guardado (puedes cambiarlas desde fuera) ---
        # self.checkpoint_dir = "checkpoints"
        #self.autosave_every = 100000//8   # ej.: 5000 para guardar cada 5k pasos; None = desactivado
        

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
        self.prev_euler = None
        #self.alpha_t = 0.0

    def calculate_s_mean(self):
        s_target=1 if self.r_mean>self.threshold else 0
        self.s_mean=self.s_mean*self.smoothing + (1-self.smoothing)*s_target
        self.s_mean=np.clip(self.s_mean,0.0,1.0)

    def calculate_reward(self, action, joint_state_properties, torque_mapping, step_count):
        """
        Método principal: calcula reward según el nivel actual
        """
        # Decido usar este método para crear varias opciones de creación de recompensas. Else, curriculo clásico
        if getattr(self, "mode", RewardMode.PROGRESSIVE).value == RewardMode.WALK3D.value:
            # return self.calculate_reward_walk3d(action, joint_state_properties, torque_mapping, step_count)
            return self.calculate_reward_walk3d(action, torque_mapping, step_count)
        else:
            raise Exception ("Solo se acepta caminar ahora")

    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        self.com_x,self.com_y,self.com_z=self.env.com_x,self.env.com_y,self.env.com_z
        self.zmp_x, self.zmp_y=self.env.zmp_x, self.env.zmp_y
        self.vel_COM=self.env.vel_COM
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        # Penalizo deriva frente y lateral
        self.dx = float(pos[0])
        self.dy = float(pos[1])
        # Caída
        if pos[2] <= self.env.init_com_z/2:
            self.last_done_reason = "fall"
            if self.env.logger:
                self.env.logger.log("main",f"❌ Episode done: Robot fell {pos[2]}")
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
                self.env.logger.log("main",f"❌ Episode done: Robot tilted too much {euler[0]}, {euler[1]}, {euler[2]}")
            return True

        # Tiempo máximo (crece con nivel)
        max_steps =  2000 # 2000 steps Creo que me he excedido con 6000 steps. Reducir a 2000 en un futuro
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
    
    # ============================================================================================================================================= #
    # ================================================= Nuevos metodos de recompensa para nuevas acciones ========================================= #
    # ============================================================================================================================================= #

    # ===================== NUEVO: Caminar 3D =====================
    def calculate_reward_walk3d(self, action, torque_mapping:dict, step_count):
        env = self.env
        #_, orn = p.getBasePositionAndOrientation(env.robot_id)
        #euler = p.getEulerFromQuaternion(orn)
        #roll, pitch, _ = euler
        #_, ang_v = p.getBaseVelocity(self.robot_id)
        
        # Torque normalizado entre [-1,1] para cada articulación. Reducir la fuerza de 
        torque_normalizado=np.array([torque_mapping[i]/self.joint_tau_max_force[i] for i in self.joint_indices])
        #num_acciones=len(action)
        vx = float(self.vel_COM[0])
        vy = float(self.vel_COM[1])
        z_star = getattr(self, "init_com_z", 0.89)
        vcmd = float(getattr(self, "_vx_target",0.6))
        #self.env.torque_max_generation(torque_mapping=torque_mapping)
        w_velocidad=0.8
        w_altura=0.3
        # Este para las acciones de y
        w_lateral=0.2
        w_smooth=0.3
        w_activos = 0.1
        # Para indicar al modelo que más tiempo igual a más recompensa
        supervivencia=0.8

        #Recompensas de ciclo del pie

        # Recompensa velocidad
        if 0<=vx<vcmd:
            reward_speed= np.exp(-(vx-vcmd)**2)
        elif vx<0:
            reward_speed=0
        else:
            reward_speed = 1
        # Se debería de incluir en supervivencia para aumentar el valor conforme riempo transcurrido
        #recompensa_supervivencia=step_count/2000
        #SI com_z esta fuera de la altura objetivo
        castigo_altura = ((self.com_z-z_star)/0.1)**2
        castigo_posicion = (self.com_y/0.1)**2
        castigo_velocidad_lateral=(vy)**2

        castigo_esfuerzo = self.castigo_effort(action, w_smooth, w_activos)
        #castigo_velocidad_joint = self.limit_speed_joint(joint_state_properties)
        #def dead_zone(pos_ang):
        #    return max(0,abs(pos_ang)-np.deg2rad(15))
        #tilt_abs =dead_zone(roll)**2 + dead_zone(pitch)**2
        #castigo_rotacion= w_rotacion * np.tanh(tilt_abs / 0.5)
        #castigo_rotacion_v =  w_rotacion_v*np.tanh(((ang_v[0])**2 + (ang_v[1])**2)/0.5)

        #castigo_angulo_limite= self.castigo_angulo_limite(joint_state_properties)

        reward= ((supervivencia + w_velocidad*reward_speed)#+ recompensa_fase + recompensa_t_aire) 
                  -(w_altura*castigo_altura+ w_lateral*castigo_posicion+ #castigo_rotacion +
                    w_lateral*castigo_velocidad_lateral+ castigo_esfuerzo)) #+ 
                    #w_velocidad_joint*castigo_velocidad_joint +castigo_rotacion_v + castigo_angulo_limite*w_angulo_limite))
        self.reawrd_step['reward_speed']   = w_velocidad*reward_speed
        self.reawrd_step['castigo_altura']  = w_altura*castigo_altura
        self.reawrd_step['castigo_posicion_y'] = w_lateral*castigo_posicion
        self.reawrd_step['castigo_velocidad_y'] =  w_lateral*castigo_velocidad_lateral
        self.reawrd_step['castigo_esfuerzo']  = castigo_esfuerzo
        # self.reawrd_step['castigo_velocidad_joint']  = w_velocidad_joint*castigo_velocidad_joint
        # self.reawrd_step['castigo_rotacion']  = castigo_rotacion
        # self.reawrd_step['castigo_rotacion_v']  = castigo_rotacion_v
        # self.reawrd_step['castigo_angulo_limite']  = castigo_angulo_limite*w_angulo_limite

        # self.reawrd_step['recompensa_fase'] = recompensa_fase
        # self.reawrd_step['recompensa_t_aire']   = recompensa_t_aire
        #tau y grf_excess_only son castigo de pain
        self.action_previous=action
        return float(reward)
    
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
    
    def castigo_effort(self,action, w_smooth, w_activos):
        # Suavidad en presiones (acciones en [0,1])
        accion_previa = self.action_previous if self.action_previous is not None else np.zeros_like(action)
        # Evita que el torque pase de +1 a -1 instantaneamente
        delta_p = np.asarray(action) - np.asarray(accion_previa)
        smooth_efectivo=float(np.mean(delta_p**2))
        # Cuenta cuantos actuadores están activos
        n_activos=float(np.mean(np.asarray(action) > 0.30))
        return w_smooth*smooth_efectivo #+ n_activos*w_activos
    
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
        alive = 0.8 #Subir a 0.8
        # Pesos de términos normalizados (ajústalos con tus logs)
        # TODO:subir w_v a 0.6
        w_v, w_post, w_z = 0.60, 0.05, 0.10
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
            # - castigo_pain
            -castigo_effort 
        )
        self._accumulate_task_term(r_vel)
        
        # Se guarda la acción previa
        self.parametro_acciones_pasadas()
        self.action_previous=np.array(action)
        # actualizar flags de contacto para siguiente paso
        return float(reward)
    
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
    

    def parametro_acciones_pasadas(self):
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


def exp_term(error, tol, r_at_tol=0.5):
    error = float(error)
    tol = max(float(tol), 1e-9)
    alpha = -np.log(r_at_tol)
    return np.exp(-alpha * (error / tol)**2)

def band_error(x, x_star, deadband):
    return max(0.0, abs(float(x) - float(x_star)) - float(deadband))

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
        
        