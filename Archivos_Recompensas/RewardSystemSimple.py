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
            return self.calculate_reward_walk3d(action, joint_state_properties, torque_mapping, step_count)
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
    def calculate_reward_walk3d(self, action, joint_state_properties, torque_mapping:dict, step_count):
        env = self.env
        _, orn = p.getBasePositionAndOrientation(env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, _ = euler
        _, ang_v = p.getBaseVelocity(self.robot_id)
        #roll_prev, pitch_prev,_ = self.prev_euler if self.prev_euler is None else (0,0,0)
        
        #normalized_torque=np.array([torque_mapping[i]/self.joint_tau_max_force[i] for i in self.joint_indices])
        #num_acciones=len(action)
        vx = float(self.vel_COM[0])
        vy = float(self.vel_COM[1])
        z_star = getattr(self, "init_com_z", 0.89)
        vcmd = float(getattr(self, "_vx_target",0.6))
        #self.env.torque_max_generation(torque_mapping=torque_mapping)
        w_velocidad=0.8
        w_altura=0.3
        w_rotacion=0.01
        w_rotacion_v=0.01
        # Este para las acciones de y
        w_lateral=0.1
        w_smooth=0.05
        w_activos = 0.1
        w_velocidad_joint = 0.01
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
        castigo_altura = ((self.com_z-z_star)/0.3)**2
        castigo_posicion = (self.com_y/0.1)**2
        castigo_velocidad_lateral=(vy)**2

        castigo_esfuerzo = self.castigo_effort(action, w_smooth, w_activos)
        castigo_velocidad_joint = self.limit_speed_joint(joint_state_properties)
        def dead_zone(pos_ang):
            return max(0,abs(pos_ang)-np.deg2rad(15))
        tilt_abs =dead_zone(roll)**2 + dead_zone(pitch)**2
        castigo_rotacion= w_rotacion * np.tanh(tilt_abs / 0.5)
        castigo_rotacion_v =  w_rotacion_v*np.tanh(((ang_v[0])**2 + (ang_v[1])**2)/0.5)

        reward= ((supervivencia + w_velocidad*reward_speed)#+ recompensa_fase + recompensa_t_aire) 
                  -(w_altura*castigo_altura+ w_lateral*castigo_posicion+ castigo_rotacion +
                    w_lateral*castigo_velocidad_lateral+ castigo_esfuerzo + w_velocidad_joint*castigo_velocidad_joint +castigo_rotacion_v))
        self.reawrd_step['reward_speed']   = w_velocidad*reward_speed
        self.reawrd_step['castigo_altura']  = w_altura*castigo_altura
        self.reawrd_step['castigo_posicion_y'] = w_lateral*castigo_posicion
        self.reawrd_step['castigo_velocidad_y'] =  w_lateral*castigo_velocidad_lateral
        self.reawrd_step['castigo_esfuerzo']  = castigo_esfuerzo
        self.reawrd_step['castigo_velocidad_joint']  = w_velocidad_joint*castigo_velocidad_joint
        self.reawrd_step['castigo_rotacion']  = castigo_rotacion
        self.reawrd_step['castigo_rotacion_v']  = castigo_rotacion_v

        # self.reawrd_step['recompensa_fase'] = recompensa_fase
        # self.reawrd_step['recompensa_t_aire']   = recompensa_t_aire
        #tau y grf_excess_only son castigo de pain
        self.action_previous=action
        return float(reward)

    def feet_airtime_reward(self, timer, t_thresh=0.25, k=1.0):
        if timer.touchdown_event:
            return float(k * max(0.0, timer.air_time_last - t_thresh))
        return 0.0

    def castigo_cocontraccion(self, action):
        flexor_pressure=np.array(action)[0:,2]
        extensor_pressure=np.array(action)[1:,2]
        co_contraction=[]
        for flexor,extensor in zip(flexor_pressure,extensor_pressure):
            # Castiga que flexor y extensor sean bajos si flexor y extensor son mayores que 0.2
            co_contraction.append(abs(flexor-extensor))
    
    def castigo_effort(self,action, w_smooth, w_activos):
        # Suavidad en presiones (acciones en [0,1])
        accion_previa = self.action_previous if self.action_previous is not None else np.zeros_like(action)
        delta_p = np.asarray(action) - np.asarray(accion_previa)
        # a = actividad ≈ acción en [0,1]; términos: (1/M)∑a^3, (1/M)∑(Δu)^2, fracción activa
        #actividad=np.asarray(action)
        #actividad_efectiva=float(np.mean(actividad**3))
        smooth_efectivo=float(np.mean(delta_p**2))
        n_activos=float(np.mean(np.asarray(action) > 0.30))
        return w_smooth*smooth_efectivo + n_activos*w_activos

    
    
    
    

    def limit_speed_joint(self, joint_state_properties):
        # |qdot| en rad/s
        qdot = np.abs(np.array([s[1] for s in joint_state_properties], dtype=np.float32))

        # límites por articulación, alineados al orden de joint_indices
        vmax = np.array([self.joint_max_angular_speed[jid]
                        for jid in self.joint_indices], dtype=np.float32)

        # Exceso sobre el límite (0 si está por debajo)
        excess = np.maximum(0.0, qdot - vmax)

        # Ancho de transición (10% del límite): controla “lo rápido” que sube el castigo
        beta = 0.1 * np.maximum(1e-6, vmax)

        # Castigo por articulación: cuadrático, 0 en el límite, ↑ con el exceso
        penalty_per_joint = (excess / beta) ** 2

        # Puedes devolver media o suma; con media suele ser más estable
        return float(penalty_per_joint.mean())

    








    
    
    