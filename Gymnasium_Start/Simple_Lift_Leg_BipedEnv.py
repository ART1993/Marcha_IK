
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
from collections import deque
from json import loads, load
import os
from enum import Enum

from Archivos_Apoyo.Configuraciones_adicionales import PAM_McKibben,Rutas_Archivos, \
                                                    seleccionar_funcion_calculo_torques
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data
from Archivos_Recompensas.RewardSystemSimple import SimpleProgressiveReward

class FootContactState(Enum):
    NONE = 0      # sin contacto
    TOUCH = 1     # rozando (0 < n_contactos <= 2 o F_total baja)
    PLANTED = 2   # apoyo plano (n_contactos > 2 y F_total alta)
           

class Simple_Lift_Leg_BipedEnv(gym.Env):
    """
        Versión expandida con 16 PAMs activos + elementos pasivos
        Indices de robot bípedo pam:
            - left hip_roll joint: 0
            - left hip_pitch joint: 1
            - left knee joint: 2
            - left ankle joint: 3
            - right hip joint: 4
            - right hip pitch joint: 5
            - right knee joint: 6
            - right ankle joint: 7
    """
    
    def __init__(self, logger=None, render_mode='human', 
                 print_env="ENV", fixed_target_leg="left",csvlog=None,
                 simple_reward_mode="progressive",allow_hops:bool=False,
                 vx_target: float = 1.2, # Bajar a 0.4 si da problemas
                 robot_name="2_legged_human_like_robot16DOF"):
        
        """
            Inicio el entorno de entrenamiento PAM
        """
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super(Simple_Lift_Leg_BipedEnv, self).__init__()
        self.robots_existentes=Rutas_Archivos.rutas_robots.value
        self.rutas_json=Rutas_Archivos.rutas_jsons.value
        self.robot_name=robot_name
        self.urdf_path = self.robots_existentes.get(f"{self.robot_name}")
        with open(self.rutas_json.get(f"{self.robot_name}"), 'r') as f:
            json_file_robot_joint_info=load(f)
            
        self.joint_indices=[]
        self.control_joint_names=[]
        # de momento este dict es solo intuitivo
        self.limit_upper_lower_angles={}
        for key, values in json_file_robot_joint_info.items():
            if values.get('type')!=4:
                self.joint_indices.append(values.get("index"))
                self.control_joint_names.append(values.get("name"))
                self.limit_upper_lower_angles[values.get("name")]={'lower':values.get("lower"),'upper':values.get("upper")}
            if values.get("link_name")=="left_foot_link":
                self.left_foot_link_id=values.get("index")
            elif values.get("link_name")=="right_foot_link":
                self.right_foot_link_id=values.get("index")

        self.footcontact_state=FootContactState
        # ===== CONFIGURACIÓN BÁSICA =====
        self.pam_muscles = PAM_McKibben(self.robot_name, self.control_joint_names)
        self.render_mode = render_mode
        self.logger=logger
        self.csvlog = csvlog
        # === Opciones de recompensa (leídas por RewardSystemSimple) ===
        self.simple_reward_mode = simple_reward_mode    # "progressive" | "walk3d" | "lift_leg" | "march_in_place"
        self.allow_hops = bool(allow_hops)              # permitir ambos pies en el aire (no-support permitido)
        self.vx_target = float(vx_target)               # objetivo de velocidad para walk3d

        
        self.muscle_names = list(self.pam_muscles.keys())
        
        self.num_active_pams = len(self.muscle_names)
        
        self.frequency_simulation=400.0
        #Probar para ver si evita tembleques
        self.frequency_control=40.0
        self.time_step = 1.0 / self.frequency_simulation
        # Action-repeat/frame-skip: aplicar una acción a 400 Hz simulación pero 50 Hz control
        self.frame_skip = max(1, int(self.frequency_simulation // self.frequency_control))
        
        # Estados PAM básicos
        self.pam_states = {
            'pressures': np.zeros(self.num_active_pams),
            'forces': np.zeros(self.num_active_pams)
        }
        
        # ===== CONFIGURACIÓN DE ESPACIOS =====
        self.recent_rewards=deque(maxlen=50)
        # Action space: self.num_active_pams presiones PAM normalizadas [0, 1]
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_active_pams,), 
            dtype=np.float32
        )
        
        # Observation space
        obs_dim=16 + 2*len(self.joint_indices)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # ===== CONFIGURACIÓN DE SIMULACIÓN =====
        
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # ===== SISTEMAS DE APOYO BÁSICOS =====
        
        self.zmp_calculator = None
        self.robot_data = None
        self.controller = None

        # Variables para tracking
        
        #self.episode_start_step = 0
        
        self.reward_system = None
        self.action_selector = None
        self.angle_expert_controller= None
        self.step_count = 0
        self.step_total=0
        self.total_reward = 0
        self.robot_id = None
        self.plane_id = None
        #self.joint_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee]
        
        

        self.joint_names=self.control_joint_names
        
        self.dict_joints= {joint_name:joint_index for joint_name, joint_index in zip(self.joint_names, self.joint_indices)}
        #ID Links de los pies (igual al de los tobillos)

        # Añadir tracking de pierna levantada
        self.fixed_target_leg = fixed_target_leg
        self.raised_leg = self.fixed_target_leg  # 'left' o 'right' - cuál pierna está levantada
        
        self.episode_reward = 0
        #Parámetros constantes que se usan en el calculo de torques
        self.parametros_torque_pam()

        self.simple_reward_system = None
        self.print_env = print_env
        self.reawrd_step={}
        self.n_episodes=0
        if self.logger:
            self.logger.log("main",f"🤖 Simplified Lift legs Environment initialized")
            self.logger.log("main",f"🤖 Environment initialized - Systems initiate in reset")
            self.logger.log("main",f"🤖 Using {self.num_active_pams=:} for {self.robot_name}" 
                  f"[{self.print_env=:}]")
    
# ========================================================================================================================================================================= #
# ===================================================== Métodos de paso y control del entorno Enhanced_PAMIKBipedEnv ====================================================== #
# ========================================================================================================================================================================= #
    
    def step(self, action):
        """
            Step SIMPLIFICADO - Solo física PAM básica + recompensa de balance
        
            1. ✅ Usar torques calculados correctamente desde PAMs
            2. ✅ Aplicar en las articulaciones correctas
            3. ✅ Configuración de fricción en PyBullet
            4. ✅ Mejor integración con sistema de recompensas
        """
        
        self.info = {"kpi": {}}
        # Obtengo posicion y orientación al inicio del paso
        self.pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        self.euler = p.getEulerFromQuaternion(orn)
        self.step_count += 1
        self.step_total += 1
        u_final = np.clip(action, 0.0, 1.0)
        # Probar los dos y ver cual da mejor resultados
        #max_step = 0.20 #SI sale mal probar 0.3 o 0.4
        #delta = np.clip(u_final - self.prev_action, -max_step, max_step)
        #delta = max_step * np.tanh(raw_delta / (max_step + 1e-6))
        
        #u_final = self.prev_action + delta
        #u_final = np.clip(u_final, 0.0, 1.0)
        
        self.ep_total_actions += 1

        # ===== NORMALIZAR Y VALIDAR ACCIÓN =====
    
        normalized_pressures = np.clip(u_final, 0.0, 1.0) 
        
        # Aplicar fuerzas PAM normalizadas
        joint_torques = self._apply_pam_forces(normalized_pressures)
        
        # ===== Paso 3: SIMULACIÓN FÍSICA =====

        # Aplicar torques
        #torque_mapping = [(joint, joint_torques[i]) for i, joint in enumerate(self.joint_indices)]
        # En el caso que quiera reducir los torques a usar y por tanto los joints no fijos #[:len(joint_torques)]
        torque_mapping = {jid: joint_torques[i] for i, jid in enumerate(self.joint_indices)}

        #self.last_tau_cmd = {jid: float(tau) for jid, tau in torque_mapping}
        for joint_id, torque in torque_mapping.items():
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )
        for _ in range(self.frame_skip):
            p.stepSimulation()
            self.sim_time += self.dt
        #parametros tras ejecutar step de simulación 
        self.pos_post, self.orn_post = p.getBasePositionAndOrientation(self.robot_id)
        self.euler_post = p.getEulerFromQuaternion(self.orn_post)
        # Tengo los datos del robot actualizados a cada paso así como la localización del COM
        self.robot_data = PyBullet_Robot_Data(self.robot_id)
        # KPI ZMP ya existente
        if self.zmp_calculator:
            try:
                zmp_xy = self.zmp_calculator.calculate_zmp()
                self.zmp_x, self.zmp_y = float(zmp_xy[0]), float(zmp_xy[1])
                # COM (usa tu helper de Pybullet_Robot_Data)
                try:
                    com_world, self.mass = self.robot_data.get_center_of_mass()
                    self.com_x, self.com_y, self.com_z = float(com_world[0]), float(com_world[1]), float(com_world[2])
                    self.vel_COM=self.robot_data.get_center_of_mass_velocity()
                except Exception:
                    self.com_x = self.com_y = self.com_z = 0.0
            except Exception:
                self.zmp_x, self.zmp_y = 0.0, 0.0
        else:
            self.zmp_x, self.zmp_y = 0.0, 0.0

        
        #if self.simple_reward_system:
        # Joint indices 
        self.joint_states_properties = p.getJointStates(self.robot_id, self.joint_indices)
        self.L_in = self.pie_tocando_suelo(self.robot_id, self.left_foot_link_id, fz_min=5.0)
        self.R_in = self.pie_tocando_suelo(self.robot_id, self.right_foot_link_id, fz_min=5.0)
        cmd_speed = float(np.hypot(self.vx_target, 0.0))
        self.left_timer.update(self.L_in, cmd_speed)
        self.right_timer.update(self.R_in, cmd_speed)
        
        # ===== CÁLCULO DE RECOMPENSAS CONSCIENTE DEL CONTEXTO =====
        done = self.simple_reward_system.is_episode_done(self.step_count)
        reward = self.simple_reward_system.calculate_reward(u_final, torque_mapping, self.step_count)
        
        # ===== PASO 4: OBSERVACIÓN Y TERMINACIÓN =====
        self.episode_reward += reward
        self.prev_action = u_final.copy()
        observation = self._get_simple_observation()

        # Info simplificado
        info = {
            'step_count': self.step_count,
            'reward': reward,
            #'action_source': action_source,
            'episode_reward': self.episode_reward
        }
        
        info, reward, done = self.function_logger_kpi(info, reward, done)

        self._debug_joint_angles_and_pressures(info, u_final, done)
        
        self.debug_rewards()
        
        return observation, reward, done, False, info
    

    def info_pam_torque(self, info):
        jt = getattr(self, "pam_states", {}).get("joint_torques", None)
        ps = getattr(self, "pam_states", {}).get("pressures", None)

        # Ejemplos: mapea índices a nombres (ver mapeo más abajo)
        if jt is not None:
            for i, joint_name in enumerate(self.control_joint_names):
                #print(i, joint_name)
                info["kpi"][f"tau_{joint_name}"]   = float(jt[i])  
            # info["kpi"]["tau_LHP"]   = float(jt[0])  # Left Hip pitch
            # info["kpi"]["tau_LHR"]   = float(jt[1])  # Left Hip roll
            # info["kpi"]["tau_LK"]    = float(jt[2])  # Left Knee
            # info["kpi"]["tau_LAP"]   = float(jt[3])  # Right ankle pitch
            # info["kpi"]["tau_LAR"]   = float(jt[4])  # Right ankle roll
            # info["kpi"]["tau_RHP"]   = float(jt[5])  # Right Hip pitch
            # info["kpi"]["tau_RHR"]    = float(jt[6])  # Right HIP roll
            # info["kpi"]["tau_RK"]    = float(jt[7])  # Right Knee
            # info["kpi"]["tau_RAP"]    = float(jt[8])  # Right ankle pitch
            # info["kpi"]["tau_RAR"]    = float(jt[9])  # Right ankle roll

        if ps is not None:
            for i, mucle_name in enumerate(self.muscle_names):
                info["kpi"][mucle_name] = float(ps[i]) 


        for joint_name, joint_index  in self.dict_joints.items():
            info["kpi"][f"q_{joint_name}"] = float(p.getJointState(self.robot_id, joint_index)[0])
        return info
    
    def function_logger_kpi(self, info, reward, done):
        if self.simple_reward_system:
            curriculum_info = self.simple_reward_system.get_info()  # Solo una llamada
            info['curriculum'] = curriculum_info  # Añadir sin reemplazar
            info['system_type'] = 'progressive'
            info['current_level'] = curriculum_info.get('level', 1)
            (left_state, n_l, F_L) = self.foot_contact_state(self.left_foot_link_id, f_min=20)
            (right_state, n_r, F_R) = self.foot_contact_state(self.right_foot_link_id, f_min=20)
            # flags "down" derivadas de estado:
            left_down  = int(left_state  == self.footcontact_state.PLANTED.value)
            right_down = int(right_state == self.footcontact_state.PLANTED.value)

            
            info["kpi"] = {
                            "global_step": int(self.step_count),
                            "reward": float(reward),
                            "roll": float(self.euler[0]),
                            "pitch": float(self.euler[1]),
                            "left_down": int(bool(left_down)),
                            "right_down": int(bool(right_down)),
                            "F_L": float(F_L),
                            "F_R": float(F_R),
                            "nL": int(n_l), "nR": int(n_r),
                            "state_L": left_state, "state_R": right_state,
                            "zmp_x": float(self.zmp_x),
                            "zmp_y": float(self.zmp_y),
                            "com_x": self.com_x,
                            "com_y": self.com_y,
                            "com_z": self.com_z,
                        }
            # Debug simple
            info = self.info_pam_torque(info)
            # opcional: incluir en info
            info["kpi"]["air_time_L"] = float(self.left_timer.air_time_last)
            info["kpi"]["air_time_R"] = float(self.right_timer.air_time_last)
            info["kpi"]["phase_L"]    = float(self.left_timer.phase)
            info["kpi"]["phase_R"]    = float(self.right_timer.phase)
            if done:
                info["ep_kpi"] = {
                                    "ep_return": float(self.episode_reward),
                                    "ep_len": int(self.step_count),
                                    "done_reason": getattr(self.simple_reward_system, "last_done_reason", None)
                                }
                self.n_episodes+=1
                if self.logger:
                    self.logger.log("main", f"📈 Episode {info['curriculum']['episodes']} | Nivel {info['curriculum']['level']} | Reward: {info['episode_reward']:.1f}")
        
        # === CSVLogger: volcado per-step (~10 Hz) ===
        if (self.step_count % (self.frequency_simulation//self.frame_skip) == 0 or done) and self.simple_reward_system:
            if self.logger:
                self.logger.log("main",f"🔍 Step {self.step_count} - Control Analysis:")
                self.logger.log("main",f"   Height: {self.pos[2]:.2f}m")
                self.logger.log("main",f"   Tilt: Roll {math.degrees(self.euler[0]):.1f}°, Pitch {math.degrees(self.euler[1]):.1f}°")
                #logger.log(f"   Action source: {action_source}")
                kpi_dbg = info.get("kpi", {})
                self.logger.log("main",f"   COM: ({kpi_dbg.get('com_x', 0.0):.3f}, {kpi_dbg.get('com_y', 0.0):.3f}, {kpi_dbg.get('com_z', 0.0):.3f}) m  ")
                self.logger.log("main",f"   ZMP: ({kpi_dbg.get('zmp_x', 0.0):.3f}, {kpi_dbg.get('zmp_y', 0.0):.3f}) m")
                #curriculum_info = self.simple_reward_system.get_info()
                self.logger.log("main",f"   Level: {info['curriculum'].get('level')}")
                self.logger.log("main",f"   COM→support: {kpi_dbg.get('com_dist_to_support', 0):.3f} m | ZMP margin: {kpi_dbg.get('zmp_margin_m', 0):+.3f} m | stable={kpi_dbg.get('com_stable_flag', 0)}")
    
            # Verificar si está cerca de límites
            max_allowed_tilt = 0.4 if self.simple_reward_system and self.simple_reward_system.level == 1 else 0.3
            if abs(self.euler[0]) > max_allowed_tilt * 0.8 or abs(self.euler[1]) > max_allowed_tilt * 0.8:
                if self.logger:
                    self.logger.log("main",f"   ⚠️ Approaching tilt limit! Max allowed: ±{math.degrees(max_allowed_tilt):.1f}°")
            

        # DEBUG TEMPORAL: Verificar timing cada cierto número de steps
            #status = self.simple_reward_system.get_info()
            elapsed_time = self.step_count / self.frequency_simulation
            #logger.log(f" {action_source} action, reward={reward:.2f}")
            if self.logger:
                self.logger.log("main",f"🕒 Step {self.step_count} ({elapsed_time:.1f}s elapsed):")
                self.logger.log("main",f"   Current level: {curriculum_info['level']}")
                self.logger.log("main",f"   Target leg: {curriculum_info.get('target_leg', 'N/A')}")
        return info, reward, done
    

    def _configure_contact_friction(self):
        """
        Configurar propiedades de fricción dinámicamente en PyBullet
        (Complementa las propiedades del URDF)
        """
        
        # ===== FRICCIÓN PARA OTROS LINKS =====
        
        # Links de piernas - fricción moderada
        for link_id in self.joint_indices:
            p.changeDynamics(
                self.robot_id,
                link_id,
                lateralFriction=0.05,    # Muy reducida de 0.6 a 0.1
                spinningFriction=0.05,  # Muy reducida de 0.4 a 0.05
                rollingFriction=0.01,   # Muy reducida de 0.05 a 0.01
                restitution=0.05
            )

        # Pie izquierdo - alta fricción para agarre
        for foot_id in (self.left_foot_link_id, self.right_foot_link_id):
            p.changeDynamics(
                self.robot_id, 
                foot_id,
                lateralFriction=0.85,                #0.9 bajar a 0.7 si hay problemas, se volvio a bajar a 0.55       
                spinningFriction=0.12,                   #0.15,       
                rollingFriction=0.01,       
                restitution=0.01,           
                contactDamping=100,         
                contactStiffness=12000,      
                frictionAnchor=1
            )
        
        # ===== FRICCIÓN DEL SUELO =====
        
        # Configurar fricción del plano del suelo
        p.changeDynamics(
            self.plane_id,
            -1,                         # -1 for base link
            lateralFriction=0.9,        # Fricción estándar del suelo 0.6
            spinningFriction=0.05,
            rollingFriction=0.005
        )
        

    def contact_with_force(self, link_id, stable_foot,min_F=20.0):
        cps = p.getContactPoints(self.robot_id, self.plane_id, link_id, -1)# -1 para el suelo
        if not cps: 
            return False
        # campo normalForce = índice 9 en PyBullet
        fuerzas_puntos=[cp[9] for cp in cps]
        num_contactos=len(fuerzas_puntos)
        F_total=sum(fuerzas_puntos)
        if self.step_count % (self.frequency_simulation//self.frame_skip) == 0:  # Cada segundos aprox
            if self.logger:
                self.logger.log("main",f"Contact force on link {link_id}: {F_total:.2f} N")
        if stable_foot:
            return (F_total > min_F) and (num_contactos>2)
        else:
            return (F_total > min_F) or (num_contactos>0)
    
    def contact_normal_force(self, link_id:int)->float:
        cps = p.getContactPoints(self.robot_id, self.plane_id, link_id, -1)
        if not cps:
            return 0, 0.0
        fuerzas_puntos=[cp[9] for cp in cps]
        num_contactos=len(fuerzas_puntos)
        F_total=sum(fuerzas_puntos)
        fuerza_contacto_y_puntos=(num_contactos, F_total)
        return fuerza_contacto_y_puntos
    
    def pie_tocando_suelo(self, robot_id, foot_link, fz_min=5.0):
        return any(cp[9] > fz_min for cp in p.getContactPoints(bodyA=robot_id, linkIndexA=foot_link))
    
    def foot_contact_state(self, link_id:int, f_min:float=20.0, n_touch:int=2):
        """
            Clasifica el estado de contacto de un link del pie:
            - NONE:   sin contacto
            - TOUCH:  rozando (<=2 puntos o fuerza baja)
            - PLANTED:apoyo plano (varios puntos y fuerza suficiente)
            Devuelve: (state_value:int, n_contactos:int, F_total:float)
        """
        n, F = self.contact_normal_force(link_id)
        if n == 0:
            state = self.footcontact_state.NONE.value
        elif n <= n_touch or F < f_min:
            state = self.footcontact_state.TOUCH.value
        else:
            state = self.footcontact_state.PLANTED.value
        return state, n, F
    
    def debug_contacts_once(self):
        for name, lid in [("L_foot", self.left_foot_link_id), ("R_foot", self.right_foot_link_id)]:
            cps = p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=lid, linkIndexB=-1)
            print(f"[DEBUG] {name} contacts: {len(cps)}")
            for i, cp in enumerate(cps[:5]):
                # cp[9]=Fnormal, cp[6]=linkA, cp[4]=linkB
                print(f"   #{i} nF={cp[9]:.2f}N  linkA={cp[6]}  linkB={cp[4]}  posA={cp[5]}")
    

# ==================================================================================================================================================================== #
# =================================================== Métodos de Aplicación de fuerzas PAM =========================================================================== #
# ==================================================================================================================================================================== #

    def _apply_pam_forces(self, pam_pressures):
        """
            Convertir presiones PAM a torques articulares usando FÍSICA REAL de PAM_McKibben
            
            ESTO ES FUNDAMENTAL - El corazón del control de actuadores PAM:
            1. Usa PAM_McKibben para calcular fuerza real según presión
            2. Considera contracción basada en ángulo articular  
            3. Aplica física biomecánica real
            
            Mapeo: 16 PAMs -> 8 articulaciones
            - PAM 0,1: cadera izquierda roll (flexor, extensor)
            - PAM 2,3: cadera derecha roll(flexor, extensor)  
            - PAM 4,5: cadera izquierda pitch(flexor, extensor)
            - PAM 6,7: cadera derecha pitch(flexor, extensor)
            - PAM 8,9: rodilla izquierda (flexor, extensor)
            - PAM 10,11: rodilla derecha (flexor, extensor)
            - PAM 12,13: tobillo izquierdo (flexor, extensor)
            - PAM 14,15: tobillo derecha (flexor, extensor)
            # MAPEO CLARO: PAM → Joint
            # joint_states[0] = left_hip_roll (joint 0)
            # joint_states[1] = left_hip_pitch (joint 1)
            # joint_states[2] = left_knee (joint 2)
            # joint_states[3] = left_ankle (joint 3) 
            # joint_states[4] = right_hip_roll (joint 4)
            # joint_states[5] = right_hip_pitch (joint 5)
            # joint_states[6] = right_knee (joint 6)
            # joint_states[7] = right_ankle (joint 7)
        """
       
        # NUEVA LÓGICA: Control automático de rodilla levantada
        joint_torques = seleccionar_funcion_calculo_torques(self, pam_pressures)
        # Activar control automatico de rodilla

        balance_info = self.current_balance_status
        if self.step_count%(self.frequency_simulation//self.frame_skip)==0:
            if self.logger:
                self.logger.log("main",f"Pierna de apoyo: {balance_info['support_leg']}")
                self.logger.log("main",f"Tiempo en equilibrio: {balance_info['balance_time']} steps")

        return joint_torques
    
    def _get_simple_observation_reset(self):
        """
        Observación específica para equilibrio en una pierna.
        Reemplaza _get_simple_observation con información más relevante.
        """
        
        obs = []
        
        # ===== ESTADO DEL TORSO (8 elementos) =====
        self.init_pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, init_ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Posición y orientación  
        obs.extend([self.init_pos[0],self.init_pos[1], self.init_pos[2], euler[0], euler[1]])  # x, y, z, roll, pitch

        # Velocidades
        #obs.extend([lin_vel[0], lin_vel[1], lin_vel[2], init_ang_vel[0], init_ang_vel[1]])  # vx, vz, wx, wy

        yaw = p.getEulerFromQuaternion(orn)[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        # rotación mundo->cuerpo (2D yaw)
        vx_b =  cy*lin_vel[0] + sy*lin_vel[1]
        vy_b = -sy*lin_vel[0] + cy*lin_vel[1]
        obs.extend([vx_b, vy_b, lin_vel[2], init_ang_vel[0], init_ang_vel[1]])
        
        
        # Observaciones ¿Por que no son posiciones del robot y V_com?
        # ===== ESTADOS ARTICULARES (4 elementos) =====
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)  # Solo joints activos
        joint_positions = [state[0] for state in joint_states]
        obs.extend(joint_positions)

        # NUEVO: velocidades
        joint_vels = [s[1] for s in joint_states]
        obs.extend(joint_vels)

        # ===== ZMP BÁSICO (2 elementos) =====
        
        if self.zmp_calculator:
            try:
                zmp_point = self.zmp_calculator.calculate_zmp()
                obs.extend([zmp_point[0], zmp_point[1]])
            except:
                obs.extend([0.0, 0.0])
        else:
            obs.extend([0.0, 0.0])
        
        # ===== INFORMACIÓN DE CONTACTO Y ALTURA DE RODILLAS (4 elementos) =====
        
        # Aquí sólo queremos saber si "hay algún contacto" → estable=False
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id, stable_foot=True, min_F=20.0)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id, stable_foot=True, min_F=20.0)
        obs.extend([float(left_contact), float(right_contact)])
        
        # Alturas de pies
        l_foot = p.getLinkState(self.robot_id, self.left_foot_link_id)[0][2]
        r_foot = p.getLinkState(self.robot_id, self.right_foot_link_id)[0][2]
        obs.extend([l_foot, r_foot])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_simple_observation(self):
        """
        Observación SIMPLIFICADA
        
        """
        
        obs = []
        
        # ===== ESTADO DEL TORSO (8 elementos) =====
        
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        
        # Posición y orientación
        obs.extend([self.pos_post[0], self.pos_post[1], self.pos_post[2], self.euler_post[0], self.euler_post[1]])  # x, z, roll, pitch

        # Velocidades
        # obs.extend([lin_vel[0], lin_vel[1], lin_vel[2], ang_vel[0], ang_vel[1]])  # vx, vz, wx, wy
        yaw = p.getEulerFromQuaternion(self.orn_post)[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        # rotación mundo->cuerpo (2D yaw)
        vx_b =  cy*lin_vel[0] + sy*lin_vel[1]
        vy_b = -sy*lin_vel[0] + cy*lin_vel[1]
        obs.extend([vx_b, vy_b, lin_vel[2], ang_vel[0], ang_vel[1]])
        
        # ===== ESTADOS ARTICULARES (4 elementos) =====
        
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_positions = [state[0] for state in joint_states]
        obs.extend(joint_positions)

        # NUEVO: velocidades
        joint_vels = [s[1] for s in joint_states]
        obs.extend(joint_vels)
        
        # ===== ZMP BÁSICO (2 elementos) =====
        
        if self.zmp_calculator:
            try:
                zmp_point = self.zmp_calculator.calculate_zmp()
                obs.extend([zmp_point[0], zmp_point[1]])
            except:
                obs.extend([0.0, 0.0])
        else:
            obs.extend([0.0, 0.0])
        
        # ===== INFORMACIÓN DE CONTACTO Y ALTURA DE RODILLAS (4 elementos) =====
        
        # Aquí sólo queremos saber si "hay algún contacto" → estable=False
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id, stable_foot=True, min_F=20.0)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id, stable_foot=True, min_F=20.0)
        obs.extend([float(left_contact), float(right_contact)])
        
        # Alturas de pies
        # eleva si prefieres pies en lugar de rodilla
        l_foot = p.getLinkState(self.robot_id, self.left_foot_link_id)[0][2]
        r_foot = p.getLinkState(self.robot_id, self.right_foot_link_id)[0][2]
        obs.extend([l_foot, r_foot])
        
        return np.array(obs, dtype=np.float32)
    
    @property
    def contacto_pies(self):
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id, stable_foot=True)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id, stable_foot=True)
        return left_contact, right_contact
    

    def reset(self, seed=None, options=None):
        """
        Reset modificado específicamente para equilibrio en una pierna.
        
        Reemplazar el método reset() del entorno original con este.
        """
        super().reset(seed=seed)
        self.prev_action = np.zeros(self.num_active_pams)
        self.ep_total_actions = 0
        self.ep_expert_weight = 0.0

        
        # ===== RESET FÍSICO =====
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        try:
            self.dt = p.getPhysicsEngineParameters().get('fixedTimeStep', self.time_step)
        except Exception:
            self.dt = self.time_step
        self.sim_time = 0.0
        self.left_timer  = FootPhaseTimer(self.dt)
        self.right_timer = FootPhaseTimer(self.dt)
        
        # Configurar solver para estabilidad
        p.setPhysicsEngineParameter(
            numSolverIterations=80,         # antes 50  
            numSubSteps=6,
            contactBreakingThreshold=0.001, #subo de 0.0005 a 0.001
            erp=0.2,                    # antes 0.9
            contactERP=0.3,            # antes 0.95
            frictionERP=0.2,            # antes  0.9
            enableConeFriction=1,        # Habilitar fricción cónica
            deterministicOverlappingPairs=1
        )
        if self.logger:
            self.logger.log("main",f"🔧 Contact friction CORRECTED for single leg balance:")
            self.logger.log("main",f"   Feet: μ=0.8 (moderate grip, less spinning)")
            self.logger.log("main",f"   Legs: μ=0.1 (very low resistance)")
            self.logger.log("main",f"   Ground: μ=0.6 (controlled)")
            self.logger.log("main",f"   Solver: Enhanced stability parameters")
        
        # Cargar entorno
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            [0, 0, 1.21],  # Altura inicial ligeramente mayor
            useFixedBase=False,
            #flags=(p.URDF_USE_SELF_COLLISION| p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        )
        self.robot_data = PyBullet_Robot_Data(self.robot_id)

        # ===== SISTEMAS ESPECÍFICOS PARA EQUILIBRIO EN UNA PIERNA =====
        # Sistemas de recompensas
        if self.simple_reward_system is None:
            self.simple_reward_system = SimpleProgressiveReward(self)
        else:
            # solo re-vincula IDs si cambiaron, sin perder contadores/racha
            self.simple_reward_system.env = self
            self.simple_reward_system.robot_id = self.robot_id
            self.simple_reward_system.fixed_target_leg = self.fixed_target_leg
            self.simple_reward_system.reset()
        # ===== CONFIGURACIÓN ARTICULAR INICIAL =====
        
        # Posiciones iniciales para equilibrio en una pierna (ligeramente asimétricas)
        if "20" in self.robot_name:
            initial_positions = {
                # Pierna izquierda
                self.joint_indices[0]: 0.0,   # left_hip_pitch_joint 0.1
                self.joint_indices[1]: 0.0,   # left_hip_roll_joint
                self.joint_indices[2]: 0.0,   # left_knee_joint      0.1
                self.joint_indices[3]: 0.0,   # left_ankle_pitch_joint
                self.joint_indices[4]: 0.0,   # left_ankle_roll_joint 0.1
                # pierna derecha
                self.joint_indices[5]: 0.0,   # right_hip_roll_joint 0.1
                self.joint_indices[6]: 0.0,   # right_hip_pitch_joint
                self.joint_indices[7]: 0.0,   # right_knee_joint    0.1
                self.joint_indices[8]: 0.0,   # right_ankle_pitch_joint
                self.joint_indices[9]: 0.0    # right_ankle_roll_joint 0.1
            }
        elif "12" in self.robot_name:
            initial_positions = {
                # Pierna izquierda
                self.joint_indices[0]: np.deg2rad(0),   # debe ser menor 0
                self.joint_indices[1]: np.deg2rad(0),   # debe ser mayor 0
                self.joint_indices[2]: np.deg2rad(0),   # debe ser menor 0
                # pierna derecha
                self.joint_indices[3]: np.deg2rad(0),   # debe ser menor 0
                self.joint_indices[4]: np.deg2rad(0),   # debe ser mayor 0
                self.joint_indices[5]: np.deg2rad(0),   # debe ser menor 0
            }
        elif "16" in self.robot_name:
            initial_positions = {
                # Pierna izquierda
                self.joint_indices[0]: 0.0,   
                self.joint_indices[1]: 0.0,   
                self.joint_indices[2]: 0.0,   
                self.joint_indices[3]: 0.0,   
                # pierna derecha
                self.joint_indices[4]: 0.0,   
                self.joint_indices[5]: 0.0,   
                self.joint_indices[6]: 0.0,   
                self.joint_indices[7]: 0.0,   
            }
        
        for joint_id, pos in initial_positions.items():
            p.resetJointState(self.robot_id, joint_id, pos)
            p.setJointMotorControl2(self.robot_id, joint_id, p.VELOCITY_CONTROL, force=0)
            p.enableJointForceTorqueSensor(self.robot_id, joint_id, enableSensor=1)
        
        # ===== CONFIGURACIÓN DE DATOS Y CALCULADORES =====
        
        
        
        # ZMP calculator (todavía útil para métricas)
        self.zmp_calculator = ZMPCalculator(
            robot_id=self.robot_id,
            left_foot_id=self.left_foot_link_id,   # left_foot_link
            right_foot_id=self.right_foot_link_id,  # right_foot_link
            frequency_simulation=self.frequency_simulation,
            robot_data=self.robot_data,
            ground_id=self.plane_id,
            contact_state_fn=self.foot_contact_state
        )
        
        self._configure_contact_friction()
        
        # NO crear ankle_control ya que los tobillos están fijos
        
        # ===== RESET DE VARIABLES =====
        
        self.episode_reward = 0
        self.step_count = 0
        self.pam_states = {
            'pressures': np.zeros(self.num_active_pams),
            'forces': np.zeros(self.num_active_pams)
        }
        #Limite de torque de las articulaciones 
        self._build_tau_limit_maps(theta_samples_per_joint=181,   # malla de 1º aprox.
                                    cocontraction_norm=0.05,       # 5% de pretensión antagonista (ajustable)
                                    safety_eta=0.85                # 85% para no sobreestimar el ideal
                                )
        
        # ===== ESTABILIZACIÓN INICIAL =====
        
        # Más pasos para estabilización inicial (equilibrio en una pierna es más difícil)
        for _ in range(int(self.frame_skip)):

            p.stepSimulation()

        if self.zmp_calculator:
            # COM (usa tu helper de Pybullet_Robot_Data)
            try:
                com_world, _ = self.robot_data.get_center_of_mass()
                self.init_com_x, self.init_com_y, self.init_com_z = float(com_world[0]), float(com_world[1]), float(com_world[2])
            except Exception:
                self.init_com_x = self.init_com_y = self.init_com_z = 0.0
        
        # Obtener observación inicial
        observation = self._get_simple_observation_reset()
        
        info = {
            'episode_reward': 0,
            'episode_length': 0,
            'target_task': 'single_leg_balance'
        }
        # Para tracking de tiempo en balance
        
        print(f"🔄 Single leg balance environment reset - Ready for training")
        
        return observation, info
    
    def close(self):
        try:
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
        except:
            pass

    @property
    def current_balance_status(self):
        """Información actual del equilibrio en una pierna"""
        left_contact, right_contact=self.contacto_pies
        support_leg = {'left':left_contact, 'right':right_contact}
        raised_leg = {'left':not left_contact, 'right':not right_contact}
        return {
            'support_leg': support_leg,
            'raised_leg': raised_leg,
            'balance_time': 0,
            #'target_knee_height': self.target_knee_height,
            'episode_step': self.step_count
        }
        
    def parametros_torque_pam(self):
        # Momentos de brazo calculados desde dimensiones reales
        self.HIP_ROLL_FLEXOR_BASE_ARM = 0.055      # 5.03cm - basado en circunferencia del muslo
        self.HIP_ROLL_FLEXOR_VARIATION = 0.008     # ±1.01cm variación por ángulo
        
        self.HIP_ROLL_EXTENSOR_BASE_ARM = 0.052    
        self.HIP_ROLL_EXTENSOR_VARIATION = 0.006 

        self.HIP_PITCH_FLEXOR_BASE_ARM = 0.050
        self.HIP_PITCH_FLEXOR_VARIATION = 0.0085#round(self.HIP_PITCH_FLEXOR_BASE_ARM/4.98, 4) 
        
        self.HIP_PITCH_EXTENSOR_BASE_ARM = 0.054#0.0628    
        self.HIP_PITCH_EXTENSOR_VARIATION = 0.007#round(self.HIP_PITCH_EXTENSOR_BASE_ARM/4.98, 4) 

        self.HIP_YAW_FLEXOR_BASE_ARM = 0.048
        self.HIP_YAW_FLEXOR_VARIATION = 0.0080#round(self.HIP_PITCH_FLEXOR_BASE_ARM/4.98, 4) 
        
        self.HIP_YAW_EXTENSOR_BASE_ARM = 0.05#0.0628    
        self.HIP_YAW_EXTENSOR_VARIATION = 0.0065#round(self.HIP_PITCH_EXTENSOR_BASE_ARM/4.98, 4) 

        self.KNEE_FLEXOR_BASE_ARM = 0.0566     
        self.KNEE_FLEXOR_VARIATION = 0.010#round(self.KNEE_FLEXOR_BASE_ARM/5, 4)  

        self.KNEE_EXTENSOR_BASE_ARM = 0.0620#0.0640     
        self.KNEE_EXTENSOR_VARIATION = 0.008#round(self.KNEE_EXTENSOR_BASE_ARM/ 5, 4)
        # Cambiar a un valor más bajo si ese es el origen del problema
        # Si está versión no sirve, reducir la variación del brazo.
        self.ankle_pitch_FLEXOR_BASE_ARM = 0.04     
        self.ankle_pitch_FLEXOR_VARIATION = 0.0102#round(self.ankle_FLEXOR_BASE_ARM/4.2, 4)    

        self.ankle_pitch_EXTENSOR_BASE_ARM = 0.044#0.055     
        self.ankle_pitch_EXTENSOR_VARIATION = 0.0085#round(self.ankle_EXTENSOR_BASE_ARM/ 4.2, 4)

        self.ankle_roll_FLEXOR_BASE_ARM = 0.04     
        self.ankle_roll_FLEXOR_VARIATION = 0.0105#round(self.ankle_FLEXOR_BASE_ARM/4.2, 4)    

        self.ankle_roll_EXTENSOR_BASE_ARM = 0.044#0.055     
        self.ankle_roll_EXTENSOR_VARIATION = 0.0085#round(self.ankle_EXTENSOR_BASE_ARM/ 4.2, 4)

        

        self.KP = 80.0   # Ganancia proporcional
        self.KD = 12.0   # Ganancia derivativa    
        
        # Parámetros de resortes pasivos (calculados desde momento gravitacional)
        self.PASSIVE_SPRING_STRENGTH = 180.5   # N⋅m 
        self.DAMPING_COEFFICIENT = 12.0        # N⋅m⋅s/rad (optimizado para masa real)
        
        # Control antagónico
        self.INHIBITION_FACTOR = 0.3           # 30% inhibición recíproca
        self.VELOCITY_DAMPING_FACTOR = 0.08    # 8% reducción por velocidad
        
        # Límites de seguridad (basados en fuerzas PAM reales calculadas)
        self.MAX_REASONABLE_TORQUE_HIP_KNEE = 200.0     # N⋅m (factor de seguridad incluido)
        self.MAX_REASONABLE_TORQUE_FEET = 40.0
        self.joint_tau_scale = {}
        #control_joint_names
        for i, jid in enumerate(self.joint_indices):
            # TODO: si tienes un dict propio o lees 'effort' del URDF, reemplázalo aquí.
            print(i, self.control_joint_names[i])
            if "ankle" in self.control_joint_names[i]:
                self.joint_tau_scale[jid]=self.MAX_REASONABLE_TORQUE_FEET
            else:
                self.joint_tau_scale[jid] = self.MAX_REASONABLE_TORQUE_HIP_KNEE

    def hip_yaw_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de cadera específico para tu robot.
        Basado en geometría real: circunferencia muslo = 0.503m
        """
        # Flexor más efectivo cuando cadera está extendida (ángulo negativo)
        angle_factor = np.cos(angle + np.pi/6.0)  # Desplazamiento para peak en extensión
        return self.HIP_ROLL_FLEXOR_BASE_ARM + self.HIP_ROLL_FLEXOR_VARIATION * angle_factor
    
    def hip_yaw_extensor_moment_arm(self, angle):
        """
        Momento de brazo del extensor de cadera (glúteos).
        Más efectivo en rango medio de flexión.
        """
        # Extensor más efectivo en flexión ligera-moderada
        angle_factor = np.cos(angle - np.pi/6)  # Peak en flexión ligera
        R_raw = self.HIP_ROLL_EXTENSOR_BASE_ARM + self.HIP_ROLL_EXTENSOR_VARIATION * angle_factor
        # theta_cut, slope = -0.15, 0.08
        # sig = 1.0 / (1.0 + np.exp((angle - theta_cut)/slope))   # ~1 si angle << theta_cut
        # atten = 0.60 + 0.40 * (1.0 - sig)  # → 0.60 en negativos, → 1.0 en positivos
        return R_raw

    def hip_roll_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de cadera específico para tu robot.
        Basado en geometría real: circunferencia muslo = 0.503m
        """
        # Flexor más efectivo cuando cadera está extendida (ángulo negativo)
        angle_factor = np.cos(angle + np.pi/6.0)  # Desplazamiento para peak en extensión
        return self.HIP_ROLL_FLEXOR_BASE_ARM + self.HIP_ROLL_FLEXOR_VARIATION * angle_factor
    
    def hip_roll_extensor_moment_arm(self, angle):
        """
        Momento de brazo del extensor de cadera (glúteos).
        Más efectivo en rango medio de flexión.
        """
        # Extensor más efectivo en flexión ligera-moderada
        angle_factor = np.cos(angle - np.pi/6)  # Peak en flexión ligera
        R_raw = self.HIP_ROLL_EXTENSOR_BASE_ARM + self.HIP_ROLL_EXTENSOR_VARIATION * angle_factor
        # theta_cut, slope = -0.15, 0.08
        # sig = 1.0 / (1.0 + np.exp((angle - theta_cut)/slope))   # ~1 si angle << theta_cut
        # atten = 0.60 + 0.40 * (1.0 - sig)  # → 0.60 en negativos, → 1.0 en positivos
        return R_raw
    
    def hip_pitch_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de cadera específico para tu robot.
        Basado en geometría real: circunferencia muslo = 0.503m
        """
        # Flexor más efectivo cuando cadera está extendida (ángulo negativo)
        angle_factor = np.cos(angle + np.pi/3)  # Desplazamiento para peak en extensión
        return self.HIP_PITCH_FLEXOR_BASE_ARM + self.HIP_PITCH_FLEXOR_VARIATION * angle_factor
    
    def hip_pitch_extensor_moment_arm(self, angle):
        """
        Momento de brazo del extensor de cadera (glúteos).
        Más efectivo en rango medio de flexión.
        """
        # Extensor más efectivo en flexión ligera-moderada
        angle_factor = np.cos(angle - np.pi/3)  # Peak en flexión ligera
        return self.HIP_PITCH_EXTENSOR_BASE_ARM + self.HIP_PITCH_EXTENSOR_VARIATION * angle_factor

    def knee_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        # angle_factor = np.cos(angle + np.pi/3)
        angle_factor = np.cos(angle + np.pi/4)
        return self.KNEE_FLEXOR_BASE_ARM + self.KNEE_FLEXOR_VARIATION * angle_factor
    
    def knee_extensor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle - np.pi/6)
        return self.KNEE_EXTENSOR_BASE_ARM + self.KNEE_EXTENSOR_VARIATION * angle_factor
    
    def ankle_roll_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle + np.pi/6)
        return self.ankle_roll_FLEXOR_BASE_ARM + self.ankle_roll_FLEXOR_VARIATION * angle_factor
    
    def ankle_roll_extensor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle - np.pi/6)
        return self.ankle_roll_EXTENSOR_BASE_ARM + self.ankle_roll_EXTENSOR_VARIATION * angle_factor
    
    def ankle_pitch_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle + np.pi/6)
        return self.ankle_pitch_FLEXOR_BASE_ARM + self.ankle_pitch_FLEXOR_VARIATION * angle_factor
    
    def ankle_pitch_extensor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle - np.pi/6)
        return self.ankle_pitch_EXTENSOR_BASE_ARM + self.ankle_pitch_EXTENSOR_VARIATION * angle_factor

    # ===== MÉTODO DE DEBUG ADICIONAL =====

    def _debug_joint_angles_and_pressures(self, info, pam_pressures, done):
        """
            ✅ MÉTODO DE DEBUG para verificar la lógica biomecánica
        
            Llama esto ocasionalmente durante el step() para verificar que la lógica funciona
        """
        if self.step_count % (self.frequency_simulation//self.frame_skip) == 0 or done:  # Cada segundo aprox
            try:
                if self.csvlog:
                    row_com={"step": int(self.step_count),
                                "episode": int(self.n_episodes),
                                "t": round(self.step_count / self.frequency_simulation, 5),}
                    row_q_angle={"step": int(self.step_count),
                                "episode": int(self.n_episodes),
                                "t": round(self.step_count / self.frequency_simulation, 5),}
                    row_v_angle={"step": int(self.step_count),
                                "episode": int(self.n_episodes),
                                "t": round(self.step_count / self.frequency_simulation, 5),}
                    row_torque_angle={"step": int(self.step_count),
                                 "episode": int(self.n_episodes),
                                 "t": round(self.step_count / self.frequency_simulation, 5),}
                    row_force_angle={"step": int(self.step_count),
                                "episode": int(self.n_episodes),
                                "t": round(self.step_count / self.frequency_simulation, 5),}
                    row_pressure_PAM={"step": int(self.step_count),
                                "episode": int(self.n_episodes),
                                "t": round(self.step_count / self.frequency_simulation, 5),}
                    for idx, (name, state) in enumerate(zip(self.dict_joints.keys(), self.joint_states_properties)):
                        pos, vel, reaction, applied = state
                        Fx,Fy,Fz,Mx,My,Mz = reaction
                        row_q_angle[f"q_{name}"]=round(pos,3)
                        row_v_angle[f"vel_{name}"]=round(vel,3)
                        row_torque_angle[f"τ_reaction_{name}_x"]=round(Mx,2)
                        row_torque_angle[f"τ_reaction_{name}_y"]=round(My,2)
                        row_torque_angle[f"τ_reaction_{name}_z"]=round(Mz,2)
                        row_force_angle[f"Forces_{name}_x"]=round(Fx,3)
                        row_force_angle[f"Forces_{name}_y"]=round(Fy,3)
                        row_force_angle[f"Forces_{name}_z"]=round(Fz,3)
                        row_pressure_PAM[f"Pressure_{name}_flexion"]=pam_pressures[idx*2]
                        row_pressure_PAM[f"Pressure_{name}_extension"]=pam_pressures[idx*2+1]
                    row_com[f"COM_x"]=round(info["kpi"]['com_x'],3)
                    row_com[f"COM_y"]=round(info["kpi"]['com_y'],3)
                    row_com[f"COM_z"]=round(info["kpi"]['com_z'],3)
                    row_com[f"ZMP_x"]=round(info["kpi"]['zmp_x'],3)
                    row_com[f"ZMP_y"]=round(info["kpi"]['zmp_y'],3)
                    row_com[f"F_L"]=round(info["kpi"]['F_L'],3)
                    row_com[f"F_R"]=round(info["kpi"]['F_R'],3)
                    row_com[f"n_l"]=int(info["kpi"]['nL'])
                    row_com[f"n_r"]=int(info["kpi"]['nR'])
                    row_com[f'Masa']=round(self.mass,1)
                    row_com[f"COM_z_inicial"]=round(self.init_com_z,3)
                    row_com['posicion_x']=round(self.pos[0],3)
                    row_com['posicion_y']=round(self.pos[1],3)
                    #row_general['zmp_margain']=round(info["kpi"]["zmp_margin_m"], 3)
                    #row_general[f"ZMP_dist_to_COM"]=round(info["kpi"]['zmp_dist_to_com'],3)
                    self.csvlog.write("COM_values", row_com)
                    self.csvlog.write("angle_values", row_q_angle)
                    self.csvlog.write("speed_values", row_v_angle)
                    self.csvlog.write("torque_values", row_torque_angle)
                    self.csvlog.write("force_values", row_force_angle)
                    self.csvlog.write("pressure", row_pressure_PAM)


            except Exception as e:
                print(f"   ❌ Debug error: {e}")

    def debug_rewards(self):
        if self.step_count % self.frame_skip==0 and len(self.reawrd_step)>0:
            if self.csvlog:
                row_rewards={
                    "step":self.step_count,
                    "episode":self.n_episodes,
                    "t": round(self.step_count / self.frequency_simulation, 5),
                }
                for reward_name, reward_value in self.reawrd_step.items():
                    row_rewards[reward_name]=reward_value
                
                self.csvlog.write("rewards", row_rewards)

    
    def _moment_arm_funcs_for_joint(self, joint_name: str):
        # Devuelve funciones (flexor, extensor) r(θ) según tipo de joint
        if "hip_roll" in joint_name:
            return self.hip_roll_flexor_moment_arm, self.hip_roll_extensor_moment_arm
        elif "hip_pitch" in joint_name:
            return self.hip_pitch_flexor_moment_arm, self.hip_pitch_extensor_moment_arm
        elif "knee" in joint_name:
            return self.knee_flexor_moment_arm, self.knee_extensor_moment_arm
        elif "ankle_roll" in joint_name:
            return self.ankle_roll_flexor_moment_arm, self.ankle_roll_extensor_moment_arm
        elif "ankle_pitch" in joint_name:
            return self.ankle_pitch_flexor_moment_arm, self.ankle_pitch_extensor_moment_arm
        else:
            # Fallback prudente
            return (lambda th: 0.05), (lambda th: 0.05)
        
    def _build_tau_limit_maps(self,
                          theta_samples_per_joint: int = 181,
                          cocontraction_norm: float = 0.00,   # 0.0 = sin pretensión
                          safety_eta: float = 0.85):          # margen por realismo
        """
        Construye tablas τ_max_flex(θ) y τ_max_ext(θ) por articulación, suponiendo
        PAM ideal (volumen vejiga const.), con límites por P_max y epsilon_max del PAM.

        - cocontraction_norm: presión normalizada basal del antagonista (0..1)
        - safety_eta: factor <1 para ajustar sobreestimación del modelo ideal.
        """
        import numpy as _np

        self.tau_limit_maps = {}         # por joint_index: dict con 'theta', 'flex', 'ext'
        self.tau_limit_interp = {}       # por joint_index: dict con funciones interp

        # Recorremos tus articulaciones controladas
        for joint_name, joint_id in self.dict_joints.items():
            # 1) límites de ángulo reales desde URDF (PyBullet)
            ji = p.getJointInfo(self.robot_id, joint_id)
            th_lo, th_hi = float(ji[8]), float(ji[9])
            if th_lo >= th_hi:           # por si el URDF no define
                th_lo, th_hi = -1.0, 1.0

            thetas = _np.linspace(th_lo, th_hi, theta_samples_per_joint)

            # 2) localizar los PAMs flexor/extensor por nombre
            flex_name = f"{joint_name}_flexor"
            ext_name  = f"{joint_name}_extensor"
            if flex_name not in self.pam_muscles or ext_name not in self.pam_muscles:
                # si este joint no tiene pareja PAM (p.ej. no controlado), saltamos
                continue
            pam_flex = self.pam_muscles[flex_name]
            pam_ext  = self.pam_muscles[ext_name]

            # 3) funciones de brazo geométrico r(θ)
            r_flex_fn, r_ext_fn = self._moment_arm_funcs_for_joint(joint_name)

            # 4) presión (real) de agonista/antagonista
            Pmax_flex = pam_flex.max_pressure
            Pmax_ext  = pam_ext.max_pressure
            Pmin_flex = pam_flex.min_pressure + cocontraction_norm*(pam_flex.max_pressure - pam_flex.min_pressure)
            Pmin_ext  = pam_ext.min_pressure  + cocontraction_norm*(pam_ext.max_pressure  - pam_ext.min_pressure)

            # 5) conversión θ -> ε para cada músculo (misma convención que usas en runtime)
            def eps_from(theta, R, pam):
                return pam.epsilon_from_angle(theta, 0.0, max(abs(R), 1e-9))

            # 6) barrido y cálculo τ_max(θ)
            tau_max_flex = []
            tau_max_ext  = []
            for th in thetas:
                Rf = float(r_flex_fn(th))
                Re = float(r_ext_fn(th))
                eps_f = eps_from(th, Rf, pam_flex)
                eps_e = eps_from(th, Re, pam_ext)

                # Fuerzas extremas del AGONISTA y pretensión mínima del ANTAGONISTA
                Ff_max = pam_flex.force_model_new(Pmax_flex, eps_f)
                Fe_min = pam_ext.force_model_new(Pmin_ext,  eps_e)
                Fe_max = pam_ext.force_model_new(Pmax_ext, eps_e)
                Ff_min = pam_flex.force_model_new(Pmin_flex, eps_f)

                # Par límite hacia flexión y extensión (agonista - antagonista)
                tau_flex_lim = Rf*Ff_max - Re*Fe_min
                tau_ext_lim  = Re*Fe_max - Rf*Ff_min

                # margen de seguridad
                tau_max_flex.append(max(0.0, safety_eta * tau_flex_lim))
                tau_max_ext.append(max(0.0, safety_eta * tau_ext_lim))

            tau_max_flex = _np.asarray(tau_max_flex)
            tau_max_ext  = _np.asarray(tau_max_ext)

            # 7) guarda tablas + interpoladores lineales
            self.tau_limit_maps[joint_id] = {
                "theta": thetas,
                "flex":  tau_max_flex,
                "ext":   tau_max_ext
            }
            def _interp_vec(x, xgrid, ygrid):
                # saturamos fuera de rango al borde más cercano
                return float(_np.interp(x, xgrid, ygrid, left=ygrid[0], right=ygrid[-1]))

            self.tau_limit_interp[joint_id] = {
                "flex": lambda th, _g=thetas, _y=tau_max_flex: _interp_vec(th, _g, _y),
                "ext":  lambda th, _g=thetas, _y=tau_max_ext:  _interp_vec(th, _g, _y),
            }

    def torque_max_generation(self, torque_mapping):
        has_maps = self.tau_limit_interp and isinstance(self.tau_limit_interp, dict) and len(self.tau_limit_interp) > 0
        # 2) Estados actuales de las articulaciones (para θ)
        try:
            joint_states = p.getJointStates(self.robot_id, self.joint_indices)
            joint_positions = [float(s[0]) for s in joint_states]
        except Exception:
            joint_positions = None
            has_maps = False

        tau_utils = []
        tau_max_values=[]
        if has_maps and joint_positions is not None:
            # === Utilización con límites asimétricos dependientes de θ
            for i, jid in enumerate(self.joint_indices):
                tau_cmd = float(torque_mapping.get(jid, 0.0))
                th_i = joint_positions[i]
                lims = self.tau_limit_interp.get(jid, None)
                if lims is None:
                    continue
                # límites positivos/negativos en el ángulo actual
                tau_flex_max = max(0.0, float(lims["flex"](th_i)))  # + (flex)
                tau_ext_max  = max(0.0, float(lims["ext"](th_i)))   # - (ext)
                denom = tau_flex_max if tau_cmd >= 0.0 else tau_ext_max
                denom = max(denom, 1e-6)  # seguridad
                tau_utils.append(abs(tau_cmd) / denom)
                tau_max_values.append(denom)
        else:
            # === Fallback: escalado global previo
            joint_tau_scale = self.joint_tau_scale
            max_reasonable = float(getattr(self, "MAX_REASONABLE_TORQUE", 240.0))
            for jid, tau_cmd in torque_mapping.items():
                scale = max_reasonable
                if isinstance(joint_tau_scale, dict):
                    scale = float(joint_tau_scale.get(jid, max_reasonable))
                tau_utils.append(abs(float(tau_cmd)) / max(scale, 1e-6))
        return tau_utils
    



#Se usa para dar un periodo al paso del pie durante recompensas
class FootPhaseTimer:
    def __init__(self, dt, f_swing_hz=1.5, z_mid=0.03, z_amp=0.04):
        self.dt=dt; self.set_freq(f_swing_hz)
        self.z_mid=z_mid; self.z_amp=z_amp
        self.is_contact=True; self.prev_contact=True
        self.phase=0.0; self.t_swing=0.0; self.air_time_last=0.0
        self.touchdown_event=False; self.liftoff_event=False
    def set_freq(self, f): self.f_swing_hz=float(np.clip(f,0.6,3.0)); self.omega=2*np.pi*self.f_swing_hz
    def update(self, in_contact, cmd_speed=None):
        if cmd_speed is not None: self.set_freq(0.8 + 1.5*abs(cmd_speed))
        self.prev_contact=self.is_contact; self.is_contact=bool(in_contact)
        self.touchdown_event=(not self.prev_contact) and self.is_contact
        self.liftoff_event  = self.prev_contact and (not self.is_contact)
        if not self.is_contact:
            if self.liftoff_event: self.phase=0.0; self.t_swing=0.0
            else: self.phase=(self.phase + self.omega*self.dt)%(2*np.pi); self.t_swing+=self.dt
        elif self.touchdown_event:
            self.air_time_last=self.t_swing; self.phase=0.0; self.t_swing=0.0
    def z_ref(self):
        return self.z_mid + 0.5*self.z_amp*(1 - np.cos(self.phase))