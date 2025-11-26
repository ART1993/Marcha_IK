
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
                 print_env="ENV", csvlog=None,
                 simple_reward_mode="progressive",allow_hops:bool=False,
                 vx_target: float = 1.2, # Bajar a 0.6 si da problemas
                 robot_name="2_legged_human_like_robot12DOF"):
        
        """
            Inicio el entorno de entrenamiento PAM
        """
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super(Simple_Lift_Leg_BipedEnv, self).__init__()
        self.robot_name=robot_name
        self.robots_existentes=Rutas_Archivos.rutas_robots.value if "blackbird" not in self.robot_name else Rutas_Archivos.rutas_blackbird.value
        self.rutas_json=Rutas_Archivos.rutas_jsons.value
        print(self.robots_existentes, self.rutas_json)
        
        self.urdf_path = self.robots_existentes.get(f"{self.robot_name}")
        print(self.robot_name, self.urdf_path)
        with open(self.rutas_json.get(f"{self.robot_name}"), 'r') as f:
            json_file_robot_joint_info=load(f)
            
        self.joint_indices=[]
        self.control_joint_names=[]
        # de momento este dict es solo intuitivo
        self.limit_upper_lower_angles={}
        self.joint_tau_max_force = {}
        self.joint_max_angular_speed = {}
        if "2_legged_minihuman_legs_robot12DOF" in self.robot_name:
            foot_name="foot_top"
            left_foot=f"left_{foot_name}"
            right_foot=f"right_{foot_name}"
        elif "blackbird" in self.robot_name:
            foot_name="foot"
            left_foot=f"l_{foot_name}"
            right_foot=f"r_{foot_name}"
        else:
            foot_name="foot_link"
            left_foot=f"left_{foot_name}"
            right_foot=f"right_{foot_name}"
        for key, values in json_file_robot_joint_info.items():
            if values.get('type')!=4:
                self.joint_indices.append(values.get("index"))
                self.control_joint_names.append(values.get("name"))
                self.limit_upper_lower_angles[values.get("index")]={'lower':values.get("lower"),'upper':values.get("upper")}
                self.joint_tau_max_force[values.get("index")]=values.get("max_force")
                self.joint_max_angular_speed[values.get("index")]=values.get("max_velocity")
            
            if values.get("link_name")==left_foot:
                self.left_foot_link_id=values.get("index")
            elif values.get("link_name")==right_foot:
                self.right_foot_link_id=values.get("index")

        self.footcontact_state=FootContactState
        # ===== CONFIGURACIÓN BÁSICA =====
        self.max_pressure=6
        self.pam_muscles = PAM_McKibben(self.robot_name, self.control_joint_names, max_pressure=self.max_pressure)
        self.render_mode = render_mode
        self.logger=logger
        self.csvlog = csvlog
        # === Opciones de recompensa (leídas por RewardSystemSimple) ===
        self.simple_reward_mode = simple_reward_mode    # "progressive" | "walk3d" | "lift_leg" | "march_in_place"
        self.allow_hops = bool(allow_hops)              # permitir ambos pies en el aire (no-support permitido)
        self.vx_target = float(vx_target)               # objetivo de velocidad para walk3d

        
        self.muscle_names = list(self.pam_muscles.keys())
        print(self.muscle_names)
        self.num_active_pams = len(self.muscle_names)
        
        self.frequency_simulation=400.0 # Pasar a 400 quizas ese sea el problema
        #Probar para ver si evita tembleques
        self.time_step = 1.0 / self.frequency_simulation
        # Action-repeat/frame-skip: aplicar una acción a 400 Hz simulación pero 50 Hz control
        self.frame_skip = 40
        
        # Estados PAM básicos
        self.pam_states = {
            'pressures': np.zeros(self.num_active_pams),
            'forces': np.zeros(self.num_active_pams)
        }
        
        # ===== CONFIGURACIÓN DE ESPACIOS =====
        # self.recent_rewards=deque(maxlen=50)
        # Action space: self.num_active_pams presiones PAM normalizadas [0, 1]
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_active_pams,), 
            dtype=np.float32
        )
        
        # Observation space
        obs_dim=18 + 2*len(self.joint_indices)
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
        
        self.reward_system = None
        self.step_count = 0
        self.step_total=0
        self.robot_id = None
        self.plane_id = None
        
        

        self.joint_names=self.control_joint_names
        
        self.dict_joints= {joint_name:joint_index for joint_name, joint_index in zip(self.joint_names, self.joint_indices)}
        #ID Links de los pies (igual al de los tobillos)
        
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
        
        self.ep_total_actions += 1

        # ===== NORMALIZAR Y VALIDAR ACCIÓN =====
    
        normalized_pressures = np.clip(u_final, 0.0, 1.0) 
        
        # Aplicar fuerzas PAM normalizadas
        self.joint_torques = self._apply_pam_forces(normalized_pressures)
        
        # ===== Paso 3: SIMULACIÓN FÍSICA =====

        # Aplicar torques
        # En el caso que quiera reducir los torques a usar y por tanto los joints no fijos #[:len(joint_torques)]
        torque_mapping = {jid: self.joint_torques[i] for i, jid in enumerate(self.joint_indices)}

        #self.last_tau_cmd = {jid: float(tau) for jid, tau in torque_mapping}
        for joint_id, torque in torque_mapping.items():
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )
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

        self.joint_states_properties = p.getJointStates(self.robot_id, self.joint_indices)
        
        # ===== CÁLCULO DE RECOMPENSAS CONSCIENTE DEL CONTEXTO =====
        done = self.simple_reward_system.is_episode_done(self.step_count)
        reward = self.simple_reward_system.calculate_reward(u_final)
        
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
        
        
        
        return observation, reward, done, False, info
    

    def info_pam_torque(self, info):
        jt = getattr(self, "pam_states", {}).get("joint_torques", None)
        ps = getattr(self, "pam_states", {}).get("pressures", None)

        # Ejemplos: mapea índices a nombres (ver mapeo más abajo)
        if jt is not None:
            for i, joint_name in enumerate(self.control_joint_names):
                info["kpi"][f"tau_{joint_name}"]   = float(jt[i])  

        if ps is not None:
            for i, mucle_name in enumerate(self.muscle_names):
                info["kpi"][mucle_name] = float(ps[i]) 


        for joint_name, joint_index  in self.dict_joints.items():
            info["kpi"][f"q_{joint_name}"] = float(p.getJointState(self.robot_id, joint_index)[0])
        return info
    
    def function_logger_kpi(self, info, reward, done):
        if self.simple_reward_system:
            info['system_type'] = 'progressive'
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
                            "yaw": float(self.euler[2]),
                            "left_down": int(bool(left_down)),
                            "right_down": int(bool(right_down)),
                            "F_L": float(F_L),
                            "F_R": float(F_R),
                            "nL": int(n_l), "nR": int(n_r),
                            "state_L": left_state, "state_R": right_state,
                            "zmp_x": float(self.zmp_x),
                            "zmp_y": float(self.zmp_y),
                            "com_x": (self.com_x),
                            "com_y": (self.com_y),
                            "com_z": self.com_z,
                        }
            # Debug simple
            info = self.info_pam_torque(info)
            # opcional: incluir en info
            if done:
                info["ep_kpi"] = {
                                    "ep_return": float(self.episode_reward),
                                    "ep_len": int(self.step_count),
                                    "done_reason": getattr(self.simple_reward_system, "last_done_reason", None)
                                }
                self.n_episodes+=1
        
        # === CSVLogger: volcado per-step (~10 Hz) ===
        if (self.step_count % (self.frame_skip) == 0 or done) and self.simple_reward_system:
            if self.logger:
                self.logger.log("main",f"🔍 Step {self.step_count} - Control Analysis:")
                self.logger.log("main",f"   Height: {self.pos[2]:.2f}m")
                self.logger.log("main",f"   Tilt: Roll {math.degrees(self.euler[0]):.1f}°, Pitch {math.degrees(self.euler[1]):.1f}°")
                kpi_dbg = info.get("kpi", {})
                self.logger.log("main",f"   COM: ({kpi_dbg.get('com_x', 0.0):.3f}, {kpi_dbg.get('com_y', 0.0):.3f}, {kpi_dbg.get('com_z', 0.0):.3f}) m  ")
                self.logger.log("main",f"   ZMP: ({kpi_dbg.get('zmp_x', 0.0):.3f}, {kpi_dbg.get('zmp_y', 0.0):.3f}) m")
            

        # DEBUG TEMPORAL: Verificar timing cada cierto número de steps
            elapsed_time = self.step_count / self.frequency_simulation
            #logger.log(f" {action_source} action, reward={reward:.2f}")
            if self.logger:
                self.logger.log("main",f"🕒 Step {self.step_count} ({elapsed_time:.1f}s elapsed):")
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
            spinningFriction=0.2,
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
        if self.step_count % (self.frame_skip) == 0:  # Cada segundos aprox
            if self.logger:
                self.logger.log("main",f"Contact force on link {link_id}: {F_total:.2f} N")
        if stable_foot:
            return (F_total > min_F) and (num_contactos>1)
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
    
    @staticmethod
    def pie_tocando_suelo(robot_id, foot_link, fz_min=5.0):
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
            Convertir presiones PAM a torques articulares
        """
       
        # NUEVA LÓGICA: Control automático de rodilla levantada
        joint_torques = seleccionar_funcion_calculo_torques(self, pam_pressures)
        # Activar control automatico de rodilla

        balance_info = self.current_balance_status
        if self.step_count%(self.frame_skip)==0:
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
        # Altura COM (si ya la calculas, úsala; si no, aprox con base z)
        com_z = getattr(self, "com_z", self.init_pos[2])
        
        # Posición y orientación  
        obs.extend([self.init_pos[0],self.init_pos[1], com_z, euler[0], euler[1],euler[2]])  # x, y, z, roll, pitch

        yaw = euler[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        _,_,vel_COM_z=self.vel_COM
        vx_b =  cy*lin_vel[0] + sy*lin_vel[1]
        vy_b = -sy*lin_vel[0] + cy*lin_vel[1]
        obs.extend([vx_b, vy_b, vel_COM_z, init_ang_vel[0], init_ang_vel[1], init_ang_vel[2]])
        
        
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
        left_contact=self.pie_tocando_suelo(self.robot_id, foot_link=self.left_foot_link_id, fz_min=5.0)
        right_contact=self.pie_tocando_suelo(self.robot_id, foot_link=self.right_foot_link_id, fz_min=5.0)
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
        com_z = getattr(self, "com_z", self.pos_post[2])
        # Posición y orientación
        obs.extend([self.pos_post[0], self.pos_post[1], com_z, self.euler_post[0], self.euler_post[1],self.euler_post[2]])
        
        # Velocidades
        yaw = self.euler_post[2]
        # rotación mundo->cuerpo (2D yaw)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        _,_,vel_COM_z=self.vel_COM
        vx_b =  cy*lin_vel[0] + sy*lin_vel[1]
        vy_b = -sy*lin_vel[0] + cy*lin_vel[1]
        obs.extend([vx_b, vy_b, vel_COM_z, ang_vel[0], ang_vel[1], ang_vel[2]])
        
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
        left_contact=self.pie_tocando_suelo(self.robot_id,foot_link=self.left_foot_link_id, fz_min=5.0)
        right_contact=self.pie_tocando_suelo(self.robot_id,foot_link=self.right_foot_link_id, fz_min=5.0)
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
            self.logger.log("main",f"🔧 Contact friction CORRECTED for walking:")
            self.logger.log("main",f"   Solver: Enhanced stability parameters")

        if "blackbird" in self.robot_name:
            pos=[0, 0, 1.10]
            orientation=p.getQuaternionFromEuler([0, 0, np.pi/2])
        elif "12DOF" in self.robot_name and "_done" not in self.robot_name:
            pos=[0, 0, 0.86]
            orientation=p.getQuaternionFromEuler([0, 0, 0])
        else:
            pos=[0, 0, 1.21]
            orientation=p.getQuaternionFromEuler([0, 0, 0])
        # Cargar entorno
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            pos,  # Altura inicial ligeramente mayor
            orientation,
            useFixedBase=False
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
            self.simple_reward_system.reset()
        # ===== CONFIGURACIÓN ARTICULAR INICIAL =====
        
        # Posiciones iniciales para equilibrio en una pierna (ligeramente asimétricas)
        if len(self.muscle_names)==20:
            initial_positions = {
                # Pierna izquierda
                self.joint_indices[0]: 0.0,
                self.joint_indices[1]: 0.0,   
                self.joint_indices[2]: 0.0,   
                self.joint_indices[3]: 0.0,   
                self.joint_indices[4]: 0.0,   
                # pierna derecha
                self.joint_indices[5]: 0.0,   
                self.joint_indices[6]: 0.0,   
                self.joint_indices[7]: 0.0,   
                self.joint_indices[8]: 0.0,   
                self.joint_indices[9]: 0.0    
            }
        if len(self.muscle_names)==24:
            initial_positions = {
                # Pierna izquierda
                self.joint_indices[0]: 0.0,
                self.joint_indices[1]: 0.0,   
                self.joint_indices[2]: 0.0,   
                self.joint_indices[3]: 0.0,   
                self.joint_indices[4]: 0.0,
                self.joint_indices[5]: 0.0,  
                # pierna derecha
                self.joint_indices[6]: 0.0,   
                self.joint_indices[7]: 0.0,   
                self.joint_indices[8]: 0.0,   
                self.joint_indices[9]: 0.0,   
                self.joint_indices[10]: 0.0,  
                self.joint_indices[11]: 0.0   
            }
        elif len(self.muscle_names)==12:
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
        elif len(self.muscle_names)==16:
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
        
        
        # ===== ESTABILIZACIÓN INICIAL =====
        
        # Más pasos para estabilización inicial (equilibrio en una pierna es más difícil)
        p.stepSimulation()

        if self.zmp_calculator:
            # COM (usa tu helper de Pybullet_Robot_Data)
            try:
                com_world, self.mass = self.robot_data.get_center_of_mass()
                self.init_com_x, self.init_com_y, self.init_com_z = float(com_world[0]), float(com_world[1]), float(com_world[2])
                self.vel_COM=self.robot_data.get_center_of_mass_velocity()
            except Exception:
                self.init_com_x = self.init_com_y = self.init_com_z = 0.0

        self.joint_states_properties = p.getJointStates(self.robot_id, self.joint_indices)
        
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
        return {
            'support_leg': support_leg,
            'balance_time': 0,
            #'target_knee_height': self.target_knee_height,
            'episode_step': self.step_count
        }
        
    def parametros_torque_pam(self):
        # Momentos de brazo calculados desde dimensiones reales
        self.HIP_ROLL_FLEXOR_BASE_ARM = 0.055      
        self.HIP_ROLL_FLEXOR_VARIATION = 0.008     
        
        self.HIP_ROLL_EXTENSOR_BASE_ARM = 0.052    
        self.HIP_ROLL_EXTENSOR_VARIATION = 0.006 

        self.HIP_PITCH_FLEXOR_BASE_ARM = 0.050
        self.HIP_PITCH_FLEXOR_VARIATION = 0.0085
        
        self.HIP_PITCH_EXTENSOR_BASE_ARM = 0.054
        self.HIP_PITCH_EXTENSOR_VARIATION = 0.007

        self.HIP_YAW_FLEXOR_BASE_ARM = 0.048
        self.HIP_YAW_FLEXOR_VARIATION = 0.0080
        
        self.HIP_YAW_EXTENSOR_BASE_ARM = 0.050
        self.HIP_YAW_EXTENSOR_VARIATION = 0.0065

        self.KNEE_FLEXOR_BASE_ARM = 0.0566     
        self.KNEE_FLEXOR_VARIATION = 0.010

        self.KNEE_EXTENSOR_BASE_ARM = 0.0620 
        self.KNEE_EXTENSOR_VARIATION = 0.008
        # Cambiar a un valor más bajo si ese es el origen del problema
        # Si está versión no sirve, reducir la variación del brazo.
        self.ankle_pitch_FLEXOR_BASE_ARM = 0.04     
        self.ankle_pitch_FLEXOR_VARIATION = 0.0102  

        self.ankle_pitch_EXTENSOR_BASE_ARM = 0.044 
        self.ankle_pitch_EXTENSOR_VARIATION = 0.0085

        self.ankle_roll_FLEXOR_BASE_ARM = 0.04     
        self.ankle_roll_FLEXOR_VARIATION = 0.0105

        self.ankle_roll_EXTENSOR_BASE_ARM = 0.044
        self.ankle_roll_EXTENSOR_VARIATION = 0.0085

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
        if self.step_count % (self.frame_skip) == 0 or done:  # Cada segundo aprox
            try:
                if self.csvlog:
                    row_com={"step": int(self.step_count),
                                "episode": int(self.n_episodes),
                                "t": round(self.step_count / self.frequency_simulation, 5),}
                    if self.print_env == "TEST":
                        self.debug_rewards()
                        row_q_angle={"step": int(self.step_count),
                                    "episode": int(self.n_episodes),
                                    "t": round(self.step_count / self.frequency_simulation, 5),}
                        row_v_angle={"step": int(self.step_count),
                                    "episode": int(self.n_episodes),
                                    "t": round(self.step_count / self.frequency_simulation, 5),}
                        
                        for name, state in zip(self.dict_joints.keys(), self.joint_states_properties):
                            pos, vel, _, _ = state
                            row_q_angle[f"q_{name}"]=round(pos,3)
                            row_v_angle[f"vel_{name}"]=round(vel,3)
                            
                        self.csvlog.write("angle_values", row_q_angle)
                        self.csvlog.write("speed_values", row_v_angle)
                        
                    
                    row_pressure_PAM={"step": int(self.step_count),
                                    "episode": int(self.n_episodes),
                                    "t": round(self.step_count / self.frequency_simulation, 5),}
                    for idx, (name, state) in enumerate(zip(self.dict_joints.keys(), self.joint_states_properties)):
                        row_pressure_PAM[f"Pressure_{name}_flexion"]=pam_pressures[idx*2]
                        row_pressure_PAM[f"Pressure_{name}_extension"]=pam_pressures[idx*2+1]
                        row_pressure_PAM[f"τ_aplicado_{name}"]=round(self.joint_torques[idx] ,2)
                    row_com[f"COM_x"]=round(info["kpi"]['com_x'],3)
                    row_com[f"COM_y"]=round(info["kpi"]['com_y'],3)
                    row_com[f"COM_z"]=round(info["kpi"]['com_z'],3)
                    
                    row_com[f"F_L"]=round(info["kpi"]['F_L'],3)
                    row_com[f"F_R"]=round(info["kpi"]['F_R'],3)
                    row_com[f"n_l"]=int(info["kpi"]['nL'])
                    row_com[f"n_r"]=int(info["kpi"]['nR'])
                    row_com[f'Masa']=round(self.mass,1)
                    row_com[f"COM_z_inicial"]=round(self.init_com_z,3)
                    row_com['posicion_x']=round(self.pos[0],3)
                    row_com['posicion_y']=round(self.pos[1],3)
                    self.csvlog.write("COM_values", row_com)
                    self.csvlog.write("pressure", row_pressure_PAM)
                    
                    


            except Exception as e:
                print(f"   ❌ Debug error: {e}")

    def debug_rewards(self):
        if self.step_count % self.frame_skip==0 and len(self.reawrd_step)>0:
            #if self.csvlog:
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