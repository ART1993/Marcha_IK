
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

from Archivos_Apoyo.Configuraciones_adicionales import PAM_McKibben,Rutas_Archivos, \
                                                    calculate_robot_specific_joint_torques_16_pam
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data

from Archivos_Recompensas.RewardSystemSimple import SimpleProgressiveReward
           

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
                 vx_target: float = 0.6,
                 robot_name="2_legged_human_like_robot16DOF"):
        
        """
            Inicio el entorno de entrenamiento PAM
        """
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super(Simple_Lift_Leg_BipedEnv, self).__init__()
        self.robots_existentes=Rutas_Archivos.rutas_robots.value
        self.rutas_json=Rutas_Archivos.rutas_jsons.value
        self.urdf_path = self.robots_existentes.get(f"{robot_name}")
        with open(self.rutas_json.get(f"{robot_name}"), 'r') as f:
            json_file_robot_joint_info=load(f)
        # ===== CONFIGURACIÓN BÁSICA =====
        self.pam_muscles = PAM_McKibben()
        self.render_mode = render_mode
        self.logger=logger
        self.csvlog = csvlog
        # === Opciones de recompensa (leídas por RewardSystemSimple) ===
        self.simple_reward_mode = simple_reward_mode    # "progressive" | "walk3d" | "lift_leg" | "march_in_place"
        self.allow_hops = bool(allow_hops)              # permitir ambos pies en el aire (no-support permitido)
        self.vx_target = float(vx_target)               # objetivo de velocidad para walk3d

        
        self.muscle_names = list(self.pam_muscles.keys())
        
        self.num_active_pams = len(self.muscle_names)

        self.contact_established = False
        self.contact_stable_steps = 0
        # Para tracking de tiempo en balance
        self._balance_start_time = 0
        # ===== CONFIGURACIÓN FÍSICA BÁSICA =====
        
        
        self.frequency_simulation=400.0
        self.switch_interval=2000  # Intervalo para cambiar pierna objetivo en curriculum
        self.time_step = 1.0 / self.frequency_simulation
        # ===== CONFIGURACIÓN PAM SIMPLIFICADA =====
        
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
        
        # Observation space SIMPLIFICADO: 20 elementos total
        # - 8: Estado del torso (pos, orient, velocidades)
        # - 8: Estados articulares básicos (posiciones)
        # - 2: ZMP básico (x, y)
        # - 2: Contactos de pies (izq, der)
        obs_dim=22
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
        self.total_reward = 0
        self.robot_id = None
        self.plane_id = None
        #self.joint_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee]
        self.joint_indices=[]
        self.control_joint_names=[]
        for key, values in json_file_robot_joint_info.items():
            self.joint_indices.append(values.get("index"))
            self.control_joint_names.append(values.get("name"))
        # self.control_joint_names = ['left_hip_roll_joint','left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 
        #                             'right_hip_roll_joint','right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint']
        self.joint_names=self.control_joint_names
        
        self.dict_joints= {joint_name:joint_index for joint_name, joint_index in zip(self.joint_names, self.joint_indices)}
        #ID Links de los pies (igual al de los tobillos)
        self.left_foot_link_id = 3
        self.right_foot_link_id = 7

        self.swing_hip_target = 0.05
        self.swing_hip_tol=0.10 

        self.swing_knee_lo = 0.40
        self.swing_knee_hi = 0.85
        # Añadir tracking de pierna levantada
        self.fixed_target_leg = fixed_target_leg
        self.raised_leg = self.fixed_target_leg  # 'left' o 'right' - cuál pierna está levantada
        
        self.target_knee_height = 0.8  # Altura objetivo de la rodilla levantada
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
            self.logger.log("main",f"🤖 Using {self.num_active_pams=:} "
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
        self.step_count += 1
        self.info = {"kpi": {}}
        # ===== DECISIÓN: EXPERTO vs RL =====
        # En env.step (o donde construyas la acción final)
        self.pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        self.euler = p.getEulerFromQuaternion(orn)
       
        u_final = np.clip(action, 0.0, 1.0)
        # Probar los dos y ver cual da mejor resultados
        #delta = np.clip(u_final - self.prev_action, -0.05, 0.05)
        #u_final = self.prev_action + delta
        #self.prev_action = u_final.copy()
        self.ep_total_actions += 1

        # ===== NORMALIZAR Y VALIDAR ACCIÓN =====
    
        normalized_pressures = np.clip(u_final, 0.0, 1.0) 
        
        # Aplicar fuerzas PAM normalizadas
        joint_torques = self._apply_pam_forces(normalized_pressures)

        # NUEVA LÍNEA: Validar comportamiento biomecánico
        is_valid = self.validate_robot_specific_behavior(normalized_pressures, joint_torques)
        
        # ===== Paso 3: SIMULACIÓN FÍSICA =====

        # Aplicar torques
        #torque_mapping = [(joint, joint_torques[i]) for i, joint in enumerate(self.joint_indices)]
        # En el caso que quiera reducir los torques a usar y por tanto los joints no fijos
        torque_mapping = [(jid, joint_torques[i]) for i, jid in enumerate(self.joint_indices[:len(joint_torques)])]

        #self.last_tau_cmd = {jid: float(tau) for jid, tau in torque_mapping}
        for joint_id, torque in torque_mapping:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )

        p.stepSimulation()
        system_used = "PROGRESSIVE"

        
        #if self.simple_reward_system:
        # Joint indices [0,1,2,4,5,6]
        self.joint_states_properties = self.obtener_estado_articulaciones()
        
        done = self.simple_reward_system.is_episode_done(self.step_count)
        reward = self.simple_reward_system.calculate_reward(u_final, self.step_count)
        # ===== CÁLCULO DE RECOMPENSAS CONSCIENTE DEL CONTEXTO =====
       
        
        
        # ===== PASO 4: OBSERVACIÓN Y TERMINACIÓN =====
        self.episode_reward += reward
        
        observation = self._get_simple_observation()

        # Info simplificado
        info = {
            'step_count': self.step_count,
            'reward': reward,
            #'action_source': action_source,
            'episode_reward': self.episode_reward
        }
        
        info, reward, done, is_valid, system_used = self.function_logger_kpi(info, reward, done, is_valid, system_used)


        self._debug_joint_angles_and_pressures(info, u_final, done)
        
        self.debug_rewards()
        
        
        
        return observation, reward, done, False, info
    
    def obtener_estado_articulaciones(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)  # sólo 8 DOF controlados
        self.left_hip_roll_angle = joint_states[0][0]
        self.left_hip_pitch_angle = joint_states[1][0]
        self.left_knee_angle = joint_states[2][0]
        self.left_ankle_angle = joint_states[3][0]
        self.right_hip_roll_angle = joint_states[4][0]
        self.right_hip_pitch_angle = joint_states[5][0]
        self.right_knee_angle = joint_states[6][0]
        self.right_ankle_angle = joint_states[7][0]
        return joint_states
    

    def info_pam_torque(self, info):
        jt = getattr(self, "pam_states", {}).get("joint_torques", None)
        ps = getattr(self, "pam_states", {}).get("pressures", None)

        # Ejemplos: mapea índices a nombres (ver mapeo más abajo)
        if jt is not None:
            info["kpi"]["tau_LHR"]   = float(jt[0])  # Left Hip Roll
            info["kpi"]["tau_LHP"]   = float(jt[1])  # Left Hip Pitch
            info["kpi"]["tau_LK"]    = float(jt[2])  # Left Knee
            info["kpi"]["tau_LA"]   = float(jt[3])  # Right ankle
            info["kpi"]["tau_RHR"]   = float(jt[4])  # Right Hip Roll
            info["kpi"]["tau_RHP"]    = float(jt[5])  # Right HIP Pitch
            info["kpi"]["tau_RK"]    = float(jt[6])  # Right Knee
            info["kpi"]["tau_RA"]    = float(jt[7])  # Right ankle

        if ps is not None:
            info["kpi"]["u_LHR_flex"] = float(ps[0])   # PAM 0: flexor cadera izq (roll)
            info["kpi"]["u_LHR_ext"]  = float(ps[1])   # PAM 1: extensor cadera izq (roll)
            info["kpi"]["u_RHR_flex"] = float(ps[2])   # PAM 2: flexor cadera der (roll)
            info["kpi"]["u_RHR_ext"]  = float(ps[3])   # PAM 3: extensor cadera der (roll)
            info["kpi"]["u_LHP_flex"] = float(ps[4])   # PAM 4: flexor cadera izq (pitch)
            info["kpi"]["u_LHP_ext"]  = float(ps[5])   # PAM 5: extensor cadera izq (pitch)
            info["kpi"]["u_RHP_flex"] = float(ps[6])   # PAM 6: flexor cadera der (pitch)
            info["kpi"]["u_RHP_ext"]  = float(ps[7])   # PAM 7: extensor cadera der (pitch)
            info["kpi"]["u_LK_flex"]  = float(ps[8])   # PAM 8: flexor rodilla izq
            info["kpi"]["u_LK_ext"]   = float(ps[9])   # PAM 9: extensor rodilla izq
            info["kpi"]["u_RK_flex"]  = float(ps[10])  # PAM 10: flexor rodilla der
            info["kpi"]["u_RK_ext"]   = float(ps[11])  # PAM 11: extensor rodilla der
            info["kpi"]["u_RA_ext"]  = float(ps[12])   # PAM 12: extensor tobillo der (pitch)
            info["kpi"]["u_LA_flex"]  = float(ps[13])   # PAM 13: flexor tobillo izq
            info["kpi"]["u_LA_ext"]   = float(ps[14])   # PAM 14: extensor tobillo izq
            info["kpi"]["u_RA_flex"]  = float(ps[15])  # PAM 15: flexor tobillo der


        if hasattr(self, "left_hip_roll_angle"):
            info["kpi"]["q_LHR"] = self.left_hip_roll_angle
            info["kpi"]["q_LHP"] = self.left_hip_pitch_angle
            info["kpi"]["q_LK"]  = self.left_knee_angle
            info["kpi"]["q_RHR"] = self.right_hip_roll_angle
            info["kpi"]["q_RHP"] = self.right_hip_pitch_angle
            info["kpi"]["q_RK"]  = self.right_knee_angle
        return info
    
    def function_logger_kpi(self, info, reward, done, is_valid, system_used):
        if self.simple_reward_system:
            curriculum_info = self.simple_reward_system.get_info()  # Solo una llamada
            info['curriculum'] = curriculum_info  # Añadir sin reemplazar
            info['system_type'] = 'progressive'
            info['current_level'] = curriculum_info.get('level', 1)
            n_l, F_L = self.contact_normal_force(self.left_foot_link_id)
            n_r, F_R = self.contact_normal_force(self.right_foot_link_id)
            left_down, right_down = self.contacto_pies

            # KPI ZMP ya existente
            if self.zmp_calculator:
                try:
                    zmp_xy = self.zmp_calculator.calculate_zmp()
                    zmp_x, zmp_y = float(zmp_xy[0]), float(zmp_xy[1])
                    # COM (usa tu helper de Pybullet_Robot_Data)
                    try:
                        com_world, _m = self.robot_data.get_center_of_mass
                        com_x, com_y, com_z = float(com_world[0]), float(com_world[1]), float(com_world[2])
                    except Exception:
                        com_x = com_y = com_z = 0.0
                except Exception:
                    zmp_x, zmp_y = 0.0, 0.0
            else:
                zmp_x, zmp_y = 0.0, 0.0
            info["kpi"] = {
                            "global_step": int(self.step_count),
                            "reward": float(reward),
                            "roll": float(self.euler[0]),
                            "pitch": float(self.euler[1]),
                            "left_down": int(bool(left_down)),
                            "right_down": int(bool(right_down)),
                            "F_L": float(F_L),
                            "F_R": float(F_R),
                            "zmp_x": float(zmp_x),
                            "zmp_y": float(zmp_y),
                            "com_x": com_x,
                            "com_y": com_y,
                            "com_z": com_z,
                        }
            # Debug simple
            info = self.info_pam_torque(info)
            if done:
                info["ep_kpi"] = {
                                    "ep_return": float(self.episode_reward),
                                    "ep_len": int(self.step_count),
                                    "done_reason": getattr(self.simple_reward_system, "last_done_reason", None)
                                }
                episode_total = info['episode_reward']  # Ya calculado arriba
                self.n_episodes+=1
                
                if self.logger:
                    self.logger.log("main",f"📈 Episode {info['curriculum']['episodes']} | Level {info['curriculum']['level']} | Reward: {episode_total:.1f}")
        
        # === CSVLogger: volcado per-step (~10 Hz) ===
        if (self.step_count % (self.frequency_simulation//10) == 0 or done) and self.simple_reward_system:
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
                self.logger.log("main",f"Step {done=:}, is_valid={is_valid}")
                self.logger.log("main",f"🎮 Active system: {system_used} at step {self.step_count}")
                self.logger.log("main",f"🕒 Step {self.step_count} ({elapsed_time:.1f}s elapsed):")
                self.logger.log("main",f"   Current level: {curriculum_info['level']}")
                self.logger.log("main",f"   Target leg: {curriculum_info.get('target_leg', 'N/A')}")
                self.logger.log("main",f"   Switch timer: {self.simple_reward_system.switch_timer}/{self.simple_reward_system.switch_interval}")
        return info, reward, done, is_valid, system_used
    

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
                lateralFriction=0.1,    # Muy reducida de 0.6 a 0.1
                spinningFriction=0.05,  # Muy reducida de 0.4 a 0.05
                rollingFriction=0.01,   # Muy reducida de 0.05 a 0.01
                restitution=0.05
            )

        # Pie izquierdo - alta fricción para agarre
        for foot_id in (self.left_foot_link_id, self.right_foot_link_id):
            p.changeDynamics(
                self.robot_id, 
                foot_id,
                lateralFriction=0.9,                #0.8,       
                spinningFriction=0.2,                   #0.15,       
                rollingFriction=0.01,       
                restitution=0.01,           
                contactDamping=100,         
                contactStiffness=15000,      
                frictionAnchor=1
            )
        
        # ===== FRICCIÓN DEL SUELO =====
        
        # Configurar fricción del plano del suelo
        p.changeDynamics(
            self.plane_id,
            -1,                         # -1 for base link
            lateralFriction=0.7,        # Fricción estándar del suelo 0.6
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
        if self.step_count % (self.frequency_simulation//10) == 0:  # Cada segundos aprox
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
    
    def _apply_automatic_knee_control(self, base_torques):
        """Control automático de la rodilla levantada basado en altura"""
        
        # Determinar qué pierna está levantada basado en contactos
        left_contact  = self.contact_with_force(link_id=self.left_foot_link_id,  stable_foot=True,  min_F= 20.0)
        right_contact = self.contact_with_force(link_id=self.right_foot_link_id,  stable_foot=True,  min_F= 20.0)

        
        
        if left_contact and not right_contact:
            # Pierna derecha levantada - controlar rodilla derecha (índice 3)
            knee_joint_id = self.dict_joints["right_knee_joint"] #self.joint_indices[controlled_knee_idx]   6     # right_knee_joint en PyBullet
            controlled_knee_idx = self.joint_indices.index(knee_joint_id)  # right_knee en joint_torques ultimo valor
        elif right_contact and not left_contact:
            # Pierna izquierda levantada - controlar rodilla izquierda (índice 1)
            knee_joint_id = self.dict_joints["left_knee_joint"] #self.joint_indices[controlled_knee_idx]    2    # left_knee_joint en PyBullet
            controlled_knee_idx = self.joint_indices.index(knee_joint_id)  # left_knee en joint_torques  tercero
        else:
            # Ambas o ninguna - no aplicar control automático
            return base_torques
        # self.target_knee_height
        # Obtener altura actual de la rodilla
        #knee link state y posición
        knee_state = p.getLinkState(self.robot_id, knee_joint_id)
        current_knee_height = knee_state[0][2]
        
        # Control PD simple hacia altura objetivo
        height_error = self.target_knee_height - current_knee_height
        # knee_joint_state y velocidad
        knee_velocity = p.getJointState(self.robot_id, knee_joint_id)[1]
        
        # Torque de control automático
        kp = self.KP  # Ganancia proporcional para h
        kd = self.KD   # Ganancia derivativa de h
        
        control_torque = kp * height_error - kd * knee_velocity
        
        # Combinar con torque base (PAM) usando peso
        base_torques[controlled_knee_idx] = (
            0.4 * base_torques[controlled_knee_idx] +  # 40% PAM
            0.6 * control_torque                        # 60% control automático
        )

        # Limitar torque final
        max_knee_torque = self.MAX_REASONABLE_TORQUE
        base_torques[controlled_knee_idx] = np.clip(
            base_torques[controlled_knee_idx], 
            -max_knee_torque, 
            max_knee_torque
        )
        
        return base_torques

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
        joint_torques = calculate_robot_specific_joint_torques_16_pam(self, pam_pressures)
        joint_torques = self._apply_automatic_knee_control(joint_torques)

        # base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)
        # roll = self.euler[0]
        # roll_rate = base_ang_vel[0]

        # KPR = 120.0    # ganancia P (sube/baja según necesidad)
        # KDR = 8.0      # ganancia D

        # tau_roll = -KPR * roll - KDR * roll_rate  # torque correctivo: empuja hacia roll=0

        # # Índices de caderas-roll en joint_torques (3D): [LHR, LHP, LK, LA, RHR, RHP, RK, RA]
        # i_LHR = 0
        # i_RHR = 4

        # # Aplica par opuesto a cada lado para recentrar la pelvis
        # joint_torques[i_LHR] += tau_roll
        # joint_torques[i_RHR] -= tau_roll

        balance_info = self.current_balance_status
        if self.step_count%(self.frequency_simulation//10)==0:
            if self.logger:
                self.logger.log("main",f"Pierna de apoyo: {balance_info['support_leg']}")
                self.logger.log("main",f"Tiempo en equilibrio: {balance_info['balance_time']} steps")

        return joint_torques
    
    def _get_single_leg_observation(self):
        """
        Observación específica para equilibrio en una pierna.
        Reemplaza _get_simple_observation con información más relevante.
        """
        
        obs = []
        
        # ===== ESTADO DEL TORSO (8 elementos) =====
        self.init_pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        init_lin_vel, init_ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Posición y orientación  
        obs.extend([self.init_pos[0],self.init_pos[1], self.init_pos[2], euler[0], euler[1]])  # x, z, roll, pitch
        
        # Velocidades
        obs.extend([init_lin_vel[0], self.init_pos[1], init_lin_vel[2], init_ang_vel[0], init_ang_vel[1]])  # vx, vz, wx, wy
        
        # ===== ESTADOS ARTICULARES (4 elementos) =====
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)  # Solo joints activos
        joint_positions = [state[0] for state in joint_states]
        obs.extend(joint_positions)
        
        # ===== INFORMACIÓN DE CONTACTO Y ALTURA DE RODILLAS (4 elementos) =====
        
        # Aquí sólo queremos saber si "hay algún contacto" → estable=False
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id, stable_foot=False, min_F=20.0)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id, stable_foot=False, min_F=20.0)
        obs.extend([float(left_contact), float(right_contact)])
        
        # Alturas de rodillas
        left_knee_state = p.getLinkState(self.robot_id, self.dict_joints["left_knee_joint"])
        right_knee_state = p.getLinkState(self.robot_id, self.dict_joints["right_knee_joint"])
        obs.extend([left_knee_state[0][2], right_knee_state[0][2]])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_simple_observation(self):
        """
        Observación SIMPLIFICADA - Solo 16 elementos esenciales
        
        ELIMINADO:
        - Estados de resortes pasivos (4 elementos)
        - ZMP history complejo (4 elementos)
        - Observation history deque
        - Métricas biomecánicas avanzadas
        """
        
        obs = []
        
        # ===== ESTADO DEL TORSO (8 elementos) =====
        
        self.pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        self.euler = p.getEulerFromQuaternion(orn)
        
        # Posición y orientación
        obs.extend([self.pos[0], self.pos[1], self.pos[2], self.euler[0], self.euler[1]])  # x, z, roll, pitch
        
        # Velocidades
        obs.extend([lin_vel[0], self.pos[1], lin_vel[2], ang_vel[0], ang_vel[1]])  # vx, vz, wx, wy
        
        # ===== ESTADOS ARTICULARES (4 elementos) =====
        
        joint_states = self.joint_states_properties
        joint_positions = [state[0] for state in joint_states]
        obs.extend(joint_positions)
        
        # ===== ZMP BÁSICO (2 elementos) =====
        
        if self.zmp_calculator:
            try:
                zmp_point = self.zmp_calculator.calculate_zmp()
                obs.extend([zmp_point[0], zmp_point[1]])
            except:
                obs.extend([0.0, 0.0])
        else:
            obs.extend([0.0, 0.0])
        
        # ===== CONTACTOS DE PIES (2 elementos) =====
        left_contact, right_contact=self.contacto_pies   

        #print ("contacto pies", left_contact, right_contact)
        obs.extend([float(left_contact), float(right_contact)])
        
        return np.array(obs, dtype=np.float32)
    
    @property
    def contacto_pies(self):
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id, stable_foot=False)
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
        # robot_joint_info=self.robot_data._get_joint_info
        # self.num_joints=p.getNumJoints(self.robot_id)
        # all_indices, all_names=[],[]
        # for j in range(self.num_joints):
        #     all_indices.append(robot_joint_info[j]['index'])
        #     all_names.append(robot_joint_info[j]['name'])
        #     p.enableJointForceTorqueSensor(self.robot_id, jointIndex=robot_joint_info[j]['index'], enableSensor=True)
        # self.dict_joints = {name: idx for name, idx in zip(all_names, all_indices)}
        # # Fijar los 8 índices en el orden esperado por el cálculo de torques
        # self.joint_names   = list(self.control_joint_names)
        # self.joint_indices = [self.dict_joints[n] for n in self.control_joint_names]
        
            
        
        # ===== SISTEMAS ESPECÍFICOS PARA EQUILIBRIO EN UNA PIERNA =====
        # Sistemas de recompensas
        if self.simple_reward_system is None:
            self.simple_reward_system = SimpleProgressiveReward(self)
        else:
            # solo re-vincula IDs si cambiaron, sin perder contadores/racha
            self.simple_reward_system.env = self
            self.simple_reward_system.robot_id = self.robot_id
            self.simple_reward_system.fixed_target_leg = self.fixed_target_leg
        # ===== CONFIGURACIÓN ARTICULAR INICIAL =====
        
        # Posiciones iniciales para equilibrio en una pierna (ligeramente asimétricas)
        initial_positions = {
            # Pierna izquierda
            self.joint_indices[0]: 0.0,   #   self.joint_indices[0] 'left_hip_roll_joint'
            self.joint_indices[1]: 0.0,   # left_hip_pitch_joint
            self.joint_indices[2]: 0.0,     # left_knee_joint
            self.joint_indices[3]: 0.0,     # left_ankle_joint
            # pierna derecha
            self.joint_indices[4]: 0.0,   # right_hip_roll_joint
            self.joint_indices[5]: 0.0,   # right_hip_pitch_joint
            self.joint_indices[6]: 0.0,     # right_knee_joint
            self.joint_indices[7]: 0.0     # right_ankle_joint
        }
        #if self.fixed_target_leg == 'left':
            #initial_positions[self.joint_indices[4]] += +0.03  # right_hip_roll_joint: inclina pelvis hacia la derecha
            #initial_positions[self.joint_indices[6]] += +0.05  # right_knee_joint: ligera flexión para absorber carga
            # Opcional: elevar un poco la cadera izquierda en pitch para facilitar clearance inicial
            #initial_positions[self.joint_indices[1]] += +0.03  # left_hip_pitch_joint
        
        for joint_id, pos in initial_positions.items():
            p.resetJointState(self.robot_id, joint_id, pos)
            p.setJointMotorControl2(self.robot_id, joint_id, p.VELOCITY_CONTROL, force=0)
        
        # ===== CONFIGURACIÓN DE DATOS Y CALCULADORES =====
        
        
        
        # ZMP calculator (todavía útil para métricas)
        self.zmp_calculator = ZMPCalculator(
            robot_id=self.robot_id,
            left_foot_id=self.left_foot_link_id,   # left_foot_link
            right_foot_id=self.right_foot_link_id,  # right_foot_link
            frequency_simulation=self.frequency_simulation,
            robot_data=self.robot_data,
            ground_id=self.plane_id
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
        for _ in range(int(self.frequency_simulation//10)):
            p.stepSimulation()
        
        # Obtener observación inicial
        observation = self._get_single_leg_observation()
        
        info = {
            'episode_reward': 0,
            'episode_length': 0,
            'target_task': 'single_leg_balance'
        }
        
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
            'target_knee_height': self.target_knee_height,
            'episode_step': self.step_count
        }
        
    def parametros_torque_pam(self):
        # Momentos de brazo calculados desde dimensiones reales
        self.HIP_ROLL_FLEXOR_BASE_ARM = 0.055      # 5.03cm - basado en circunferencia del muslo
        self.HIP_ROLL_FLEXOR_VARIATION = 0.008     # ±1.01cm variación por ángulo
        
        self.HIP_ROLL_EXTENSOR_BASE_ARM = 0.052    
        self.HIP_ROLL_EXTENSOR_VARIATION = 0.004 

        self.HIP_PITCH_FLEXOR_BASE_ARM = 0.045
        self.HIP_PITCH_FLEXOR_VARIATION = 0.0085#round(self.HIP_PITCH_FLEXOR_BASE_ARM/4.98, 4) 
        
        self.HIP_PITCH_EXTENSOR_BASE_ARM = 0.054#0.0628    
        self.HIP_PITCH_EXTENSOR_VARIATION = 0.007#round(self.HIP_PITCH_EXTENSOR_BASE_ARM/4.98, 4) 

        self.KNEE_FLEXOR_BASE_ARM = 0.0566     
        self.KNEE_FLEXOR_VARIATION = 0.010#round(self.KNEE_FLEXOR_BASE_ARM/5, 4)    

        self.KNEE_EXTENSOR_BASE_ARM = 0.0620#0.0640     
        self.KNEE_EXTENSOR_VARIATION = 0.008#round(self.KNEE_EXTENSOR_BASE_ARM/ 5, 4)

        self.ankle_FLEXOR_BASE_ARM = 0.05     
        self.ankle_FLEXOR_VARIATION = 0.0105#round(self.ankle_FLEXOR_BASE_ARM/4.2, 4)    

        self.ankle_EXTENSOR_BASE_ARM = 0.054#0.055     
        self.ankle_EXTENSOR_VARIATION = 0.0085#round(self.ankle_EXTENSOR_BASE_ARM/ 4.2, 4)

        self.KP = 80.0   # Ganancia proporcional
        self.KD = 12.0   # Ganancia derivativa    
        
        # Parámetros de resortes pasivos (calculados desde momento gravitacional)
        self.PASSIVE_SPRING_STRENGTH = 180.5   # N⋅m 
        self.DAMPING_COEFFICIENT = 12.0        # N⋅m⋅s/rad (optimizado para masa real)
        
        # Control antagónico
        self.INHIBITION_FACTOR = 0.3           # 30% inhibición recíproca
        self.MAX_CONTRACTION_RATIO = 0.25      # 25% contracción máxima segura
        self.VELOCITY_DAMPING_FACTOR = 0.08    # 8% reducción por velocidad
        
        # Límites de seguridad (basados en fuerzas PAM reales calculadas)
        self.MAX_REASONABLE_TORQUE = 240.0     # N⋅m (factor de seguridad incluido)

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
    
    def ankle_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle + np.pi/6)
        return self.ankle_FLEXOR_BASE_ARM + self.ankle_FLEXOR_VARIATION * angle_factor
    
    def ankle_extensor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle - np.pi/6)
        return self.ankle_EXTENSOR_BASE_ARM + self.ankle_EXTENSOR_VARIATION * angle_factor

    # ===== MÉTODO DE DEBUG ADICIONAL =====

    def _debug_joint_angles_and_pressures(self, info, pam_pressures, done):
        """
            ✅ MÉTODO DE DEBUG para verificar la lógica biomecánica
        
            Llama esto ocasionalmente durante el step() para verificar que la lógica funciona
        """
        
        if self.step_count % (self.frequency_simulation//10) == 0 or done:  # Cada segundo aprox
            try:
                if self.csvlog:
                    row_general={"step": int(self.step_count),
                                "episode": int(self.n_episodes),
                                "t": round(self.step_count / self.frequency_simulation, 5),}
                    for (name, idx), state in zip(self.dict_joints.items(), self.joint_states_properties):
                        pos, vel, reaction, applied = state
                        Fx,Fy,Fz,Mx,My,Mz = reaction
                        row_general[f"q_{name}"]=round(pos,3)
                        row_general[f"vel_{name}"]=round(vel,3)
                        row_general[f"τ_reaction_{name}_x"]=round(Mx,2)
                        row_general[f"τ_reaction_{name}_y"]=round(My,2)
                        row_general[f"τ_reaction_{name}_z"]=round(Mz,2)
                        row_general[f"Forces_{name}_x"]=round(Fx,3)
                        row_general[f"Forces_{name}_y"]=round(Fy,3)
                        row_general[f"Forces_{name}_z"]=round(Fz,3)
                        row_general[f"Pressure_{name}flexion"]=pam_pressures[idx*2]
                        row_general[f"Pressure_{name}extension"]=pam_pressures[idx*2+1]
                    row_general[f"COM_x"]=round(info["kpi"]['com_x'],3)
                    row_general[f"COM_y"]=round(info["kpi"]['com_y'],3)
                    row_general[f"COM_z"]=round(info["kpi"]['com_z'],3)
                    row_general[f"ZMP_x"]=round(info["kpi"]['zmp_x'],3)
                    row_general[f"ZMP_y"]=round(info["kpi"]['zmp_y'],3)
                    #row_general[f"ZMP_dist_to_COM"]=round(info["kpi"]['zmp_dist_to_com'],3)
                    self.csvlog.write("general_values", row_general)

            except Exception as e:
                print(f"   ❌ Debug error: {e}")

    def debug_rewards(self):
        if self.step_count % 10==0 and len(self.reawrd_step)>0:
            if self.csvlog:
                row_rewards={
                    "step":self.step_count,
                    "episode":self.n_episodes,
                    "t": round(self.step_count / self.frequency_simulation, 5),
                }
                for reward_name, reward_value in self.reawrd_step.items():
                    row_rewards[reward_name]=reward_value
                
                self.csvlog.write("rewards", row_rewards)
                

            

    # Validación de robot:
    def validate_robot_specific_behavior(self, pam_pressures, joint_torques):
        """
        Validación específica para tu robot de 25kg y 1.20m altura
        """
        
        warnings = []
        
        # ===== VALIDAR CO-CONTRACCIÓN EXCESIVA =====
        
        # Caderas: detectar activación simultánea alta
        if pam_pressures[0] > 0.7 and pam_pressures[1] > 0.7:
            cocontraction_level = (pam_pressures[0] + pam_pressures[1]) / 2
            warnings.append(f"Left hip co-contraction: {cocontraction_level:.1%}")
        
        if pam_pressures[2] > 0.7 and pam_pressures[3] > 0.7:
            cocontraction_level = (pam_pressures[2] + pam_pressures[3]) / 2
            warnings.append(f"Right hip co-contraction: {cocontraction_level:.1%}")
            # Rodillas: 
        if pam_pressures[4] > 0.7 and pam_pressures[5] > 0.7:
            cocontraction_level = (pam_pressures[4] + pam_pressures[5]) / 2
            warnings.append(f"Left Knee co-contraction: {cocontraction_level:.1%}")
        if pam_pressures[6] > 0.7 and pam_pressures[7] > 0.7:
            cocontraction_level = (pam_pressures[6] + pam_pressures[7]) / 2
            warnings.append(f"Right Knee co-contraction: {cocontraction_level:.1%}")
        if pam_pressures[8] > 0.7 and pam_pressures[9] > 0.7:
            cocontraction_level = (pam_pressures[8] + pam_pressures[9]) / 2
            warnings.append(f"Left Knee co-contraction: {cocontraction_level:.1%}")
        if pam_pressures[10] > 0.7 and pam_pressures[11] > 0.7:
            cocontraction_level = (pam_pressures[10] + pam_pressures[11]) / 2
            warnings.append(f"Right Knee co-contraction: {cocontraction_level:.1%}")
        if pam_pressures[12] > 0.7 and pam_pressures[13] > 0.7:
            cocontraction_level = (pam_pressures[12] + pam_pressures[13]) / 2
            warnings.append(f"Left Knee co-contraction: {cocontraction_level:.1%}")
        if pam_pressures[14] > 0.7 and pam_pressures[15] > 0.7:
            cocontraction_level = (pam_pressures[14] + pam_pressures[15]) / 2
            warnings.append(f"Right Knee co-contraction: {cocontraction_level:.1%}")
        
        
        # ===== VALIDAR TORQUES DENTRO DE CAPACIDAD FÍSICA =====
        
        # Para el robot específico:
        for i, torque in enumerate(joint_torques):
            if abs(torque) > self.MAX_REASONABLE_TORQUE*0.6:  # Warning de uso excesivo de torques
                warnings.append(f"{self.joint_names[i]}: High torque {torque:.1f} N⋅m")
        
        # ===== VALIDAR EFICIENCIA ENERGÉTICA =====
        
        # Para tu robot de 25kg, activación total >4.0 es ineficiente
        total_activation = np.sum(pam_pressures)
        n_pams = self.num_active_pams
        if total_activation > n_pams * 2.0/3.0:  # >75% activación total
            efficiency = (n_pams - total_activation) / max(n_pams,1) * 100.0  # % aprox
            warnings.append(f"Energy efficiency: {efficiency:.1f}% (high activation)")
        
        # ===== VALIDAR ESTABILIDAD BIOMECÁNICA =====
        
        # Para equilibrio en una pierna, verificar asimetría apropiada
        left_activation = np.sum(pam_pressures[0:2]) + pam_pressures[4] + \
        (pam_pressures[self.muscle_names.index('left_knee_extensor')])  # Cadera izq + rodilla izq
        right_activation = np.sum(pam_pressures[2:4]) + pam_pressures[5] + \
        (pam_pressures[self.muscle_names.index('right_knee_extensor')])  # Cadera der + rodilla der
        
        asymmetry = abs(left_activation - right_activation)
        if asymmetry < 0.5:  # Muy simétrico para equilibrio en una pierna
            warnings.append(f"Low asymmetry: {asymmetry:.2f} (may indicate poor single-leg balance)")
        else:
            if self.logger:
                self.logger.log("main","Correct_asimetry")
        
        # ===== LOGGING CONDICIONAL =====
        
        if warnings and self.step_count % (self.frequency_simulation//10) == 0:  # Cada 0.5 segundos aprox
            if self.logger:
                self.logger.log("main",f"🤖 Robot-specific validation (Step {self.step_count}):")
                for warning in warnings:
                    self.logger.log("main",f"   ⚠️ {warning}")
                
                # Info adicional útil
                self.logger.log("main",f"   Total mass: 25kg, Height: 1.20m")
                self.logger.log("main",f"   Current torques: τ_cmd={[f'{t:.1f}' for t in joint_torques]} N⋅m")
        
        return len(warnings) == 0
    
def configure_robot_specific_pam_system(env:Simple_Lift_Leg_BipedEnv):
    """
    Configurar el sistema PAM específicamente para tu robot.
    Llamar una vez después de crear el entorno.
    """
    
    # Verificar que las dimensiones coinciden
    expected_mass = 24.1  # kg
    expected_height = 1.20  # m
    if env.logger:
        env.logger.log("main","🤖 Configuring PAM system for your specific robot:")
        env.logger.log("main",f"   Expected mass: {expected_mass}kg")
        env.logger.log("main",f"   Expected height: {expected_height}m")
        env.logger.log("main",f"   PAM configuration: {env.num_active_pams} muscles "
                f"(hips antagonistic + knees {'flex+ext'})")
        env.logger.log("main",f"   Moment arms: Hip 5.0-6.3cm, Knee 5.7cm")
        env.logger.log("main",f"   Passive springs: {env.PASSIVE_SPRING_STRENGTH} N⋅m (gravity-compensated)")
    
    # Configurar parámetros específicos en el entorno
    env.robot_specific_configured = True
    env.expected_robot_mass = expected_mass
    env.expected_robot_height = expected_height
    
    # Reemplazar el método de cálculo de torques
    env._calculate_basic_joint_torques = env._calculate_robot_specific_joint_torques_16_pam
    if env.logger:
        env.logger.log("main","✅ Robot-specific PAM system configured!")
    
    return True




