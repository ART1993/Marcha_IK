
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
from collections import deque

from Archivos_Apoyo.Configuraciones_adicionales import PAM_McKibben, \
                                                    calculate_robot_specific_joint_torques_8_pam
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data
from Archivos_Apoyo.simple_log_redirect import log_print, both_print

from Archivos_Mejorados.RewardSystemSimple import SimpleProgressiveReward
from Archivos_Mejorados.SingleLegActionSelector import SingleLegActionSelector
from Archivos_Mejorados.AngleBasedExpertController import AngleBasedExpertController           

class Simple_Lift_Leg_BipedEnv(gym.Env):
    """
        Versión expandida con 6 PAMs activos + elementos pasivos
        - 4 PAMs antagónicos en caderas (flexor/extensor bilateral)  
        - 2 PAMs flexores en rodillas + resortes extensores pasivos
        - Resortes pasivos en tobillos para estabilización
        Indices de robot bípedo pam:
            - left hip joint: 0
            - left knee joint: 1
            - left ankle joint: 2
            - right hip joint: 3
            - right knee joint: 4
            - right anckle joint: 5
    """
    
    def __init__(self, render_mode='human', testeo_movimiento=False,
                 enable_curriculum=True, print_env="ENV", probe_expert_only=False):
        
        """
            Inicio el entorno de entrenamiento PAM
        """
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super(Simple_Lift_Leg_BipedEnv, self).__init__()

        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.render_mode = render_mode
        
        # self.action_space_type = action_space  # Solo "pam"
        self.probe_expert_only = probe_expert_only
        self.testeo_movimiento=testeo_movimiento

        self.muscle_names = ['left_hip_flexor', 'left_hip_extensor', 'right_hip_flexor', 
                            'right_hip_extensor', 'left_knee_flexor','left_knee_extensor', 
                            'right_knee_flexor','right_knee_extensor', 'left_anckle_flexor',
                            'left_anckle_extensor', 'right_anckle_flexor', 'right_anckle_extensor']
        
        self.num_active_pams = len(self.muscle_names)

        self.pam_control_active = False
        self.contact_established = False
        self.contact_stable_steps = 0
        # Para tracking de tiempo en balance
        self._balance_start_time = 0
        # ===== CONFIGURACIÓN FÍSICA BÁSICA =====
        
        self.urdf_path = "2_legged_human_like_robot.urdf"
        self.frequency_simulation=400.0
        self.switch_interval=2000  # Intervalo para cambiar pierna objetivo en curriculum
        self.time_step = 1.0 / self.frequency_simulation
        # ===== CONFIGURACIÓN PAM SIMPLIFICADA =====
        

        self.pam_muscles = PAM_McKibben()
        
        # Estados PAM básicos
        self.pam_states = {
            'pressures': np.zeros(self.num_active_pams),
            'forces': np.zeros(self.num_active_pams)
        }
        
        # ===== CONFIGURACIÓN DE ESPACIOS =====
        self.recent_rewards=deque(maxlen=50)
        # Action space: 6 presiones PAM normalizadas [0, 1]
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_active_pams,), 
            dtype=np.float32
        )
        
        # Observation space SIMPLIFICADO: 16 elementos total
        # - 8: Estado del torso (pos, orient, velocidades)
        # - 4: Estados articulares básicos (posiciones)
        # - 2: ZMP básico (x, y)
        # - 2: Contactos de pies (izq, der)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,), # Ver si es 18 o 16
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

        self.step_count = 0
        self.total_reward = 0
        self.robot_id = None
        self.plane_id = None
        self.joint_indices =  list(range(6))#[0, 1, 2, 3, 4, 5]  # left_hip, left_knee, right_hip, right_knee
        self.joint_names = ['left_hip', 'left_knee', 'left_anckle', 'right_hip', 'right_knee', 'right_anckle']
        self.dict_joints= {joint_name:joint_id for joint_name, joint_id in zip(self.joint_names, self.joint_indices)}
        self.left_foot_link_id = 2
        self.right_foot_link_id = 5

        # Añadir tracking de pierna levantada
        self.raised_leg = 'left'  # 'left' o 'right' - cuál pierna está levantada
        self.target_knee_height = 0.8  # Altura objetivo de la rodilla levantada
        self.episode_reward = 0
        #Parámetros constantes que se usan en el calculo de torques
        self.parametros_torque_pam()

        self.enable_curriculum = enable_curriculum
        self.simple_reward_system = None
        self.print_env = print_env
        log_print(f"🤖 Simplified Lift legs Environment initialized")
        log_print(f"🤖 Environment initialized - Systems initiate in reset")
        log_print(f"🤖 Using {self.num_active_pams=:} "
                  f"{enable_curriculum=:} [{self.print_env=:}]")
    
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
        # ===== DECISIÓN: EXPERTO vs RL =====
        # En env.step (o donde construyas la acción final)
        self.pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        self.euler = p.getEulerFromQuaternion(orn)
        if self.enable_curriculum:
            u_expert = self.action_selector.get_expert_action()            # [0,1]^6
            u_rl = np.clip(action, 0.0, 1.0) 
            
            if self.probe_expert_only:                    # [0,1]^6
                u_final = u_expert
                self.ep_expert_weight += 1.0
            else:
                assist = self.action_selector.expert_help_ratio                # 0.85→0.0
                tilt_boost = 0.0
                current_tilt = abs(self.euler[0]) + abs(self.euler[1])
                if current_tilt > 0.20: tilt_boost = 0.30
                elif current_tilt > 0.15: tilt_boost = 0.15
                assist = float(np.clip(assist + tilt_boost, 0.0, 0.95))
                
                u_final = assist * u_expert + (1.0 - assist) * u_rl
                self.ep_expert_weight += float(np.clip(assist, 0.0, 1.0))
        else:
            u_final = np.clip(action, 0.0, 1.0)

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
        torque_mapping = [(joint, joint_torques[i]) for i, joint in enumerate(self.joint_indices)]
        
        #self.last_tau_cmd = {jid: float(tau) for jid, tau in torque_mapping}
        for joint_id, torque in torque_mapping:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )

        p.stepSimulation()

        # ✅ LLAMAR DEBUG OCASIONALMENTE
        self._debug_joint_angles_and_pressures(u_final)

        
        #if self.simple_reward_system:
        reward = self.simple_reward_system.calculate_reward(u_final, self.step_count)
        done = self.simple_reward_system.is_episode_done(self.step_count, self.testeo_movimiento)
        system_used = "PROGRESSIVE"
        # ===== CÁLCULO DE RECOMPENSAS CONSCIENTE DEL CONTEXTO =====
       
        
        # ===== PASO 4: OBSERVACIÓN Y TERMINACIÓN =====
        self.episode_reward += reward
        
        observation = self._get_simple_observation()

        # ===== APLICAR ACCIÓN PAM =====
        self.action_selector.update_after_step(reward)
        
        # Info simplificado
        info = {
            'step_count': self.step_count,
            'reward': reward,
            #'action_source': action_source,
            'episode_reward': self.episode_reward
        }

        if self.simple_reward_system:
            curriculum_info = self.simple_reward_system.get_info()  # Solo una llamada
            info['curriculum'] = curriculum_info  # Añadir sin reemplazar
            info['system_type'] = 'progressive'
            info['current_level'] = curriculum_info.get('level', 1)
            # Debug simple
            if done:
                expert_pct = 100.0 * self.ep_expert_weight  / max(1, self.ep_total_actions)
                log_print(f"🎯 Expert usage this episode: {expert_pct:.1f}% "
                        f"({self.ep_expert_weight }/{self.ep_total_actions})")
                info['expert_usage_pct'] = expert_pct
                episode_total = info['episode_reward']  # Ya calculado arriba
                self.simple_reward_system.update_after_episode(episode_total)
                log_print(f"📈 Episode {info['curriculum']['episodes']} | Level {info['curriculum']['level']} | Reward: {episode_total:.1f}")
        
        # CONSERVAR tu debug existente 
        if self.step_count % (self.frequency_simulation//10) == 0 or done:
            log_print(f"🔍 Step {self.step_count} - Control Analysis:")
            log_print(f"   Height: {self.pos[2]:.2f}m")
            log_print(f"   Tilt: Roll {math.degrees(self.euler[0]):.1f}°, Pitch {math.degrees(self.euler[1]):.1f}°")
            #log_print(f"   Action source: {action_source}")
            
            if self.simple_reward_system:
                curriculum_info = self.simple_reward_system.get_info()
                log_print(f"   Level: {info['curriculum'].get('level')}, Target: {self.action_selector.current_action}")
    
            # Verificar si está cerca de límites
            max_allowed_tilt = 0.4 if self.simple_reward_system and self.simple_reward_system.level == 1 else 0.3
            if abs(self.euler[0]) > max_allowed_tilt * 0.8 or abs(self.euler[1]) > max_allowed_tilt * 0.8:
                log_print(f"   ⚠️ Approaching tilt limit! Max allowed: ±{math.degrees(max_allowed_tilt):.1f}°")
            

        # DEBUG TEMPORAL: Verificar timing cada cierto número de steps
        if self.step_count % (self.frequency_simulation//10) == 0 and self.simple_reward_system:  # Cada 5 segundos aprox
            status = self.simple_reward_system.get_info()
            elapsed_time = self.step_count / self.frequency_simulation
            #log_print(f" {action_source} action, reward={reward:.2f}")
            log_print(f"Step {done=:}, is_valid={is_valid}")
            log_print(f"🎮 Active system: {system_used} at step {self.step_count}")
            log_print(f"🕒 Step {self.step_count} ({elapsed_time:.1f}s elapsed):")
            log_print(f"   Current level: {status['level']}")
            log_print(f"   Target leg: {status.get('target_leg', 'N/A')}")
            log_print(f"   Switch timer: {self.simple_reward_system.switch_timer}/{self.simple_reward_system.switch_interval}")
        
        return observation, reward, done, False, info
    

    def _configure_contact_friction(self):
        """
        Configurar propiedades de fricción dinámicamente en PyBullet
        (Complementa las propiedades del URDF)
        """
        
        # ===== FRICCIÓN ESPECÍFICA PARA PIES =====
        
        # Pie izquierdo - alta fricción para agarre
        for foot_id in (self.left_foot_link_id, self.right_foot_link_id):
            p.changeDynamics(
                self.robot_id, 
                foot_id,
                lateralFriction=0.8,        # Reducido de 1.2 a 0.8
                spinningFriction=0.15,       # Reducido de 0.8 a 0.15
                rollingFriction=0.01,       # Reducido de 0.1 a 0.01
                restitution=0.01,           # Reducido de 0.05 a 0.01 (menos rebote)
                contactDamping=100,         # Aumentado de 50 a 100 (más amortiguación)
                contactStiffness=15000,      # Aumentado de 10000 a 15000 (más rigidez)
                frictionAnchor=1
            )

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
        
        # ===== FRICCIÓN DEL SUELO =====
        
        # Configurar fricción del plano del suelo
        p.changeDynamics(
            self.plane_id,
            -1,                         # -1 for base link
            lateralFriction=0.6,        # Fricción estándar del suelo
            spinningFriction=0.2,
            rollingFriction=0.005
        )
        
    def contact_with_force(self, link_id, min_F=20.0):
        cps = p.getContactPoints(self.robot_id, self.plane_id, link_id, -1)# -1 para el suelo
        if not cps: 
            return False
        # campo normalForce = índice 9 en PyBullet
        totalF = sum(cp[9] for cp in cps)
        if self.step_count % (self.frequency_simulation//10) == 0:  # Cada segundos aprox
            log_print(f"Contact force on link {link_id}: {totalF:.2f} N")
        return totalF > min_F
    
    def contact_normal_force(self, link_id:int)->float:
        cps = p.getContactPoints(self.robot_id, self.plane_id, link_id, -1)
        return 0.0 if not cps else sum(cp[9] for cp in cps)
    
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
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id)

        
        
        if left_contact and not right_contact:
            # Pierna derecha levantada - controlar rodilla derecha (índice 3)
            controlled_knee_idx = 3  # right_knee en joint_torques
            knee_joint_id = 4        # right_knee_joint en PyBullet
        elif right_contact and not left_contact:
            # Pierna izquierda levantada - controlar rodilla izquierda (índice 1)
            controlled_knee_idx = 1  # left_knee en joint_torques  
            knee_joint_id = 1        # left_knee_joint en PyBullet
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
            
            Mapeo: 6 PAMs -> 4 articulaciones
            - PAM 0,1: cadera izquierda (flexor, extensor)
            - PAM 2,3: cadera derecha (flexor, extensor)  
            - PAM 4: rodilla izquierda (flexor)
            - PAM 5: rodilla derecha (flexor)
            # MAPEO CLARO: PAM → Joint
            # joint_states[0] = left_hip (joint 0)
            # joint_states[1] = left_knee (joint 1) 
            # joint_states[2] = right_hip (joint 3)
            # joint_states[3] = right_knee (joint 4)
        """
       
        # NUEVA LÓGICA: Control automático de rodilla levantada
        joint_torques = calculate_robot_specific_joint_torques_8_pam(self, pam_pressures)
        

        balance_info = self.current_balance_status
        if self.step_count%(self.frequency_simulation//10)==0:

            log_print(f"Pierna de apoyo: {balance_info['support_leg']}")
            log_print(f"Tiempo en equilibrio: {balance_info['balance_time']} steps")

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
        obs.extend([self.init_pos[0], self.init_pos[2], euler[0], euler[1]])  # x, z, roll, pitch
        
        # Velocidades
        obs.extend([init_lin_vel[0], init_lin_vel[2], init_ang_vel[0], init_ang_vel[1]])  # vx, vz, wx, wy
        
        # ===== ESTADOS ARTICULARES (4 elementos) =====
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)  # Solo joints activos
        joint_positions = [state[0] for state in joint_states]
        obs.extend(joint_positions)
        
        # ===== INFORMACIÓN DE CONTACTO Y ALTURA DE RODILLAS (4 elementos) =====
        
        # Contactos
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id)
        obs.extend([float(left_contact), float(right_contact)])
        
        # Alturas de rodillas
        left_knee_state = p.getLinkState(self.robot_id, 1)
        right_knee_state = p.getLinkState(self.robot_id, 4)
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
        obs.extend([self.pos[0], self.pos[2], self.euler[0], self.euler[1]])  # x, z, roll, pitch
        
        # Velocidades
        obs.extend([lin_vel[0], lin_vel[2], ang_vel[0], ang_vel[1]])  # vx, vz, wx, wy
        
        # ===== ESTADOS ARTICULARES (4 elementos) =====
        
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
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
        left_contact, right_contact=self.contacto_pies_log   

        #print ("contacto pies", left_contact, right_contact)
        obs.extend([float(left_contact), float(right_contact)])
        
        return np.array(obs, dtype=np.float32)
    
    @property
    def contacto_pies_log(self):
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id)
        if self.step_count<=150 and self.step_count%(self.frequency_simulation//10)==0:
            log_print(f"Contactos pie izquierdo: {left_contact}")
            log_print(f"Contactos pie derecho: {right_contact}")
        return left_contact, right_contact
    
    @property
    def contacto_pies(self):
        left_contact=self.contact_with_force(link_id=self.left_foot_link_id)
        right_contact=self.contact_with_force(link_id=self.right_foot_link_id)
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
        
        # Actualizar reward system y action selector del episodio anterior si existen
        if self.action_selector is not None and hasattr(self, 'episode_reward'):
            self.action_selector.on_episode_end(self.episode_reward)

        
        # ===== RESET FÍSICO =====
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Configurar solver para estabilidad
        p.setPhysicsEngineParameter(
            numSolverIterations=20,
            numSubSteps=6,
            contactBreakingThreshold=0.0005,
            erp=0.9,
            contactERP=0.95,
            frictionERP=0.9,
            enableConeFriction=1,        # Habilitar fricción cónica
            deterministicOverlappingPairs=1
        )
        log_print(f"🔧 Contact friction CORRECTED for single leg balance:")
        log_print(f"   Feet: μ=0.8 (moderate grip, less spinning)")
        log_print(f"   Legs: μ=0.1 (very low resistance)")
        log_print(f"   Ground: μ=0.6 (controlled)")
        log_print(f"   Solver: Enhanced stability parameters")
        
        # Cargar entorno
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            [0, 0, 1.21],  # Altura inicial ligeramente mayor
            # useFixedBase=False,
            useFixedBase=False
        )
        for j in self.joint_indices:
            p.enableJointForceTorqueSensor(self.robot_id, jointIndex=j, enableSensor=True)
        
        # ===== SISTEMAS ESPECÍFICOS PARA EQUILIBRIO EN UNA PIERNA =====
        # Sistemas de recompensas
        if self.simple_reward_system is None:
            self.simple_reward_system = SimpleProgressiveReward(self)
        else:
            # solo re-vincula IDs si cambiaron, sin perder contadores/racha
            self.simple_reward_system.robot_id = self.robot_id
            self.simple_reward_system.plane_id = self.plane_id
            self.simple_reward_system.env=self
                 
        # Nuevo selector de acciones
        if self.action_selector is None:
            log_print("🎯 Initializing action_selector for single leg balance")
            self.action_selector = SingleLegActionSelector(self)
            self.angle_expert_controller = self.action_selector.angle_controller
        else:
            # Actualiza el env dentro del selector (por si cambian intervalos/refs)
            self.action_selector.env = self
            self.action_selector.target_switch_time = self.switch_interval
            self.action_selector.angle_controller = AngleBasedExpertController(self)
        self.angle_expert_controller = self.action_selector.angle_controller
        
        # ===== CONFIGURACIÓN ARTICULAR INICIAL =====
        
        # Posiciones iniciales para equilibrio en una pierna (ligeramente asimétricas)
        initial_positions = {
            0: -0.05,   # left_hip - ligera flexión
            1: 0.05,   # left_knee - extendida (pierna de soporte)
            3: -0.05,   # right_hip - más flexión
            4: 0.05,   # right_knee - flexionada (pierna levantada)
        }
        
        for joint_id, pos in initial_positions.items():
            p.resetJointState(self.robot_id, joint_id, pos)
            p.setJointMotorControl2(self.robot_id, joint_id, p.VELOCITY_CONTROL, force=0)
        
        # ===== CONFIGURACIÓN DE DATOS Y CALCULADORES =====
        
        self.robot_data = PyBullet_Robot_Data(self.robot_id)
        
        # ZMP calculator (todavía útil para métricas)
        self.zmp_calculator = ZMPCalculator(
            robot_id=self.robot_id,
            left_foot_id=2,   # left_foot_link
            right_foot_id=5,  # right_foot_link
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

        self.KP = 80.0   # Ganancia proporcional
        self.KD = 12.0   # Ganancia derivativa

        self.HIP_FLEXOR_BASE_ARM = 0.0503      # 5.03cm - basado en circunferencia del muslo
        self.HIP_FLEXOR_VARIATION = self.HIP_FLEXOR_BASE_ARM/4.98     # ±1.01cm variación por ángulo 
        
        self.HIP_EXTENSOR_BASE_ARM = 0.0628    
        self.HIP_EXTENSOR_VARIATION = self.HIP_EXTENSOR_BASE_ARM/ 4.98 #+-1.26cm   

        self.KNEE_FLEXOR_BASE_ARM = 0.0566     
        self.KNEE_FLEXOR_VARIATION = self.KNEE_FLEXOR_BASE_ARM/5 # +-1.13 cm    

        self.KNEE_EXTENSOR_BASE_ARM = 0.0640     
        self.KNEE_EXTENSOR_VARIATION = self.KNEE_EXTENSOR_BASE_ARM/5 # +-1.20 cm    

        self.ANCKLE_FLEXOR_BASE_ARM = 0.0610     
        self.ANCKLE_FLEXOR_VARIATION = self.ANCKLE_FLEXOR_BASE_ARM/5.2 # +-1.17 cm    

        self.ANCKLE_EXTENSOR_BASE_ARM = 0.0680     
        self.ANCKLE_EXTENSOR_VARIATION = self.ANCKLE_EXTENSOR_BASE_ARM/5.2 # +-1.30 cm 
        
        # Parámetros de resortes pasivos (calculados desde momento gravitacional)
        self.PASSIVE_SPRING_STRENGTH = 180.5   # N⋅m 
        self.DAMPING_COEFFICIENT = 12.0        # N⋅m⋅s/rad (optimizado para masa real)
        
        # Control antagónico
        self.INHIBITION_FACTOR = 0.3           # 30% inhibición recíproca
        self.MAX_CONTRACTION_RATIO = 0.25      # 25% contracción máxima segura
        self.VELOCITY_DAMPING_FACTOR = 0.08    # 8% reducción por velocidad
        
        # Límites de seguridad (basados en fuerzas PAM reales calculadas)
        self.MAX_REASONABLE_TORQUE = 240.0     # N⋅m (factor de seguridad incluido)

    def hip_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de cadera específico para tu robot.
        Basado en geometría real: circunferencia muslo = 0.503m
        """
        # Flexor más efectivo cuando cadera está extendida (ángulo negativo)
        angle_factor = np.cos(angle + np.pi/3)  # Desplazamiento para peak en extensión
        return self.HIP_FLEXOR_BASE_ARM + self.HIP_FLEXOR_VARIATION * angle_factor
    
    def hip_extensor_moment_arm(self, angle):
        """
        Momento de brazo del extensor de cadera (glúteos).
        Más efectivo en rango medio de flexión.
        """
        # Extensor más efectivo en flexión ligera-moderada
        angle_factor = np.cos(angle - np.pi/6)  # Peak en flexión ligera. Probar distintos angulos
        return self.HIP_EXTENSOR_BASE_ARM + self.HIP_EXTENSOR_VARIATION * angle_factor

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
    
    def anckle_flexor_moment_arm(self, angle):
        """
            Momento de brazo de flexión de talon
        """
        angle_factor = np.cos(angle + np.pi/6)
        return self.ANCKLE_FLEXOR_BASE_ARM + self.ANCKLE_FLEXOR_VARIATION*angle_factor
    
    def anckle_extensor_moment_arm(self, angle):
        """
            Momento de brazo de extensión de talon
        """
        angle_factor = np.cos(angle - np.pi/6)
        return self.ANCKLE_EXTENSOR_BASE_ARM + self.ANCKLE_EXTENSOR_VARIATION*angle_factor


    # ===== MÉTODO DE DEBUG ADICIONAL =====

    def _debug_joint_angles_and_pressures(self, pam_pressures):
        """
            ✅ MÉTODO DE DEBUG para verificar la lógica biomecánica
        
            Llama esto ocasionalmente durante el step() para verificar que la lógica funciona
        """
        
        if self.step_count % (self.frequency_simulation//10) == 0:  # Cada segundo aprox
            try:
                # Joint indices [0,1,3,4]
                joint_states = p.getJointStates(self.robot_id, self.joint_indices)  # rodillas
                left_hip_angle = joint_states[0][0]
                left_knee_angle = joint_states[1][0]
                right_hip_angle = joint_states[2][0]
                right_knee_angle = joint_states[3][0]
                for idx, state in zip(self.joint_indices, joint_states):
                    pos, vel, reaction, _ = state
                    Fx,Fy,Fz,Mx,My,Mz = reaction
                    both_print(f"Joint {idx}: q={pos:.3f}, vel=({vel:.3f}),τ_reaction=({Mx:.2f},{My:.2f},{Mz:.2f})," \
                               f"Forces=({Fx:.3f},{Fy:.3f},{Fz:.3f})") # , τ_motor={applied:.2f} es cero siempre por lo que no importa

                log_print(f"\n🔍 Biomechanical Debug (Step {self.step_count=:}):")
                log_print(f"   Left hip: {left_hip_angle:.3f} rad ({math.degrees(left_hip_angle):.1f}°)")
                log_print(f"   Right hip: {right_hip_angle:.3f} rad ({math.degrees(right_hip_angle):.1f}°)")
                log_print(f"   Left knee: {left_knee_angle:.3f} rad ({math.degrees(left_knee_angle):.1f}°)")
                log_print(f"   Right knee: {right_knee_angle:.3f} rad ({math.degrees(right_knee_angle):.1f}°)")
                log_print(f"   L Hip flex/ext: {pam_pressures[0]:.3f} / {pam_pressures[1]:.3f}")
                log_print(f"   R Hip flex/ext: {pam_pressures[2]:.3f} / {pam_pressures[3]:.3f}")
                log_print(f"   L knee flex/ext: {pam_pressures[4]:.3f} / {pam_pressures[5]:.3f}")
                log_print(f"   R knee flex/ext: {pam_pressures[6]:.3f} / {pam_pressures[7]:.3f}")
                #log_print(f"[XHIP] eL={eL:.3f} appL={appL} | eR={eR:.3f} appR={appR}")
            
            except Exception as e:
                print(f"   ❌ Debug error: {e}")

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
        
        # ===== LOGGING CONDICIONAL =====
        
        if warnings and self.step_count % (self.frequency_simulation//10) == 0:  # Cada 0.5 segundos aprox
            log_print(f"🤖 Robot-specific validation (Step {self.step_count}):")
            for warning in warnings:
                log_print(f"   ⚠️ {warning}")
            
            # Info adicional útil
            log_print(f"   Total mass: 25kg, Height: 1.20m")
            log_print(f"   Current torques: τ_cmd={[f'{t:.1f}' for t in joint_torques]} N⋅m")
        
        return len(warnings) == 0
    
def configure_robot_specific_pam_system(env):
    """
    Configurar el sistema PAM específicamente para tu robot.
    Llamar una vez después de crear el entorno.
    """
    
    # Verificar que las dimensiones coinciden
    expected_mass = 25.0  # kg
    expected_height = 1.20  # m
    
    log_print("🤖 Configuring PAM system for your specific robot:")
    log_print(f"   Expected mass: {expected_mass}kg")
    log_print(f"   Expected height: {expected_height}m")
    log_print(f"   PAM configuration: {env.num_active_pams} muscles "
              f"(hips antagonistic + knees {'flex+ext'})")
    log_print(f"   Moment arms: Hip 5.0-6.3cm, Knee 5.7cm")
    log_print(f"   Passive springs: {env.PASSIVE_SPRING_STRENGTH} N⋅m (gravity-compensated)")
    
    # Configurar parámetros específicos en el entorno
    env.robot_specific_configured = True
    env.expected_robot_mass = expected_mass
    env.expected_robot_height = expected_height
    
    # Reemplazar el método de cálculo de torques
    env._calculate_basic_joint_torques = env._calculate_robot_specific_joint_torques_8_pam
    
    log_print("✅ Robot-specific PAM system configured!")
    
    return True




