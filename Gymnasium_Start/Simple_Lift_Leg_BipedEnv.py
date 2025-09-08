
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
from collections import deque

from Archivos_Apoyo.Configuraciones_adicionales import PAM_McKibben
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data
from Archivos_Apoyo.simple_log_redirect import log_print, both_print

from Archivos_Mejorados.RewardSystemSimple import AngleBasedExpertController, \
                                                    SingleLegActionSelector, SimpleProgressiveReward           

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
    
    def __init__(self, render_mode='human', action_space="pam", enable_curriculum=True):
        
        """
            Inicio el entorno de entrenamiento PAM
        """
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super(Simple_Lift_Leg_BipedEnv, self).__init__()

        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.render_mode = render_mode
        self.action_space_type = action_space  # Solo "pam"
        self.muscle_names = ['left_hip_flexor', 'left_hip_extensor', 'right_hip_flexor', 
                        'right_hip_extensor', 'left_knee_flexor', 'right_knee_flexor']

        self.pam_control_active = False
        self.contact_established = False
        self.contact_stable_steps = 0
        # Para tracking de tiempo en balance
        self._balance_start_time = 0
        
        # ===== CONFIGURACIÓN FÍSICA BÁSICA =====
        
        self.urdf_path = "2_legged_human_like_robot.urdf"
        self.frecuency_simulation=400.0
        self.switch_interval=2000  # Intervalo para cambiar pierna objetivo en curriculum
        self.time_step = 1.0 / self.frecuency_simulation
        # ===== CONFIGURACIÓN PAM SIMPLIFICADA =====
        self.num_active_pams = 6

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
            shape=(16,),
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
        self.joint_indices = [0, 1, 3, 4]  # left_hip, left_knee, right_hip, right_knee
        self.left_foot_link_id = 2
        self.right_foot_link_id = 5

        # Añadir tracking de pierna levantada
        self.raised_leg = 'left'  # 'left' o 'right' - cuál pierna está levantada
        self.target_knee_height = 0.8  # Altura objetivo de la rodilla levantada
        self.episode_reward = 0
        #Parámetros constantes que se usan en el calculo de torques
        self.parametros_torque_pam()

        self.use_simple_progressive = enable_curriculum
        self.simple_reward_system = None
        
        log_print(f"🤖 Simplified Lift legs Environment initialized")
        log_print(f"🤖 Environment initialized - Systems initiate in reset")
    
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
        use_expert = False
    
        if self.action_selector is not None and self.action_selector.should_use_expert_action():
            use_expert = True

        # NUEVA LÓGICA: Más ayuda experta en situaciones críticas
        if self.simple_reward_system:
            curriculum_info = self.simple_reward_system.get_info()
            current_level = curriculum_info.get('level', 1)
            episodes_completed = curriculum_info.get('episodes', 0)

            # Obtener estado actual para decidir ayuda experta
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            current_tilt = abs(euler[0]) + abs(euler[1])
            # LÓGICA INTELIGENTE: Más ayuda cuando hay problemas
            base_expert_probability = {
                1: 0.85,  # 85% en nivel 1
                2: 0.65,  # 65% en nivel 2
                3: 0.45   # 45% en nivel 3
            }.get(current_level, 0.5)
            
            ## Aumentar ayuda si hay inestabilidad
            if current_tilt > 0.2:  # Más de 11.5 grados total
                base_expert_probability += 0.3
            elif current_tilt > 0.15:  # Más de 8.6 grados total
                base_expert_probability += 0.15
            
            # Más ayuda en episodios tempranos de cada nivel
            if episodes_completed < 20:
                base_expert_probability += 0.1
        
            use_expert = np.random.random() < min(0.95, base_expert_probability)
        
        if use_expert and hasattr(self, 'angle_expert_controller'):
            actual_action = self.angle_expert_controller.get_expert_action_for_level(self.simple_reward_system)
            action_source = "ANGLE_EXPERT"
        elif use_expert and self.action_selector is not None:
            actual_action = self.action_selector.get_expert_action()
            action_source = "BASIC_EXPERT"  
        else:
            actual_action = action
            action_source = "RL"

        self.ep_total_actions += 1
        if action_source in ("ANGLE_EXPERT", "BASIC_EXPERT"):
            self.ep_expert_actions += 1

        # ===== NORMALIZAR Y VALIDAR ACCIÓN =====
    
        normalized_pressures = np.clip(actual_action, 0.0, 1.0) 
        
        # Aplicar fuerzas PAM normalizadas
        joint_torques = self._apply_pam_forces(normalized_pressures)

        # NUEVA LÍNEA: Validar comportamiento biomecánico
        is_valid = self.validate_robot_specific_behavior(normalized_pressures, joint_torques)
        
        # ===== Paso 3: SIMULACIÓN FÍSICA =====

        # Aplicar torques
        torque_mapping = [
            (0, joint_torques[0]),  # left_hip_joint
            (1, joint_torques[1]),  # left_knee_joint  
            (3, joint_torques[2]),  # right_hip_joint
            (4, joint_torques[3])   # right_knee_joint
        ]

        for joint_id, torque in torque_mapping:
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )

        p.stepSimulation()

        # ✅ LLAMAR DEBUG OCASIONALMENTE
        self._debug_joint_angles_and_pressures(actual_action)

        
        #if self.simple_reward_system:
        reward = self.simple_reward_system.calculate_reward(actual_action, self.step_count)
        done = self.simple_reward_system.is_episode_done(self.step_count)
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
            'action_source': action_source,
            'episode_reward': self.episode_reward
        }

        if self.simple_reward_system:
            curriculum_info = self.simple_reward_system.get_info()  # Solo una llamada
            info['curriculum'] = curriculum_info  # Añadir sin reemplazar
            info['system_type'] = 'progressive'
            info['current_level'] = curriculum_info.get('level', 1)
            # Debug simple
            if done:
                expert_pct = 100.0 * self.ep_expert_actions / max(1, self.ep_total_actions)
                log_print(f"🎯 Expert usage this episode: {expert_pct:.1f}% "
                        f"({self.ep_expert_actions}/{self.ep_total_actions})")
                info['expert_usage_pct'] = expert_pct
                episode_total = info['episode_reward']  # Ya calculado arriba
                self.simple_reward_system.update_after_episode(episode_total)
                log_print(f"📈 Episode {info['curriculum']['episodes']} | Level {info['curriculum']['level']} | Reward: {episode_total:.1f}")
        #else:
            #info['current_task'] = current_task
            #info['system_type'] = 'fallback'
            #log_print(f"{self.current_task=:}")
        
        # CONSERVAR tu debug existente 
        if self.step_count % self.frecuency_simulation//2 == 0 or done:
            log_print(f"🔍 Step {self.step_count} - Control Analysis:")
            log_print(f"   Height: {pos[2]:.2f}m")
            log_print(f"   Tilt: Roll {math.degrees(euler[0]):.1f}°, Pitch {math.degrees(euler[1]):.1f}°")
            log_print(f"   Action source: {action_source}")
            
            if self.simple_reward_system:
                curriculum_info = self.simple_reward_system.get_info()
                log_print(f"   Level: {curriculum_info.get('level')}, Target: {curriculum_info.get('target_leg', 'N/A')}")
    
            # Verificar si está cerca de límites
            max_allowed_tilt = 0.4 if self.simple_reward_system and self.simple_reward_system.level == 1 else 0.3
            if abs(euler[0]) > max_allowed_tilt * 0.8 or abs(euler[1]) > max_allowed_tilt * 0.8:
                log_print(f"   ⚠️ Approaching tilt limit! Max allowed: ±{math.degrees(max_allowed_tilt):.1f}°")
            

        # DEBUG TEMPORAL: Verificar timing cada cierto número de steps
        if self.step_count % self.frecuency_simulation == 0 and self.simple_reward_system:  # Cada 5 segundos aprox
            status = self.simple_reward_system.get_info()
            elapsed_time = self.step_count / self.frecuency_simulation
            log_print(f" {action_source} action, reward={reward:.2f}")
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
        p.changeDynamics(
            self.robot_id, 
            self.left_foot_link_id,
            lateralFriction=0.8,        # Reducido de 1.2 a 0.8
            spinningFriction=0.3,       # Reducido de 0.8 a 0.3
            rollingFriction=0.02,       # Reducido de 0.1 a 0.02
            restitution=0.01,           # Reducido de 0.05 a 0.01 (menos rebote)
            contactDamping=100,         # Aumentado de 50 a 100 (más amortiguación)
            contactStiffness=15000      # Aumentado de 10000 a 15000 (más rigidez)
        )
        
        # Pie derecho - mismas propiedades
        p.changeDynamics(
            self.robot_id,
            self.right_foot_link_id, 
            lateralFriction=0.8,
            spinningFriction=0.3,
            rollingFriction=0.02,
            restitution=0.01,
            contactDamping=100,
            contactStiffness=15000
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
        
        log_print(f"🔧 Contact friction configured:")
        log_print(f"   Feet: μ=1.2 (high grip)")
        log_print(f"   Legs: μ=0.6 (moderate)")
        log_print(f"   Ground: μ=1.0 (standard)")
    

# ==================================================================================================================================================================== #
# =================================================== Métodos de Aplicación de fuerzas PAM =========================================================================== #
# ==================================================================================================================================================================== #
    
    def _apply_automatic_knee_control(self, base_torques):
        """Control automático de la rodilla levantada basado en altura"""
        
        # Determinar qué pierna está levantada basado en contactos
        left_contact = len(p.getContactPoints(self.robot_id, self.plane_id, 2, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.plane_id, 5, -1)) > 0
        
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
        self.target_knee_height
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
        joint_torques = self._calculate_robot_specific_joint_torques(pam_pressures)
        
        # Aplicar control automático de altura de rodilla
        joint_torques = self._apply_automatic_knee_control(joint_torques)

        balance_info = self.current_balance_status
        if self.step_count%100==0:
            log_print(f"Pierna de apoyo: {balance_info['support_leg']}")
            log_print(f"Tiempo en equilibrio: {balance_info['balance_time']} steps")

        return joint_torques
    
    def _calculate_robot_specific_joint_torques(self, pam_pressures):
        """
        Calcular torques básicos de articulaciones desde presiones PAM.
        
        Este método reemplaza la parte inicial de _apply_pam_forces
        antes del control automático de rodilla.
        """
        
        # Obtener estados articulares (solo joints activos: caderas y rodillas)
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)  
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # Calcular fuerzas PAM reales
        pam_forces = np.zeros(6, dtype=float)
        
        def eps_from(theta, R_abs,muscle_name):
            # θ0=0 como referencia; el método ya hace clip a [0, εmax]
            return self.pam_muscles[muscle_name].epsilon_from_angle(theta, 0.0, max(abs(R_abs), 1e-9))
        
        P = np.array([self.pam_muscles[muscle_names].real_pressure_PAM(u) for muscle_names,u in zip(self.muscle_names, pam_pressures)], dtype=float)

        # Cadera izquierda j0
        flexor_cadera_L, extensor_cadera_L=self.muscle_names[0], self.muscle_names[1]
        thL = joint_positions[0]
        R_flex_L = self.hip_flexor_moment_arm(thL)
        R_ext_L  = self.hip_extensor_moment_arm(thL)
        eps_flex_L = eps_from(thL, R_flex_L,flexor_cadera_L)
        eps_ext_L  = eps_from(thL, R_ext_L, extensor_cadera_L)
        pam_forces[0] = self.pam_muscles[flexor_cadera_L].force_model_new(P[0], eps_flex_L)  # flexor L
        pam_forces[1] = self.pam_muscles[extensor_cadera_L].force_model_new(P[1], eps_ext_L)   # extensor L

        # Cadera derecha j2
        flexor_cadera_R, extensor_cadera_R=self.muscle_names[2], self.muscle_names[3]
        thR = joint_positions[2]
        R_flex_R = self.hip_flexor_moment_arm(thR)
        R_ext_R  = self.hip_extensor_moment_arm(thR)
        eps_flex_R = eps_from(thR, R_flex_R,flexor_cadera_R)
        eps_ext_R  = eps_from(thR, R_ext_R,extensor_cadera_R)
        pam_forces[2] = self.pam_muscles[flexor_cadera_R].force_model_new(P[2], eps_flex_R)  # flexor L
        pam_forces[3] = self.pam_muscles[extensor_cadera_R].force_model_new(P[3], eps_ext_R)   # extensor L

        # RODILLAS (solo flexores activos)  (j1 = left_knee, j3 = right_knee)
        flexor_rodilla_L, flexor_rodilla_R=self.muscle_names[4], self.muscle_names[5]
        thK_L = joint_positions[1]
        thK_R = joint_positions[3]
        R_knee_L = self.knee_flexor_moment_arm(thK_L)
        R_knee_R = self.knee_flexor_moment_arm(thK_R)
        eps_knee_L = eps_from(thK_L, R_knee_L,flexor_rodilla_L)
        eps_knee_R = eps_from(thK_R, R_knee_R,flexor_rodilla_R)
        pam_forces[4] = self.pam_muscles[flexor_rodilla_L].force_model_new(P[4], eps_knee_L)  # flexor rodilla L
        pam_forces[5] = self.pam_muscles[flexor_rodilla_R].force_model_new(P[5], eps_knee_R)  # flexor rodilla R

        

        # Aplicar a las caderas (tienen músculos antagónicos)
        pam_forces[0], pam_forces[1] = apply_reciprocal_inhibition(pam_forces[0], 
                                                                   pam_forces[1],
                                                                   self.INHIBITION_FACTOR)  # Cadera izq
        pam_forces[2], pam_forces[3] = apply_reciprocal_inhibition(pam_forces[2], 
                                                                   pam_forces[3],
                                                                   self.INHIBITION_FACTOR)  # Cadera der


            
        # Convertir a torques articulares
        joint_torques = np.zeros(4)

         # CADERA IZQUIERDA (antagónica: flexor vs extensor)
        #left_hip_angle = joint_positions[0]
        #flexor_arm = self.hip_flexor_moment_arm(left_hip_angle)
        #extensor_arm = self.hip_extensor_moment_arm(left_hip_angle)
        
        #flexor_torque = pam_forces[0] * flexor_arm
        #extensor_torque = -pam_forces[1] * extensor_arm  # Negativo (dirección opuesta)
        #joint_torques[0] = flexor_torque + extensor_torque

        # Cadera izquierda: flexión positiva por flexor, extensión por extensor
        joint_torques[0]  = ( pam_forces[0] * R_flex_L) + (-pam_forces[1] * R_ext_L)

        # Rodilla izquierda: flexor + resorte/damping pasivos (como tenías)
        passive_spring  = - self.PASSIVE_SPRING_STRENGTH * np.sin(thK_L)
        passive_damping = - self.DAMPING_COEFFICIENT    * joint_velocities[1]
        joint_torques[1]  = (pam_forces[4] * R_knee_L) + passive_spring + passive_damping
        
        # Cadera derecha
        joint_torques[2]  = ( pam_forces[2] * R_flex_R) + (-pam_forces[3] * R_ext_R)
        
       # Rodilla derecha
        passive_spring_R  = - self.PASSIVE_SPRING_STRENGTH * np.sin(thK_R)
        passive_damping_R = - self.DAMPING_COEFFICIENT    * joint_velocities[3]
        joint_torques[3]  = (pam_forces[5] * R_knee_R) + passive_spring_R + passive_damping_R
        
        joint_torques = np.clip(joint_torques, -self.MAX_REASONABLE_TORQUE, self.MAX_REASONABLE_TORQUE)
    
        # ===== PASO 6: ACTUALIZAR ESTADOS PARA DEBUGGING =====
        
        self.pam_states = {
            'pressures': pam_pressures.copy(),
            'forces': np.abs(pam_forces),
            'raw_forces': pam_forces,
            'joint_torques': joint_torques.copy(),
            'moment_arms': {
                'left_hip_flexor': R_flex_L,
                'left_hip_extensor': R_ext_L,
                'right_hip_flexor': R_flex_R,
                'right_hip_extensor': R_ext_R,
                'left_knee_flexor': R_knee_L,
                'right_knee_flexor': R_knee_R
            },
            'inhibition_applied': True,
            'robot_specific_params': True
        }
        
        return joint_torques
    
    def seleccion_joint(self, i, joint_positions, joint_velocities):
        if i in [0, 1]:  # Cadera izquierda
            return joint_positions[0], joint_velocities[0]
        elif i in [2, 3]:  # Cadera derecha
            return joint_positions[2], joint_velocities[2]
        elif i == 4:  # Rodilla izquierda
            return joint_positions[1], joint_velocities[1]
        elif i == 5:  # Rodilla derecha
            return joint_positions[3], joint_velocities[3]
    
    def _get_single_leg_observation(self):
        """
        Observación específica para equilibrio en una pierna.
        Reemplaza _get_simple_observation con información más relevante.
        """
        
        obs = []
        
        # ===== ESTADO DEL TORSO (8 elementos) =====
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Posición y orientación  
        obs.extend([pos[0], pos[2], euler[0], euler[1]])  # x, z, roll, pitch
        
        # Velocidades
        obs.extend([lin_vel[0], lin_vel[2], ang_vel[0], ang_vel[1]])  # vx, vz, wx, wy
        
        # ===== ESTADOS ARTICULARES (4 elementos) =====
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)  # Solo joints activos
        joint_positions = [state[0] for state in joint_states]
        obs.extend(joint_positions)
        
        # ===== INFORMACIÓN DE CONTACTO Y ALTURA DE RODILLAS (4 elementos) =====
        
        # Contactos
        left_contact = len(p.getContactPoints(self.robot_id, self.plane_id, 2, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.plane_id, 5, -1)) > 0
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
        euler = p.getEulerFromQuaternion(orn)
        
        # Posición y orientación
        obs.extend([self.pos[0], self.pos[2], euler[0], euler[1]])  # x, z, roll, pitch
        
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
        left_contact, right_contact=self.contacto_pies    

        #print ("contacto pies", left_contact, right_contact)
        obs.extend([float(left_contact), float(right_contact)])
        
        return np.array(obs, dtype=np.float32)
    
    @property
    def contacto_pies(self):
        left_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_link_id, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_link_id, -1)) > 0
        if self.step_count<=150 and self.step_count%10==0:
            log_print(f"Contactos pie izquierdo: {left_contact}")
            log_print(f"Contactos pie derecho: {right_contact}")
        return left_contact, right_contact
    

    def reset(self, seed=None, options=None):
        """
        Reset modificado específicamente para equilibrio en una pierna.
        
        Reemplazar el método reset() del entorno original con este.
        """
        super().reset(seed=seed)

        self.ep_total_actions = 0
        self.ep_expert_actions = 0
        
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
            useFixedBase=False
        )
        
        # ===== SISTEMAS ESPECÍFICOS PARA EQUILIBRIO EN UNA PIERNA =====
        # Sistemas de recompensas
        if self.use_simple_progressive and self.simple_reward_system is None:
            self.simple_reward_system = SimpleProgressiveReward(self.robot_id, self.plane_id, 
                                                                self.frecuency_simulation,
                                                                switch_interval=self.switch_interval)
        else:
            # solo re-vincula IDs si cambiaron, sin perder contadores/racha
            self.simple_reward_system.robot_id = self.robot_id
            self.simple_reward_system.plane_id = self.plane_id
                 
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
            0: 0.0,   # left_hip - ligera flexión
            1: 0.0,   # left_knee - extendida (pierna de soporte)
            3: 0.0,   # right_hip - más flexión
            4: 0.0,   # right_knee - flexionada (pierna levantada)
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
            dt=self.time_step,
            robot_data=self.robot_data
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
        for _ in range(150):
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
        return {
            'support_leg': None,
            'raised_leg': None,
            'balance_time': 0,
            'target_knee_height': self.target_knee_height,
            'episode_step': self.step_count
        }
        
    def parametros_torque_pam(self):
        # Momentos de brazo calculados desde dimensiones reales
        self.HIP_FLEXOR_BASE_ARM = 0.0503      # 5.03cm - basado en circunferencia del muslo
        self.HIP_FLEXOR_VARIATION = 0.0101     # ±1.01cm variación por ángulo

        self.KP = 80.0   # Ganancia proporcional
        self.KD = 12.0   # Ganancia derivativa
        
        self.HIP_EXTENSOR_BASE_ARM = 0.0628    # 6.28cm - extensores más potentes (glúteos)
        self.HIP_EXTENSOR_VARIATION = 0.0126   # ±1.26cm variación por ángulo
        
        self.KNEE_FLEXOR_BASE_ARM = 0.0566     # 5.66cm - basado en circunferencia pantorrilla
        self.KNEE_FLEXOR_VARIATION = 0.0113    # ±1.13cm variación por ángulo
        
        # Parámetros de resortes pasivos (calculados desde momento gravitacional)
        self.PASSIVE_SPRING_STRENGTH = 180.5   # N⋅m 
        self.DAMPING_COEFFICIENT = 12.0        # N⋅m⋅s/rad (optimizado para masa real)
        
        # Control antagónico
        self.INHIBITION_FACTOR = 0.3           # 30% inhibición recíproca
        self.MAX_CONTRACTION_RATIO = 0.25      # 25% contracción máxima segura
        self.VELOCITY_DAMPING_FACTOR = 0.08    # 8% reducción por velocidad
        
        # Límites de seguridad (basados en fuerzas PAM reales calculadas)
        self.MAX_REASONABLE_TORQUE = 120.0     # N⋅m (factor de seguridad incluido)

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
        angle_factor = np.cos(angle - np.pi/6)  # Peak en flexión ligera
        return self.HIP_EXTENSOR_BASE_ARM + self.HIP_EXTENSOR_VARIATION * angle_factor

    def knee_flexor_moment_arm(self, angle):
        """
        Momento de brazo del flexor de rodilla (isquiotibiales).
        Basado en geometría real: circunferencia pantorrilla = 0.377m
        """
        # Flexor de rodilla más efectivo cerca de extensión
        angle_factor = np.cos(angle + np.pi/4)
        return self.KNEE_FLEXOR_BASE_ARM + self.KNEE_FLEXOR_VARIATION * angle_factor

    # ===== MÉTODO DE DEBUG ADICIONAL =====

    def _debug_joint_angles_and_pressures(self, pam_pressures):
        """
            ✅ MÉTODO DE DEBUG para verificar la lógica biomecánica
        
            Llama esto ocasionalmente durante el step() para verificar que la lógica funciona
        """
        
        if self.step_count % self.frecuency_simulation == 0:  # Cada segundo aprox
            try:
                joint_states = p.getJointStates(self.robot_id, [1, 4])  # rodillas
                left_knee_angle = joint_states[0][0]
                right_knee_angle = joint_states[1][0]

                log_print(f"\n🔍 Biomechanical Debug (Step {self.step_count}):")
                log_print(f"   Left knee: {left_knee_angle:.3f} rad ({math.degrees(left_knee_angle):.1f}°)")
                log_print(f"   Right knee: {right_knee_angle:.3f} rad ({math.degrees(right_knee_angle):.1f}°)")
                
                
                log_print(f"\n🔍 Biomechanical Debug (Step {self.step_count}):")
                log_print(f"   Left knee: {left_knee_angle:.3f} rad ({math.degrees(left_knee_angle):.1f}°)")
                log_print(f"   Right knee: {right_knee_angle:.3f} rad ({math.degrees(right_knee_angle):.1f}°)")
                log_print(f"   Left knee flexor pressure: {pam_pressures[4]:.3f}")
                log_print(f"   Right knee flexor pressure: {pam_pressures[5]:.3f}")
                
                # Verificar lógica biomecánica
                if left_knee_angle > 0.05 and pam_pressures[4] > 0.01:
                    log_print(f"   ⚠️ Warning: Left knee flexed but flexor active!")
                elif left_knee_angle > 0.05 and pam_pressures[4] <= 0.01:
                    log_print(f"   ✅ Correct: Left knee flexed, flexor inactive")
                    
                if right_knee_angle > 0.05 and pam_pressures[5] > 0.01:
                    log_print(f"   ⚠️ Warning: Right knee flexed but flexor active!")
                elif right_knee_angle > 0.05 and pam_pressures[5] <= 0.01:
                    log_print(f"   ✅ Correct: Right knee flexed, flexor inactive")
            
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
        
        # ===== VALIDAR TORQUES DENTRO DE CAPACIDAD FÍSICA =====
        
        # Para tu robot específico: torques >120 N⋅m son físicamente imposibles
        for i, torque in enumerate(joint_torques):
            if abs(torque) > 100.0:  # Warning a 100 N⋅m (antes del límite de 120)
                joint_names = ['left_hip', 'left_knee', 'right_hip', 'right_knee']
                warnings.append(f"{joint_names[i]}: High torque {torque:.1f} N⋅m")
        
        # ===== VALIDAR EFICIENCIA ENERGÉTICA =====
        
        # Para tu robot de 25kg, activación total >4.0 es ineficiente
        total_activation = np.sum(pam_pressures)
        if total_activation > 4.0:
            efficiency = (6.0 - total_activation) / 6.0 * 100  # % de eficiencia
            warnings.append(f"Energy efficiency: {efficiency:.1f}% (high activation)")
        
        # ===== VALIDAR ESTABILIDAD BIOMECÁNICA =====
        
        # Para equilibrio en una pierna, verificar asimetría apropiada
        left_activation = np.sum(pam_pressures[0:2]) + pam_pressures[4]  # Cadera izq + rodilla izq
        right_activation = np.sum(pam_pressures[2:4]) + pam_pressures[5]  # Cadera der + rodilla der
        
        asymmetry = abs(left_activation - right_activation)
        if asymmetry < 0.5:  # Muy simétrico para equilibrio en una pierna
            warnings.append(f"Low asymmetry: {asymmetry:.2f} (may indicate poor single-leg balance)")
        
        # ===== LOGGING CONDICIONAL =====
        
        if warnings and self.step_count % self.frecuency_simulation//2 == 0:  # Cada 0.5 segundos aprox
            log_print(f"🤖 Robot-specific validation (Step {self.step_count}):")
            for warning in warnings:
                log_print(f"   ⚠️ {warning}")
            
            # Info adicional útil
            log_print(f"   Total mass: 25kg, Height: 1.20m")
            log_print(f"   Current torques: {[f'{t:.1f}' for t in joint_torques]} N⋅m")
        
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
    log_print(f"   PAM configuration: 6 muscles (4 hip antagonistic + 2 knee flexors)")
    log_print(f"   Moment arms: Hip 5.0-6.3cm, Knee 5.7cm")
    log_print(f"   Passive springs: 32.5 N⋅m (gravity-compensated)")
    
    # Configurar parámetros específicos en el entorno
    env.robot_specific_configured = True
    env.expected_robot_mass = expected_mass
    env.expected_robot_height = expected_height
    
    # Reemplazar el método de cálculo de torques
    env._calculate_basic_joint_torques = env._calculate_robot_specific_joint_torques
    
    log_print("✅ Robot-specific PAM system configured!")
    
    return True

def apply_reciprocal_inhibition(flexor_force, extensor_force, INHIBITION_FACTOR):
    """
        Inhibición recíproca calibrada para tu robot.
        Basada en estudios neurológicos: cuando un músculo se activa fuerte,
        el sistema nervioso inhibe parcialmente su antagonista.
    """
    total_activation = flexor_force + extensor_force
    if total_activation > 0:
        # Reducir la fuerza del músculo menos activo
        flexor_ratio = flexor_force / total_activation
        extensor_ratio = extensor_force / total_activation

        if flexor_ratio > 0.6:
            extensor_force *= (1.0 - INHIBITION_FACTOR * flexor_ratio)
        elif extensor_ratio > 0.6:
            flexor_force *= (1.0 - INHIBITION_FACTOR * extensor_ratio)
    
    return flexor_force, extensor_force




