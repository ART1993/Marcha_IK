
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

from Archivos_Mejorados.RewardSystemSimple import SingleLegBalanceRewardSystem, \
                                                    SingleLegActionSelector             

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
            - right hip joint: 4
            - right hip joint: 5
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
        self.enable_curriculum=enable_curriculum

        self.pam_control_active = False
        self.contact_established = False
        self.contact_stable_steps = 0
        # Para tracking de tiempo en balance
        self._balance_start_time = 0
        
        # ===== CONFIGURACIÓN FÍSICA BÁSICA =====
        
        self.urdf_path = "2_legged_human_like_robot.urdf"
        self.frecuency_simulation=1500.0
        self.time_step = 1.0 / self.frecuency_simulation
        # ===== CONFIGURACIÓN PAM SIMPLIFICADA =====
        self.num_active_pams = 6
        self.min_pressure = 101325  # 1 atm
        self.max_pressure = 5 * self.min_pressure  # 5 atm

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
        
        log_print(f"🤖 Simplified Lift legs Environment initialized")
        log_print(f"🤖 Environment initialized - Systems initiate in reset")


    def _calculate_reward(self, action_applied):
        """
            Calcular recompensa según el modo de control activo
        """
        
        if self.pam_control_active:
            # Usar sistema de recompensas PAM completo
            self.reward_system.pam_states = self.pam_states
            reward, reward_components = self.reward_system.calculate_simple_reward(
                action=action_applied,
                pam_forces=self.pam_states['forces']
            )
        else:
            # Recompensa simplificada para modo STANDING
            reward, reward_components = self._calculate_standing_reward()
        
        return reward, reward_components

    def _calculate_standing_reward(self):
        """
        Recompensa para modo STANDING_POSITION
        """
        
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        reward_components = {}
        
        # Recompensa por mantener altura
        height_reward = max(0, pos[2] - 0.8) * 5.0
        reward_components['height'] = height_reward
        
        # Recompensa por orientación vertical
        orientation_penalty = (abs(euler[0]) + abs(euler[1])) * 10.0
        reward_components['orientation'] = -orientation_penalty
        
        # Recompensa por progreso hacia contacto
        left_contacts = p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_id, -1)
        right_contacts = p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_id, -1)
        
        contact_reward = 0
        if len(left_contacts) > 0 and len(right_contacts) > 0:
            contact_reward = 10.0  # Bonificación por contacto bilateral
        elif len(left_contacts) > 0 or len(right_contacts) > 0:
            contact_reward = 3.0   # Bonificación menor por contacto parcial
        
        reward_components['contact_progress'] = contact_reward
        
        # Recompensa base
        reward_components['survival'] = 1.0
        
        total_reward = sum(reward_components.values())
        
        return total_reward, reward_components
    
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
        
        if self.action_selector.should_use_expert_action():
            actual_action = self.action_selector.get_expert_action()
            action_source = "EXPERT"
        else:
            actual_action = action
            action_source = "RL"

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

        current_task = self.action_selector.current_action.value
        reward = self.reward_system.calculate_reward(actual_action, current_task)
        # ===== CÁLCULO DE RECOMPENSAS CONSCIENTE DEL CONTEXTO =====
       
        
        # ===== PASO 4: OBSERVACIÓN Y TERMINACIÓN =====
        self.episode_reward += reward
        
        done = self.reward_system.is_episode_done(self.step_count, self.frecuency_simulation)
        observation = self._get_simple_observation()

        # ===== APLICAR ACCIÓN PAM =====

        self.episode_reward += reward
        self.action_selector.update_after_step(reward)
        
        # Info simplificado
        info = {
            'step_count': self.step_count,
            'reward': reward,
            'action_source': action_source,
            'current_task': current_task,
            'episode_reward': self.episode_reward
        }
        
        # CONSERVAR tu debug existente
        if self.step_count % 1500 == 0 or done:
            log_print(f"{self.step_count=:}: {action_source} action, reward={reward:.2f}, task={current_task}")
            log_print(f"Step {done=:}")
        
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
            lateralFriction=1.2,        # Fricción lateral alta
            spinningFriction=0.8,       # Fricción rotacional
            rollingFriction=0.1,        # Fricción de rodadura baja
            restitution=0.05,           # Poco rebote
            contactDamping=50,          # Amortiguación de contacto
            contactStiffness=10000      # Rigidez de contacto
        )
        
        # Pie derecho - mismas propiedades
        p.changeDynamics(
            self.robot_id,
            self.right_foot_link_id, 
            lateralFriction=1.2,
            spinningFriction=0.8,
            rollingFriction=0.1,
            restitution=0.05,
            contactDamping=50,
            contactStiffness=10000
        )
        
        # ===== FRICCIÓN PARA OTROS LINKS =====
        
        # Links de piernas - fricción moderada
        for link_id in self.joint_indices:
            p.changeDynamics(
                self.robot_id,
                link_id,
                lateralFriction=0.6,
                spinningFriction=0.4,
                rollingFriction=0.05,
                restitution=0.1
            )
        
        # ===== FRICCIÓN DEL SUELO =====
        
        # Configurar fricción del plano del suelo
        p.changeDynamics(
            self.plane_id,
            -1,                         # -1 for base link
            lateralFriction=1.0,        # Fricción estándar del suelo
            spinningFriction=0.5,
            rollingFriction=0.01
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
        kp = 80.0  # Ganancia proporcional para h
        kd = 12.0   # Ganancia derivativa de h
        
        control_torque = kp * height_error - kd * knee_velocity
        
        # Combinar con torque base (PAM) usando peso
        base_torques[controlled_knee_idx] = (
            0.4 * base_torques[controlled_knee_idx] +  # 40% PAM
            0.6 * control_torque                        # 60% control automático
        )

        # Limitar torque final
        max_knee_torque = 100.0
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
        pam_forces = np.zeros(6)
        muscle_names = ['left_hip_flexor', 'left_hip_extensor', 'right_hip_flexor', 
                        'right_hip_extensor', 'left_knee_flexor', 'right_knee_flexor']
        
        for i, pressure_normalized in enumerate(pam_pressures):
            # Determinar articulación correspondiente
            #joint_angle, joint_velocity=self.seleccion_joint(i, joint_positions, joint_velocities)
            
            # Presión real
            real_pressure = self.min_pressure + pressure_normalized * (self.max_pressure - self.min_pressure)

            # Contracción comandada (no basada en posición actual)
            commanded_contraction = pressure_normalized * self.MAX_CONTRACTION_RATIO
            
            
            # Calcular fuerza desde modelo PAM
            muscle_name = muscle_names[i]
            pam_muscle = self.pam_muscles[muscle_name]
            raw_force = pam_muscle.force_model_new(real_pressure, commanded_contraction)
            
            ## Damping por velocidad específico para robot
            joint_idx = 0 if i < 2 else (2 if i < 4 else (1 if i == 4 else 3))
            velocity_damping = 1.0 - self.VELOCITY_DAMPING_FACTOR * abs(joint_velocities[joint_idx])
            velocity_damping = np.clip(velocity_damping, 0.6, 1.0)
            
            pam_forces[i] = raw_force * velocity_damping

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
        left_hip_angle = joint_positions[0]
        flexor_arm = self.hip_flexor_moment_arm(left_hip_angle)
        extensor_arm = self.hip_extensor_moment_arm(left_hip_angle)
        
        flexor_torque = pam_forces[0] * flexor_arm
        extensor_torque = -pam_forces[1] * extensor_arm  # Negativo (dirección opuesta)
        joint_torques[0] = flexor_torque + extensor_torque
        
        # Rodilla izquierda (flexor + resorte pasivo)
        left_knee_angle = joint_positions[1]
        knee_arm = self.knee_flexor_moment_arm(left_knee_angle)
        flexor_torque = pam_forces[4] * knee_arm
        # Resorte pasivo calculado para contrarrestar gravedad de tu robot específico
        passive_spring = -self.PASSIVE_SPRING_STRENGTH * np.sin(left_knee_angle)
        passive_damping = -self.DAMPING_COEFFICIENT * joint_velocities[1]
        joint_torques[1] = flexor_torque + passive_spring + passive_damping
        
        # CADERA DERECHA (antagónica: flexor vs extensor)
        right_hip_angle = joint_positions[2]
        flexor_arm = self.hip_flexor_moment_arm(right_hip_angle)
        extensor_arm = self.hip_extensor_moment_arm(right_hip_angle)

        flexor_torque = pam_forces[2] * flexor_arm
        extensor_torque = -pam_forces[3] * extensor_arm
        joint_torques[2] = flexor_torque + extensor_torque
        
        # RODILLA DERECHA (flexor PAM + resorte extensor pasivo)
        right_knee_angle = joint_positions[3]
        knee_arm = self.knee_flexor_moment_arm(right_knee_angle)
        
        flexor_torque = pam_forces[5] * knee_arm
        passive_spring = -self.PASSIVE_SPRING_STRENGTH * np.sin(right_knee_angle)
        passive_damping = -self.DAMPING_COEFFICIENT * joint_velocities[3]
        joint_torques[3] = flexor_torque + passive_spring + passive_damping
        
        joint_torques = np.clip(joint_torques, -self.MAX_REASONABLE_TORQUE, self.MAX_REASONABLE_TORQUE)
    
        # ===== PASO 6: ACTUALIZAR ESTADOS PARA DEBUGGING =====
        
        self.pam_states = {
            'pressures': pam_pressures.copy(),
            'forces': np.abs(pam_forces),
            'raw_forces': pam_forces,
            'joint_torques': joint_torques.copy(),
            'moment_arms': {
                'left_hip_flexor': self.hip_flexor_moment_arm(left_hip_angle),
                'left_hip_extensor': self.hip_extensor_moment_arm(left_hip_angle),
                'right_hip_flexor': self.hip_flexor_moment_arm(right_hip_angle),
                'right_hip_extensor': self.hip_extensor_moment_arm(right_hip_angle),
                'left_knee_flexor': self.knee_flexor_moment_arm(left_knee_angle),
                'right_knee_flexor': self.knee_flexor_moment_arm(right_knee_angle)
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
        left_contact = len(p.getContactPoints(self.robot_id, 0, 2, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, 0, 5, -1)) > 0
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
        
        # Actualizar reward system y action selector del episodio anterior si existen
        if hasattr(self, 'action_selector') and self.action_selector and hasattr(self, 'episode_reward'):
            self.action_selector.update_after_episode(self.episode_reward)

        
        # ===== RESET FÍSICO =====
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Configurar solver para estabilidad
        p.setPhysicsEngineParameter(
            numSolverIterations=12,
            numSubSteps=4,
            contactBreakingThreshold=0.001,
            erp=0.8,
            contactERP=0.9,
            frictionERP=0.8,
        )
        
        # Cargar entorno
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            [0, 0, 1.21],  # Altura inicial ligeramente mayor
            useFixedBase=False
        )
        
        # ===== SISTEMAS ESPECÍFICOS PARA EQUILIBRIO EN UNA PIERNA =====
        
        # Nuevo sistema de recompensas
        self.reward_system = SingleLegBalanceRewardSystem(self.robot_id, self.plane_id)
        
        # Nuevo selector de acciones
        if not hasattr(self, 'action_selector') or not self.action_selector:
            self.action_selector = SingleLegActionSelector(self)
        
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
        if hasattr(self, 'reward_system') and self.reward_system:
            return {
                'support_leg': self.reward_system.current_support_leg,
                'raised_leg': self.reward_system.raised_leg,
                'balance_time': self.reward_system.single_leg_time,
                'target_knee_height': self.target_knee_height,
                'episode_step': self.step_count
            }
        else:
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
        
        self.HIP_EXTENSOR_BASE_ARM = 0.0628    # 6.28cm - extensores más potentes (glúteos)
        self.HIP_EXTENSOR_VARIATION = 0.0126   # ±1.26cm variación por ángulo
        
        self.KNEE_FLEXOR_BASE_ARM = 0.0566     # 5.66cm - basado en circunferencia pantorrilla
        self.KNEE_FLEXOR_VARIATION = 0.0113    # ±1.13cm variación por ángulo
        
        # Parámetros de resortes pasivos (calculados desde momento gravitacional)
        self.PASSIVE_SPRING_STRENGTH = 32.5   # N⋅m (120% del momento gravitacional real)
        self.DAMPING_COEFFICIENT = 10.0        # N⋅m⋅s/rad (optimizado para masa real)
        
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
        
        if warnings and self.step_count % 750 == 0:  # Cada 0.5 segundos aprox
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




