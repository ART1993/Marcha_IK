
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
from collections import deque

from Controlador.discrete_action_controller import ActionType 
from Controlador.ankle_control_and_curriculum_fixes import OptimizedCurriculumSelector, IntelligentAnkleControl


from Archivos_Apoyo.Configuraciones_adicionales import PAM_McKibben
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data

from Archivos_Mejorados.Simplified_BalanceSquat_RewardSystem import Simplified_BalanceSquat_RewardSystem
from Archivos_Mejorados.AntiFlexionController import AntiFlexionController, configure_enhanced_ankle_springs    
from Archivos_Mejorados.phase_aware_postural_control import PhaseAwareEnhancedController, MovementPhase             

class Simple_BalanceSquat_BipedEnv(gym.Env):
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
            Initialize the enhanced PAM biped environment.
            
            Args:
                render_mode: 'human' or 'direct'
                action_space: Only "pam" supported in this version
        """
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super(Simple_BalanceSquat_BipedEnv, self).__init__()

        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.render_mode = render_mode
        self.action_space_type = action_space  # Solo "pam"
        self.enable_curriculum=enable_curriculum
        if enable_curriculum:
            self.curriculum = OptimizedCurriculumSelector()
        else:
            self.curriculum = None

        self.pam_control_active = False
        self.contact_established = False
        self.contact_stable_steps = 0
        self.min_stable_steps = 50
        
        # ===== CONFIGURACIÓN FÍSICA BÁSICA =====
        
        self.urdf_path = "2_legged_human_like_robot.urdf"
        self.time_step = 1.0 / 1500.0
        
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
        
        # ===== VARIABLES DE SEGUIMIENTO BÁSICAS =====
        
        self.step_count = 0
        self.total_reward = 0
        self.robot_id = None
        self.plane_id = None
        self.left_foot_id = 2
        self.right_foot_id = 5
        
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
        self.episode_reward = 0
        self.episode_start_step = 0
        
        # Sistema de recompensas simplificado
        self.reward_system = Simplified_BalanceSquat_RewardSystem()
        
        print(f"🤖 Simplified Balance & Squat Environment initialized")
        print(f"   Action space: 6 PAM pressures")
        print(f"   Observation space: 16 elements")
        print(f"   Target: Balance + Sentadillas")
        print(f"🤖 Environment initialized - Starting in STANDING_POSITION mode")


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
        # ===== DECISIÓN: EXPERTO vs RL =====
        
        if self.curriculum and self.curriculum.should_use_expert_action(self.step_count):
            # 🎓 Usar acción EXPERTA
            actual_action = self.controller.get_expert_action(self.time_step)
            action_source = "EXPERT"
        else:
            # 🤖 Usar acción de la RED NEURONAL
            actual_action = action
            action_source = "RL_AGENT"
        
        # ===== DECISIÓN: BALANCE vs SQUAT =====
        
        if self.curriculum and self.curriculum.should_transition_to_squat():
            if self.controller.current_action != ActionType.SQUAT:
                self.controller.set_action(ActionType.SQUAT)
                print(f"   🏋️ Transitioning to SQUAT mode (Episode {self.curriculum.episode_count})")



        self.step_count += 1

        # ===== PASO 1: NORMALIZAR Y VALIDAR ACCIÓN =====
    
        normalized_pressures = np.clip(actual_action, 0.0, 1.0)

        # Validar que tenemos 6 presiones PAM
        if len(normalized_pressures) != self.num_active_pams:
            raise ValueError(f"Expected 6 PAM pressures, got {len(normalized_pressures)}")

        

        # NUEVO: Aplicar inhibición recíproca
        if not hasattr(self, 'anti_flexion'):
            self.anti_flexion = AntiFlexionController()

        # ===== PASO 2: APLICAR LÓGICA DE CONTROL SEGÚN ESTADO =====
        joint_states = p.getJointStates(self.robot_id, [0, 1, 3, 4])
        joint_positions = [state[0] for state in joint_states]
        
        # Corregir presiones PAM usando principio de reciprocidad
        corrected_pressures = self.anti_flexion.apply_reciprocal_inhibition(
            normalized_pressures, joint_positions
        )
        
        # Aplicar fuerzas PAM corregidas
        joint_torques = self._apply_pam_forces(corrected_pressures)
        
        # NUEVO: Aplicar control de tobillos mejorado
        configure_enhanced_ankle_springs(self.robot_id)

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

        # NUEVO: Control inteligente de tobillos
        if self.ankle_control:
            left_ankle_torque, right_ankle_torque = self.ankle_control.calculate_ankle_torques(
                self.robot_data, self.zmp_calculator
            )
            
            # Aplicar torques calculados
            p.setJointMotorControl2(self.robot_id, self.left_foot_id, p.TORQUE_CONTROL, force=left_ankle_torque)  # left_ankle
            p.setJointMotorControl2(self.robot_id, self.right_foot_id, p.TORQUE_CONTROL, force=right_ankle_torque)  # right_ankle
        
        p.stepSimulation()

        # ✅ LLAMAR DEBUG OCASIONALMENTE
        self._debug_joint_angles_and_pressures(actual_action)

    
        # Pasar información PAM al sistema de recompensas
        self.reward_system.pam_states = self.pam_states
        #reward, reward_components = self._calculate_reward(action_applied)
        reward, reward_components = self.reward_system.calculate_simple_reward(
            action=normalized_pressures,
            pam_forces=self.pam_states['forces']
        )

        # ===== PASO 4: OBSERVACIÓN Y TERMINACIÓN =====
        self.episode_reward += reward
        observation = self._get_simple_observation()
        done = self._is_done()

        # ===== APLICAR ACCIÓN PAM =====
        
        # Info básico
        info = {
            'step_count': self.step_count,
            'reward': reward,
            'reward_components': reward_components,
            'action_source': action_source,  # ✅ NUEVO
            'pam_pressures': normalized_pressures.tolist(),
            'episode_reward': self.episode_reward
        }
        
        # Añadir info de curriculum
        #if self.curriculum:
        #    info['curriculum'] = self.curriculum.get_curriculum_info()

        # NUEVAS LÍNEAS: Añadir información del sistema postural
        if hasattr(self.controller, 'get_comprehensive_performance_stats'):
            info['postural_control'] = {
                'current_phase': self.controller.postural_system.current_phase.value,
                'performance_stats': self.controller.get_comprehensive_performance_stats(),
                'total_corrections': self.controller.total_corrections
            }
        
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
            self.left_foot_id,
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
            self.right_foot_id, 
            lateralFriction=1.2,
            spinningFriction=0.8,
            rollingFriction=0.1,
            restitution=0.05,
            contactDamping=50,
            contactStiffness=10000
        )
        
        # ===== FRICCIÓN PARA OTROS LINKS =====
        
        # Links de piernas - fricción moderada
        leg_links = [0, 1, 3, 4]  # caderas y rodillas (si tienen collision)
        for link_id in leg_links:
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
            -1,  # -1 for base link
            lateralFriction=1.0,        # Fricción estándar del suelo
            spinningFriction=0.5,
            rollingFriction=0.01
        )
        
        print(f"🔧 Contact friction configured:")
        print(f"   Feet: μ=1.2 (high grip)")
        print(f"   Legs: μ=0.6 (moderate)")
        print(f"   Ground: μ=1.0 (standard)")

    def _apply_control_logic(self, action, both_feet_contact):
        """
        ✅ LÓGICA DE CONTROL CORREGIDA
        
        Estados:
        1. STANDING_POSITION: Robot se estabiliza con control de posición
        2. CONTACT_STABLE: Contacto bilateral establecido y estable
        3. PAM_ACTIVE: Control por torques PAM activo
        """
        
        # ===== MANEJO DE TRANSICIONES DE ESTADO =====
        
        if both_feet_contact:
            self.contact_stable_steps += 1
            if not self.contact_established:
                print(f"   🦶 Step {self.step_count}: Contacto bilateral detectado")
                self.contact_established = True
        else:
            self.contact_stable_steps = 0
            if self.pam_control_active:
                print(f"   ⚠️ Step {self.step_count}: Contacto perdido - Volviendo a STANDING_POSITION")
                self.pam_control_active = False
                self.contact_established = False

        # ===== ACTIVACIÓN DE CONTROL PAM =====
        
        if (self.contact_established and 
            self.contact_stable_steps >= self.min_stable_steps and 
            not self.pam_control_active):
            
            print(f"   🔥 Step {self.step_count}: Contacto estable {self.contact_stable_steps} steps - ACTIVANDO PAMs")
            self.pam_control_active = True

        # ===== APLICAR CONTROL SEGÚN MODO =====
        
        if self.pam_control_active:
            # 🔥 MODO PAM: Control por torques
            action_applied = self._apply_pam_control(action)
            control_mode = 'PAM_TORQUE'
            
            
        else:
            # 🤖 MODO STANDING: Control por posición
            action_applied = self._apply_standing_position_control()
            control_mode = 'POSITION_STANDING'

        return control_mode, action_applied
    
    
    def _apply_standing_position_control(self):
        """
        ✅ CONTROL DE POSICIÓN: Mantener standing position hasta contacto estable
        """
        
        for i, target_pos in self.neutral_positions.items():
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=50,  # Fuerza suficiente para mantener posición
                maxVelocity=1.0  # Velocidad moderada
            )
        
        # Actualizar estados PAM (inactivos)
        self.pam_states = {
            'pressures': np.zeros(self.num_active_pams),
            'forces': np.zeros(self.num_active_pams)
        }
        
        # Retornar acción neutral para info
        return np.array([0.3, 0.35, 0.3, 0.35, 0.2, 0.2])  # Presiones base

    def _apply_pam_control(self, action):
        """
        ✅ CONTROL PAM: Aplicar torques calculados desde presiones PAM
        """
        
        # Normalizar acción
        normalized_pressures = np.clip(action, 0.0, 1.0)
        
        # Calcular torques PAM
        joint_torques = self._apply_pam_forces(normalized_pressures)
        
        # Aplicar torques a articulaciones principales
        for i, joint_idx in enumerate([0, 1, 3, 4]):  # caderas y rodillas
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=joint_torques[i]
            )
        
        # Los tobillos pueden mantener control de posición suave
        for ankle_idx in [2, 5]:
            p.setJointMotorControl2(
                self.robot_id,
                ankle_idx,
                p.POSITION_CONTROL,
                targetPosition=0.0,  # Neutral
                force=50  # Muy suave
            )
        
        # Actualizar estados PAM
        self.pam_states['pressures'] = normalized_pressures

        return normalized_pressures

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
        # Obtener estados articulares actuales
        joint_states = p.getJointStates(self.robot_id, [0, 1, 3, 4])  # caderas y rodillas
        # para joint states cada estado representa:
        
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        pam_forces = np.zeros(6)  # Fuerzas reales de cada PAM

        # ===== CALCULAR FUERZAS REALES PAM =====
    
        muscle_names = ['left_hip_flexor', 'left_hip_extensor', 'right_hip_flexor', 
                        'right_hip_extensor', 'left_knee_flexor', 'right_knee_flexor']
        
        for i, pressure_normalized in enumerate(pam_pressures):
            # ===== PASO 1: DETERMINAR ARTICULACIÓN Y ÁNGULO =====
        
            if i in [0, 1]:  # Cadera izquierda
                joint_angle = joint_positions[0]  # left_hip
                joint_velocity = joint_velocities[0]
            elif i in [2, 3]:  # Cadera derecha  
                joint_angle = joint_positions[2]  # right_hip
                joint_velocity = joint_velocities[2]
            elif i == 4:  # Rodilla izquierda
                joint_angle = joint_positions[1]  # left_knee
                joint_velocity = joint_velocities[1]
            elif i == 5:  # Rodilla derecha
                joint_angle = joint_positions[3]  # right_knee
                joint_velocity = joint_velocities[3]

            # ===== PASO 2: CALCULAR CONTRACCIÓN REALISTA =====
        
            # Convertir presión normalizada [0,1] a presión real [Pa]
            real_pressure = self.min_pressure + pressure_normalized * (self.max_pressure - self.min_pressure)
            
            # Calcular contracción basada en activación muscular realista
            if i in [0, 2]:  # Flexores de cadera
                # Flexores se activan con flexión positiva (hacia adelante)
                max_flexion = 1.2  # rad (~69 grados)
                activation = max(0, joint_angle) / max_flexion
                contraction_ratio = activation * 0.25  # Máximo 25% contracción
                
            elif i in [1, 3]:  # Extensores de cadera  
                # Extensores se activan con extensión (ángulo negativo) 
                max_extension = 1.2  # rad
                activation = max(0, -joint_angle) / max_extension
                contraction_ratio = activation * 0.25
                
            elif i in [4, 5]:  # Flexores de rodilla
                # Rodillas solo flexión (0 a ~90 grados)
                max_knee_flexion = 1.571  # rad (90 grados)
                activation = max(0, joint_angle) / max_knee_flexion
                contraction_ratio = activation * 0.3  # Rodillas pueden contraerse más
            
            # Limitar contracción
            contraction_ratio = np.clip(contraction_ratio, 0, 0.3)  # Seguro
            
            muscle_name = muscle_names[i]
            pam_muscle = self.pam_muscles[muscle_name]
            
            # Calcular fuerza base del modelo físico
            raw_force = pam_muscle.force_model_new(real_pressure, contraction_ratio)
    
                
            # Modular por velocidad articular (damping biomecánico)
            velocity_damping = 1.0 - 0.1 * abs(joint_velocity)  # Reducir fuerza con velocidad alta
            velocity_damping = np.clip(velocity_damping, 0.5, 1.0)
            
            raw_force *= velocity_damping
            
            pam_forces[i] = raw_force
            
            # Debug detallado cada 1500 pasos (1 segundo aprox)
            if self.step_count % 1500 == 0 and i < 2:  # Solo primeros 2 PAMs para no saturar
                print(f"   PAM {i} ({muscle_name}): "
                    f"P={real_pressure/101325:.1f}atm, "
                    f"θ={joint_angle:.2f}rad, "
                    f"ε={contraction_ratio:.3f}, "
                    f"F={raw_force:.1f}N")
                
        # ===== PASO 2: CONVERTIR FUERZAS PAM A TORQUES ARTICULARES =====
        
        # Convertir fuerzas PAM a torques articulares
        moment_arm = 0.05  # Brazo de palanca típico (5cm)
        joint_torques = np.zeros(4)
        
        # ✅ CADERA IZQUIERDA: Antagónico (flexor vs extensor)
        # Torque resultante = momento_flexor - momento_extensor
        flexor_moment = pam_forces[0] * moment_arm    # PAM 0: flexor
        extensor_moment = -pam_forces[1] * moment_arm # PAM 1: extensor (ya negativo)
        joint_torques[0] = flexor_moment + extensor_moment  # Suma algebraica
        
        # ✅ RODILLA IZQUIERDA: Flexor + resorte pasivo de extensión
        flexor_moment = pam_forces[4] * moment_arm    # PAM 4: flexor
        passive_spring_torque = -150.0 * (joint_positions[1] - 0.1)  # Resorte a 0.1 rad
        damping_torque = -10.0 * joint_velocities[1]  # Damping proporcional a velocidad
        joint_torques[1] = flexor_moment + passive_spring_torque + damping_torque
            
        # ✅ CADERA DERECHA: Antagónico (flexor vs extensor)
        flexor_moment = pam_forces[2] * moment_arm    # PAM 2: flexor  
        extensor_moment = -pam_forces[3] * moment_arm # PAM 3: extensor
        joint_torques[2] = flexor_moment + extensor_moment
        
        # ✅ RODILLA DERECHA: Flexor + resorte pasivo
        flexor_moment = pam_forces[5] * moment_arm    # PAM 5: flexor
        passive_spring_torque = -150.0 * (joint_positions[3] - 0.1)
        damping_torque = -10.0 * joint_velocities[3]  # Damping proporcional a velocidad
        joint_torques[3] = flexor_moment + passive_spring_torque + damping_torque
        
        # Debug torques
        if self.step_count % 1500 == 0:
            print(f"Joint Torques: LH={joint_torques[0]:.1f}, LK={joint_torques[1]:.1f}, "
                f"RH={joint_torques[2]:.1f}, RK={joint_torques[3]:.1f}")
        
        # ===== PASO 6: APLICAR LÍMITES DE SEGURIDAD =====
    
        max_torque = 100.0  # Nm - límite de seguridad
        joint_torques = np.clip(joint_torques, -max_torque, max_torque)
        
        # Debug de torques
        if self.step_count % 1500 == 0:
            print(f"   Joint Torques: "
                f"LH={joint_torques[0]:.1f}, "
                f"LK={joint_torques[1]:.1f}, "
                f"RH={joint_torques[2]:.1f}, "
                f"RK={joint_torques[3]:.1f} Nm")
        
        # ===== ACTUALIZAR ESTADOS PAM =====
        
        self.pam_states = {
            'pressures': pam_pressures.copy(),
            'forces': np.abs(pam_forces),  # Magnitudes para observación
            'raw_forces': pam_forces,      # Con signo para debug
            'joint_torques': joint_torques.copy()
        }

        return joint_torques
    
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
        
        joint_states = p.getJointStates(self.robot_id, [0, 1, 3, 4])
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
        left_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_id, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_id, -1)) > 0
        return left_contact, right_contact
    
    def _calculate_simple_balance_reward(self):
        """
        Recompensa SIMPLIFICADA enfocada solo en balance y estabilidad
        
        ELIMINADO:
        - Biomechanical reward components complejos
        - Expert action similarity
        - Energy efficiency calculations
        - Coordination metrics
        """
        
        reward = 0.0
        
        # ===== RECOMPENSA POR ESTAR DE PIE =====
        
        self.pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        left_contact, right_contact=self.contacto_pies
        contacto_pies=all((left_contact, right_contact))
        # Altura del torso (mantenerse erguido)
        if contacto_pies:
            height_reward = max(0, self.pos[2] - 0.8) * 10.0  # Recompensar altura > 0.8m
            reward += height_reward
        
        # Orientación vertical (roll y pitch pequeños)
        orientation_penalty = abs(euler[0]) + abs(euler[1])  # roll + pitch
        reward -= orientation_penalty * 20.0
        
        # ===== RECOMPENSA POR ESTABILIDAD ZMP =====
        
        if self.zmp_calculator:
            try:
                zmp_point = self.zmp_calculator.calculate_zmp()
                is_stable = self.zmp_calculator.is_stable(zmp_point)
                
                if is_stable:
                    reward += 5.0  # Bonificación por estabilidad
                else:
                    reward -= 10.0  # Penalización por inestabilidad
                    
            except:
                reward -= 5.0  # Penalización si no se puede calcular ZMP
        
        # ===== RECOMPENSA POR CONTACTO CON EL SUELO =====
        
        left_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_id, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_id, -1)) > 0
        
        if contacto_pies:
            reward += 2.0  # Ambos pies en el suelo
        elif contacto_pies ==False and self.step_count>500:
            reward -= 10.0  # Penalización severa por estar en el aire
        
        # ===== PENALIZACIÓN POR MOVIMIENTO EXCESIVO =====
        
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        # Solo es apleciable el bamboleo u la bajada
        velocity_penalty = (abs(lin_vel[0])+ abs(ang_vel[2])) * 2.0
        reward -= velocity_penalty
        
        # ===== RECOMPENSA BASE POR SUPERVIVENCIA =====
        
        reward += 1.0  # Recompensa base por cada step exitoso
        
        return reward
    


# ===================================================================================================================================================================== #
# ================================================ Componentes existentes en BIPEDIKPAMENv ============================================================================ #
# ===================================================================================================================================================================== #

                

    def _is_done(self):
        """Condiciones de terminación unificadas"""

        self.pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        if self.pos[2] < 0.4 or self.pos[2] > 3.0:
            if self.contacto_pies is False:
                print("to high or too low", euler, self.pos)
                return True
            
        # Terminar si la inclinación lateral es excesiva  
        #if abs(euler[1]) > math.pi/4 + 0.2 or abs(euler[0]) > math.pi/4 + 0.2:
        #    print("rotated", euler)
        #    return True



            
        # Límite de tiempo
        if self.step_count > 1500*3: # 5 segundos a 1500 Hz
            print("fuera de t")
            return True
            
        return False
    

    def reset(self, seed=None, options=None):
        """
            Reset SIMPLIFICADO - Solo configuración esencial para balance
        """
        super().reset(seed=seed)

        # Actualizar curriculum con rendimiento del episodio anterior
        if self.curriculum and hasattr(self, 'episode_reward'):
            episode_length = self.step_count - self.episode_start_step
            self.curriculum.update_after_episode(self.episode_reward, episode_length)
        
        # ===== RESET FÍSICO BÁSICO =====
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        # En reset():
        self.pam_control_active = False
        self.contact_established = False
        self.contact_stable_steps = 0# Ejemplo, no hace falta ya que puedo usar count steps y ya

        # Configurar solver para mejor estabilidad
        p.setPhysicsEngineParameter(
            numSolverIterations=10,        # Más iteraciones = más estable
            numSubSteps=4,                 # Más substeps = más preciso
            contactBreakingThreshold=0.001, # Umbral de contacto más sensible
            erp=0.8,                       # Error Reduction Parameter
            contactERP=0.9,                # ERP específico para contactos
            frictionERP=0.8,               # ERP para fricción
        )
        
        # Cargar entorno
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            [0, 0, 1.21],  # Posición inicial de pie
            # [0, 0, 0, 1],  # Orientación neutral
            useFixedBase=False
        )
        
        # ===== CONFIGURACIÓN PARA BALANCE ESTÁTICO =====
        
        # Posiciones articulares para estar de pie (balance neutro)
        self.neutral_positions = {
            0:0.0,   # left_hip - neutral
            1:0.0,   # left_knee - ligeramente flexionada para estabilidad
            2:0.0,   # left_anckle. Por si el resorte lo dejo con angulo no nulo
            3:0.0,   # right_hip - neutral  
            4:0.0,   # right_knee - ligeramente flexionada
            5:0.0,   # right_anckle - lo mismo que antes
        }
        
        for i, pos in self.neutral_positions.items():
            p.resetJointState(self.robot_id, i, pos)

        # ===== CONFIGURACIÓN PAM =====

        for i, target_pos in self.neutral_positions.items():  # Solo caderas y rodillas
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.VELOCITY_CONTROL,
                force=0
            )
        
        
        # SIN velocidad inicial - queremos balance estático
        
        # ===== Sistemas de apoyo ===== #
        # Robot data para métricas básicas
        self.robot_data = PyBullet_Robot_Data(self.robot_id)
         # ZMP calculator para estabilidad
        self.zmp_calculator = ZMPCalculator(
            robot_id=self.robot_id,
            left_foot_id=self.left_foot_id,
            right_foot_id=self.right_foot_id,
            robot_data=self.robot_data
        )
        self._configure_contact_friction()
        # Controller para acciones discretas (BALANCE_STANDING, SQUAT)
        self.controller = PhaseAwareEnhancedController(self)
        self.ankle_control = IntelligentAnkleControl(self.robot_id)
        #self.controller.set_action(ActionType.BALANCE_STANDING)  # Empezar con balance
        self.controller.set_action(MovementPhase.STATIC_BALANCE)

        # Configurar sistema de recompensas
        self.reward_system.redefine_robot(self.robot_id, self.plane_id)
        
        # ===== VARIABLES DE SEGUIMIENTO =====
        
        self.total_reward = 0
        self._friction_configured = False  # Flag para configuración de fricción

        # Estabilización inicial con física mejorada
        for _ in range(100):  # Más steps para estabilización
            p.stepSimulation()

        # ===== INICIALIZAR VARIABLES DE CONTROL =====
        self.pam_control_active = False
        self.contact_established = False
        self.contact_stable_steps = 0
        self.step_count = 0
        
        # Inicializar estados PAM en modo neutro
        self.pam_states = {
            'pressures': np.zeros(self.num_active_pams),
            'forces': np.zeros(self.num_active_pams)
        }

        print(f"   🤖 Robot inicializado en modo STANDING POSITION")
        print(f"   🦶 Esperando contacto bilateral para activar PAMs...")
        
        # Estabilización inicial
        #for _ in range(50):
        #    p.stepSimulation()
        
        # Obtener observación inicial
        observation = self._get_simple_observation()
        info = {
                    'episode_reward': 0,
                    'episode_length': 0,
                }
        
        print(f"   🔄 Environment reset - Ready for balance/squat training")
        
        return observation, info
    
    def close(self):
        try:
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
        except:
            pass

    # ===== MÉTODO DE DEBUG ADICIONAL =====

    def _debug_joint_angles_and_pressures(self, pam_pressures):
        """
        ✅ MÉTODO DE DEBUG para verificar la lógica biomecánica
        
        Llama esto ocasionalmente durante el step() para verificar que la lógica funciona
        """
        
        if self.step_count % 1500 == 0:  # Cada segundo aprox
            try:
                joint_states = p.getJointStates(self.robot_id, [1, 4])  # rodillas
                left_knee_angle = joint_states[0][0]
                right_knee_angle = joint_states[1][0]
                
                print(f"\n🔍 Biomechanical Debug (Step {self.step_count}):")
                print(f"   Left knee: {left_knee_angle:.3f} rad ({math.degrees(left_knee_angle):.1f}°)")
                print(f"   Right knee: {right_knee_angle:.3f} rad ({math.degrees(right_knee_angle):.1f}°)")
                print(f"   Left knee flexor pressure: {pam_pressures[4]:.3f}")
                print(f"   Right knee flexor pressure: {pam_pressures[5]:.3f}")
                
                # Verificar lógica biomecánica
                if left_knee_angle > 0.05 and pam_pressures[4] > 0.01:
                    print(f"   ⚠️ Warning: Left knee flexed but flexor active!")
                elif left_knee_angle > 0.05 and pam_pressures[4] <= 0.01:
                    print(f"   ✅ Correct: Left knee flexed, flexor inactive")
                    
                if right_knee_angle > 0.05 and pam_pressures[5] > 0.01:
                    print(f"   ⚠️ Warning: Right knee flexed but flexor active!")
                elif right_knee_angle > 0.05 and pam_pressures[5] <= 0.01:
                    print(f"   ✅ Correct: Right knee flexed, flexor inactive")
            
            except Exception as e:
                print(f"   ❌ Debug error: {e}")

# ===== FUNCIÓN DE USO FÁCIL =====

def create_simple_balance_squat_env(render_mode='human'):
    """
    Crear entorno simplificado para balance y sentadillas
    """
    
    env = Simple_BalanceSquat_BipedEnv(render_mode=render_mode)
    
    print(f"✅ Simple Balance & Squat Environment created")
    print(f"   Focus: Balance de pie + Sentadillas")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Observation space: {env.observation_space.shape}")
    
    return env

def test_simple_balance_env(duration_steps=1000):
    """
    Test básico del entorno simplificado
    """
    
    print("🧪 Testing Simple Balance Environment...")
    
    env = create_simple_balance_squat_env(render_mode='human')
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(duration_steps):
        # Acción aleatoria o acción de balance básica
        if step < 100:
            # Primeros pasos: acción neutra para balance
            action = np.array([0.3, 0.4, 0.3, 0.4, 0.2, 0.2])  # Presiones base
        else:
            # Después: acciones aleatorias suaves
            action = env.action_space.sample() * 0.5 + 0.25  # Suavizar
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 200 == 0:
            print(f"   Step {step}: Reward = {reward:.2f}, Total = {total_reward:.2f}")
        
        if done:
            print(f"   Episode terminado en step {step}")
            break
    
    env.close()
    print(f"🎉 Test completado. Recompensa total: {total_reward:.2f}")
    
    return total_reward


# ===== EJEMPLO DE USO =====

if __name__ == "__main__":
    
    print("🎯 SIMPLE BALANCE & SQUAT ENVIRONMENT")
    print("=" * 50)
    print("Objetivo: Entrenar balance de pie y sentadillas")
    print("Enfoque: Simplificado - Solo lo esencial")
    print("=" * 50)
    
    # Test del entorno
    test_simple_balance_env(duration_steps=500)