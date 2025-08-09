
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
from collections import deque


from Archivos_Apoyo.dinamica_pam import PAMMcKibben
from Archivos_Apoyo.ImproveRewards import ImprovedRewardSystem#, PAMTrainingConfig
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Apoyo.SimplifiedWalkingController import SimplifiedWalkingController
from Archivos_Mejorados.Enhanced_SimplifiedWalkingController import Enhanced_SimplifiedWalkingController
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data

class Enhanced_PAMIKBipedEnv(gym.Env):
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
    
    def __init__(self, render_mode='human', action_space="pam", 
                 num_actors_per_leg=3, num_articulaciones_pierna=2):
        
        """
            Initialize the enhanced PAM biped environment.
            
            Args:
                render_mode: 'human' or 'direct'
                action_space: Only "pam" supported in this version
                enable_curriculum: Enable expert curriculum support
        """
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super(Enhanced_PAMIKBipedEnv, self).__init__()

        self.action_space = action_space
        self.render_mode = render_mode
        self.phase=0
        self.num_actors_per_leg=num_actors_per_leg
        self.num_articulaciones_pierna=num_articulaciones_pierna
        # Conf inicial
        self.generar_simplified_space()
        self.limites_sistema()
        self.variables_seguimiento()
        self.configuracion_simulacion_1()
        
        # Redefinir para 6 actuadores activos
        self.num_active_pams = 6
        self.redefine_pam_system()
        self.setup_passive_elements()
        
        # Actualizar espacios de acción y observación
        self.update_action_observation_spaces()
    
    def redefine_pam_system(self):
        """Reconfigurar el sistema PAM para 6 actuadores antagónicos"""
        
        # Definir los 6 PAMs activos con parámetros optimizados por articulación
        self.pam_muscles = {
            # Caderas - Control antagónico completo
            'left_hip_flexor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),   # Más potente
            'left_hip_extensor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            'right_hip_flexor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            'right_hip_extensor': PAMMcKibben(L0=0.6, r0=0.025, alpha0=np.pi/4),
            
            # Rodillas - Solo flexores (extensión pasiva)
            'left_knee_flexor': PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4),
            'right_knee_flexor': PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4),
        }
        
        # Actualizar límites articulares para el nuevo sistema
        self.joint_limits = {
            'left_hip_joint': (-1.2, 1.0),      # Rango ampliado para flexión/extensión
            'left_knee_joint': (0.0, 1.571),    # Solo flexión (extensión por resorte)
            'right_hip_joint': (-1.2, 1.0),
            'right_knee_joint': (0.0, 1.571),
            'left_ankle_joint': (-0.5, 0.5),    # Controlado por resortes
            'right_ankle_joint': (-0.5, 0.5),
        }
        
        # Estados internos para 6 PAMs
        self.pam_states = {
            'pressures': np.zeros(6),
            'contractions': np.zeros(6),
            'forces': np.zeros(6)
        }
    
    def setup_passive_elements(self):
        """Configurar parámetros de elementos pasivos (resortes)"""
        
        self.passive_springs = {
            # Resortes extensores de rodilla (devuelven la rodilla a posición neutra)
            'left_knee_extensor': {
                'k_spring': 15.0,      # Rigidez del resorte (Nm/rad)
                'rest_angle': 0.1,     # Ángulo de reposo (ligeramente flexionado)
                'damping': 2.0         # Amortiguación
            },
            'right_knee_extensor': {
                'k_spring': 15.0,
                'rest_angle': 0.1, 
                'damping': 2.0
            },
            
            # Resortes de tobillo (estabilización pasiva) # ¿Debería de usarse para el flexor y el extensor?
            'left_ankle_spring': {
                'k_spring': 8.0,       # Más suave para permitir adaptación al terreno
                'rest_angle': 0.0,     # Pie horizontal
                'damping': 1.5
            },
            'right_ankle_spring': {
                'k_spring': 8.0,
                'rest_angle': 0.0,
                'damping': 1.5
            }
        }
    
    def update_action_observation_spaces(self):
        """Actualizar espacios para 6 PAMs activos"""
        
        # Espacio de acción: 6 presiones PAM normalizadas [-1, 1]

        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Observación ampliada: estado base + 6 PAMs + info de resortes
        base_obs_size = 28  # Tu observación base actual
        pam_obs_size = 6    # 6 presiones PAM normalizadas
        spring_obs_size = 4 # 4 fuerzas de resortes pasivos
        
        total_obs_size = base_obs_size + pam_obs_size + spring_obs_size
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )

    # Parece sustituir al antiguo _calculate_pam_forces
    def _calculate_antagonistic_forces(self, pam_pressures, joint_states):
        """
        Calcula fuerzas considerando pares antagónicos en caderas
        """
        forces = []
        muscle_names = list(self.pam_muscles.keys())
        
        # Procesar por pares antagónicos y músculos individuales
        for i, (pressure, joint_state) in enumerate(zip(pam_pressures, joint_states)):
            muscle_name = muscle_names[i]
            pam = self.pam_muscles[muscle_name]
            
            # Obtener información de la articulación
            joint_pos = joint_state[0]
            #joint_vel = joint_state[1]
            #joint_range = self.joint_limits[list(self.joint_limits.keys())[i]]
            
            # Calcular ratio de contracción según el tipo de músculo
            if 'hip' in muscle_name:
                # Para caderas: calcular según si es flexor o extensor
                if 'flexor' in muscle_name:
                    # Flexor se activa con ángulos positivos (flexión)
                    normalized_pos = max(0, joint_pos) / 1.2  # Normalizar a rango [0,1]
                else:  # extensor Ver si hay que cambiarlo para x, y, z
                    # Extensor se activa con ángulos negativos (extensión)
                    normalized_pos = max(0, -joint_pos) / 1.2
                    
            elif 'knee' in muscle_name:
                # Para rodillas: solo flexor, se activa con flexión
                joint_range = self.joint_limits[f"{'left' if 'left' in muscle_name else 'right'}_knee_joint"]
                normalized_pos = (joint_pos - joint_range[0]) / (joint_range[1] - joint_range[0])
            
            contraction_ratio = np.clip(normalized_pos, 0, 0.8)
            
            # Calcular fuerza PAM
            pam_force = pam.force_model_new(pressure, contraction_ratio)
            
            # Aplicar dirección según tipo de músculo
            if 'extensor' in muscle_name:
                pam_force = -pam_force  # Los extensores aplican fuerza negativa
            
            forces.append(pam_force)
            
            # Actualizar estados internos
            self.pam_states['pressures'][i] = pressure
            self.pam_states['contractions'][i] = contraction_ratio
            self.pam_states['forces'][i] = pam_force
        
        return forces
    
    def _calculate_joint_torques_from_pam_forces(self, pam_forces):
        """
        Convierte fuerzas PAM individuales en torques articulares netos
        considerando pares antagónicos
        """
        joint_torques = []
        
        # Cadera izquierda: combinar flexor y extensor
        left_hip_torque = pam_forces[0] + pam_forces[1]  # flexor + extensor (extensor ya es negativo)
        joint_torques.append(left_hip_torque)
        
        # Rodilla izquierda: solo flexor
        left_knee_torque = pam_forces[4]
        joint_torques.append(left_knee_torque)
        
        # Cadera derecha: combinar flexor y extensor  
        right_hip_torque = pam_forces[2] + pam_forces[3]
        joint_torques.append(right_hip_torque)
        
        # Rodilla derecha: solo flexor
        right_knee_torque = pam_forces[5]
        joint_torques.append(right_knee_torque)
        
        return joint_torques
    
    def _apply_passive_spring_torques(self):
        """
        Aplica torques de resortes pasivos en rodillas y tobillos
        """
        # Obtener estados articulares actuales
        joint_states = p.getJointStates(self.robot_id, range(6))  # 6 articulaciones total
        
        # Resortes extensores de rodillas (índices 1 y 4)
        for i, knee_id in enumerate([1, 4]):  # left_knee=1, right_knee=4
            joint_pos = joint_states[knee_id][0]
            joint_vel = joint_states[knee_id][1]
            
            side = 'left' if i == 0 else 'right'
            spring_params = self.passive_springs[f'{side}_knee_extensor']
            
            # Torque de resorte: k * (rest_angle - current_angle) - damping * velocity
            spring_force = (spring_params['k_spring'] * 
                          (spring_params['rest_angle'] - joint_pos) - 
                          spring_params['damping'] * joint_vel)
            
            p.setJointMotorControl2(
                self.robot_id,
                knee_id,
                p.TORQUE_CONTROL,
                force=spring_force
            )
        
        # Resortes de tobillos (índices 2 y 5) - ya implementado en _apply_ankle_spring
        self._apply_ankle_spring(k_spring=8.0)
    
    def _apply_enhanced_control(self, action):
        """
        Control mejorado con 6 PAMs activos + elementos pasivos
        """
        action = np.asarray(action, dtype=np.float32)
        
        # Convertir acciones a presiones PAM (de [-1,1] a presiones reales)
        pam_pressures = (action + 1.0) / 2.0 * self.max_pressure
        
        # Obtener estados articulares (solo las 4 articulaciones activas)
        joint_states_pam = p.getJointStates(self.robot_id, [0,1,3,4])
        
        # Calcular fuerzas PAM considerando antagonismo
        pam_forces = self._calculate_antagonistic_forces(pam_pressures, joint_states_pam)
        
        # Convertir a torques articulares netos
        joint_torques = self._calculate_joint_torques_from_pam_forces(pam_forces)
        
        # Aplicar torques a las articulaciones activas
        for i, torque in enumerate(joint_torques):
            p.setJointMotorControl2(
                self.robot_id,
                i,  # joint indices 0-3 (hip_left, knee_left, hip_right, knee_right)  
                p.TORQUE_CONTROL,
                force=torque
            )
        
        # Aplicar elementos pasivos
        self._apply_passive_spring_torques()
        
        return joint_torques

    def set_training_phase(self, phase, phase_timesteps):
        """Configurar fase de entrenamiento para currículo experto"""
        self.phase = phase
        self.total_phase_timesteps = phase_timesteps
        
        if phase == 0:
            # Fase de equilibrio básico
            self.use_walking_cycle = False
            self.walking_controller = None
            self.imitation_weight = 0.0
        elif phase == 1:
            # Fase de imitación con walking cycle
            self.use_walking_cycle = True
            if hasattr(self, 'walking_controller') and self.walking_controller:
                self.walking_controller.mode = "pressure"
            self.imitation_weight = 1.0
        elif phase == 2:
            # Fase de exploración guiada
            self.use_walking_cycle = True
            self.imitation_weight = 0.3
        else:
            # Fases avanzadas - RL puro
            self.use_walking_cycle = False
            self.walking_controller = None
            self.imitation_weight = 0.0
        
        # Actualizar sistema de recompensas
        if hasattr(self, 'sistema_recompensas'):
            self.sistema_recompensas.set_curriculum_phase(phase)
    
    def _enhanced_observation(self):
        """
        Observación mejorada incluyendo estados de 6 PAMs + resortes pasivos
        """
        # Obtener observación base (28 elementos)
        base_obs = self._stable_observation()
        
        # Estados PAM (6 elementos): presiones normalizadas
        pam_obs = []
        if hasattr(self, 'pam_states'):
            normalized_pressures = self.pam_states['pressures'] / self.max_pressure
            pam_obs.extend(normalized_pressures.tolist())
        else:
            pam_obs.extend([0.0] * 6)
        
        # Estados de resortes pasivos (4 elementos): fuerzas normalizadas
        spring_obs = []
        try:
            joint_states = p.getJointStates(self.robot_id, [1, 4, 2, 5])  # rodillas + tobillos
            
            for i, joint_state in enumerate(joint_states):
                joint_pos = joint_state[0]
                if i < 2:  # rodillas
                    side = 'left' if i == 0 else 'right'
                    spring_params = self.passive_springs[f'{side}_knee_extensor']
                    spring_force = spring_params['k_spring'] * (spring_params['rest_angle'] - joint_pos)
                    normalized_force = np.clip(spring_force / 20.0, -1.0, 1.0)  # Normalizar
                else:  # tobillos  
                    side = 'left' if i == 2 else 'right'
                    spring_params = self.passive_springs[f'{side}_ankle_spring']
                    spring_force = spring_params['k_spring'] * (spring_params['rest_angle'] - joint_pos)
                    normalized_force = np.clip(spring_force / 10.0, -1.0, 1.0)
                
                spring_obs.append(normalized_force)
        except:
            spring_obs = [0.0] * 4

        # Información ZMP (4 elementos)
        zmp_info = [0.0, 0.0, 0.0, 0.0]
        if self.zmp_calculator:
            try:
                zmp_point = self.zmp_calculator.calculate_zmp()
                is_stable = self.zmp_calculator.is_stable(zmp_point)
                margin = self.zmp_calculator.stability_margin_distance(zmp_point)
                
                zmp_info = [
                    zmp_point[0], zmp_point[1], 
                    float(is_stable), 
                    np.clip(margin, -1.0, 1.0)
                ]
            except:
                pass
        
        # Combinar todas las observaciones
        enhanced_obs = np.concatenate([base_obs, pam_obs, spring_obs, zmp_info])
        
        return enhanced_obs
    
    def step(self, action):
        """Step sobrescrito para usar el control mejorado"""
        self.step_count += 1

        # Combinar acción del ciclo con RL si está habilitado
        if self.walking_controller and self.use_walking_cycle:
            base_action = self.walking_controller.get_enhanced_walking_actions(self.time_step)
            modulation_factor = 0.3 if self.imitation_weight > 0 else 1.0
            final_action = self._safe_blend_actions(base_action, action, modulation_factor)
        else:
            final_action = action
        
        # Aplicar control mejorado con 6 PAMs
        joint_torques = self._apply_enhanced_control(final_action)
        
        # Simular
        p.stepSimulation()
        
        # Obtener observación mejorada
        observation = self._enhanced_observation()
        
        # Calcular recompensa (usar el sistema existente)
        reward, _ = self.sistema_recompensas._calculate_balanced_reward(
            action=action, 
            pam_forces=joint_torques  # Pasar los torques netos
        )

        # Reward shaping por imitación si está habilitado
        if self.imitation_weight > 0 and self.walking_controller:
            expert_action = self.walking_controller.get_enhanced_walking_actions(self.time_step)
            if expert_action is not None:
                imitation_penalty = np.linalg.norm(final_action - expert_action)
                reward -= self.imitation_weight * imitation_penalty
        
        # ZMP y otras métricas (código existente)
        if self.zmp_calculator is not None:
            zmp_reward, self.zmp_history, self.max_zmp_history, \
            self.stability_bonus, self.instability_penalty, self.zmp_reward_weight \
            = self.zmp_calculator._calculate_zmp_reward(
                self.zmp_history, self.max_zmp_history, 
                self.stability_bonus, self.instability_penalty, self.zmp_reward_weight
            )
            reward += zmp_reward
        
        self.total_reward += reward

        # Actualizar estado
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.previous_position = current_pos
        
        # Info de estado
        info = {
            'episode_reward': self.total_reward,
            'episode_length': self.step_count,
            'control_mode': 'enhanced_pam_6',
            'num_active_pams': self.num_active_pams,
            'pam_pressures': self.pam_states['pressures'].tolist(),
            'joint_torques': joint_torques
        }

        # Añadir información ZMP si está disponible
        if self.zmp_calculator and self.zmp_history:
            latest_zmp = self.zmp_history[-1]
            info.update({
                'zmp_stable': latest_zmp['stable'],
                'zmp_margin': latest_zmp['margin'],
                'zmp_reward': zmp_reward,
                'zmp_position': latest_zmp['zmp'].tolist()
            })

        # Recompensa adicional basada en estabilidad del robot
        stability = self.robot_data.get_stability_metrics
        com_height = stability['com_height']
        stable = stability['is_stable']
        reward += 1.0 if stable else -5.0
        reward += max(0, com_height - 0.4)
        
        done = self._is_done()
        
        return observation, reward, done, False, info
    
##############################################################################################################################################################
############################################################## Generacion Variables constantes ###############################################################
##############################################################################################################################################################

    def generar_simplified_space(self):
        """
            Genera el espacio de acción simplificado 
            y las propiedades del entorno.
        """
        self.urdf_path = "2_legged_human_like_robot.urdf"
        self.time_step = 1.0 / 1500.0

        # Variables específicas de IK
        self.last_ik_success = False
        self.last_ik_error = 0.0
        self.last_hybrid_weight = 0.5
        # Número total de joints articulados que usan PAMS
        # Control mode siempre PAM para enhanced version
        self.control_mode = "pam"
        self.num_joints = 4

    def configuracion_simulacion_1(self):
        """MODIFICADO: IDs de articulaciones corregidos"""
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            import threading
            self._thread_id = threading.get_ident()
            self.physics_client = p.connect(p.DIRECT)
        # IDs corregidos para el robot bípedo
        self.left_hip_id = 0
        self.left_knee_id = 1
        self.left_foot_id = 2   # ID Pie izquierdo End effector pie izquierdo
        self.left_ankle_id=self.left_foot_id

        self.right_hip_id = 3
        self.right_knee_id = 4
        self.right_foot_id = 5  # ID pie derecho End effector pie derecho
        self.right_ankle_id=self.right_foot_id
        
        # Variables de tracking mejoradas
        self.velocity_history = deque(maxlen=10)
        self.previous_action = None
        

        # Fuerzas específicas para cada articulación (en Newtons)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # IDs de end-effectors (enlaces de pies)
        

        # Número total de joints articulados
        self.sistema_recompensas=ImprovedRewardSystem(self.left_foot_id, 
                                                    self.right_foot_id,
                                                    self.num_joints,
                                                    None, # Se inicia después
                                                    num_pams=6)

        # Límites más amplios
        self.foot_workspace = {
            'x_range': (-1.2, 1.2),    # Adelante/atrás
            'y_range': (-0.8, 0.8),    # Izquierda/derecha 
            'z_range': (-0.3, 1.0)     # Altura del suelo
        }

    def limites_sistema(self):
        """Define los límites del sistema y las propiedades de los músculos PAM."""
        self.min_pressure = 101325 # 1 atm: Suponemos que esa es la presión ambiente
        # Parámetros de control PAM
        self.max_pressure = 500000+ self.min_pressure  # 5 bar en Pa
        # Es esta variable realmente necesaria
        self.joint_forces = {
            'left_hip_joint': 150,    # Cadera necesita más fuerza
            'left_knee_joint': 120,   # Rodilla fuerza moderada
            'right_hip_joint': 150,
            'right_knee_joint': 120,
        }
        self.joint_force_array = [150, 120, 150, 120]
        self.imitation_weight = 1.0

    def variables_seguimiento(self):
        # Variables para seguimiento de rendimiento
        self.step_count = 0
        self.total_reward = 0
        self.previous_position = None
        self.previous_velocity = None

        # Parámetros mejorados para LSTM
        self.history_length = 5  # Mantener historial de 5 observaciones pasadas
        self.observation_history = deque(maxlen=self.history_length)
        
        # Variables para recompensas
        self.previous_contacts = [False, False]
        self.previous_action = None
        
        # IDs de objetos (se inicializan en reset)
        self.robot_id = None
        self.plane_id = None

        self.zmp_calculator = None
        self.zmp_history = []
        self.max_zmp_history = 20
        
        # Parámetros para recompensas ZMP
        self.zmp_reward_weight = 0.2
        self.stability_bonus = 20.0
        self.instability_penalty = -15.0

        # Añadir al final:
        self.walking_controller = None
        self.use_walking_cycle = True


# ===================================================================================================================================================================== #
# ================================================ Componentes existentes en BIPEDIKPAMENv ============================================================================ #
# ===================================================================================================================================================================== #

    def _setup_motors_for_force_control(self):
        """Configura motores para control de fuerza"""
        if self.robot_id is not None:
            # 4 articulaciones
            for i in [0, 1, 3, 4]:  # Caderas y rodillas
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    force=0  # Desactivar motores por defecto
                )

    def _safe_blend_actions(self, base_action, action, modulation_factor=1.0):
        """Combina de forma segura acciones de base y RL"""
        base_action = np.asarray(base_action)
        action = np.asarray(action)

        # Si shapes coinciden, suma directa
        if base_action.shape == action.shape:
            return base_action + modulation_factor * action

        # Si base_action tiene menos dimensiones
        if base_action.size < action.size:
            new_base = np.zeros_like(action)
            new_base[-base_action.size:] = base_action
            return new_base + modulation_factor * action

        # Si base_action tiene más dimensiones
        if base_action.size > action.size:
            base_trimmed = base_action[-action.size:]
            return base_trimmed + modulation_factor * action

        raise ValueError(f"Incompatible shapes: {base_action.shape}, {action.shape}")

    def _apply_simplified_control(self, action):
        """Control PAM simple - MÉTODO FALTANTE (adaptado para 6 PAMs)"""
        action = np.asarray(action, dtype=np.float32)
        
        # Convertir acciones normalizadas a presiones PAM
        pam_pressures = (action + 1.0) / 2.0 * self.max_pressure
        
        # Obtener estados articulares (solo articulaciones activas)
        joint_states_pam = p.getJointStates(self.robot_id, [0, 1, 3, 4])  # caderas y rodillas
        
        # Calcular y aplicar fuerzas PAM
        pam_forces = self._calculate_antagonistic_forces(pam_pressures, joint_states_pam)
        joint_torques = self._calculate_joint_torques_from_pam_forces(pam_forces)
        
        # Aplicar torques
        for i, torque in enumerate(joint_torques):
            joint_idx = [0, 1, 3, 4][i]  # Mapear a índices reales
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torque
            )
        
        # Aplicar elementos pasivos
        self._apply_passive_spring_torques()
        
        return joint_torques

    def _validate_joint_limits(self, joint_positions):
        """Verifica que las posiciones articulares estén dentro de límites."""
        active_joints = [0, 1, 3, 4]
        #antes joint positions de todos: joint_positions[0, 1, 3, 4]
        joint_names = ['left_hip_joint', 'left_knee_joint', 'right_hip_joint', 'right_knee_joint']
        
        for i, joint_id in enumerate(active_joints):  # Solo caderas y rodillas
            if i < len(joint_positions) and i<len(joint_id):
                joint_name = joint_names[i]
                pos = joint_positions[i]
                low, high = self.joint_limits[joint_name]
                if pos < low or pos > high:
                    return False
                
        return True
                

    def _is_done(self):
        """Condiciones de terminación unificadas"""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, _ = p.getBaseVelocity(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        if pos[2] < 0.5:
            print("down", euler, pos)
            return True
            
        # Terminar si la inclinación lateral es excesiva  
        if abs(euler[1]) > math.pi/2 + 0.2:
            print("rotated", euler)
            return True
        
        # Velocidad hacia atrás prolongada
        if lin_vel[0] < 0.0 and pos[0] <= 0.0 and self.step_count > 1000:
            self.repeat += 1
            #print("repeat", self.repeat, lin_vel, pos)
            if self.repeat > 200:
                print("marcha atrás", lin_vel, pos)
                return True
        else:
            self.repeat=0
            
        # Terminar si se sale del área
        if pos[0] > 1000 or pos[0] < -2.0 or abs(pos[1]) > 20:
            print("Fuera del area")
            return True
        
        # Nueva condición: ZMP fuera del polígono por mucho tiempo
        if self.zmp_calculator and len(self.zmp_history) >= 10:
            recent_unstable = sum(1 for entry in self.zmp_history[-10:] 
                                if not entry['stable'])
            
            # Si 8 de los últimos 10 steps son inestables
            print(recent_unstable)
            if recent_unstable >= 8:
                print("nivel inestabilidad",recent_unstable)
                return True
            
        # Límite de tiempo
        if self.step_count > 9000:
            print("fuera de t")
            return True
            
        return False
    

    def reset(self, seed=None, options=None):
        """Reset unificado para ambos entornos (base e híbrido PAM)."""
        super().reset(seed=seed)
        
        self.setup_reset_simulation

        # Dentro de PAMIKBipedEnv.reset()
        self.robot_data = PyBullet_Robot_Data(self.robot_id)
        self.zmp_calculator = ZMPCalculator(
            robot_id=self.robot_id,
            left_foot_id=self.left_foot_id,
            right_foot_id=self.right_foot_id,
            robot_data=self.robot_data
        )

        # (Opcional) Inclinación inicial hacia adelante
        if self.use_walking_cycle:
            # Usar enhanced walking controller para 6 PAMs
            self.walking_controller = Enhanced_SimplifiedWalkingController(self)
            self.walking_controller.reset()

        self.sistema_recompensas.redefine_robot(self.robot_id, self.plane_id)
        
        # Configurar propiedades físicas
        

        # Configurar posición inicial de articulaciones para caminar
        initial_joint_positions = [
            0.00,   # left_hip - ligeramente flexionado
            0.00,  # left_knee - flexionado
            0.00,  # right_hip - extendido
            0.00,  # right_knee - ligeramente flexionado
        ]

        self.esperando_contacto = True
    
        for i, pos in enumerate(initial_joint_positions):
            p.resetJointState(self.robot_id, i, pos)

        p.resetBaseVelocity(self.robot_id, [0.08, 0, 0], [0, 0, 0])
        
        # Configurar control de fuerza (solo para PAM)
        if hasattr(self, 'pam_muscles'):
            self._setup_motors_for_force_control()
            # Resetear estados PAM
            self.pam_states = {
                'pressures': np.zeros(6),
                'contractions': np.zeros(6),
                'forces': np.zeros(6)
            }
        
        # Resetear variables de seguimiento comunes
        self.step_count = 0
        self.total_reward = 0
        self.observation_history.clear()
        self.previous_position = [0,0,1.2]
        self.previous_contacts = [False, False]
        self.previous_action = None
        self.repeat = 0
        
        self.zmp_history.clear()

        observation = self._stable_observation()

        info = {'episode_reward': 0, 'episode_length': 0}

        return observation, info
    
    def close(self):
        try:
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
        except:
            pass

    @property
    def setup_reset_simulation(self):
        """
            Configuración inicial del robot y simulación al reiniciar el entorno.
            - Reinicia la simulación y establece gravedad.
            Da cierta randomización para fortalecer el modelo
        """
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSubSteps=4)
        

        # Cargar entorno con fricción random
        random_friction = np.random.uniform(0.8, 1.1)
        correction_quaternion = p.getQuaternionFromEuler([0, 0, 0])
        self.previous_position = [0,0,1.2]
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=random_friction)
        self.robot_id = p.loadURDF(
            self.urdf_path,
            self.previous_position,
            correction_quaternion,
            useFixedBase=False
        )
        self.setup_physics_properties()

        # Esperar a que la simulación se estabilice
        for _ in range(20):
            p.stepSimulation()
        self.repeat = 0
        # Randomización de masa en links del robot
        # Esto ayuda a robustecer el modelo ante variaciones físicas
        for link_id in range(p.getNumJoints(self.robot_id)):
            orig_mass = p.getDynamicsInfo(self.robot_id, link_id)[0]
            rand_mass = orig_mass * np.random.uniform(0.8, 1.1)
            p.changeDynamics(self.robot_id, link_id, mass=rand_mass)

    def _stable_observation(self):
        """Observación base estable - MÉTODO FALTANTE"""
        obs = []
        
        try:
            # Estado del torso (8 elementos)
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # Sanitizar valores
            pos = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in pos]
            euler = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in euler]
            lin_vel = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in lin_vel]
            ang_vel = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in ang_vel]
            
            obs.extend([pos[0], pos[2], euler[0], euler[1], 
                       lin_vel[0], lin_vel[1], ang_vel[0], ang_vel[1]])
            
            # Estados articulares (12 elementos - 6 articulaciones)
            joint_states = p.getJointStates(self.robot_id, range(6))
            for state in joint_states:
                pos_val = state[0] if not (math.isnan(state[0]) or math.isinf(state[0])) else 0.0
                vel_val = state[1] if not (math.isnan(state[1]) or math.isinf(state[1])) else 0.0
                obs.extend([pos_val, vel_val])
            
            # Contactos y tiempo (4 elementos)
            left_contact = len(p.getContactPoints(self.robot_id, 0, self.left_foot_id)) > 0
            right_contact = len(p.getContactPoints(self.robot_id, 0, self.right_foot_id)) > 0
            # Antes era 200 pero step_count es 1/1500 por lo que para normalizarlo deberái de ser 1500
            # Normalizar el tiempo al rango [0, 1] para 1500 pasos
            normalized_time = (self.step_count % 1500) / 1500.0
            obs.extend([float(left_contact), float(right_contact), 
                       normalized_time, float(self.step_count > 0)])
            
            # Estados PAM básicos (4 elementos - compatibilidad)
            if hasattr(self, 'pam_states'):
                # Usar solo los primeros 4 para compatibilidad con observación base
                # ¿Por qué no se pueden usar los seis?
                normalized_pressures = self.pam_states['pressures'][:4] / self.max_pressure
                obs.extend(normalized_pressures.tolist())
            else:
                obs.extend([0.0] * 4)
            
        except Exception as e:
            print(f"Error en observación: {e}")
            obs = [0.0] * 28
        
        return np.array(obs, dtype=np.float32)
    

    def setup_physics_properties(self):
        """Configurar propiedades físicas de fricción para mayor estabilidad"""
        # Aumentar fricción en los pies
        p.changeDynamics(self.robot_id, self.left_foot_id,
                        lateralFriction=1.0,
                        restitution=0.0)
        p.changeDynamics(self.robot_id, self.right_foot_id,
                        lateralFriction=1.0,
                        restitution=0.0)
        
    def set_walking_cycle(self, enabled=True):
        """Activar/desactivar el ciclo de paso"""
        self.use_walking_cycle = enabled
        if not enabled:
            self.walking_controller = None


    def _apply_joint_forces(self, forces):
        """
            Aplica fuerzas tensoriales a las articulaciones usando PyBullet
        """
        active_joints = [0, 1, 3, 4]  # caderas y rodillas
        for i, force in enumerate(forces):
            if i < len(active_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    active_joints[i],
                    p.TORQUE_CONTROL,
                    force=force
                )


    def _apply_ankle_spring(self, k_spring=8.0):
        """
        Aplica un torque de resorte pasivo en los tobillos izquierdo y derecho
        para mantener los pies paralelos al suelo pero permitiendo movimiento.
        Llama a esta función en cada paso antes de p.stepSimulation().
        k_spring: rigidez del resorte [Nm/rad]
        """
        # Asegúrate de que los nombres/índices son correctos
        for ankle_joint in [2, 5]:  # left_ankle=2, right_ankle=5
            theta = p.getJointState(self.robot_id, ankle_joint)[0]
            torque = -k_spring * theta
            p.setJointMotorControl2(
                self.robot_id,
                ankle_joint,
                controlMode=p.TORQUE_CONTROL,
                force=torque
            )
    # ¿Debería de crearse un set training phase?
    

# ===================================================================================================================================================================== #
# =================================================Pruebo el testeo==================================================================================================== #
# ===================================================================================================================================================================== #


def test_6pam_system():
    """Script de prueba para verificar que el sistema de 6 PAMs funciona"""
    
    print("🔧 Testing Enhanced PAM System (6 actuators)")
    
    # Crear entorno de prueba
    env = Enhanced_PAMIKBipedEnv(render_mode='human', action_space="pam")
    
    # Test básico
    obs, info = env.reset()
    print(f"✅ Environment created successfully")
    print(f"   - Action space: {env.action_space.shape}")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Active PAMs: {env.num_active_pams}")
    
    # Test de acciones aleatorias
    for step in range(100):
        action = env.action_space.sample()  # Acción aleatoria de 6 dimensiones
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"   Step {step}: Reward = {reward:.2f}, PAM pressures = {info['pam_pressures']}")
        
        if done:
            print(f"   Episode ended at step {step}")
            obs, info = env.reset()
    
    env.close()
    print("🎉 Test completed successfully!")