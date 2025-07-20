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
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data

class PAMIKBipedEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode='human', action_space="hybrid"):
        """
            Entorno de robot bípedo con control híbrido IK + control directo.
    
            El espacio de acciones se expande para incluir:
            - 6 coordenadas objetivo para pies (x,y,z para cada pie) - IK
            - 6 ajustes finos articulares - Control directo
            - 1 peso híbrido (qué tanto usar IK vs control directo)
        """
        super(PAMIKBipedEnv, self).__init__()

        self.action_space=action_space
        self.render_mode = render_mode

        self.generar_simplified_space

        # conf simulacion
        self.configuracion_simulacion_1

        #p.setPhysicsEngineParameter(solverResidualThreshold=1e-6)
        #p.setPhysicsEngineParameter(numSolverIterations=200)

        # limites sistema
        self.limites_sistema
        
        base_obs_size = 28  # Expandido para incluir más información Paso de 34 a 26 ver si serían 28 (elimino 4 de PAM y de IK)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_obs_size,), 
            dtype=np.float32
        )

        # Variables_seguimiento

        self.variables_seguimiento

        print(f"Observation space: {self.observation_space.shape}")
    
    def _setup_motors_for_force_control(self):
        """Configura motores para control de fuerza"""
        if self.robot_id is not None:
            for i in range(4):  # 6 articulaciones
                p.setJointMotorControl2(
                    self.robot_id,
                    i,
                    p.VELOCITY_CONTROL,
                    force=0  # Desactivar motores por defecto
                )
        
    
    def _validate_joint_limits(self, joint_positions):
        """Verifica que las posiciones articulares estén dentro de límites."""
        joint_names = list(self.joint_limits.keys())
        
        for i, pos in enumerate(joint_positions[:4]):
            if i < len(joint_names):
                joint_name = joint_names[i]
                low, high = self.joint_limits[joint_name]
                if pos < low or pos > high:
                    return False
        return True
    
    def step(self, action):
        """Step simplificado con recompensas positivas"""
        self.step_count += 1

        # NUEVO: Combinar acción del ciclo con RL
        if self.walking_controller and self.use_walking_cycle:
            base_action = self.walking_controller.get_next_action()
            modulation_factor = 0.3 if self.imitation_weight > 0 else 1.0
            final_action=_safe_blend_actions(base_action, action, modulation_factor)
        else:
            final_action = action
        
        # 1. VALIDAR ACCIÓN IK (si es control híbrido)
        # Aplicar control según el modo
        if self.control_mode == "hybrid" and len(action) >= 8:
            #print("HYBRID_PAM")
            foot_targets = final_action[:4] * 0.3  # Escalar a workspace
            if not self._validate_foot_targets(foot_targets):
                # Aplicar penalización por targets inválidos
                reward = -10.0
                observation = self._stable_observation()
                return observation, reward, True, False, {'invalid_targets': True}
        # 2. APLICAR CONTROL (IK o PAM según acción)
            pam_forces = self._apply_hybrid_ik_control(action)
        else:
            # Control PAM simple
            print("PAM_SIMPLE")
            pam_forces = self._apply_simplified_control(action)

        # 3. Simular
        p.stepSimulation()
        
        # Obtener observación y recompensa
        observation = self._stable_observation()
        reward, _ = self.sistema_recompensas._calculate_balanced_reward(action=action, pam_forces=pam_forces)

        # --------- REWARD SHAPING POR IMITACIÓN ---------
        if self.imitation_weight > 0 and self.walking_controller:
            expert_action = self.walking_controller.get_expert_action()
            expert_action = np.asarray(expert_action)
            action = np.asarray(action)
            if expert_action.shape[0] != action.shape[0]:
                # Por ejemplo, compara solo PAM:
                imitation_penalty = np.linalg.norm(action[-expert_action.shape[0]:] - expert_action)
            else:
                imitation_penalty = np.linalg.norm(action - expert_action)
            reward -= self.imitation_weight * imitation_penalty
        
        # 5. ACTUALIZAR TRACKING IK (si aplica)
        if self.control_mode == "hybrid" and len(action) >= 8:
            self._update_ik_tracking()
        # 6. ACTUALIZAR Estado
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.previous_position = current_pos
        
        info={
            'episode_reward': self.total_reward,
            'episode_length': self.step_count,
            'control_mode': self.control_mode
        }

        # Añado parámetros ZMP
        if self.zmp_calculator is None:
        # Añadir recompensa ZMP
            zmp_reward, self.zmp_history, self.max_zmp_history, \
            self.stability_bonus, self.instability_penalty, self.zmp_reward_weight\
            = self.zmp_calculator._calculate_zmp_reward(self.zmp_history, self.max_zmp_history, 
            self.stability_bonus, self.instability_penalty, self.zmp_reward_weight)
            reward = reward + zmp_reward
            self.total_reward += reward
        
        # Actualizar info
        if self.zmp_calculator and self.zmp_history:
            latest_zmp = self.zmp_history[-1]
            info.update({
                'zmp_stable': latest_zmp['stable'],
                'zmp_margin': latest_zmp['margin'],
                'zmp_reward': zmp_reward,
                'zmp_position': latest_zmp['zmp'].tolist()
            })

        # Obtener métricas del robot
        stability = self.robot_data.get_stability_metrics
        com_height = stability['com_height']
        stable = stability['is_stable']

        # Recompensa adicional basada en estabilidad
        reward += 1.0 if stable else -5.0
        reward += max(0, com_height - 0.4)  # Favorecer que se mantenga erguido

        done = self._is_done()
        
        return observation, reward, done, False, info
    
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
        self.walking_controller = SimplifiedWalkingController(self)
        self.walking_controller.reset()

        self.sistema_recompensas.redefine_robot(self.robot_id, self.plane_id)
        
        # Configurar propiedades físicas
        

        # Configurar posición inicial de articulaciones para caminar
        initial_joint_positions = [
            -0.00,   # left_hip - ligeramente flexionado
            0.01,  # left_knee - flexionado
            0.01,  # right_hip - extendido
            -0.00,  # right_knee - ligeramente flexionado
        ]
    
        for i, pos in enumerate(initial_joint_positions):
            p.resetJointState(self.robot_id, i, pos)

        # Impulso inicial hacia adelante De momento lo dejo nulo en caso de que sea la fuente de desequilibrios
        p.resetBaseVelocity(self.robot_id, [0.05, 0, 0], [0, 0, 0])
        
        # Configurar control de fuerza (solo para PAM)
        if hasattr(self, 'pam_muscles'):
            self._setup_motors_for_force_control()
            # Resetear estados PAM
            self.pam_states = {
                'pressures': np.zeros(4),
                'contractions': np.zeros(4),
                'forces': np.zeros(4)
            }
        
        # Resetear variables de seguimiento comunes
        self.step_count = 0
        self.total_reward = 0
        self.observation_history.clear()
        self.previous_position = [0,0,1.2]
        self.ik_success_rate.clear()
        
        # Variables específicas de IK
        self.last_ik_success = False
        self.last_ik_error = 0.0
        self.last_hybrid_weight = 0.5
        self.previous_contacts = [False, False]
        self.previous_action = None

        # Inicializar calculador ZMP después de cargar el robot
        if self.robot_id is not None:
            self.zmp_calculator = ZMPCalculator(
                self.robot_id, 
                self.left_foot_id, 
                self.right_foot_id,
                robot_data=self.robot_data
            )
            self.zmp_history.clear()

        # Permitir estabilización inicial - ejecutar varios steps sin evaluar ZMP
            for _ in range(15):  # Más pasos de estabilización
                p.stepSimulation()
        #if self.use_walking_cycle:
        #    self.walking_controller = SimplifiedWalkingController(self)
        # Obtener observación (compatible con ambos entornos)
        observation = self._stable_observation()
        info = {'episode_reward': 0, 'episode_length': 0}
        if self.use_walking_cycle:
            self.walking_controller = SimplifiedWalkingController(self)
        return observation, info
    
    def close(self):
        try:
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
        except:
            pass

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
        
########################################################################################################################
#########################################Sección PAM####################################################################
########################################################################################################################

    def _apply_joint_forces(self, forces):
        """
        Aplica fuerzas tensoriales a las articulaciones usando PyBullet
        """
        for i, force in enumerate(forces):
            torque = force
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.TORQUE_CONTROL,
                force=torque
            )

    def _calculate_pam_forces(self, pressures, joint_states):
        """
            Calcula fuerzas de músculos PAM basado en presiones y estados articulares
        """
        muscle_names = list(self.pam_muscles.keys())
        forces = []
        
        for i, (pressure, joint_state) in enumerate(zip(pressures, joint_states)):
            muscle_name = muscle_names[i]
            pam = self.pam_muscles[muscle_name]
            
            # Calcular ratio de contracción basado en posición articular
            joint_pos = joint_state[0]  # Posición actual
            joint_range = self.joint_limits[list(self.joint_limits.keys())[i]]
            
            # Normalizar posición articular a ratio de contracción (0-1)
            normalized_pos = (joint_pos - joint_range[0]) / (joint_range[1] - joint_range[0])
            contraction_ratio = np.clip(normalized_pos, 0, 0.8)  # Límite seguro
            
            # Calcular fuerza del músculo PAM
            pam_force = pam.force_model_new(pressure, contraction_ratio)
            forces.append(pam_force)
            
            # Actualizar estado interno
            self.pam_states['pressures'][i] = pressure
            self.pam_states['contractions'][i] = contraction_ratio
            self.pam_states['forces'][i] = pam_force
        return forces

    def _simplified_action_space(self, shape=6):
        """Simplificar el espacio de acción para facilitar el aprendizaje"""
        # Solo 6 dimensiones: presiones PAM directas
        action_space = spaces.Box(low=-1.0, high=1.0, 
                                    shape=(shape,), 
                                    dtype=np.float32)
        return action_space
        
    def _apply_simplified_control(self, action):
        """Control simplificado solo con PAM"""
        action=np.asarray(action, dtype=np.float32)
        # Convertir acciones normalizadas a presiones PAM
        if len(action)==12:
            pam_pressures = (action[4:] + 1.0) / 2.0 * self.max_pressure
        else:
            pam_pressures = (action+ 1.0) / 2.0 * self.max_pressure
        # Obtener estados articulares
        joint_states_pam = p.getJointStates(self.robot_id, range(4))
        
        # Calcular y aplicar fuerzas PAM
        pam_forces = self._calculate_pam_forces(pam_pressures, joint_states_pam)
        self._apply_joint_forces(pam_forces)

    def _stable_observation(self):
        """Observación simplificada y estable para LSTM"""
        obs = []
        
        try:
            # Estado del torso (8 elementos) - más info temporal
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # Sanitizar valores
            pos = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in pos]
            euler = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in euler]
            lin_vel = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in lin_vel]
            ang_vel = [x if not (math.isnan(x) or math.isinf(x)) else 0.0 for x in ang_vel]
            
            # Incluir más info de velocidad angular para LSTM
            obs.extend([pos[0], pos[2], euler[0], euler[1], 
                    lin_vel[0], lin_vel[1], ang_vel[0], ang_vel[1]])
            
            # Estados articulares (12 elementos)
            joint_states = p.getJointStates(self.robot_id, range(4))
            for state in joint_states:
                pos_val = state[0] if not (math.isnan(state[0]) or math.isinf(state[0])) else 0.0
                vel_val = state[1] if not (math.isnan(state[1]) or math.isinf(state[1])) else 0.0
                obs.extend([pos_val, vel_val])
            
            # Contactos y tiempo (4 elementos) - importante para secuencias
            left_contact = len(p.getContactPoints(self.robot_id, 0, self.left_foot_id)) > 0
            right_contact = len(p.getContactPoints(self.robot_id, 0, self.right_foot_id)) > 0
            
            # Normalizar step_count para LSTM
            normalized_time = (self.step_count % 200) / 200.0  # Ciclo de 200 steps
            obs.extend([float(left_contact), float(right_contact), 
                    normalized_time, float(self.step_count > 0)])
            
            # Estados PAM (6 elementos)
            if hasattr(self, 'pam_states'):
                normalized_pressures = self.pam_states['pressures'] / self.max_pressure
                obs.extend(normalized_pressures.tolist())
            else:
                obs.extend([0.0] * 4)
            
        except Exception as e:
            print(f"Error en observación: {e}")
            obs = [0.0] * 30  # Observación de seguridad (actualizada)

        # Añadir información ZMP (4 elementos adicionales)
        zmp_info = [0.0, 0.0, 0.0, 0.0]  # zmp_x, zmp_y, is_stable, margin
        
        if self.zmp_calculator:
            try:
                zmp_point = self.zmp_calculator.calculate_zmp()
                is_stable = self.zmp_calculator.is_stable(zmp_point)
                margin = self.zmp_calculator.stability_margin_distance(zmp_point)
                
                zmp_info = [
                    zmp_point[0], zmp_point[1], 
                    float(is_stable), 
                    np.clip(margin, -1.0, 1.0)  # Normalizar margen
                ]
            except:
                pass

        base_obs=np.array(obs, dtype=np.float32)
        obs=np.concatenate([base_obs, zmp_info])
        
        return obs
    
    # Inverse Kinematics
    def _apply_ik_control(self, target_positions):
        """Aplica control IK para posicionar los pies en coordenadas objetivo"""
        left_target = target_positions[:3]   # x,y,z pie izquierdo
        right_target = target_positions[3:6] # x,y,z pie derecho
        
        # IK para pie izquierdo
        left_joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.left_foot_id,
            left_target,
            solver=self.ik_solver,
            maxNumIterations=self.max_ik_iterations,
            residualThreshold=self.ik_residual_threshold
        )
        
        # IK para pie derecho  
        right_joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.right_foot_id,
            right_target,
            solver=self.ik_solver,
            maxNumIterations=self.max_ik_iterations,
            residualThreshold=self.ik_residual_threshold
        )
        
        return left_joint_poses[:3], right_joint_poses[:3]  # Solo articulaciones de cada pierna
    
    # Para aplicar control de IK + PAM
    def _apply_hybrid_ik_control(self, action):
        # Primeros 6: coordenadas objetivo pies (x,y,z cada uno)
        foot_targets = action[:6] * 0.3  # Escalar a workspace
        
        # Aplicar IK
        left_joints, right_joints = self._apply_ik_control(foot_targets)
        target_positions = np.concatenate([left_joints, right_joints])

        # Obtener posiciones actuales
        joint_states = p.getJointStates(self.robot_id, range(4))
        current_positions = [state[0] for state in joint_states]

        # Control PID simple hacia objetivos IK
        ik_forces = []
        for _, (current, target) in enumerate(zip(current_positions, target_positions)):
            error = target - current
            pid_force = error * 50.0  # Ganancia proporcional
            ik_forces.append(pid_force)
         # Combinar con fuerzas PAM si hay ajustes adicionales
        total_forces = np.array(ik_forces)
        
        if len(action)>4:
            pam_adjustments = (action[4:8] + 1.0) / 2.0 * self.max_pressure  # Ajustes finos PAM
            pam_forces = self._calculate_pam_forces(pam_adjustments, joint_states)

            # Peso híbrido (si existe)
            hybrid_weight = 0.7 if len(action) <= 8 else action[8]
            total_forces = hybrid_weight * np.array(ik_forces) + (1 - hybrid_weight) * np.array(pam_forces)
        # Aplicar solo control de fuerza
        self._apply_joint_forces(total_forces)
        return total_forces #No hay fuerzas PAM adicionales
    
    def _update_ik_tracking(self):
        """Actualiza métricas de seguimiento IK"""
        # Verificar si IK fue exitosa comparando posiciones actuales vs objetivos
        left_pos = p.getLinkState(self.robot_id, self.left_foot_id)[0]
        right_pos = p.getLinkState(self.robot_id, self.right_foot_id)[0]
        
        # Calcular error de posicionamiento (simplified)
        self.last_ik_error = np.linalg.norm(np.array(left_pos) - np.array(right_pos))
        self.last_ik_success = self.last_ik_error < 0.05  # 5cm de tolerancia
        
        # Actualizar tasa de éxito
        self.ik_success_rate.append(float(self.last_ik_success))

    
    def _validate_foot_targets(self, targets):
        """Valida que los objetivos de pies estén en workspace válido"""
        left_target, right_target = targets[:2], targets[2:4]
        
        for target in [left_target, right_target]:
            if not (self.foot_workspace['x_range'][0] <= target[0] <= self.foot_workspace['x_range'][1] and
                    #self.foot_workspace['y_range'][0] <= target[1] <= self.foot_workspace['y_range'][1] and
                    self.foot_workspace['z_range'][0] <= target[1] <= self.foot_workspace['z_range'][1]):
                return False
        return True
    


##############################################################################################################################################################
############################################################## Generacion Variables constantes ###############################################################
##############################################################################################################################################################

    @property
    def generar_simplified_space(self):
        """Genera el espacio de acción simplificado y las propiedades del entorno."""
        self.urdf_path = "2_legged_human_like_robot.urdf"
        self.time_step = 1.0 / 1500.0

        # Variables específicas de IK
        self.last_ik_success = False
        self.last_ik_error = 0.0
        self.last_hybrid_weight = 0.5
        # Número total de joints articulados
        self.num_joints = 4
        if self.action_space=="pam":
            self.control_mode = "pam"
            action_dims=1*self.num_joints
        else:
            self.control_mode = "hybrid"
            action_dims=2*self.num_joints

        self.action_space=self._simplified_action_space(shape=action_dims)

    @property
    def configuracion_simulacion_1(self):
        """Configuración de simulación inicial"""
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            import threading
            self._thread_id = threading.get_ident()
            self.physics_client = p.connect(p.DIRECT)
        
        # Variables de tracking mejoradas
        self.velocity_history = deque(maxlen=10)
        self.previous_action = None
        

        # Fuerzas específicas para cada articulación (en Newtons)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # IDs de end-effectors (enlaces de pies)
        self.left_foot_id = 2   # ID Pie izquierdo
        self.right_foot_id = 5  # ID pie derecho

        # Número total de joints articulados
        self.sistema_recompensas=ImprovedRewardSystem(self.left_foot_id, 
                                                    self.right_foot_id,
                                                    self.num_joints)
        # Configuración de IK
        self.ik_solver = p.IK_DLS  # Damped Least Squares (más estable)
        self.max_ik_iterations = 100
        self.ik_residual_threshold = 1e-3
        self.ik_success_rate = deque(maxlen=100)

        # Límites más amplios
        self.foot_workspace = {
            'x_range': (-1.2, 1.2),    # Adelante/atrás
            'y_range': (-0.8, 0.8),    # Izquierda/derecha 
            'z_range': (-0.3, 1.0)     # Altura del suelo
        }

    @property
    def limites_sistema(self):
        """Define los límites del sistema y las propiedades de los músculos PAM."""
        # Configuración PAM
        self.pam_muscles = {
            'left_hip_joint': PAMMcKibben(L0=0.6, r0=0.02, alpha0=np.pi/4),
            'left_knee_joint': PAMMcKibben(L0=0.5, r0=0.015, alpha0=np.pi/4),
            'right_hip_joint': PAMMcKibben(L0=0.6, r0=0.02, alpha0=np.pi/4),
            'right_knee_joint': PAMMcKibben(L0=0.5, r0=0.015, alpha0=np.pi/4),
        }

        # Parámetros de control PAM
        self.max_pressure = 500000  # 5 bar en Pa
        self.min_pressure = 0

        # Estado interno de los músculos PAM
        self.pam_states = {
            'pressures': np.zeros(4),
            'contractions': np.zeros(4),
            'forces': np.zeros(4)
        }

        # Límites articulares reales del URDF para normalización
        self.joint_limits = {
            'left_hip_joint': (-1.0, 1.0),
            'left_knee_joint': (0.0, 1.571),
            'right_hip_joint': (-1.0, 1.0),
            'right_knee_joint': (0.0, 1.571),
        }

        self.joint_forces = {
            'left_hip_joint': 150,    # Cadera necesita más fuerza
            'left_knee_joint': 120,   # Rodilla fuerza moderada
            'right_hip_joint': 150,
            'right_knee_joint': 120,
        }
        self.joint_force_array = [150, 120, 150, 120]
        self.imitation_weight = 1.0

    @property
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

########################################################################################################################
###########################Preparacion params simulacion y PAM##########################################################
########################################################################################################################

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
        random_friction = np.random.uniform(0.5, 1.5)
        correction_quaternion = p.getQuaternionFromEuler([0, 0, 0])
        #print(correction_quaternion)
        self.previous_position = [0,0,1.2]
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=random_friction)
        self.robot_id = p.loadURDF(
            self.urdf_path,
            self.previous_position,
            correction_quaternion,
            useFixedBase=False
        )
        aabb_min, aabb_max = p.getAABB(self.robot_id, -1)
        #altura = aabb_max[2] - aabb_min[2]
        #print(f"Altura del robot: {altura:.2f} m")
        self.setup_physics_properties()

        # Esperar a que la simulación se estabilice
        for _ in range(20):
            p.stepSimulation()
        self.repeat = 0
        # Randomización de masa en links del robot
        # Esto ayuda a robustecer el modelo ante variaciones físicas
        for link_id in range(p.getNumJoints(self.robot_id)):
            orig_mass = p.getDynamicsInfo(self.robot_id, link_id)[0]
            rand_mass = orig_mass * np.random.uniform(0.8, 1.2)
            p.changeDynamics(self.robot_id, link_id, mass=rand_mass)

    def set_training_phase(self, phase):
        # Por ejemplo:
        if phase == 1:
            self.use_walking_cycle = True
            self.walking_controller = SimplifiedWalkingController(self, mode="trajectory")
            self.imitation_weight = 1.0
        elif phase == 2:
            self.use_walking_cycle = True
            self.walking_controller = SimplifiedWalkingController(self, mode="blend")
            self.imitation_weight = 0.2
        else:
            self.walking_controller = SimplifiedWalkingController(self, mode="pressure")
            self.use_walking_cycle = False
            self.walking_controller = None
            self.imitation_weight = 0.0

    def obtener_posicion_inicial_robot(self):
        robot_id = p.loadURDF("tu_robot.urdf", [0, 0, 0], useFixedBase=True)
        for i in range(6):
            print(i, p.getJointInfo(robot_id, i)[12].decode())

######################################################################################################################
##################################Ajuste de base_action para control híbrido##############################################
#######################################################################################################################

def _safe_blend_actions(base_action, action, modulation_factor=1.0):
    # combino de forma segura IK y PAM
    base_action = np.asarray(base_action)
    action = np.asarray(action)

    # Si shapes ya coinciden, suma directa
    if base_action.shape == action.shape:
        return base_action + modulation_factor * action

    # Si base_action tiene menos dimensiones (ejemplo: 6 y action es 12)
    if base_action.size < action.size:
        # Asumimos que base_action son los últimos elementos (por ejemplo, PAM)
        new_base = np.zeros_like(action)
        new_base[-base_action.size:] = base_action
        return new_base + modulation_factor * action

    # Si base_action tiene más dimensiones (raro, pero prevenimos)
    if base_action.size > action.size:
        # Usamos solo las últimas dimensiones de base_action
        base_trimmed = base_action[-action.size:]
        return base_trimmed + modulation_factor * action

    # Fallback, por si acaso
    raise ValueError(f"No compatible shapes for blending: {base_action.shape}, {action.shape}")

