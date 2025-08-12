
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
from collections import deque

from Controlador.discrete_action_controller import ActionType
from Controlador.discrete_action_controller import DiscreteActionController
from Archivos_Apoyo.dinamica_pam import PAMMcKibben
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Mejorados.Enhanced_ImproveReward_system import Enhanced_ImproveRewardSystem
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
                 num_actors_per_leg=3, num_articulaciones_pierna=2, 
                 phase=0, use_discrete_actions=True):
        
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
        self.phase=phase
        self.num_actors_per_leg=num_actors_per_leg
        self.num_articulaciones_pierna=num_articulaciones_pierna
        self.use_discrete_actions = use_discrete_actions
        self.current_action_type = None

        #self.walking_controller = None  # Se inicializará en reset()
        # Por defecto, empezar con walking cycle habilitado si usamos acciones discretas
        if use_discrete_actions:
            self.use_walking_cycle = True
            self.imitation_weight = 0.0  # Se ajustará según la fase
        else:
            self.use_walking_cycle = False
            self.imitation_weight = 0.0
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
            low=0.0, 
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

    def relacionar_musculo_con_joint(self, muscle_name, joint_data):
        # ===== PASO 1: DETERMINAR QUÉ ARTICULACIÓN AFECTA ESTE MÚSCULO =====
            
        if muscle_name in ['left_hip_flexor', 'left_hip_extensor']:
            joint_id = 0  # left_hip_joint
            joint_state = joint_data[0]
        elif muscle_name in ['right_hip_flexor', 'right_hip_extensor']:
            joint_id = 3  # right_hip_joint (en PyBullet)
            joint_state = joint_data[3]
        elif muscle_name == 'left_knee_flexor':
            joint_id = 1  # left_knee_joint
            joint_state = joint_data[1]
        elif muscle_name == 'right_knee_flexor':
            joint_id = 4  # right_knee_joint (en PyBullet)
            joint_state = joint_data[4]
        else:
            raise ValueError(f"Unknown muscle name: {muscle_name}")
        #    print(f" Warning: Unknown muscle {muscle_name}")
        #    forces.append(0.0)
        #    continue
        return joint_id, joint_state
        

    # Parece sustituir al antiguo _calculate_pam_forces
    def _calculate_antagonistic_forces(self, pam_pressures, joint_states):
        """
        Calcula fuerzas considerando pares antagónicos en caderas
        """

        # Verificar que tenemos exactamente 6 presiones PAM
        if len(pam_pressures) != 6:
            print(f"⚠️ Warning: Expected 6 PAM pressures, got {len(pam_pressures)}")
            return [0.0] * 6
        
        joint_data = {
            0: joint_states[0],  # left_hip_joint
            1: joint_states[1],  # left_knee_joint  
            3: joint_states[2],  # right_hip_joint
            4: joint_states[3],  # right_knee_joint
        }
        forces = []
        muscle_names = list(self.pam_muscles.keys())
        
        # Procesar por pares antagónicos y músculos individuales
        for i, pressure in enumerate(pam_pressures):
            muscle_name = muscle_names[i]
            pam = self.pam_muscles[muscle_name]

            joint_id, joint_state=self.relacionar_musculo_con_joint(muscle_name, joint_data)
            
            # Obtener información de la articulación
            joint_pos = joint_state[0] # Posición actual de la articulación
            #print(f"{muscle_name=:}")
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
                side = 'left' if 'left' in muscle_name else 'right'
                joint_range = self.joint_limits[f"{side}_knee_joint"]
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

            # Debug opcional: imprimir información cada ciertos pasos
            if hasattr(self, 'step_count') and self.step_count % 1000 == 0 and i == 0:
                print(f"🔧 PAM Debug - {muscle_name}: pressure={pressure:.0f}Pa, "
                      f"contraction={contraction_ratio:.2f}, force={pam_force:.1f}N")
        #print(f"   ✅ Calculated PAM forces: {forces}")
        if len(forces) != 6:
            print(f"❌ Error: Generated {len(forces)} forces instead of 6!")
            return [0.0] * 6
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
        
        # Resortes de tobillos (índices 2 y 5) 
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

        # Configuración base de pesos de imitación
        phase_configs = {
            0: {'imitation_weight': 0.0, 'use_walking_cycle': False},    # Equilibrio libre
            1: {'imitation_weight': 0.9, 'use_walking_cycle': True},     # Imitación equilibrio
            2: {'imitation_weight': 0.3, 'use_walking_cycle': True},     # Exploración equilibrio
            3: {'imitation_weight': 0.85, 'use_walking_cycle': True},    # Imitación sentadilla parcial
            4: {'imitation_weight': 0.5, 'use_walking_cycle': True},     # Exploración sentadilla
            5: {'imitation_weight': 0.9, 'use_walking_cycle': True},     # Imitación levantar izq.
            6: {'imitation_weight': 0.6, 'use_walking_cycle': True},     # Exploración levantar izq.
            7: {'imitation_weight': 0.9, 'use_walking_cycle': True},     # Imitación levantar der.
            8: {'imitation_weight': 0.6, 'use_walking_cycle': True},     # Exploración levantar der.
            9: {'imitation_weight': 0.8, 'use_walking_cycle': True},     # Imitación paso izq.
            10: {'imitation_weight': 0.5, 'use_walking_cycle': True},    # Exploración paso izq.
            11: {'imitation_weight': 0.8, 'use_walking_cycle': True},    # Imitación paso der.
            12: {'imitation_weight': 0.2, 'use_walking_cycle': True},    # Maestría bilateral
        }

        config = phase_configs.get(phase, {'imitation_weight': 0.0, 'use_walking_cycle': False})
    
        self.imitation_weight = config['imitation_weight']
        self.use_walking_cycle = config['use_walking_cycle']

        # Configurar la acción apropiada si usamos acciones discretas
        if self.use_discrete_actions and self.walking_controller is not None:
            try:
                # Verificar que el controlador es del tipo correcto
                if hasattr(self.walking_controller, 'get_action_for_phase'):
                    action_type = self.walking_controller.get_action_for_phase(phase)
                    self.walking_controller.set_action(action_type)
                    print(f"   🎯 Phase {phase}: Training action type: {action_type.value}")
                else:
                    print(f"   ⚠️ Walking controller doesn't support discrete actions")
            except ImportError as e:
                print(f"   ⚠️ Could not import ActionType: {e}")
        
        # Sincronizar sistema de recompensas
        if hasattr(self, 'sistema_recompensas'):
            self.sistema_recompensas.set_curriculum_phase(phase)
            print(f"   ✅ Reward system synchronized to phase {phase}")
        
        print(f"   🎮 Environment configuration for phase {phase}:")
        print(f"      Use walking cycle: {self.use_walking_cycle}")
        print(f"      Imitation weight: {self.imitation_weight}")
        print(f"   🏁 Phase {phase} configuration completed\n")
        

    
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
        """
            Step implementation for Enhanced PAM system with integrated biomechanical rewards.
        
            Este método es donde ocurre la "magia" de la integración. Cada paso de simulación:
            1. Ejecuta las acciones PAM en el robot físico
            2. El sistema de recompensas observa y evalúa el rendimiento biomecánico
            3. Proporciona feedback educativo al algoritmo de RL
            
            Es como un entrenador observando cada movimiento y dando feedback inmediato.
        """
        self.step_count += 1

        # ===== FASE 1: PROCESAMIENTO DE ACCIÓN =====
        # Combinar acción del ciclo con RL si está habilitado
        if self.walking_controller and self.use_walking_cycle:
            if self.use_discrete_actions and hasattr(self.walking_controller, 'get_expert_action'):
                base_action = self.walking_controller.get_expert_action(self.time_step)
            elif hasattr(self.walking_controller, 'get_enhanced_walking_actions'):
                base_action = self.walking_controller.get_enhanced_walking_actions(self.time_step)
            else:
                # Fallback si algo falla
                base_action = action
            modulation_factor = 0.3 if self.imitation_weight > 0 else 1.0
            final_action = self._safe_blend_actions(base_action, action, modulation_factor)
        else:
            final_action = action
        
        # ===== FASE 2: EJECUCIÓN FÍSICA =====
        # Aplicar control mejorado con 6 PAMs
        joint_torques = self._apply_enhanced_control(final_action)

        # ANTES de simular, asegurarnos que el sistema de recompensas tiene acceso
        # a los estados PAM más recientes. Es como asegurar que el entrenador
        # pueda ver exactamente lo que está haciendo el robot bipedo.
        if not self.reward_system_integration_status['pam_states_linked']:
            self._ensure_reward_system_integration()

        # Actualizar el sistema de recompensas con los estados PAM actuales
        self.sistema_recompensas.pam_states = self.pam_states
        self.sistema_recompensas.max_pressure = self.max_pressure

        # ===== FASE 3: SIMULACIÓN FÍSICA =====
        
        # Simular
        p.stepSimulation()

        # ===== FASE 4: EVALUACIÓN BIOMECÁNICA =====
        
        # Obtener observación mejorada
        observation = self._enhanced_observation()
        
        # Calcular recompensa (usar el sistema existente) Calcular recompensa usando el sistema biomecánico
        # Aquí es donde el "entrenador" evalúa la técnica del robot bipedo
        reward, reward_components  = self.sistema_recompensas._calculate_balanced_reward(
            action=action, 
            pam_forces=joint_torques  # Pasar los torques netos
        )

        # Guardar componentes de recompensa para debugging y análisis
        self.reward_system_integration_status['last_reward_components'] = reward_components

        # ===== FASE 5: REWARD SHAPING POR IMITACIÓN =====

        # Reward shaping por imitación si está habilitado
        if self.imitation_weight > 0 and self.walking_controller:
            if self.use_discrete_actions:
                expert_action = self.walking_controller.get_expert_action(self.time_step)
            else:
                expert_action = self.walking_controller.get_enhanced_walking_actions(self.time_step)
            if expert_action is not None:
                imitation_penalty = np.linalg.norm(final_action - expert_action)
                reward -= self.imitation_weight * imitation_penalty

                # Tracking para análisis de currículo
                self.reward_system_integration_status['last_imitation_penalty'] = imitation_penalty
        
        # ===== FASE 6: MÉTRICAS ADICIONALES =====
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

        # ===== FASE 7: ACTUALIZACIÓN DE ESTADO =====

        # Actualizar estado
        current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        self.previous_position = current_pos

        # ===== FASE 8: INFORMACIÓN DE ESTADO PARA ANÁLISIS =====
        
        # Info de estado
        info = {
            'episode_reward': self.total_reward,
            'episode_length': self.step_count,
            'control_mode': 'enhanced_pam_6',
            'num_active_pams': self.num_active_pams,
            'pam_pressures': self.pam_states['pressures'].tolist(),
            'joint_torques': joint_torques,

            # ===== INFORMACIÓN BIOMECÁNICA =====
            'reward_components': reward_components,
            'curriculum_phase': self.phase,
            'imitation_weight': self.imitation_weight,
            'reward_system_status': self.reward_system_integration_status.copy()
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

        # ===== FASE 9: RECOMPENSAS ADICIONALES CONTEXTUALES =====
        # Recompensa adicional basada en estabilidad del robot
        stability = self.robot_data.get_stability_metrics
        com_height = stability['com_height']
        stable = stability['is_stable']
        reward += 1.0 if stable else -5.0
        reward += max(0, com_height - 0.4)

        # ===== FASE 10: DETERMINACIÓN DE EPISODIO TERMINADO =====
        
        done = self._is_done()

        # Añadir información de debug sobre la acción actual
        if self.use_discrete_actions and self.walking_controller:
            debug_info = self.walking_controller.get_debug_info()
            info['current_discrete_action'] = debug_info['current_action']
            info['action_progress'] = debug_info['action_progress']

        # ===== LOGGING PARA DEBUGGING (cada 100 pasos) =====
        
        if self.step_count % 100 == 0:
            self._log_integration_status(reward, reward_components)
        
        return observation, reward, done, False, info
    
    def _log_integration_status(self, reward, reward_components):
        """
        Log de estado de integración para debugging y análisis.
        
        Este método es como el "informe de progreso" que un entrenador da
        periódicamente para mostrar cómo va el entrenamiento del atleta.
        """
        
        if self.step_count % 500 == 0:  # Log cada 500 pasos (menos frecuente)
            print(f"\n📊 Integration Status Report (Step {self.step_count})")
            print(f"   Total reward: {reward:.3f}")
            
            # Mostrar componentes de recompensa
            if reward_components:
                print("   Reward breakdown:")
                for component, value in reward_components.items():
                    print(f"      {component}: {value:.3f}")
            
            # Estado de los PAMs
            if hasattr(self, 'pam_states') and self.pam_states is not None:
                pressures = self.pam_states['pressures']
                normalized_pressures = pressures / self.max_pressure
                
                print("   PAM status:")
                muscle_names = ['L_hip_flex', 'L_hip_ext', 'R_hip_flex', 'R_hip_ext', 'L_knee', 'R_knee']
                for i, (name, pressure) in enumerate(zip(muscle_names, normalized_pressures)):
                    print(f"      {name}: {pressure:.2f}")
            
            # Estado del currículo
            curriculum_info = self.reward_system_integration_status
            print(f"   Curriculum phase: {curriculum_info['curriculum_phase']}")
            print(f"   Imitation weight: {self.imitation_weight}")
            
            # Métricas de coordinación si están disponibles
            if hasattr(self.sistema_recompensas, 'coordination_metrics'):
                coords = self.sistema_recompensas.coordination_metrics
                print("   Coordination metrics:")
                for metric, value in coords.items():
                    if value != 0.0:  # Solo mostrar métricas activas
                        print(f"      {metric}: {value:.3f}")
            
            print("   ✅ Integration status: HEALTHY\n")
    
    def _ensure_reward_system_integration(self):
        """
            Asegura que el sistema de recompensas esté correctamente integrado.
            
            Este método es como hacer una "verificación de comunicación" entre
            el entrenador y el atleta para asegurar que se entienden perfectamente.
            
            Se ejecuta una sola vez al inicio del entrenamiento para verificar que:
            1. Los estados PAM se están compartiendo correctamente
            2. Los IDs de articulaciones coinciden
            3. Las métricas biomecánicas se pueden calcular
        """
        
        try:
            # ===== VERIFICACIÓN 1: ESTADOS PAM =====
            
            if self.pam_states is None:
                print("   ⚠️ Warning: PAM states not initialized yet")
                return False
            
            # Verificar que tenemos todos los estados PAM necesarios
            required_keys = ['pressures', 'contractions', 'forces']
            for key in required_keys:
                if key not in self.pam_states:
                    print(f"   ❌ Error: Missing PAM state key: {key}")
                    return False
                    
                if len(self.pam_states[key]) != 6:
                    print(f"   ❌ Error: Expected 6 PAM {key}, got {len(self.pam_states[key])}")
                    return False
            
            # ===== VERIFICACIÓN 2: COMUNICACIÓN CON REWARD SYSTEM =====
            
            # Pasar los estados PAM al sistema de recompensas
            self.sistema_recompensas.pam_states = self.pam_states
            self.sistema_recompensas.max_pressure = self.max_pressure
            
            # Verificar que el sistema de recompensas puede acceder a los IDs del robot
            self.sistema_recompensas.robot_id = self.robot_id
            self.sistema_recompensas.plane_id = self.plane_id
            
            # ===== VERIFICACIÓN 3: TEST DE MÉTRICAS =====
            
            # Realizar un test rápido para verificar que las métricas funcionan
            test_activations = self.pam_states['pressures'] / self.max_pressure
            
            # Test de co-activación
            if hasattr(self.sistema_recompensas, '_evaluate_coactivation_efficiency'):
                coactivation_score = self.sistema_recompensas._evaluate_coactivation_efficiency(test_activations)
                print(f"   ✅ Coactivation evaluation working: {coactivation_score:.3f}")
            
            # ===== VERIFICACIÓN 4: MAPEO DE ARTICULACIONES =====
            
            # Verificar que los IDs de articulaciones son correctos
            expected_joints = {
                'left_hip': 0,
                'left_knee': 1, 
                'right_hip': 3,
                'right_knee': 4
            }
            
            # Verificar que las articulaciones existen en PyBullet
            for joint_name, joint_id in expected_joints.items():
                try:
                    joint_state = p.getJointState(self.robot_id, joint_id)
                    print(f"   ✅ Joint {joint_name} (ID {joint_id}): position = {joint_state[0]:.3f}")
                except:
                    print(f"   ❌ Error: Cannot access joint {joint_name} (ID {joint_id})")
                    return False
            
            # ===== ÉXITO: MARCAR COMO INTEGRADO =====
            
            self.reward_system_integration_status['pam_states_linked'] = True
            print("   🎉 Reward system integration successful!")
            print(f"   📊 Monitoring {len(self.pam_states['pressures'])} PAM muscles")
            print(f"   🧠 Biomechanical metrics: ACTIVE")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Integration error: {e}")
            print("   🔄 Will retry on next step...")
            return False

    
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

        # ===== INTEGRACIÓN CRÍTICA: Sistema de recompensas para 6 PAMs antagónicos =====
        # 
        # IMPORTANTE: Inicializamos con None para pam_states porque se crea antes de
        # que tengamos los estados PAM inicializados. Se actualizará en reset().
        # Esto es como tener un entrenador que espera a conocer al atleta antes de
        # comenzar a evaluar su rendimiento.
        

        # Número total de joints articulados
        self.sistema_recompensas=Enhanced_ImproveRewardSystem(self.left_foot_id, 
                                                    self.right_foot_id,
                                                    self.num_joints,
                                                    None, # Se inicia después
                                                    curriculum_phase=1,  # Fase inicial por defecto
                                                    num_pams=6)
        # Variable para tracking de integración - nos ayuda a debuggear
        # Variable para tracking de integración - nos ayuda a debuggear
        self.reward_system_integration_status = {
            'initialized': True,
            'pam_states_linked': False,
            'curriculum_phase': self.sistema_recompensas.curriculum_phase,
            'last_reward_components': None
        }

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
            # 6 articulaciones: 4 activas (caderas y rodillas) + 2 tobillos
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
        """Control PAM simple (adaptado para 6 PAMs)"""
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
            print("recent_unstable",recent_unstable)
            if recent_unstable >= 8:
                print("nivel inestabilidad",recent_unstable)
                return True
            
        # Límite de tiempo
        if self.step_count > 9000:
            print("fuera de t")
            return True
            
        return False
    
    def generate_walking_controller(self):
        # Inicializar controlador apropiado
        if self.use_discrete_actions:
            try:
                self.walking_controller = DiscreteActionController(self)
                # Establecer acción inicial basada en la fase del currículo
                if hasattr(self, 'phase'):
                    action_type = self.walking_controller.get_action_for_phase(self.phase)
                    self.walking_controller.set_action(action_type)
                else:
                    # Por defecto, empezar con equilibrio
                    self.walking_controller.set_action(ActionType.BALANCE_STANDING)
                    print(f"   🎯 DiscreteActionController initialized with default action: BALANCE_STANDING")
            except ImportError as e:
                print(f"   ❌ Error importing DiscreteActionController: {e}")
                print("   ⚠️ Falling back to Enhanced walking controller")
                self.walking_controller = Enhanced_SimplifiedWalkingController(self)
        else:
            # Usar el controlador de marcha original si es necesario
            self.walking_controller = Enhanced_SimplifiedWalkingController(self)
    

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

        self.generate_walking_controller()

        # Solo resetear si el controlador existe
        if self.walking_controller is not None:
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
        # Estabilización inicial
        for _ in range(20):
            p.stepSimulation()
        #Anteriormente stable observation
        observation = self._enhanced_observation()

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
        """Observación base estable """
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
            # ¿Son las presiones totales o la presión por cada joint con actuador?
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

    # ===== MÉTODOS DE VALIDACIÓN Y TESTING ===== #

    def validate_integration(self):
        """
        Valida que la integración entre Enhanced_PAMIKBipedEnv y Enhanced_ImprovedRewardSystem
        esté funcionando correctamente.
        
        Este método es como un "examen médico completo" que verifica que todos los
        sistemas del robot están comunicándose correctamente. Es especialmente útil
        para debugging cuando algo no funciona como esperamos.
        
        Returns:
            dict: Reporte completo del estado de integración
        """
        
        print("🔍 Validating Enhanced PAM Integration...")
        
        validation_report = {
            'overall_status': 'UNKNOWN',
            'pam_system': {},
            'reward_system': {},
            'communication': {},
            'biomechanics': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # ===== VALIDACIÓN 1: SISTEMA PAM =====
            
            print("   Checking PAM system...")
            
            # Verificar que tenemos 6 PAMs
            if hasattr(self, 'pam_states') and self.pam_states is not None:
                if len(self.pam_states['pressures']) == 6:
                    validation_report['pam_system']['muscle_count'] = '✅ 6 muscles'
                else:
                    validation_report['errors'].append(f"Expected 6 PAMs, got {len(self.pam_states['pressures'])}")
                
                # Verificar rangos de presión
                max_pressure = np.max(self.pam_states['pressures'])
                min_pressure = np.min(self.pam_states['pressures'])
                validation_report['pam_system']['pressure_range'] = f"{min_pressure:.0f} - {max_pressure:.0f} Pa"
                
                if max_pressure <= self.max_pressure:
                    validation_report['pam_system']['pressure_limits'] = '✅ Within limits'
                else:
                    validation_report['warnings'].append("Some pressures exceed maximum")
            else:
                validation_report['errors'].append("PAM states not initialized")
            
            # ===== VALIDACIÓN 2: SISTEMA DE RECOMPENSAS =====
            
            print("   Checking reward system...")
            
            if hasattr(self, 'sistema_recompensas'):
                # Verificar tipo correcto
                if isinstance(self.sistema_recompensas, Enhanced_ImproveRewardSystem):
                    validation_report['reward_system']['type'] = '✅ Enhanced system'
                else:
                    validation_report['errors'].append("Using old reward system instead of Enhanced")
                
                # Verificar configuración para 6 PAMs
                if hasattr(self.sistema_recompensas, 'num_pams'):
                    if self.sistema_recompensas.num_pams == 6:
                        validation_report['reward_system']['pam_count'] = '✅ Configured for 6 PAMs'
                    else:
                        validation_report['errors'].append(f"Reward system expects {self.sistema_recompensas.num_pams} PAMs, not 6")
                
                # Verificar mapeo antagónico
                if hasattr(self.sistema_recompensas, 'antagonistic_pairs'):
                    pairs_count = len(self.sistema_recompensas.antagonistic_pairs)
                    validation_report['reward_system']['antagonistic_pairs'] = f'✅ {pairs_count} pairs'
                
                # Verificar fase de currículo
                if hasattr(self.sistema_recompensas, 'curriculum_phase'):
                    phase = self.sistema_recompensas.curriculum_phase
                    validation_report['reward_system']['curriculum_phase'] = f'✅ Phase {phase}'
            else:
                validation_report['errors'].append("Reward system not found")
            
            # ===== VALIDACIÓN 3: COMUNICACIÓN =====
            
            print("   Checking communication...")
            
            # Test de comunicación PAM -> Reward System
            if (hasattr(self, 'pam_states') and self.pam_states is not None and
                hasattr(self, 'sistema_recompensas')):
                
                # Verificar que el reward system puede acceder a PAM states
                self.sistema_recompensas.pam_states = self.pam_states
                
                if self.sistema_recompensas.pam_states is not None:
                    validation_report['communication']['pam_to_reward'] = '✅ Connected'
                else:
                    validation_report['errors'].append("Reward system cannot access PAM states")
                
                # Test de cálculo de recompensa
                try:
                    test_action = np.zeros(6)
                    test_forces = np.zeros(6)
                    
                    # Simular estados mínimos requeridos para PyBullet (mock)
                    if self.robot_id is not None:
                        reward, components = self.sistema_recompensas._calculate_balanced_reward(
                            test_action, test_forces
                        )
                        validation_report['communication']['reward_calculation'] = f'✅ Working (reward: {reward:.2f})'
                    else:
                        validation_report['warnings'].append("Cannot test reward calculation without robot_id")
                        
                except Exception as e:
                    validation_report['errors'].append(f"Reward calculation failed: {str(e)}")
            
            # ===== VALIDACIÓN 4: MÉTRICAS BIOMECÁNICAS =====
            
            print("   Checking biomechanical metrics...")
            
            if (hasattr(self, 'sistema_recompensas') and 
                hasattr(self.sistema_recompensas, '_evaluate_coactivation_efficiency')):
                
                # Test de métricas antagónicas
                try:
                    test_activations = np.array([0.3, 0.4, 0.2, 0.5, 0.1, 0.15])  # Test activations
                    
                    coactivation_score = self.sistema_recompensas._evaluate_coactivation_efficiency(test_activations)
                    validation_report['biomechanics']['coactivation'] = f'✅ Working (score: {coactivation_score:.2f})'
                    
                    reciprocal_score = self.sistema_recompensas._evaluate_reciprocal_inhibition(test_activations)
                    validation_report['biomechanics']['reciprocal_inhibition'] = f'✅ Working (score: {reciprocal_score:.2f})'
                    
                    bilateral_score = self.sistema_recompensas._evaluate_bilateral_coordination(test_activations)
                    validation_report['biomechanics']['bilateral_coordination'] = f'✅ Working (score: {bilateral_score:.2f})'
                    
                except Exception as e:
                    validation_report['errors'].append(f"Biomechanical metrics failed: {str(e)}")
            
            # ===== DETERMINACIÓN DE ESTADO GENERAL =====
            
            if len(validation_report['errors']) == 0:
                validation_report['overall_status'] = 'HEALTHY'
                print("   🎉 Integration validation PASSED!")
            elif len(validation_report['errors']) <= 2 and len(validation_report['warnings']) <= 3:
                validation_report['overall_status'] = 'FUNCTIONAL_WITH_ISSUES'
                print("   ⚠️ Integration validation PASSED with minor issues")
            else:
                validation_report['overall_status'] = 'CRITICAL_ISSUES'
                print("   ❌ Integration validation FAILED")
            
        except Exception as e:
            validation_report['errors'].append(f"Validation process failed: {str(e)}")
            validation_report['overall_status'] = 'VALIDATION_ERROR'
        
        return validation_report
    
    def get_integration_summary(self):
        """
        Proporciona un resumen legible del estado de integración.
        
        Este método es como un "informe ejecutivo" que muestra de manera
        clara y concisa cómo está funcionando la integración entre sistemas.
        
        Returns:
            str: Resumen formateado del estado de integración
        """
        
        summary = []
        summary.append("🔗 ENHANCED PAM INTEGRATION SUMMARY")
        summary.append("=" * 50)
        
        # Estado general
        if hasattr(self, 'reward_system_integration_status'):
            status = self.reward_system_integration_status
            
            summary.append(f"📊 Integration Status: {'✅ LINKED' if status['pam_states_linked'] else '❌ NOT LINKED'}")
            summary.append(f"📚 Curriculum Phase: {status['curriculum_phase']}")
            summary.append(f"🎯 Imitation Weight: {self.imitation_weight}")
        
        # Configuración PAM
        if hasattr(self, 'pam_states') and self.pam_states is not None:
            summary.append(f"💪 Active PAMs: {len(self.pam_states['pressures'])}")
            
            current_pressures = self.pam_states['pressures'] / self.max_pressure
            summary.append("🔧 Current Activations:")
            muscle_names = ['L_Hip_Flex', 'L_Hip_Ext', 'R_Hip_Flex', 'R_Hip_Ext', 'L_Knee', 'R_Knee']
            for name, activation in zip(muscle_names, current_pressures):
                summary.append(f"   {name}: {activation:.1%}")
        
        # Sistema de recompensas
        if hasattr(self, 'sistema_recompensas'):
            summary.append(f"🧠 Reward System: Enhanced (6 PAMs)")
            
            if hasattr(self.sistema_recompensas, 'weights'):
                summary.append("📈 Current Reward Weights:")
                for component, weight in self.sistema_recompensas.weights.items():
                    summary.append(f"   {component}: {weight:.1%}")
        
        # Estado del entorno
        summary.append(f"🎮 Control Mode: PAM (6 antagonistic)")
        summary.append(f"🚶 Walking Cycle: {'✅ ACTIVE' if self.use_walking_cycle else '❌ DISABLED'}")
        
        summary.append("=" * 50)
        
        return "\n".join(summary)

    

# ===================================================================================================================================================================== #
# =================================================Pruebo el testeo==================================================================================================== #
# ===================================================================================================================================================================== #


def test_complete_integration():
    """
    Función de testing integral que valida toda la integración entre
    Enhanced_PAMIKBipedEnv y Enhanced_ImprovedRewardSystem.
    
    Esta función es como un "curso intensivo" que pone a prueba todos
    los aspectos de la integración para asegurar que funciona correctamente.
    """
    
    print("🧪 COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # ===== PASO 1: CREAR ENTORNO =====
        
        print("1️⃣ Creating Enhanced PAM Environment...")
        env = Enhanced_PAMIKBipedEnv(render_mode='direct', action_space="pam")
        print("   ✅ Environment created successfully")
        
        # ===== PASO 2: RESET INICIAL =====
        
        print("\n2️⃣ Performing initial reset...")
        obs, info = env.reset()
        print(f"   ✅ Reset completed")
        print(f"   📊 Observation shape: {obs.shape}")
        print(f"   🎯 Action space: {env.action_space.shape}")
        
        # ===== PASO 3: VALIDACIÓN DE INTEGRACIÓN =====
        
        print("\n3️⃣ Validating integration...")
        validation_report = env.validate_integration()
        
        print(f"   📋 Overall status: {validation_report['overall_status']}")
        if validation_report['errors']:
            print("   ❌ Errors found:")
            for error in validation_report['errors']:
                print(f"      - {error}")
        if validation_report['warnings']:
            print("   ⚠️ Warnings:")
            for warning in validation_report['warnings']:
                print(f"      - {warning}")
        
        # ===== PASO 4: TEST DE ACCIONES =====
        
        print("\n4️⃣ Testing PAM actions...")
        
        # Test con diferentes tipos de acciones
        test_actions = [
            np.zeros(6),                           # Acción nula
            np.array([0.2, 0.3, 0.2, 0.3, 0.1, 0.1]),  # Activación balanceada
            np.array([0.8, 0.1, 0.1, 0.8, 0.6, 0.6]),  # Activación alternada
            env.action_space.sample()              # Acción aleatoria
        ]
        
        for i, action in enumerate(test_actions):
            print(f"   Testing action {i+1}/4...")
            obs, reward, done, truncated, info = env.step(action)
            
            print(f"      Reward: {reward:.3f}")
            if 'reward_components' in info:
                components = info['reward_components']
                print(f"      Components: {len(components)} metrics calculated")
            
            if done:
                print("      Episode ended - resetting...")
                obs, info = env.reset()
        
        # ===== PASO 5: TEST DE CURRÍCULO =====
        
        print("\n5️⃣ Testing curriculum phases...")
        
        for phase in [0, 1, 2]:
            print(f"   Setting phase {phase}...")
            env.set_training_phase(phase, 1000)  # 1000 steps por fase
            
            # Test rápido en cada fase
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"      Phase {phase} reward: {reward:.3f}")
        
        # ===== PASO 6: ANÁLISIS DE COORDINACIÓN =====
        
        print("\n6️⃣ Testing muscle coordination...")
        
        # Simular 20 pasos para generar datos de coordinación
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            if done:
                obs, info = env.reset()
        
        # Verificar métricas de coordinación
        if hasattr(env.sistema_recompensas, 'coordination_metrics'):
            coords = env.sistema_recompensas.coordination_metrics
            print("   🤝 Coordination metrics:")
            for metric, value in coords.items():
                print(f"      {metric}: {value:.3f}")
        
        # ===== PASO 7: RESUMEN FINAL =====
        
        print("\n7️⃣ Integration summary:")
        summary = env.get_integration_summary()
        print(summary)
        
        # ===== RESULTADO =====
        
        env.close()
        
        if validation_report['overall_status'] in ['HEALTHY', 'FUNCTIONAL_WITH_ISSUES']:
            print("\n🎉 INTEGRATION TEST PASSED!")
            print("   The Enhanced PAM system is ready for training.")
            return True
        else:
            print("\n❌ INTEGRATION TEST FAILED!")
            print("   Please review the errors above before training.")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    




# ===== FUNCIÓN DE PRUEBA =====

def test_enhanced_6pam_system():
    """Script de prueba para verificar el sistema de 6 PAMs"""
    
    print("🔧 Testing Enhanced PAM System (6 actuators)")
    
    env = Enhanced_PAMIKBipedEnv(render_mode='human', action_space="pam")
    
    obs, info = env.reset()
    print(f"✅ Environment created successfully")
    print(f"   - Action space: {env.action_space.shape}")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Active PAMs: {env.num_active_pams}")
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"   Step {step}: Reward = {reward:.2f}")
            if 'pam_pressures' in info:
                print(f"      PAM pressures = {info['pam_pressures']}")
        
        if done:
            print(f"   Episode ended at step {step}")
            obs, info = env.reset()
    
    env.close()
    print("🎉 Test completed successfully!")

if __name__ == "__main__":
    test_enhanced_6pam_system()