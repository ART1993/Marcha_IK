
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import numpy as np
import pybullet as p
from Archivos_Apoyo.dinamica_pam import PAMMcKibben
from bided_pam_IK import PAMIKBipedEnv

class Enhanced_PAMIKBipedEnv(PAMIKBipedEnv):
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
        
        # Llamar al constructor padre pero sobrescribir configuración PAM
        super().__init__(render_mode, action_space, num_actors_per_leg, num_articulaciones_pierna)
        
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
            
            # Resortes de tobillo (estabilización pasiva)
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
        from gymnasium import spaces
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
            joint_vel = joint_state[1]
            
            # Calcular ratio de contracción según el tipo de músculo
            if 'hip' in muscle_name:
                # Para caderas: calcular según si es flexor o extensor
                if 'flexor' in muscle_name:
                    # Flexor se activa con ángulos positivos (flexión)
                    normalized_pos = max(0, joint_pos) / 1.2  # Normalizar a rango [0,1]
                else:  # extensor
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
        
        # Combinar todas las observaciones
        enhanced_obs = np.concatenate([base_obs, pam_obs, spring_obs])
        
        return enhanced_obs
    
    def step(self, action):
        """Step sobrescrito para usar el control mejorado"""
        self.step_count += 1
        
        # Aplicar control mejorado con 6 PAMs
        joint_torques = self._apply_enhanced_control(action)
        
        # Simular
        p.stepSimulation()
        
        # Obtener observación mejorada
        observation = self._enhanced_observation()
        
        # Calcular recompensa (usar el sistema existente)
        reward, _ = self.sistema_recompensas._calculate_balanced_reward(
            action=action, 
            pam_forces=joint_torques  # Pasar los torques netos
        )
        
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
        
        # Info de estado
        info = {
            'episode_reward': self.total_reward,
            'episode_length': self.step_count,
            'control_mode': 'enhanced_pam_6',
            'num_active_pams': self.num_active_pams,
            'pam_pressures': self.pam_states['pressures'].tolist(),
            'joint_torques': joint_torques
        }
        
        done = self._is_done()
        
        return observation, reward, done, False, info
    

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