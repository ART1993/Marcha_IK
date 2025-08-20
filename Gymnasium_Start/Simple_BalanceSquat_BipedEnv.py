
# Modificaciones para PAMIKBipedEnv - Sistema de 6 PAMs + elementos pasivos

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
from collections import deque

from Controlador.discrete_action_controller import DiscreteActionController, ActionType
from Archivos_Apoyo.Configuraciones_adicionales import PAM_McKibben
from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Mejorados.Simplified_BalanceSquat_RewardSystem import Simplified_BalanceSquat_RewardSystem
from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data                 

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
    
    def __init__(self, render_mode='human', action_space="pam"):
        
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
            'pressures': np.zeros(6),
            'forces': np.zeros(6)
        }
        
        # ===== CONFIGURACIÓN DE ESPACIOS =====
        
        # Action space: 6 presiones PAM normalizadas [0, 1]
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(6,), 
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
        
        # Sistema de recompensas simplificado
        self.reward_system = Simplified_BalanceSquat_RewardSystem()
        
        print(f"🤖 Simplified Balance & Squat Environment initialized")
        print(f"   Action space: 6 PAM pressures")
        print(f"   Observation space: 16 elements")
        print(f"   Target: Balance + Sentadillas")
    
# ========================================================================================================================================================================= #
# ===================================================== Métodos de paso y control del entorno Enhanced_PAMIKBipedEnv ====================================================== #
# ========================================================================================================================================================================= #
    
    def step(self, action):
        """
            Step SIMPLIFICADO - Solo física PAM básica + recompensa de balance
        
            ELIMINADO:
            - 10 fases complejas de procesamiento
            - Walking controller blending
            - Expert action imitation
            - Biomechanical metrics logging
            - Complex reward components
        """
        self.step_count += 1

        # ✅ VERIFICAR CONTACTO BILATERAL SIMPLE
        left_contacts = p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_id, -1)
        right_contacts = p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_id, -1)

        both_feet_contact = len(left_contacts) > 0 and len(right_contacts) > 0

        # ✅ LÓGICA DE TRANSICIÓN DE CONTROL
        if both_feet_contact and not self.pam_control_active:
            # 🎯 PRIMERA VEZ CON CONTACTO BILATERAL → Activar PAMs
            print(f"   🔥 Step {self.step_count}: Contacto bilateral detectado - ACTIVANDO PAMs")
            self.contact_established = True
            self.pam_control_active = True
            
        elif not both_feet_contact and self.pam_control_active:
            # ⚠️ PERDIDO CONTACTO → Volver a standing position
            print(f"   ⚠️ Step {self.step_count}: Contacto perdido - VOLVIENDO A STANDING POSITION")
            self.pam_control_active = False
            
        # ✅ APLICAR CONTROL SEGÚN MODO
        if self.pam_control_active:
            # 🔥 MODO PAM: Control por torques
            self._apply_pam_control(action)
            control_mode = 'PAM_TORQUE'
            
        else:
            # 🤖 MODO STANDING: Control por posición
            self._apply_standing_position_control()
            control_mode = 'POSITION_STANDING'

        # ===== SIMULACIÓN FÍSICA =====
        
        p.stepSimulation()

        # ✅ CALCULAR RECOMPENSA (solo si PAMs activos)
        if self.pam_control_active:
            reward = self._calculate_simple_balance_reward()
        else:
            # Sin penalización mientras se establece contacto
            reward = 0.0

        observation = self._get_simple_observation()
        done = self._is_done()

        # ===== APLICAR ACCIÓN PAM =====
        
        # Info básico
        info = {
                'step_count': self.step_count,
                'reward': reward,
                'contact_established': self.contact_established,
                'pam_control_active': self.pam_control_active,
                'control_mode': control_mode,
                'both_feet_contact': both_feet_contact
            }
        
        return observation, reward, done, False, info
    
    def _apply_standing_position_control(self):
        """
        ✅ CONTROL DE POSICIÓN: Mantener standing position
        """
        
        for i, target_pos in self.neutral_positions.items():
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=50,  # Fuerza suficiente para mantener posición
                maxVelocity=0.5
            )
        
        # Actualizar estados PAM (inactivos)
        self.pam_states['pressures'] = np.zeros(6)
        self.pam_states['forces'] = np.zeros(6)

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
                force=10  # Muy suave
            )
        
        # Actualizar estados PAM
        self.pam_states['pressures'] = normalized_pressures

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
        
        pam_forces = np.zeros(6)  # Fuerzas reales de cada PAM
    
        # ===== PASO 1: CALCULAR FUERZAS REALES USANDO MODELO PAM_MCKIBBEN =====
        
        for i, pressure_normalized in enumerate(pam_pressures):
            # Determinar qué músculo y articulación
            if i in [0, 1]:  # Cadera izquierda
                joint_angle = joint_positions[0]  # left_hip
            elif i in [2, 3]:  # Cadera derecha  
                joint_angle = joint_positions[2]  # right_hip
            elif i == 4:  # Rodilla izquierda
                joint_angle = joint_positions[1]  # left_knee
            elif i == 5:  # Rodilla derecha
                joint_angle = joint_positions[3]  # right_knee
            # Convertir presión normalizada [0,1] a presión real [Pa]
            real_pressure = self.min_pressure + pressure_normalized * (self.max_pressure - self.min_pressure)
            
            # Calcular ratio de contracción basado en ángulo articular
            if i in [0, 2]:  # Flexores de cadera
                contraction_ratio = max(0, joint_angle) / 1.2  # Flexión positiva
            elif i in [1, 3]:  # Extensores de cadera  
                contraction_ratio = max(0, -joint_angle) / 1.2  # Extensión negativa
            elif i in [4, 5]:  # Flexores de rodilla
                contraction_ratio = max(0, joint_angle) / 1.571  # Solo flexión
            
            # Limitar contracción
            contraction_ratio = np.clip(contraction_ratio, 0, 0.3)  # Seguro
            
            # ===== CALCULAR FUERZA REAL USANDO PAM_MCKIBBEN =====
            muscle_names = ['left_hip_flexor', 'left_hip_extensor', 'right_hip_flexor', 
                        'right_hip_extensor', 'left_knee_flexor', 'right_knee_flexor']
            muscle_name = muscle_names[i]
            pam_muscle = self.pam_muscles[muscle_name]
            
            # ¡AQUÍ ESTÁ LA CLAVE! Usar el modelo físico real
            pam_force = pam_muscle.force_model_new(real_pressure, contraction_ratio)
            
            # Aplicar dirección (extensores son negativos)
            if 'extensor' in muscle_name:
                pam_force = -pam_force
                
            pam_forces[i] = pam_force
            
            # Debug
            if self.step_count % 1500 == 0:  # Debug cada segundo aprox
                print(f"PAM {i} ({muscle_name}): P={real_pressure/101325:.1f}atm, "
                    f"ε={contraction_ratio:.2f}, F={pam_force:.1f}N")
                
        # ===== PASO 2: CONVERTIR FUERZAS PAM A TORQUES ARTICULARES =====
        
        # Convertir fuerzas PAM a torques articulares
        moment_arm = 0.05  # Brazo de palanca típico (5cm)
        joint_torques = np.zeros(4)
        
        # Cadera izquierda: diferencia entre extensor y flexor
        joint_torques[0] = moment_arm  * (pam_pressures[1] + pam_pressures[0])
        
        # Rodilla izquierda: solo flexor + resorte pasivo de extensión
        passive_spring_force_left = -15.0 * (joint_positions[1] - 0.1)  # Resorte a posición neutral
        joint_torques[1] = moment_arm  * (pam_pressures[4] + passive_spring_force_left)  # resorte pasivo
        
        # Cadera derecha: diferencia entre extensor y flexor  
        joint_torques[2] = moment_arm  * (pam_pressures[3] + pam_pressures[2])
        
        # Rodilla derecha: solo flexor + resorte pasivo
        passive_spring_force_right = -15.0 * (joint_positions[3] - 0.1)
        joint_torques[3] = moment_arm  * (pam_pressures[5] + passive_spring_force_right)  # resorte pasivo
        
        # Debug torques
        if self.step_count % 1500 == 0:
            print(f"Joint Torques: LH={joint_torques[0]:.1f}, LK={joint_torques[1]:.1f}, "
                f"RH={joint_torques[2]:.1f}, RK={joint_torques[3]:.1f}")
        
        # ===== PASO 3: APLICAR TORQUES CORRECTAMENTE =====
    
        # CRÍTICO: Usar índices correctos de PyBullet
        joint_indices = [0, 1, 3, 4]  # left_hip, left_knee, right_hip, right_knee
        
        for _, (joint_idx, torque) in enumerate(zip(joint_indices, joint_torques)):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torque
            )
        
        # Actualizar estados PAM para observación
        self.pam_states['pressures'] = pam_pressures
        self.pam_states['forces'] = np.abs(pam_forces)

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
        
        left_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.left_foot_id, -1)) > 0
        right_contact = len(p.getContactPoints(self.robot_id, self.plane_id, self.right_foot_id, -1)) > 0
        print ("contacto pies", left_contact, right_contact)
        obs.extend([float(left_contact), float(right_contact)])
        
        return np.array(obs, dtype=np.float32)
    
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
        
        # Altura del torso (mantenerse erguido)
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
        
        if left_contact and right_contact:
            reward += 2.0  # Ambos pies en el suelo
        elif left_contact or right_contact:
            reward += 1.0  # Al menos un pie en el suelo
        else:
            reward -= 15.0  # Penalización severa por estar en el aire
        
        # ===== PENALIZACIÓN POR MOVIMIENTO EXCESIVO =====
        
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        velocity_penalty = (abs(lin_vel[0]) + abs(lin_vel[1]) + abs(ang_vel[2])) * 2.0
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
            print("to high or too low", euler, self.pos)
            return True
            
        # Terminar si la inclinación lateral es excesiva  
        if abs(euler[1]) > math.pi/4 + 0.2 or abs(euler[1]) > math.pi/4 + 0.2:
            print("rotated", euler)
            return True
        
        # Desplazamiento lateral excesivo
        if abs(self.pos[1]) > 2.0:
            return True

            
        # Límite de tiempo
        if self.step_count > 1500*5: # 5 segundos a 1500 Hz
            print("fuera de t")
            return True
            
        return False
    

    def reset(self, seed=None, options=None):
        """
            Reset SIMPLIFICADO - Solo configuración esencial para balance
        """
        super().reset(seed=seed)
        
        # ===== RESET FÍSICO BÁSICO =====
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Cargar entorno
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(
            self.urdf_path,
            [0, 0, 1.3],  # Posición inicial de pie
            # [0, 0, 0, 1],  # Orientación neutral
            useFixedBase=False
        )
        
        # ===== CONFIGURACIÓN PARA BALANCE ESTÁTICO =====
        
        # Posiciones articulares para estar de pie (balance neutro)
        self.neutral_positions = {
            0:0.0,   # left_hip - neutral
            1:0.1,   # left_knee - ligeramente flexionada para estabilidad
            2:0.0,   # left_anckle. Por si el resorte lo dejo con angulo no nulo
            3:0.0,   # right_hip - neutral  
            4:0.1,   # right_knee - ligeramente flexionada
            5:0.0,   # right_anckle - lo mismo que antes
        }
        
        for i, pos in self.neutral_positions.items():
            p.resetJointState(self.robot_id, i, pos)
        
        # SIN velocidad inicial - queremos balance estático
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        
        # ===== CONFIGURACIÓN PAM =====

        for i, target_pos in self.neutral_positions.items():  # Solo caderas y rodillas
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=50,
                maxVelocity=0.5
            )
        
        

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

        # Controller para acciones discretas (BALANCE_STANDING, SQUAT)
        self.controller = DiscreteActionController(self)
        self.controller.set_action(ActionType.BALANCE_STANDING)  # Empezar con balance

        # Configurar sistema de recompensas
        self.reward_system.redefine_robot(self.robot_id, self.plane_id)
        
        # ===== VARIABLES DE SEGUIMIENTO =====
        
        self.step_count = 0
        self.total_reward = 0
        self.contact_established = False
        self.pam_control_active = False  # Nueva variable clave

        # Resetear estados PAM a neutro
        self.pam_states = {
            'pressures': np.zeros(6),  # Presión base para mantener postura
            'forces': np.zeros(6)
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
                    'contact_established': self.contact_established,
                    'pam_control_active': self.pam_control_active,
                    'initial_height': self.pos[2],
                    'control_mode': 'POSITION_STANDING'
                }
        
        print(f"   🔄 Environment reset - Ready for balance/squat training")
        
        return observation, info
    
    def close(self):
        try:
            if hasattr(self, 'physics_client'):
                p.disconnect(self.physics_client)
        except:
            pass

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