import numpy as np
import pybullet as p
from Archivos_Apoyo.SimpleWalkingCycle import SimpleWalkingCycle


class SimplifiedWalkingController:
    """
    Controlador simplificado que integra el ciclo de paso con el entorno
    """
    
    def __init__(self, env, mode="trajectory", blend_factor=0.0):
        self.env = env
        self.mode = mode
        self.blend_factor = blend_factor
        self.walking_cycle = SimpleWalkingCycle(robot_id=env.robot_id)
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0
        self.last_valid_action = np.zeros(4, dtype=np.float32)  # Inicializa acción previa
        self.last_phase = 0.0
        
    def get_next_action(self):
        """
            Obtiene la siguiente acción del ciclo de paso
            Dependiendo del modo de entrenamiento actual
        """
        print(self.mode)
        if not self.is_initialized:
            return self._get_initialization_action()
        else:
            alpha = self.walking_cycle.phase
            phase_direction = "right" if alpha < 0.5 else "left"
            swing_foot_id = self.env.right_foot_id if phase_direction == "right" else self.env.left_foot_id
            # Si se aproxima cambio de fase, esperar contacto
            if abs(alpha - 0.5) < 0.02:
                contacts = p.getContactPoints(self.env.robot_id, 0, swing_foot_id)
                if not contacts:
                    return self.last_valid_action  # Esperar hasta que haya contacto
                
            if self.mode == "trajectory":
                # Usa la trayectoria de pie (IK)
                action = self.walking_cycle.get_trajectory_walking_actions(
                    self.env.time_step,
                    self.env.left_foot_id,
                    self.env.right_foot_id
                )
            elif self.mode == "pressure":
                # Acciones sinusoidales en presión PAM
                action = self.walking_cycle.get_simple_walking_actions(self.env.time_step)
            elif self.mode == "blend":
                traj = self.walking_cycle.get_trajectory_walking_actions(
                    self.env.time_step, self.env.left_foot_id, self.env.right_foot_id
                )
                press = self.walking_cycle.get_simple_walking_actions(self.env.time_step)
                action (1-self.blend_factor) * traj + self.blend_factor * press
            else:
                raise ValueError(f"Modo de walking controller no válido: {self.mode}")
            
            # Refuerzo al final del swing
            phase = self.walking_cycle.phase
            if 0.48 < phase < 0.5 or 0.98 < phase < 1.0:
                action = action * 1.2  # Aumenta temporalmente presión/acción

            self.last_valid_action = action
            return action
            
    def _get_initialization_action(self):
        """
            Maneja la secuencia de inicialización
        """
        if self.init_sequence is None:
            self.init_sequence = self.walking_cycle.get_initialization_sequence()
        
        if self.init_step < len(self.init_sequence):
            action = self.init_sequence[self.init_step]
            self.init_step += 1
            return action
        else:
            self.is_initialized = True
            return self.get_next_action()
    
    def reset(self):
        """
            Reinicia el controlador
        """
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0
        self.walking_cycle.phase = 0.0
        self.last_valid_action = np.zeros(4, dtype=np.float32)

    def get_expert_action(self):
        # Útil para reward shaping por imitación (fase 1)
        return self.walking_cycle.get_trajectory_walking_actions(
            self.env.time_step, self.env.left_foot_id, self.env.right_foot_id
        )
    def get_expert_action_pressures(self):
        # Ahora devuelve las presiones expertas basadas en dinámica inversa
        return self.get_inverse_dynamics_pressures()

    
    def get_inverse_dynamics_pressures(self):
        """
        Calcula presiones PAM aproximadas usando dinámica inversa sobre posiciones IK
        """
        joint_states = p.getJointStates(self.env.robot_id, range(4))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        desired_accelerations = [0.0 for _ in joint_positions]  # simplificación

        torques = p.calculateInverseDynamics(
            self.env.robot_id,
            joint_positions,
            joint_velocities,
            desired_accelerations
        )

        pressures = []
        for i, torque in enumerate(torques[:4]):
            joint_name = list(self.env.joint_limits.keys())[i]
            joint_range = self.env.joint_limits[joint_name]
            normalized_pos = (joint_positions[i] - joint_range[0]) / (joint_range[1] - joint_range[0])
            contraction_ratio = np.clip(normalized_pos, 0.0, 0.8)

            pam = self.env.pam_muscles[joint_name]
            if pam.F_max_factor > 0:
                pressure = torque / (pam.area * pam.F_max_factor)
            else:
                pressure = 0.0
            pressure = np.clip(pressure, 0.0, self.env.max_pressure)
            normalized_pressure = (pressure / self.env.max_pressure) * 2.0 - 1.0
            pressures.append(normalized_pressure)

        return np.array(pressures, dtype=np.float32)


# Ejemplo de uso en tu entorno
def integrate_walking_cycle_in_env(env):
    """
    Ejemplo de cómo integrar el ciclo de paso en tu entorno
    """
    controller = SimplifiedWalkingController(env)
    
    # En tu loop de entrenamiento:
    obs, info = env.reset()
    controller.reset()
    
    for step in range(10000):
        # Obtener acción del ciclo de paso
        walking_action = controller.get_next_action()
        
        # Opcional: añadir ruido para exploración
        exploration_noise = np.random.normal(0, 0.1, size=walking_action.shape)
        action = walking_action + exploration_noise
        action = np.clip(action, -1.0, 1.0)
        
        # Ejecutar en el entorno
        obs, reward, done, truncated, info = env.step(action)
        
        if done:
            break
    
    return obs, reward, done, info