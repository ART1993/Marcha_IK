import numpy as np
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
        
    def get_next_action(self):
        """
            Obtiene la siguiente acción del ciclo de paso
            Dependiendo del modo de entrenamiento actual
        """
        if not self.is_initialized:
            return self._get_initialization_action()
        else:
            if self.mode == "trajectory":
                # Usa la trayectoria de pie (IK)
                return self.walking_cycle.get_trajectory_walking_actions(
                    self.env.time_step,
                    self.env.left_foot_id,
                    self.env.right_foot_id
                )
            elif self.mode == "pressure":
                # Acciones sinusoidales en presión PAM
                return self.walking_cycle.get_simple_walking_actions(self.env.time_step)
            elif self.mode == "blend":
                traj = self.walking_cycle.get_trajectory_walking_actions(
                    self.env.time_step, self.env.left_foot_id, self.env.right_foot_id
                )
                press = self.walking_cycle.get_simple_walking_actions(self.env.time_step)
                return (1-self.blend_factor) * traj + self.blend_factor * press
            else:
                raise ValueError(f"Modo de walking controller no válido: {self.mode}")
            
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

    def get_expert_action(self):
        # Útil para reward shaping por imitación (fase 1)
        return self.walking_cycle.get_trajectory_walking_actions(
            self.env.time_step, self.env.left_foot_id, self.env.right_foot_id
        )


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