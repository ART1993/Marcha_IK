import numpy as np
from Archivos_Apoyo.SimpleWalkingCycle import SimpleWalkingCycle


class SimplifiedWalkingController:
    """
    Controlador simplificado que integra el ciclo de paso con el entorno
    """
    
    def __init__(self, env):
        self.env = env
        self.walking_cycle = SimpleWalkingCycle(robot_id=env.robot_id)
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0
        
    def get_next_action(self):
        """
        Obtiene la siguiente acción del ciclo de paso
        """
        if not self.is_initialized:
            return self._get_initialization_action()
        else:
            return self.walking_cycle.get_simple_walking_actions(self.env.time_step)
    
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
            return self.walking_cycle.get_simple_walking_actions(self.env.time_step)
    
    def reset(self):
        """
        Reinicia el controlador
        """
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0
        self.walking_cycle.phase = 0.0
        #com, _ = self.robot_data.get_center_of_mass
        #height = com[2]
        #self.walking_cycle.step_length = 0.25 + 0.05 * (height - 0.5)

        #stability = self.robot_data.get_stability_metrics()
        #if not stability['is_stable']:
            # Disminuir amplitud del paso si es inestable
        #    for joint in self.walking_cycle.modulation_amplitudes:
        #        self.walking_cycle.modulation_amplitudes[joint] *= 0.8


# Ejemplo de uso en tu entorno
def integrate_walking_cycle_in_env(env):
    """
    Ejemplo de cómo integrar el ciclo de paso en tu entorno
    """
    controller = SimplifiedWalkingController(env)
    
    # En tu loop de entrenamiento:
    obs, info = env.reset()
    controller.reset()
    
    for step in range(1000):
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