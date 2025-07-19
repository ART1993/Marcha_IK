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
        self.walking_cycle = SimpleWalkingCycle(robot_id=env.robot_id, 
                                                plane_id=env.plane_id,
                                                robot_data =env.robot_data,
                                                zmp_calculator=env.zmp_calculator,
                                                blend_factor=blend_factor)
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0
        
    def get_next_action(self):
        """
            Obtiene la siguiente acción del ciclo de paso
            Dependiendo del modo de entrenamiento actual
        """
        #print(self.mode)
        if not self.is_initialized:
            print("is initiated")
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
                print("inicio_traj")
                traj = self.walking_cycle.get_trajectory_walking_actions(
                    self.env.time_step, 
                    self.env.left_foot_id, 
                    self.env.right_foot_id
                )
                if traj is None:
                    return traj
                press = self.walking_cycle.get_simple_walking_actions(self.env.time_step)
                traj = np.asarray(traj, dtype=np.float32)
                press = np.asarray(press, dtype=np.float32)
                print("press",press*self.blend_factor)
                print("traj",self.blend_factor, (1-self.blend_factor)*traj)
                # press y traj deben de ser trayectoria experta y de 
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

        self.walking_cycle.swing_leg=None
        self.walking_cycle.stand_leg=None
        if hasattr(self.walking_cycle, 'double_support_counter'):
            self.walking_cycle.double_support_counter = 0

    def get_expert_action(self):
        # Útil para reward shaping por imitación (fase 1)
        return self.walking_cycle.get_trajectory_walking_actions(
            self.env.time_step, self.env.left_foot_id, self.env.right_foot_id
        )