import time

import pybullet as p
import numpy as np  


class PyBullet_Robot_Data:

    def __init__(self, robot_id):
        self.robot_id=robot_id
        self.is_initialized = False
        self.init_sequence = None
        self.init_step = 0

        # Obtener información del robot
        self.robot_info = self._get_robot_info
        self.joint_info = self._get_joint_info
        self.link_info = self._get_link_info

    @property
    def _get_robot_info(self) -> dict:
        """Obtiene información general del robot"""
        num_joints = p.getNumJoints(self.robot_id)
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        
        return {
            'robot_id': self.robot_id,
            'num_joints': num_joints,
            'base_position': np.array(base_pos),
            'base_orientation': np.array(base_orn)
        }
    
    @property
    def _get_joint_info(self) -> dict:
        """Obtiene información detallada de todas las articulaciones"""
        joint_info = {}
        
        for i in range(self.robot_info['num_joints']):
            info = p.getJointInfo(self.robot_id, i)
            joint_info[i] = {
                'index': info[0],
                'name': info[1].decode('utf-8'),
                'type': info[2],  # 0=revolute, 1=prismatic, 4=fixed
                'damping': info[6],
                'friction': info[7],
                'lower_limit': info[8],
                'upper_limit': info[9],
                'max_force': info[10],
                'max_velocity': info[11],
                'link_name': info[12].decode('utf-8'),
                'joint_axis': info[13],
                'parent_frame_pos': info[14],
                'parent_frame_orn': info[15],
                'parent_index': info[16]
            }
        
        return joint_info

    @property
    def _get_link_info(self) -> dict:
        """Obtiene información de los links"""
        link_info = {}
        
        # Información del link base
        base_mass = p.getDynamicsInfo(self.robot_id, -1)[0]  # -1 para base
        link_info[-1] = {
            'name': 'base_link',
            'mass': base_mass,
            'index': -1
        }
        
        # Información de otros links
        for i in range(self.robot_info['num_joints']):
            dynamics_info = p.getDynamicsInfo(self.robot_id, i)
            link_info[i] = {
                'name': self.joint_info[i]['link_name'],
                'mass': dynamics_info[0],
                'lateral_friction': dynamics_info[1],
                'local_inertia_diagonal': dynamics_info[2],
                'local_inertial_pos': dynamics_info[3],
                'local_inertial_orn': dynamics_info[4],
                'restitution': dynamics_info[5],
                'rolling_friction': dynamics_info[6],
                'spinning_friction': dynamics_info[7],
                'contact_damping': dynamics_info[8],
                'contact_stiffness': dynamics_info[9],
                'index': i
            }
        
        return link_info
    
    @property
    def get_joint_position_velocities_and_torques(self) -> dict[str, float]:
        """
        Obtiene las velocidades actuales de todas las articulaciones
        
        Returns:
            Diccionario con nombre_articulación: velocidad
        """
        velocities = {}
        torques = {}
        positions = {}
        for joint_index, joint_data in self.joint_info.items():
            if joint_data['type'] in [0, 1]:  # Revolute o prismatic
                joint_state = p.getJointState(self.robot_id, joint_index)
                positions[joint_data['name']] = joint_state[0]  # Posición
                velocities[joint_data['name']] = joint_state[1]  # Velocidad
                # if pVELOCITIES o p.TORQUE usado
                torques[joint_data['name']] = joint_state[3]  # Torque de reacción
        
        return positions, velocities, torques
    
    @property
    def get_center_of_mass(self) -> tuple[np.ndarray, float]:
        """
        Calcula el centro de masas global del robot
        
        Returns:
            Tuple con (posición_COM, masa_total)
        """
        total_mass = 0.0
        weighted_position = np.zeros(3)
        
        # Contribución del link base
        base_mass = self.link_info[-1]['mass']
        if base_mass > 0:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_com_local = p.getDynamicsInfo(self.robot_id, -1)[3]  # pos inercial local
            base_com_world = p.multiplyTransforms(base_pos, base_orn, base_com_local, [0,0,0,1])[0]
            weighted_position += base_mass * np.array(base_com_world)
            total_mass += base_mass
        
        # Contribución de otros links
        for joint_index, joint_data in self.joint_info.items():
            link_mass = self.link_info[joint_index]['mass']
            if link_mass > 0:
                # getLinkState devuelve la posición del COM del link
                 # getLinkState: [2] = world COM position del link
                ls = p.getLinkState(self.robot_id, joint_index, computeForwardKinematics=True)
                # En quickstart guide veo que es 0
                com_pos = np.array(ls[2] if ls[2] is not None else ls[0])
                
                weighted_position += link_mass * com_pos
                total_mass += link_mass
        
        if total_mass > 0:
            center_of_mass = weighted_position / total_mass
        else:
            center_of_mass = np.zeros(3)
        
        return center_of_mass, total_mass