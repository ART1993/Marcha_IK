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
    def get_joint_position_velocities_and_torques(self) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
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
                torques[joint_data['name']] = joint_state[3]  # Par de motor aplicado
                # reaction_6d = joint_state[2]  # (Fx,Fy,Fz,Mx,My,Mz) reacción del constraint
        
        return positions, velocities, torques
    
    def get_center_of_mass(self):
        """
            Cálculo del COM global del robot.
            PyBullet ya entrega la **posición mundial del COM del link** en `ls[0]`
            cuando llamas a `getLinkState(..., computeForwardKinematics=True)`;
            no hay que volver a sumar el offset inercial local (`ls[2]`).

            Para la base (-1), transformamos su marco inercial local a mundo una sola vez.
        """
        total_mass = 0.0
        weighted_position = np.zeros(3, dtype=float)

        # --- BASE (-1) ---
        base_mass = self.link_info[-1]['mass']
        if base_mass > 0:
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
            base_com_local, base_com_local_orn = p.getDynamicsInfo(self.robot_id, -1)[3:5]
            base_com_world, _ = p.multiplyTransforms(base_pos, base_orn, base_com_local, base_com_local_orn)
            weighted_position += base_mass * np.array(base_com_world, dtype=float)
            total_mass += base_mass


        for joint_index, joint_data in self.joint_info.items():
            link_mass = self.link_info[joint_index]['mass']
            if link_mass > 0:
                ls = p.getLinkState(self.robot_id, joint_index, computeForwardKinematics=True)
                com_world = ls[0]  # COM del link en mundo
                weighted_position += link_mass * np.array(com_world, dtype=float)
                total_mass += link_mass

        return (weighted_position / max(total_mass, 1e-9)).tolist(), total_mass
    

    def get_center_of_mass_old(self) -> tuple[np.ndarray, float]:
        """
        NOTA: para cada link, getLinkState(..., computeForwardKinematics=True)
            devuelve:
            ls[0] = pos del marco del link en MUNDO
            ls[1] = orn del marco del link en MUNDO (quat)
            ls[2] = offset inercial LOCAL del COM (en el marco del link)
            Por tanto, hay que transformar ese offset a mundo.
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
                ls = p.getLinkState(self.robot_id, joint_index, computeForwardKinematics=True)
                # ls[0], ls[1]: pose del link en mundo; ls[2]: COM local inercial
                # En quickstart guide veo que es 0
                link_pos_world  = ls[0]
                link_orn_world  = ls[1]
                com_local_inert = ls[2]
                 # Transformar offset inercial local -> mundo
                com_world = p.multiplyTransforms(link_pos_world, link_orn_world,
                                             com_local_inert, [0,0,0,1])[0]
                weighted_position += link_mass * np.array(com_world)
                total_mass += link_mass
        
        if total_mass > 0:
            center_of_mass = weighted_position / total_mass
        else:
            center_of_mass = np.zeros(3)
        
        return center_of_mass, total_mass
    
    def get_center_of_mass_velocity(self):
        """
        Opción A: media ponderada por masa de las velocidades lineales del COM de cada link.
        Requiere computeLinkVelocity=True para los links.
        """
        total_mass = 0.0
        weighted_vel = np.zeros(3)

        # --- Base (-1): v_base + omega x r_com_world ---
        base_mass = self.link_info[-1]['mass']
        if base_mass > 0:
            (base_pos, base_orn) = p.getBasePositionAndOrientation(self.robot_id)
            (v_lin_base, v_ang_base) = p.getBaseVelocity(self.robot_id)

            base_com_local = p.getDynamicsInfo(self.robot_id, -1)[3]  # offset local del COM base
            # r_com_world = R(base_orn) * base_com_local
            r_com_world = p.multiplyTransforms([0,0,0], base_orn, base_com_local, [0,0,0,1])[0]
            v_base_com = np.array(v_lin_base) + np.cross(np.array(v_ang_base), np.array(r_com_world))

            weighted_vel += base_mass * v_base_com
            total_mass  += base_mass

        # --- Resto de links: usar v_lin del COM que da getLinkState ---
        for jid in self.joint_info.keys():
            link_mass = self.link_info[jid]['mass']
            if link_mass > 0:
                ls = p.getLinkState(self.robot_id, jid,
                                    computeForwardKinematics=True,
                                    computeLinkVelocity=True)
                # ls[6] = vel lineal del COM del link en mundo (PyBullet)
                # ls[7] = vel angular del link en mundo
                v_lin_com = np.array(ls[6], dtype=float)
                weighted_vel += link_mass * v_lin_com
                total_mass  += link_mass

        if total_mass <= 0:
            return np.zeros(3)
        return weighted_vel / total_mass

    # ------- Fallback por diferencias finitas + suavizado opcional -------
    _prev_com = None
    _prev_com_v = np.zeros(3)


    def get_center_of_mass_velocity_fd(self, dt, alpha=0.2):
        """
        Opción B: diferencias finitas del COM global, con EMA para suavizar.
        dt: paso de integración real (tu sim_dt).
        alpha: 0..1 (más alto = menos suavizado).
        """
        com, _ = self.get_center_of_mass()
        if self._prev_com is None or dt <= 0:
            self._prev_com = np.array(com, dtype=float)
            self._prev_com_v = np.zeros(3)
            return np.zeros(3)

        v = (np.array(com) - self._prev_com) / float(dt)
        self._prev_com = np.array(com, dtype=float)
        # EMA
        self._prev_com_v = (1 - alpha) * self._prev_com_v + alpha * v
        return self._prev_com_v