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
    def get_all_joint_states(self) -> dict[str, dict]:
        """
        Obtiene todos los estados de las articulaciones de una vez (más eficiente)
        
        Returns:
            Diccionario con información completa de cada articulación
        """
        # Obtener todos los índices de articulaciones activas
        joint_indices = [i for i, data in self.joint_info.items() 
                        if data['type'] in [0, 1]]
        
        # Método más eficiente: getJointStates (múltiples articulaciones)
        joint_states = p.getJointStates(self.robot_id, joint_indices)
        
        all_states = {}
        for i, joint_index in enumerate(joint_indices):
            joint_name = self.joint_info[joint_index]['name']
            state = joint_states[i]
            
            all_states[joint_name] = {
                'position': state[0],
                'velocity': state[1],
                'reaction_forces': state[2],  # Fuerzas de reacción (6 valores)
                'applied_torque': state[3],   # Torque aplicado por el motor
                'index': joint_index
            }
        
        return all_states
    
    @property
    def get_link_positions_orientations(self) -> dict[str, np.ndarray]:
        """
        Obtiene las posiciones globales de todos los links
        
        Returns:
            Diccionario con nombre_link: posición_global
        """
        positions = {}
        orientations = {}
        # Posición del link base
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        positions['base_link'] = np.array(base_pos)
        orientations['base_link'] = np.array(base_orn)
        # Posiciones de otros links
        for joint_index, joint_data in self.joint_info.items():
            link_state = p.getLinkState(self.robot_id, joint_index)
            link_name = joint_data['link_name']
            positions[link_name] = np.array(link_state[0])  # Posición del COM del link
            orientations[link_name] = np.array(link_state[1]) 
        
        return positions, orientations
    
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
            base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            weighted_position += base_mass * np.array(base_pos)
            total_mass += base_mass
        
        # Contribución de otros links
        for joint_index, joint_data in self.joint_info.items():
            link_mass = self.link_info[joint_index]['mass']
            if link_mass > 0:
                # getLinkState devuelve la posición del COM del link
                link_state = p.getLinkState(self.robot_id, joint_index)
                com_pos = np.array(link_state[0])
                
                weighted_position += link_mass * com_pos
                total_mass += link_mass
        
        if total_mass > 0:
            center_of_mass = weighted_position / total_mass
        else:
            center_of_mass = np.zeros(3)
        
        return center_of_mass, total_mass
    
    @property
    def get_stability_metrics(self) -> dict:
        """
        Calcula métricas de estabilidad del robot
        
        Returns:
            Diccionario con métricas de estabilidad
        """
        com, total_mass = self.get_center_of_mass
        
        # Obtener posiciones de los pies (asumiendo nombres específicos)
        link_positions, orientation_positions = self.get_link_positions_orientations
        
        # Buscar links de los pies
        left_foot_pos = link_positions.get('left_foot_link', np.array([0, 0.25, 0]))
        right_foot_pos = link_positions.get('right_foot_link', np.array([0, -0.25, 0]))
        
        # Calcular polígono de soporte (simplificado)
        foot_width = 0.15
        foot_length = 0.3
        
        support_polygon = np.array([
            [left_foot_pos[0] - foot_length/2, left_foot_pos[1] - foot_width/2],
            [left_foot_pos[0] + foot_length/2, left_foot_pos[1] - foot_width/2],
            [right_foot_pos[0] + foot_length/2, right_foot_pos[1] + foot_width/2],
            [right_foot_pos[0] - foot_length/2, right_foot_pos[1] + foot_width/2]
        ])
        
        # Verificar estabilidad (COM dentro del polígono)
        com_2d = com[:2]
        is_stable = self._point_in_polygon(com_2d, support_polygon)
        
        return {
            'center_of_mass': com,
            'total_mass': total_mass,
            'left_foot_position': left_foot_pos,
            'right_foot_position': right_foot_pos,
            'support_polygon': support_polygon,
            'is_stable': is_stable,
            'com_height': com[2]
        }
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Verifica si un punto está dentro de un polígono usando ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def set_joint_positions(self, target_positions: dict[str, float]):
        """
        Establece posiciones objetivo para las articulaciones
        
        Args:
            target_positions: Diccionario con nombre_articulación: posición_objetivo
        """
        for joint_name, target_pos in target_positions.items():
            # Encontrar índice de la articulación
            joint_index = None
            for index, data in self.joint_info.items():
                if data['name'] == joint_name:
                    joint_index = index
                    break
            
            if joint_index is not None:
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=self.joint_info[joint_index]['max_force']
                )
    
    def monitor_robot_state(self, duration: float = 10.0, frequency: float = 30.0):
        """
        Monitorea el estado del robot en tiempo real
        
        Args:
            duration: Duración del monitoreo en segundos
            frequency: Frecuencia de actualización en Hz
        """
        dt = 1.0 / frequency
        start_time = time.time()
        
        print("Iniciando monitoreo del robot...")
        print("Tiempo | COM_X | COM_Y | COM_Z | Masa | Estable")
        print("-" * 50)
        
        while time.time() - start_time < duration:
            # Obtener estado actual
            stability = self.get_stability_metrics
            com = stability['center_of_mass']
            
            # Mostrar información
            elapsed = time.time() - start_time
            print(f"{elapsed:6.2f} | {com[0]:5.2f} | {com[1]:5.2f} | {com[2]:5.2f} | "
                  f"{stability['total_mass']:4.1f} | {stability['is_stable']}")
            
            # Avanzar simulación
            p.stepSimulation()
            time.sleep(dt)
    
    def print_robot_summary(self):
        """Imprime un resumen completo del robot"""
        print("=== RESUMEN DEL ROBOT EN PYBULLET ===\n")
        
        print(f"Robot ID: {self.robot_id}")
        print(f"Número de articulaciones: {self.robot_info['num_joints']}")
        print(f"Posición base: {self.robot_info['base_position']}")
        print()
        
        print("ARTICULACIONES:")
        for index, data in self.joint_info.items():
            if data['type'] in [0, 1]:  # Solo articulaciones móviles
                joint_type = "Revolute" if data['type'] == 0 else "Prismatic"
                print(f"  {data['name']} (ID: {index})")
                print(f"    Tipo: {joint_type}")
                print(f"    Límites: {data['lower_limit']:.3f} a {data['upper_limit']:.3f}")
                print(f"    Fuerza máxima: {data['max_force']:.1f}")
                print()
        
        print("LINKS:")
        for index, data in self.link_info.items():
            print(f"  {data['name']} (ID: {index})")
            print(f"    Masa: {data['mass']:.2f} kg")
            print()
        
        # Estado actual
        com, total_mass = self.get_center_of_mass
        print(f"ESTADO ACTUAL:")
        print(f"  Centro de masas: {com}")
        print(f"  Masa total: {total_mass:.2f} kg")
        
        stability = self.get_stability_metrics
        print(f"  Estable: {stability['is_stable']}")
    
    #def __del__(self):
    #    """Cleanup al destruir el objeto"""
    #    try:
    #        p.disconnect()
    #    except:
    #        pass

# Ejemplo de uso
if __name__ == "__main__":
    # Crear analizador (cambiar por tu ruta URDF)
    # No sirve porque se usa robot_id
    analyzer = PyBullet_Robot_Data("2_legged_human_like_robot.urdf")
    
    # Mostrar resumen
    analyzer.print_robot_summary()
    
    # Obtener estados actuales
    print("\n=== ESTADOS ACTUALES ===")
    positions, _, _ = analyzer.get_joint_position_velocities_and_torques
    print("Posiciones de articulaciones:")
    for name, pos in positions.items():
        print(f"  {name}: {pos:.3f}")
    
    # Obtener centro de masas
    com, mass = analyzer.get_center_of_mass
    print(f"\nCentro de masas: {com}")
    print(f"Masa total: {mass:.2f} kg")
    
    # Establecer una nueva posición
    new_positions = {
        'left_hip_joint': 0.2,
        'left_knee_joint': -0.4,
        'right_hip_joint': 0.2,
        'right_knee_joint': -0.4
    }
    
    print("\nEstableciendo nueva posición...")
    analyzer.set_joint_positions(new_positions)
    
    # Simular por un tiempo
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)
    
    # Verificar nueva posición
    new_com, _ = analyzer.get_center_of_mass
    print(f"Nuevo centro de masas: {new_com}")
    
    # Opcional: monitorear en tiempo real
    # analyzer.monitor_robot_state(duration=5.0)
    
    print("\nPresiona Enter para cerrar...")
    input()