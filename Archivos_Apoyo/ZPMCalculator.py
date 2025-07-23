import numpy as np
import pybullet as p

from Archivos_Apoyo.Pybullet_Robot_Data import PyBullet_Robot_Data


class ZMPCalculator:
    """
    Calculador de Zero Moment Point para robot bípedo
    Implementa las ecuaciones que proporcionaste
    """
    
    def __init__(self, robot_id, left_foot_id, right_foot_id, robot_data):
        self.robot_id = robot_id
        self.left_foot_id = left_foot_id
        self.right_foot_id = right_foot_id
        self.robot_data=robot_data

        # Dimensiones formato (x,y,z,masa)//(m,m,m,kg) para cubo
        # para cilindro_
        self.dimensiones_robot={
            "cubic":{
                "waist":(0.3,0.8,0.1,6),
                "foot_left":(0.5,0.35,0.05,1.5),
                "foot_right":(0.5,0.35,0.05,1.5)},
            "cilinder":{
                "thigh_left":(0.08,0.6,4),
                "thight_right":(0.08,0.6,4),
                "shin_left":(0.06,0.5,4),
                "shin_right":(0.06,0.5,4)
            }
        }
        # Selecciona la altura del centro de masa
        com, _ = self.robot_data.get_center_of_mass
        height = com[2]
        
        # Parámetros del modelo (ajustar según tu robot)
        self.g = 9.81  # gravedad
        self.l = height   # altura aproximada del centro de masa selecciono entre 0.8 y 0.4
        
        # Historia para cálculo de aceleraciones
        self.com_history = []
        self.max_history = 5
        
        # Límites de estabilidad (margen de seguridad)
        self.stability_margin = 0.25  # 5cm de margen

        self.initialization_steps = 0
        self.min_step_stability=15
        
    def update_com_history(self, com_position):
        """Actualiza el historial del centro de masa"""
        self.com_history.append(com_position)
        if len(self.com_history) > self.max_history:
            self.com_history.pop(0)
    
    def calculate_com_acceleration(self, dt=1.0/1500.0):
        """Calcula aceleración del centro de masa usando diferencias finitas"""
        if len(self.com_history) < 3:
            return np.array([0.0, 0.0, 0.0])
        
        # Usar diferencias finitas de segundo orden
        pos_current = np.array(self.com_history[-1])
        pos_prev = np.array(self.com_history[-2])
        pos_prev2 = np.array(self.com_history[-3])
        
        # Aceleración = (pos_current - 2*pos_prev + pos_prev2) / dt^2
        acceleration = (pos_current - 2*pos_prev + pos_prev2) / (dt**2)

        # Filtrar valores extremos durante inicialización
        acceleration = np.clip(acceleration, -50.0, 50.0)
        return acceleration
    
    def get_support_polygon(self):
        """
            Obtiene el polígono de soporte basado en contactos con el suelo
        """
        left_contacts = p.getContactPoints(self.robot_id, 0, self.left_foot_id)
        right_contacts = p.getContactPoints(self.robot_id, 0, self.right_foot_id)
        
        support_points = []
        
        # Obtener posiciones de contacto
        if left_contacts:
            left_pos = p.getLinkState(self.robot_id, self.left_foot_id)[0]
            support_points.append([left_pos[0], left_pos[1]])  # Solo x,y
            
        if right_contacts:
            right_pos = p.getLinkState(self.robot_id, self.right_foot_id)[0]
            support_points.append([right_pos[0], right_pos[1]])  # Solo x,y
        
        return np.array(support_points) if support_points else np.array([[0, 0]])
    
    def calculate_zmp(self, dt=1.0/1500.0):
        """
            Calcula ZMP usando las ecuaciones proporcionadas
        """
        # Obtener posición actual del centro de masa
        #com_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        com_pos, _ = self.robot_data.get_center_of_mass
        com_pos = np.array(com_pos)
        
        # Actualizar historial
        self.update_com_history(com_pos)
        
        # Calcular aceleración
        com_acceleration = self.calculate_com_acceleration(dt)
        
        # Ecuaciones ZMP del TFM:
        # X_zmp = X_mc - (l/g) * l^2 * X_mc_ddot
        # Simplificando: X_zmp = X_mc - (l^3/g) * X_mc_ddot
        
        zmp_x = com_pos[0] - (self.l**3 / self.g) * com_acceleration[0]
        zmp_y = com_pos[1] - (self.l**3 / self.g) * com_acceleration[1]
        
        # Para la ecuación lateral (si hubiera movimiento oscilatorio):
        # Y_zmp = A(1 + l*omega^2/g)*sin(omega*t)
        # Esta es más compleja y requiere conocer la frecuencia omega
        
        return np.array([zmp_x, zmp_y])
    
    def is_stable(self, zmp_point=None):
        """
            Verifica si el ZMP está dentro del polígono de soporte
        """

        if self.initialization_steps < self.min_step_stability:
            self.initialization_steps += 1
            return True
        
        if zmp_point is None:
            zmp_point = self.calculate_zmp()
        
        support_polygon = self.get_support_polygon()
        
        if len(support_polygon) == 0:
            return False  # Sin contacto = inestable
        
        if len(support_polygon) == 1:
            # Solo un pie en contacto - verificar distancia
            distance = np.linalg.norm(zmp_point - support_polygon[0])
            return distance < self.stability_margin
        
        # Dos pies - verificar si ZMP está entre ellos (versión simplificada)
        min_x = min(support_polygon[:, 0]) - self.stability_margin
        max_x = max(support_polygon[:, 0]) + self.stability_margin
        min_y = min(support_polygon[:, 1]) - self.stability_margin
        max_y = max(support_polygon[:, 1]) + self.stability_margin
        
        return (min_x <= zmp_point[0] <= max_x and 
                min_y <= zmp_point[1] <= max_y)
    
    def stability_margin_distance(self, zmp_point=None):
        """
            Calcula la distancia del ZMP al borde del polígono de soporte
            Positivo = dentro, negativo = fuera
        """
        if zmp_point is None:
            zmp_point = self.calculate_zmp()
        
        support_polygon = self.get_support_polygon()
        
        if len(support_polygon) == 0:
            return -1.0  # Muy inestable
        
        if len(support_polygon) == 1:
            return self.stability_margin - np.linalg.norm(zmp_point - support_polygon[0])
        
        # Calcular distancia al rectángulo de soporte
        min_x, max_x = min(support_polygon[:, 0]), max(support_polygon[:, 0])
        min_y, max_y = min(support_polygon[:, 1]), max(support_polygon[:, 1])
        
        # Distancia al borde más cercano
        dist_x = min(zmp_point[0] - min_x, max_x - zmp_point[0])
        dist_y = min(zmp_point[1] - min_y, max_y - zmp_point[1])
        
        return min(dist_x, dist_y)
    
    def _calculate_zmp_reward(self, zmp_history, max_zmp_history, 
                              stability_bonus, instability_penalty, zmp_reward_weight):
        """
            Calcula recompensa basada en estabilidad ZMP
        """
        try:
            zmp_point = self.calculate_zmp()
            is_stable = self.is_stable(zmp_point)
            margin_distance = self.stability_margin_distance(zmp_point)
            
            # Guardar en historial
            zmp_history.append({
                'zmp': zmp_point,
                'stable': is_stable,
                'margin': margin_distance
            })
            
            if len(zmp_history) > max_zmp_history:
                zmp_history.pop(0)
            
            # Calcular recompensa
            if is_stable:
                # Bonificación por estabilidad + bonificación por margen
                reward = stability_bonus + max(0, margin_distance * 10)
            else:
                # Penalización por inestabilidad
                reward = instability_penalty + margin_distance * 10  # margin_distance es negativo

            new_reward=reward * zmp_reward_weight

            salida = (new_reward,zmp_history, max_zmp_history, 
                              stability_bonus, instability_penalty, zmp_reward_weight)
            return salida
            
        except Exception as e:
            print(f"Error calculando ZMP reward: {e}")
            return 0.0