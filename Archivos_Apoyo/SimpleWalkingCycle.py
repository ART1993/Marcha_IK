import pybullet as p
import numpy as np
import math
from random import randint
import time

class SimpleWalkingCycle:
    """
    Ciclo de paso sencillo para robot blando con músculos PAM
    Mantiene 6 articulaciones pero con patrones simplificados
    """
    
    def __init__(self, robot_id, plane_id, robot_data, 
                 zmp_calculator, step_frequency=1.0, step_length=0.1,
                 blend_factor=0.0):
        self.step_frequency = step_frequency  # Hz
        self.step_length = step_length        # metros
        self.robot_id=robot_id
        self.plane_id=plane_id
        self.robot_data = robot_data
        self.zmp_calculator = zmp_calculator
        self.blend_factor = blend_factor
        
        self.warking_cycle_params
        
    def get_simple_walking_actions(self, time_step):
        """
        Genera acciones de paso sencillas basadas en patrones senoidales
        Returns: array de 6 presiones PAM normalizadas [-1, 1]
        """
        self.update_phase(time_step)
        
        # Patrones básicos usando senos desfasados
        left_leg_phase = self.phase * 2 * math.pi
        right_leg_phase = (self.phase + 0.5) * 2 * math.pi  # Desfase de 180°
        
        actions = []
        
        # PIERNA IZQUIERDA
        # Cadera izquierda: flexión/extensión
        hip_left = self.base_pressures['left_hip'] + \
                  self.modulation_amplitudes['left_hip'] * math.sin(left_leg_phase)
        
        # Rodilla izquierda: flexión durante swing
        knee_left = self.base_pressures['left_knee'] + \
                   self.modulation_amplitudes['left_knee'] * max(0, math.sin(left_leg_phase))
        
        # Tobillo izquierdo: estabilización
        ankle_left = self.base_pressures['left_ankle'] + \
                    self.modulation_amplitudes['left_ankle'] * 0.5 * math.sin(left_leg_phase)
        
        # PIERNA DERECHA (desfasada 180°)
        hip_right = self.base_pressures['right_hip'] + \
                   self.modulation_amplitudes['right_hip'] * math.sin(right_leg_phase)
        
        knee_right = self.base_pressures['right_knee'] + \
                    self.modulation_amplitudes['right_knee'] * max(0, math.sin(right_leg_phase))
        
        ankle_right = self.base_pressures['right_ankle'] + \
                     self.modulation_amplitudes['right_ankle'] * 0.5 * math.sin(right_leg_phase)
        
        # Compilar acciones y normalizar a [-1, 1]
        pressures = [hip_left, knee_left, ankle_left, hip_right, knee_right, ankle_right]
        
        # Convertir de [0,1] a [-1,1] para el entorno
        actions = [2.0 * pressure - 1.0 for pressure in pressures]
        
        # Asegurar límites
        actions = [max(-1.0, min(1.0, action)) for action in actions]
        
        return np.array(actions, dtype=np.float32)
    
    def get_initialization_sequence(self, duration=10.0, dt=0.01):
        """
        Genera secuencia de inicialización para estabilizar el robot
        """
        steps = int(duration / dt)
        sequence = []
        
        for i in range(steps):
            # Ramp-up gradual de presiones
            ramp_factor = min(1.0, i / (steps * 0.3))  # 30% del tiempo para ramp-up
            
            # Presiones simétricas iniciales
            base_pressure = 0.3 * ramp_factor
            
            actions = [
                base_pressure, #*self.blend_factor,      # left_hip
                base_pressure, #*self.blend_factor, # left_knee  
                base_pressure, #*self.blend_factor, # left_ankle
                base_pressure, #*self.blend_factor,      # right_hip
                base_pressure, #*self.blend_factor, # right_knee
                base_pressure #*self.blend_factor  # right_ankle
            ]
            
            # Normalizar a [-1,1]
            actions = [2.0 * action - 1.0 for action in actions]
            sequence.append(np.array(actions, dtype=np.float32))
        
        return sequence
    
################################################################################################################################
###############################################Parabola simple para trayectorias de pie#########################################
################################################################################################################################
    
    def foot_parabola(self, start_pos, end_pos, height, alpha):
        """
        Trayectoria parabólica simple entre dos puntos en el espacio.
        Args:
            start_pos: posición inicial [x, y, z]
            end_pos: posición final [x, y, z]
            height: altura máxima del pie
            alpha: progreso del swing [0,1]
        Returns:
            posición 3D en [x, y, z]
        """
        x = (1 - alpha) * start_pos[0] + alpha * end_pos[0]
        y = (1 - alpha) * start_pos[1] + alpha * end_pos[1]
        z = (1 - alpha) * start_pos[2] + alpha * end_pos[2] + 4 * height * alpha * (1 - alpha)
        return [x, y, z]
    
    def get_trajectory_walking_actions(self, time_step, left_foot_index, right_foot_index):
        """
        Genera acciones de paso basadas en trayectoria del pie y cinemática inversa.
        Requiere índices de links de pies izquierdo y derecho.
        Returns: array de posiciones articulares
        """
        self.left_foot_index = left_foot_index
        self.right_foot_index = right_foot_index
        self.update_phase(time_step)
        alpha = (self.phase % 1.0)  # [0,1]
        step_length = self.step_length
        step_height = self.step_height  # altura del pie durante swing
        # Obtener posiciones actuales de los pies
        # --- INICIO DEL NUEVO BLOQUE ---
        plane_id = self.plane_id

        # Si left_foot_touchdown o right_foot_touchdown son True, significa que el pie está en contacto con el suelo
        #Empezaría con ambos true entonces si ambos true ó right es true y left es false, que se mueva el pie izquierdo
        # Si ambos vuelven a ser true y antes se movia la izquierda, entonces se mueve la pierna derecha y la izquierda permanece como estabilizador
        left_foot_touchdown, right_foot_touchdown = self.detectar_contacto_pies_suelo
        if left_foot_touchdown==False and right_foot_touchdown==False:
            # Acción neutral del mismo shape
            action = np.zeros(6, dtype=np.float32) # Fallback
            return action
        print(f"{left_foot_touchdown=:}", f"{right_foot_touchdown=:}")
        print(self.swing_leg, self.stand_leg)
        if self.swing_leg == "right" and self.stand_leg == "left":
            swing_foot_id = self.right_foot_index
            support_foot_id = self.left_foot_index
        elif self.swing_leg == "left" and self.stand_leg == "right":
            swing_foot_id = self.left_foot_index
            support_foot_id = self.right_foot_index

        link_swing_state = p.getLinkState(self.robot_id, swing_foot_id, computeLinkVelocity=1)
        link_support_state = p.getLinkState(self.robot_id, support_foot_id)

        vz= link_swing_state[6][2]  # Velocidad vertical del pie en swing

        #if left_foot_touchdown>0 and self.phase < 0.5:
        #    self.fase = 0.5  # Forzar cambio de pierna de soporte
        #elif right_foot_touchdown>0 and self.phase >= 0.5:
        #    self.fase = 0.0

        contact_points = p.getContactPoints(self.robot_id, plane_id, support_foot_id)
        en_contacto = len(contact_points) > 0



        if vz > 0:
            print("Ascenso")
        else:
            print("Descenso")

        if en_contacto:
            print("El pie de soporte ESTÁ en contacto con el suelo")
        else:
            print("El pie de soporte NO está en contacto con el suelo")

        ground_z = 0.0  # O usa la altura real del plano si es diferente

        print(self.robot_id)
        #left_start_link = p.getLinkState(self.robot_id, left_foot_index, computeLinkVelocity=1)
        #right_start_link = p.getLinkState(self.robot_id, right_foot_index, computeLinkVelocity=1)
        #left_start, left_speed = left_start_link[0], left_start_link[6]
        #right_start, right_speed = right_start_link[0], right_start_link[6]
        #vzl, vzr = left_speed[2], right_speed[2]  # velocidad vertical de los pies


        joint_positions = self.selection_swang_and_stance(alpha, link_swing_state, link_support_state, 
                                                          step_length, step_height, swing_foot_id, support_foot_id)


        return joint_positions  # posición deseada de las articulaciones
    

############################################################################################################################
########################################Agregar al entorno de simulación en futuro##########################################
############################################################################################################################
    @ property
    def detectar_contacto_pies_suelo(self):
        """
            Genera una tupla en la que determina el contacto de los pies con el suelo\n
            Fuerzo al robot a seleccionar aleatoriamente al inicio si iniciar pierna derecha o izquierda con un random\n
            Cuando se inicie, indicare en cada paso del pie cual debe de seleccionar para el swing y cual para el stance con los condicionales\n
            Returns:
                True si ambos pies están en contacto, False de lo contrario.
        """
        left_foot_touchdown = len(p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.left_foot_index, linkIndexB=-1))
        right_foot_touchdown = len(p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=self.right_foot_index, linkIndexB=-1))
        for i in range(p.getNumJoints(self.robot_id)):
            contacts = p.getContactPoints(self.robot_id, self.plane_id, linkIndexA=i, linkIndexB=-1)
            if contacts:
                print(f"Link {i} ({p.getJointInfo(self.robot_id, i)[12].decode()}): tiene contacto con el suelo")
        if left_foot_touchdown>0 and right_foot_touchdown>0 and self.swing_leg == "right" and self.stand_leg == "right":
            # Ambos pies en contacto, no mover ninguno
            valor= randint(0, 1)
            if valor == 0:
                self.swing_leg = "left"
                self.stand_leg = "right"
            else:
                self.swing_leg = "right"
                self.stand_leg = "left"
        elif left_foot_touchdown and right_foot_touchdown and self.swing_leg == "right" and self.stand_leg == "left":
            self.swing_leg = "left"
            self.stand_leg = "right"
        elif left_foot_touchdown and right_foot_touchdown and self.swing_leg == "left" and self.stand_leg == "right":
            self.swing_leg = "right"
            self.stand_leg = "left"
        elif left_foot_touchdown is True and right_foot_touchdown is False:
            self.swing_leg = "right"
            self.stand_leg = "left"
            print("Pie izquierdo en contacto, moviendo pie derecho")
        elif left_foot_touchdown is False and right_foot_touchdown is True:
            self.swing_leg = "left"
            self.stand_leg = "right"
            print("Pie derecho en contacto, moviendo pie izquierdo")
        else:
            print("Ningún pie en contacto, problematico ,moviendo el pie derecho por defecto")
        
        return left_foot_touchdown>0, right_foot_touchdown>0

    @property
    def warking_cycle_params(self):
        self.phase = 0.0                      # Fase del ciclo (0-1)
        # Configuración simplificada del ciclo
        self.cycle_phases = {
            'stance_left': (0.0, 0.5),   # Pie izquierdo en suelo
            'stance_right': (0.5, 1.0),  # Pie derecho en suelo  
        }
        
        # Patrones de presión base para cada articulación (0-1)
        self.base_pressures = {
            'left_hip': 0.3*self.blend_factor,
            'left_knee': 0.2*self.blend_factor,
            'left_ankle': 0.4*self.blend_factor,
            'right_hip': 0.3*self.blend_factor,
            'right_knee': 0.2*self.blend_factor,
            'right_ankle': 0.4*self.blend_factor
        }
        
        # Amplitudes de modulación para cada articulación
        self.modulation_amplitudes = {
            'left_hip': 0.4*(1-self.blend_factor),
            'left_knee': 0.6*(1-self.blend_factor),
            'left_ankle': 0.3*(1-self.blend_factor),
            'right_hip': 0.4*(1-self.blend_factor),
            'right_knee': 0.6*(1-self.blend_factor),
            'right_ankle': 0.3*(1-self.blend_factor)
        }

        self.left_foot_index = 2
        self.right_foot_index = 5
        self.step_height = 0.10
        self.phase = 0.0  # [0, 1)
        self.fase = 0.0  # Fase del ciclo de paso (0-1)
        self.swing_leg = "right"  # O "left", alterna cada medio ciclo
        self.stand_leg = "right"

    def update_phase(self, dt):
        """Actualiza la fase del ciclo de paso"""
        self.phase += dt * self.step_frequency
        self.phase = self.phase % 1.0  # Mantener en rango [0,1]

    def selection_swang_and_stance(self, alpha, link_swing_state, link_support_state, 
                                   step_length, step_height, swing_foot_id, support_foot_id):
        start_swing=link_swing_state[0]  # Posición inicial del pie en swing
        start_support=link_support_state[0]  # Posición del pie en stance
        end = [start_swing[0] + self.step_length, start_swing[1], start_swing[2]]
        ctrl1 = [start_swing[0] + 0.1, start_swing[1], start_swing[2] + step_height]
        ctrl2 = [end[0] - 0.1, end[1], end[2] + step_height]
        target_pos = foot_bezier_parabola(start_swing, end, ctrl1, ctrl2, alpha, step_height)
        swing_joint_positions = p.calculateInverseKinematics(self.robot_id, swing_foot_id, target_pos)

        support_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # 3. ¿El COM está sobre el área de soporte?
        stability_metrics = self.robot_data.get_stability_metrics
        com = stability_metrics['center_of_mass']
        is_stable_com = stability_metrics['is_stable']

        # 4. ¿El ZMP está dentro del polígono de soporte?
        zmp_point = self.zmp_calculator.calculate_zmp()
        zmp_estable = self.zmp_calculator.is_stable(zmp_point)

        if zmp_estable:
            print("El ZMP está DENTRO del polígono de soporte (ZMP ESTABLE)")
        else:
            print("¡ALERTA! El ZMP está FUERA del polígono de soporte")

        if is_stable_com:
            print("El centro de masa está sobre el área de soporte (COM ESTABLE)")
        else:
            print("¡CUIDADO! El COM está fuera del área de soporte")

        # 4.3. Si está cerca del límite del polígono de soporte, corrige el pie de soporte hacia el ZMP (o COM)
        correction_factor = 0.08
        support_pos=list(start_support)  # Posición del pie en stance
        if not zmp_estable or not is_stable_com:
            # Calcula la diferencia entre el soporte (stance) y el ZMP proyectado al suelo
            delta_x = zmp_point[0] - support_pos[0]
            delta_y = zmp_point[1] - support_pos[1]
            # Aplica solo una fracción para evitar movimientos bruscos
            support_pos[0] += correction_factor * delta_x
            support_pos[1] += correction_factor * delta_y
        
        support_joint_positions = p.calculateInverseKinematics(self.robot_id, support_foot_id, start_support, support_orientation)
        
        if self.swing_leg =="right" and self.stand_leg == "left":
            # Pierna derecha en swing, izquierda en stance
            joint_positions = [
                support_joint_positions[0], support_joint_positions[1], support_joint_positions[2], # left leg
                swing_joint_positions[3], swing_joint_positions[4], swing_joint_positions[5]      # right leg
            ]
        else:
            joint_positions = [swing_joint_positions[0], swing_joint_positions[1], swing_joint_positions[2],      # left leg
                                support_joint_positions[3], support_joint_positions[4], support_joint_positions[5], # right leg
        ]
        return joint_positions
    
    def condicionar_pie_paralelo_suelo(self): 
        """
        Asegura que los pies estén paralelos al suelo.
        Args:
            robot_id: ID del robot en PyBullet
            left_foot_index: índice del pie izquierdo
            right_foot_index: índice del pie derecho
        """
        plane_id = self.plane_id

        # Si left_foot_touchdown o right_foot_touchdown son True, significa que el pie está en contacto con el suelo
        #Empezaría con ambos true entonces si ambos true ó right es true y left es false, que se mueva el pie izquierdo
        # Si ambos vuelven a ser true y antes se movia la izquierda, entonces se mueve la pierna derecha y la izquierda permanece como estabilizador
        left_foot_touchdown = len(p.getContactPoints(self.robot_id, plane_id, linkIndexA=self.left_foot_index)) > 0
        right_foot_touchdown = len(p.getContactPoints(self.robot_id, plane_id, linkIndexA=self.right_foot_index)) > 0
        # Obtener orientación actual de los pies
        left_orientation = p.getLinkState(self.robot_id, self.left_foot_index)[1]
        right_orientation = p.getLinkState(self.robot_id, self.right_foot_index)[1]
        
        # Calcular rotación necesaria para alinear con el plano horizontal
        target_orientation = p.getQuaternionFromEuler([0, 0, 0])  # Orientación plana
        
        # Aplicar corrección a ambos pies
        p.resetBasePositionAndOrientation(self.robot_id, self.left_foot_index, [0, 0, 0], target_orientation)
        p.resetBasePositionAndOrientation(self.robot_id, self.right_foot_index, [0, 0, 0], target_orientation)

def foot_bezier_parabola(start, end, ctrl1, ctrl2, alpha, height):
    """
        Trayectoria mixta: curva Bézier para (x, y), parábola para z.
        - start, end: posiciones 3D (inicio y fin del swing)
        - ctrl1, ctrl2: puntos de control Bézier (3D)
        - alpha: progreso de swing [0, 1]
        - height: altura máxima de swing (aplica a z)
    """
    # Bézier para X e Y (puedes usarlo también para Z base si quieres)
    def bezier(p0, p1, p2, p3, t):
        return ((1 - t)**3) * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    # Bézier en x, y (z puede quedarse lineal o usar solo inicio y fin)
    x = bezier(start[0], ctrl1[0], ctrl2[0], end[0], alpha)
    y = bezier(start[1], ctrl1[1], ctrl2[1], end[1], alpha)

    # Parábola para z
    z_base = (1 - alpha) * start[2] + alpha * end[2]
    z = z_base + height * 4 * alpha * (1 - alpha)

    return [x, y, z]


