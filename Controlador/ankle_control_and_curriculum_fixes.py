# ====================================================================
# SISTEMA HÍBRIDO DE CONTROL DE TOBILLOS
# Resortes inteligentes + control adaptativo para balance estable
# ====================================================================

import numpy as np
import math
from collections import deque
import pybullet as p

from Archivos_Apoyo.simple_log_redirect import log_print, both_print

class IntelligentAnkleControl:
    """
    Sistema híbrido de control de tobillos que combina:
    1. Resortes pasivos para soporte estructural básico
    2. Control activo adaptativo para balance fino
    3. Detección de perturbaciones para respuesta rápida
    
    Los tobillos humanos proporcionan ~70% del control postural en posición erguida.
    Este sistema imita esa funcionalidad crítica sin necesidad de PAMs adicionales.
    """
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # Parámetros de resortes base (más fuertes que rodillas)
        self.base_spring_stiffness = 120.0  # Más rígido que rodillas (eran 150)
        self.base_damping = 8.0  # Amortiguación moderada
        
        # Parámetros de control activo
        self.balance_gain = 0.8  # Ganancia para corrección de balance
        self.max_active_torque = 25.0  # Máximo torque activo adicional
        
        # Sistema de detección de perturbaciones
        self.com_history = deque(maxlen=5)
        self.ankle_angle_history = deque(maxlen=5)
        
        # Umbral para activación de control activo
        self.activation_threshold = 0.02  # rad (~1.1 grados)
        
        log_print(f"🦶 Sistema Inteligente de Control de Tobillos inicializado")
        log_print(f"   Rigidez base: {self.base_spring_stiffness} N⋅m/rad")
        log_print(f"   Control activo: Habilitado para balance fino")
    
    def calculate_ankle_torques(self, robot_data, zmp_calculator=None):
        """
        Calcula torques de tobillos usando sistema híbrido:
        1. Resortes pasivos para soporte base
        2. Control activo para correcciones de balance
        
        Args:
            robot_data: Datos del robot (PyBullet_Robot_Data)
            zmp_calculator: Calculador ZMP para información de estabilidad
            
        Returns:
            tuple: (left_ankle_torque, right_ankle_torque)
        """
        
        # Obtener estados de tobillos
        ankle_states = p.getJointStates(self.robot_id, [2, 5])  # left_ankle, right_ankle
        left_ankle_pos = ankle_states[0][0]
        right_ankle_pos = ankle_states[1][0]
        left_ankle_vel = ankle_states[0][1]
        right_ankle_vel = ankle_states[1][1]
        
        # ===== COMPONENTE 1: RESORTES PASIVOS BÁSICOS =====
        
        # Torques de resorte hacia posición neutral (0.0)
        left_spring_torque = -self.base_spring_stiffness * left_ankle_pos
        right_spring_torque = -self.base_spring_stiffness * right_ankle_pos
        
        # Amortiguación proporcional a velocidad
        left_damping_torque = -self.base_damping * left_ankle_vel
        right_damping_torque = -self.base_damping * right_ankle_vel
        
        # Torques pasivos totales
        left_passive_torque = left_spring_torque + left_damping_torque
        right_passive_torque = right_spring_torque + right_damping_torque
        
        # ===== COMPONENTE 2: CONTROL ACTIVO PARA BALANCE =====
        
        # Obtener información de balance si está disponible
        if zmp_calculator and robot_data:
            try:
                # Detectar si necesitamos control activo
                balance_correction = self._calculate_balance_correction(
                    robot_data, zmp_calculator, left_ankle_pos, right_ankle_pos
                )
                
                # Añadir corrección activa si es necesaria
                left_active_torque = balance_correction['left_correction']
                right_active_torque = balance_correction['right_correction']
                
            except Exception as e:
                # Fallback si hay problemas con cálculos avanzados
                left_active_torque = 0.0
                right_active_torque = 0.0
        else:
            # Sin información de balance, solo usar control postural básico
            left_active_torque = self._basic_postural_correction(left_ankle_pos)
            right_active_torque = self._basic_postural_correction(right_ankle_pos)
        
        # ===== COMBINACIÓN FINAL =====
        
        total_left_torque = left_passive_torque + left_active_torque
        total_right_torque = right_passive_torque + right_active_torque
        
        # Limitar torques para seguridad
        max_total_torque = 80.0  # N⋅m
        total_left_torque = np.clip(total_left_torque, -max_total_torque, max_total_torque)
        total_right_torque = np.clip(total_right_torque, -max_total_torque, max_total_torque)
        
        return total_left_torque, total_right_torque
    
    def _calculate_balance_correction(self, robot_data, zmp_calculator, left_ankle_pos, right_ankle_pos):
        """
        Calcula correcciones activas basadas en información de balance y ZMP
        
        Esta función implementa control postural específicamente diseñado
        para tobillos, usando información del centro de masas y ZMP
        """
        
        # Obtener centro de masas
        try:
            com_position, total_mass = robot_data.get_center_of_mass
            self.com_history.append(com_position)
        except:
            com_position = np.array([0, 0, 1.1])
            self.com_history.append(com_position)
        
        # Calcular movimiento del COM
        if len(self.com_history) >= 2:
            com_velocity = self.com_history[-1] - self.com_history[-2]
            com_velocity_magnitude = np.linalg.norm(com_velocity[:2])  # Solo x,y
        else:
            com_velocity_magnitude = 0.0
        
        # Obtener información ZMP
        try:
            zmp_point = zmp_calculator.calculate_zmp()
            is_stable = zmp_calculator.is_stable(zmp_point)
            stability_margin = zmp_calculator.stability_margin_distance(zmp_point)
        except:
            zmp_point = np.array([0.0, 0.0])
            is_stable = True
            stability_margin = 0.1
        
        # ===== LÓGICA DE CORRECCIÓN INTELIGENTE =====
        
        correction_factor = 0.0
        
        # Si el robot está perdiendo estabilidad, activar control
        if not is_stable or stability_margin < 0.05:
            # Corrección basada en posición del ZMP
            zmp_error_x = zmp_point[0]  # Desviación lateral del ZMP
            correction_factor = self.balance_gain * zmp_error_x
            
        # Si el COM se está moviendo rápido, anticipar corrección
        elif com_velocity_magnitude > 0.05:  # m/s
            com_error_x = com_position[0]  # Desviación del COM
            correction_factor = self.balance_gain * 0.5 * com_error_x
        
        # Aplicar corrección de manera diferencial (más en el tobillo que necesita trabajar más)
        if abs(correction_factor) > 0.01:
            # Tobillo que está más comprometido trabaja más
            left_load_factor = 1.0 + abs(left_ankle_pos)
            right_load_factor = 1.0 + abs(right_ankle_pos)
            total_load = left_load_factor + right_load_factor
            
            left_correction = correction_factor * (left_load_factor / total_load)
            right_correction = correction_factor * (right_load_factor / total_load)
        else:
            left_correction = 0.0
            right_correction = 0.0
        
        # Limitar correcciones activas
        left_correction = np.clip(left_correction, -self.max_active_torque, self.max_active_torque)
        right_correction = np.clip(right_correction, -self.max_active_torque, self.max_active_torque)
        
        return {
            'left_correction': left_correction,
            'right_correction': right_correction,
            'zmp_stable': is_stable,
            'correction_magnitude': abs(correction_factor)
        }
    
    def _basic_postural_correction(self, ankle_angle):
        """
        Corrección postural básica cuando no hay información avanzada disponible
        
        Usa solo el ángulo del tobillo para determinar si necesita corrección
        """
        
        if abs(ankle_angle) > self.activation_threshold:
            # Corrección proporcional simple
            correction = -self.balance_gain * ankle_angle
            return np.clip(correction, -self.max_active_torque, self.max_active_torque)
        
        return 0.0