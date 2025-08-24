import numpy as np
from collections import deque
import math
import pybullet as p

from Controlador.discrete_action_controller import DiscreteActionController

class PosturalControlSystem:
    """
    Sistema de control postural que imita el sistema vestibular humano.
    
    Este sistema detecta desviaciones posturales y aplica micro-correcciones
    para mantener la cadera paralela al suelo y el equilibrio centrado.
    
    Principios biomecánicos implementados:
    - Control postural reactivo (reacción a perturbaciones)
    - Integración sensorial (múltiples fuentes de información)
    - Adaptación (aprende de correcciones exitosas)
    """
    
    def __init__(self):
        # Parámetros de control postural
        self.target_hip_angle = 0.0  # Ángulo objetivo de cadera (paralelo al suelo)
        self.deadzone = 0.02  # ±1.1 grados - zona sin corrección
        self.max_correction = 0.15  # Máxima corrección aplicable
        
        # Ganancias proporcionales (ajustables según respuesta)
        self.kp_hip = 2.0      # Ganancia para corrección de cadera
        self.kd_hip = 0.5      # Ganancia derivativa (amortiguación)
        
        # Historia para control derivativo
        self.hip_angle_history = deque(maxlen=5)
        self.correction_history = deque(maxlen=10)
        
        # Sistema de aprendizaje simple
        self.successful_corrections = []
        self.learning_rate = 0.1
        
        print(f"🧠 Sistema de Control Postural inicializado")
        print(f"   Zona muerta: ±{math.degrees(self.deadzone):.1f}°")
        print(f"   Corrección máxima: {self.max_correction:.2f}")
    
    def detect_postural_deviation(self, robot_id):
        """
        Detecta desviaciones posturales analizando múltiples sensores
        
        Returns:
            dict: Información completa sobre el estado postural
        """
        # Obtener estado del robot
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(orn)
        
        # Obtener ángulos de cadera
        joint_states = p.getJointStates(robot_id, [0, 3])  # left_hip, right_hip
        left_hip_angle = joint_states[0][0]
        right_hip_angle = joint_states[1][0]
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2.0
        
        # Actualizar historia para control derivativo
        self.hip_angle_history.append(avg_hip_angle)
        
        # Calcular velocidad angular de cadera (derivada discreta)
        if len(self.hip_angle_history) >= 2:
            hip_angular_velocity = (self.hip_angle_history[-1] - self.hip_angle_history[-2]) * 1500  # Hz
        else:
            hip_angular_velocity = 0.0
        
        # Detectar tipo de desviación
        deviation_info = {
            'avg_hip_angle': avg_hip_angle,
            'hip_error': avg_hip_angle - self.target_hip_angle,
            'hip_velocity': hip_angular_velocity,
            'torso_pitch': euler[1],  # Inclinación hacia adelante/atrás
            'torso_roll': euler[0],   # Inclinación lateral
            'center_of_mass_x': pos[0],
            'needs_correction': abs(avg_hip_angle - self.target_hip_angle) > self.deadzone
        }
        
        return deviation_info
    
    def calculate_postural_correction(self, deviation_info):
        """
        Calcula las correcciones necesarias usando control PD
        
        Args:
            deviation_info: Información de desviación del método anterior
            
        Returns:
            dict: Correcciones a aplicar en las presiones PAM
        """
        hip_error = deviation_info['hip_error']
        hip_velocity = deviation_info['hip_velocity']
        
        # Si estamos dentro de la zona muerta, no hacer nada
        if not deviation_info['needs_correction']:
            return {
                'left_hip_flexor_adj': 0.0,
                'left_hip_extensor_adj': 0.0,
                'right_hip_flexor_adj': 0.0,
                'right_hip_extensor_adj': 0.0,
                'correction_applied': False
            }
        
        # Control PD: Proporcional + Derivativo
        proportional_term = self.kp_hip * hip_error
        derivative_term = self.kd_hip * hip_velocity
        
        # Corrección total (PD)
        total_correction = proportional_term + derivative_term
        
        # Limitar corrección máxima
        total_correction = np.clip(total_correction, -self.max_correction, self.max_correction)
        
        # Interpretar la corrección:
        # Si hip_error > 0: cadera está hacia adelante (flexionada)
        #   -> Necesitamos más extensión, menos flexión
        # Si hip_error < 0: cadera está hacia atrás (extendida)
        #   -> Necesitamos más flexión, menos extensión
        
        if hip_error > 0:  # Cadera muy flexionada -> más extensión
            flexor_adjustment = -abs(total_correction)  # Reducir flexores
            extensor_adjustment = +abs(total_correction)  # Aumentar extensores
        else:  # Cadera muy extendida -> más flexión  
            flexor_adjustment = +abs(total_correction)  # Aumentar flexores
            extensor_adjustment = -abs(total_correction)  # Reducir extensores
        
        corrections = {
            'left_hip_flexor_adj': flexor_adjustment,
            'left_hip_extensor_adj': extensor_adjustment,
            'right_hip_flexor_adj': flexor_adjustment,  # Simétrico
            'right_hip_extensor_adj': extensor_adjustment,  # Simétrico
            'correction_applied': True,
            'total_correction': total_correction,
            'error_magnitude': abs(hip_error)
        }
        
        # Guardar en historia para aprendizaje
        self.correction_history.append({
            'error': hip_error,
            'correction': total_correction,
            'timestamp': len(self.correction_history)
        })
        
        return corrections
    
    def apply_postural_learning(self, success_feedback):
        """
        Sistema de aprendizaje simple que ajusta las ganancias según el éxito
        
        Args:
            success_feedback: bool, True si la corrección fue exitosa
        """
        if success_feedback:
            # Si la corrección fue exitosa, aumentar ligeramente la confianza
            self.kp_hip *= (1 + self.learning_rate * 0.1)
            self.kp_hip = min(self.kp_hip, 3.0)  # Límite superior
        else:
            # Si no fue exitosa, reducir la agresividad
            self.kp_hip *= (1 - self.learning_rate * 0.05)  
            self.kp_hip = max(self.kp_hip, 0.5)  # Límite inferior
    
    def get_diagnostic_info(self):
        """Información de diagnóstico para debugging"""
        return {
            'kp_hip': self.kp_hip,
            'kd_hip': self.kd_hip,
            'corrections_applied': len(self.correction_history),
            'avg_hip_history': list(self.hip_angle_history),
            'recent_corrections': list(self.correction_history)[-3:] if self.correction_history else []
        }
    
class EnhancedDiscreteActionController(DiscreteActionController):
    """
    Controlador mejorado con sistema de control postural integrado
    
    Extiende el controlador original añadiendo:
    - Detección de desviaciones posturales
    - Corrección automática de presiones PAM
    - Aprendizaje de patrones exitosos
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Integrar sistema de control postural
        self.postural_system = PosturalControlSystem()
        
        # Tracking de rendimiento del sistema postural
        self.postural_corrections_applied = 0
        self.successful_corrections = 0
        
        print(f"🎯 Controlador Mejorado con Control Postural inicializado")
        print(f"   Sistema reactivo: Activo")
        print(f"   Aprendizaje adaptativo: Habilitado")
    
    def get_expert_action(self, time_step):
        """
        Versión mejorada que incluye correcciones posturales automáticas
        
        Flujo:
        1. Obtener acción base del controlador original
        2. Detectar desviaciones posturales
        3. Calcular correcciones necesarias
        4. Aplicar correcciones a las presiones base
        5. Devolver acción corregida
        """
        # PASO 1: Obtener acción base (patrón experto original)
        base_action = super().get_expert_action(time_step)
        
        # PASO 2: Detectar si necesitamos correcciones posturales
        deviation_info = self.postural_system.detect_postural_deviation(self.env.robot_id)
        
        # PASO 3: Calcular correcciones si son necesarias
        corrections = self.postural_system.calculate_postural_correction(deviation_info)
        
        # PASO 4: Aplicar correcciones a la acción base
        if corrections['correction_applied']:
            corrected_action = self._apply_postural_corrections(base_action, corrections)
            self.postural_corrections_applied += 1
            
            # Debug cada 100 correcciones
            if self.postural_corrections_applied % 100 == 0:
                print(f"   🧠 Corrección postural #{self.postural_corrections_applied}")
                print(f"      Error cadera: {deviation_info['hip_error']:.3f} rad")
                print(f"      Corrección aplicada: {corrections['total_correction']:.3f}")
        else:
            corrected_action = base_action
        
        return corrected_action
    
    def _apply_postural_corrections(self, base_action, corrections):
        """
        Aplica las correcciones posturales a la acción base
        
        Esta función modifica sutilmente las presiones PAM para corregir
        desviaciones posturales manteniendo el patrón base intacto
        """
        corrected_action = base_action.copy()
        
        # Mapear correcciones a índices PAM
        # Recordar: [0,1,2,3,4,5] = [LHF, LHE, RHF, RHE, LKF, RKF]
        pam_adjustments = {
            0: corrections['left_hip_flexor_adj'],    # Left Hip Flexor
            1: corrections['left_hip_extensor_adj'],  # Left Hip Extensor  
            2: corrections['right_hip_flexor_adj'],   # Right Hip Flexor
            3: corrections['right_hip_extensor_adj'], # Right Hip Extensor
            # Rodillas no se ajustan en correcciones posturales básicas
            4: 0.0,  # Left Knee Flexor
            5: 0.0,  # Right Knee Flexor
        }
        
        # Aplicar ajustes manteniendo límites [0,1]
        for pam_idx, adjustment in pam_adjustments.items():
            corrected_action[pam_idx] += adjustment
            corrected_action[pam_idx] = np.clip(corrected_action[pam_idx], 0.0, 1.0)
        
        return corrected_action
    
    def evaluate_correction_success(self, previous_deviation, current_deviation):
        """
        Evalúa si una corrección postural fue exitosa
        
        Criterios de éxito:
        - Error de cadera se redujo
        - No se crearon nuevos problemas (inestabilidad)
        """
        error_improved = abs(current_deviation['hip_error']) < abs(previous_deviation['hip_error'])
        stability_maintained = current_deviation['needs_correction'] == False or \
                             abs(current_deviation['hip_error']) < 0.05
        
        success = error_improved and stability_maintained
        
        if success:
            self.successful_corrections += 1
        
        # Aplicar aprendizaje al sistema postural
        self.postural_system.apply_postural_learning(success)
        
        return success
    
    def get_postural_performance_stats(self):
        """Estadísticas del rendimiento del sistema postural"""
        success_rate = (self.successful_corrections / max(1, self.postural_corrections_applied)) * 100
        
        return {
            'corrections_applied': self.postural_corrections_applied,
            'successful_corrections': self.successful_corrections,
            'success_rate_percent': success_rate,
            'postural_gains': {
                'kp_hip': self.postural_system.kp_hip,
                'kd_hip': self.postural_system.kd_hip
            }
        }