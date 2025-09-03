#!/usr/bin/env python3
"""
CONTROLADOR ESPECIALIZADO DE SENTADILLAS - VERSIÓN SIMPLIFICADA
Compatible con tu arquitectura actual simplificada
"""

import os
import time
import numpy as np
import pybullet as p
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import json
from collections import deque

# Importar solo las clases que realmente existen en tu código
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import Simple_BalanceSquat_BipedEnv

class SquatPhase(Enum):
    """Fases específicas de una sentadilla"""
    PREPARATION = "preparation"          # Preparación inicial
    DESCENT = "descent"                 # Descenso controlado
    BOTTOM_HOLD = "bottom_hold"         # Mantener posición baja
    ASCENT = "ascent"                   # Ascenso controlado
    RECOVERY = "recovery"               # Recuperación a posición inicial
    COMPLETED = "completed"             # Sentadilla completada

class SquatQuality(Enum):
    """Evaluación de calidad de la sentadilla"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class SquatMetrics:
    """Métricas de una sentadilla ejecutada"""
    duration: float                     # Duración total
    max_depth: float                   # Profundidad máxima alcanzada
    stability_score: float             # Puntuación de estabilidad (0-1)
    smoothness_score: float            # Puntuación de suavidad (0-1)
    completion_rate: float             # Porcentaje de completitud (0-1)
    quality: SquatQuality             # Evaluación general
    phase_times: Dict[str, float]     # Tiempo en cada fase

@dataclass
class SquatConfig:
    """Configuración para las sentadillas"""
    target_depth: float = 0.7         # Profundidad objetivo (altura del torso)
    descent_speed: float = 0.3         # Velocidad de descenso (multiplicador)
    ascent_speed: float = 0.4          # Velocidad de ascenso (multiplicador)
    hold_duration: float = 2.0         # Tiempo en posición baja (segundos)
    max_squat_duration: float = 15.0   # Duración máxima total
    stability_threshold: float = 0.15  # Umbral de estabilidad (rad)
    use_balance_action: bool = True    # Usar acción de balance entre sentadillas

class SquatController:
    """
    Controlador especializado para ejecutar sentadillas con el robot bípedo entrenado.
    
    VERSIÓN SIMPLIFICADA - Compatible con tu arquitectura actual:
    - Usa solo tu modelo RL entrenado (RecurrentPPO)
    - Compatible con Simple_BalanceSquat_BipedEnv
    - Sistema de fases propio para monitoreo
    - Métricas y evaluación de calidad
    - Sistema de seguridad básico integrado
    """
    
    def __init__(self, 
                 model_path: str,
                 normalize_path: Optional[str] = None,
                 config: Optional[SquatConfig] = None,
                 render_mode: str = 'human'):
        
        # ===== CONFIGURACIÓN BÁSICA =====
        
        self.model_path = model_path
        self.normalize_path = normalize_path
        self.config = config or SquatConfig()
        self.render_mode = render_mode
        
        # ===== ESTADO DEL CONTROLADOR =====
        
        self.current_phase = SquatPhase.PREPARATION
        self.squat_start_time = 0.0
        self.phase_start_time = 0.0
        self.squat_count = 0
        self.is_active = False
        
        # ===== DATOS DE SEGUIMIENTO =====
        
        self.phase_history = deque(maxlen=100)
        self.stability_history = deque(maxlen=50)
        self.height_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.squat_metrics_history: List[SquatMetrics] = []
        
        # ===== SISTEMA DE MÉTRICAS =====
        
        self.current_metrics = {
            'max_depth_reached': 1.2,
            'current_stability': 1.0,
            'phase_duration': 0.0,
            'total_duration': 0.0,
            'average_reward': 0.0
        }
        
        # ===== INICIALIZACIÓN =====
        
        self._initialize_components()
        
        print(f"🏋️ SquatController (Simplified) initialized successfully")
        print(f"   Model: {os.path.basename(self.model_path)}")
        print(f"   Target depth: {self.config.target_depth}m")
        print(f"   Architecture: RL Model Only (Simplified)")
    
    def _initialize_components(self):
        """Inicializar todos los componentes necesarios"""
        
        # ===== CREAR ENTORNO =====
        
        print("🏗️ Initializing simplified squat environment...")
        self.env = Simple_BalanceSquat_BipedEnv(
            render_mode=self.render_mode,
            action_space="pam",
            enable_curriculum=False  # No curriculum para ejecución
        )
        
        # ===== CARGAR MODELO RL =====
        
        print("🧠 Loading trained RL model...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            self.rl_model = RecurrentPPO.load(self.model_path)
            print(f"✅ RL model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading RL model: {e}")
            raise
        
        # ===== VARIABLES DE ESTADO LSTM =====
        
        self.lstm_states = None  # Para RecurrentPPO
        self.last_observation = None
        
        print("✅ Components initialized (Simplified Architecture)")
    
    def start_squat_session(self) -> bool:
        """
        Iniciar una nueva sesión de sentadillas
        
        Returns:
            bool: True si la sesión se inició correctamente
        """
        
        if self.is_active:
            print("⚠️ Squat session already active")
            return False
        
        print("🚀 Starting squat session...")
        
        # Resetear entorno
        obs, info = self.env.reset()
        self.last_observation = obs
        
        # Resetear estado del controlador
        self._reset_squat_state()
        
        # Estabilización inicial usando el modelo RL
        print("⚖️ Initial stabilization...")
        for i in range(150):  # ~3 segundos de estabilización
            
            # Usar modelo RL con acción de balance (neutral/conservadora)
            action = self._get_balance_action(obs)
            
            obs, reward, done, truncated, info = self.env.step(action)
            self.last_observation = obs
            
            # Mostrar progreso cada 50 pasos
            if i % 50 == 0:
                height = obs[1] if len(obs) > 1 else 1.0
                print(f"   Stabilizing... step {i+1}/150, height: {height:.2f}m")
            
            if done:
                print("⚠️ Robot fell during stabilization")
                return False
        
        self.is_active = True
        self.current_phase = SquatPhase.PREPARATION
        self.squat_start_time = time.time()
        
        print("✅ Squat session started successfully")
        return True
    
    def execute_single_squat(self) -> SquatMetrics:
        """
        Ejecutar una sentadilla completa usando únicamente el modelo RL
        
        Returns:
            SquatMetrics: Métricas de la sentadilla ejecutada
        """
        
        if not self.is_active:
            raise RuntimeError("Squat session not active. Call start_squat_session() first.")
        
        print(f"🏋️ Executing squat #{self.squat_count + 1} (RL Model Only)...")
        
        # Resetear métricas para nueva sentadilla
        self._reset_squat_metrics()
        
        squat_start = time.time()
        completed = False
        
        # ===== BUCLE PRINCIPAL DE EJECUCIÓN =====
        
        step_count = 0
        while time.time() - squat_start < self.config.max_squat_duration:
            
            # Obtener observación actual
            if self.last_observation is None:
                print("⚠️ No observation available")
                break
            
            # Actualizar fase actual basada en métricas
            self._update_squat_phase()
            
            # Obtener acción del modelo RL (adaptada según la fase)
            action = self._get_squat_action(self.last_observation)
            
            # Ejecutar acción
            obs, reward, done, truncated, info = self.env.step(action)
            self.last_observation = obs
            step_count += 1
            
            # Actualizar métricas
            self._update_metrics(obs, info, reward)
            
            # Debug cada 300 pasos (~5 segundos)
            if step_count % 300 == 0:
                height = obs[1] if len(obs) > 1 else 1.0
                stability = self.current_metrics['current_stability']
                print(f"   Step {step_count}: Phase {self.current_phase.value}, "
                      f"Height {height:.2f}m, Stability {stability:.2f}")
            
            # Verificar condiciones de terminación
            if done:
                print("⚠️ Episode terminated during squat")
                break
            
            # Verificar si la sentadilla se completó
            if self._is_squat_completed():
                completed = True
                print(f"✅ Squat cycle completed in {time.time() - squat_start:.1f}s")
                break
            
            # Verificar seguridad
            if not self._safety_check(obs):
                print("🛑 Safety check failed - aborting squat")
                break
        
        # ===== CALCULAR MÉTRICAS FINALES =====
        
        final_metrics = self._calculate_final_metrics(completed, time.time() - squat_start)
        self.squat_metrics_history.append(final_metrics)
        self.squat_count += 1
        
        self.current_phase = SquatPhase.COMPLETED
        
        print(f"📊 Squat completed - Quality: {final_metrics.quality.value}, "
              f"Depth: {final_metrics.max_depth:.2f}m, "
              f"Stability: {final_metrics.stability_score:.2f}")
        
        return final_metrics
    
    def execute_squat_sequence(self, num_squats: int, rest_time: float = 3.0) -> List[SquatMetrics]:
        """
        Ejecutar una secuencia de sentadillas
        
        Args:
            num_squats: Número de sentadillas a ejecutar
            rest_time: Tiempo de descanso entre sentadillas (segundos)
            
        Returns:
            List[SquatMetrics]: Lista de métricas para cada sentadilla
        """
        
        if not self.start_squat_session():
            return []
        
        print(f"🏋️ Starting squat sequence: {num_squats} squats (RL Model)")
        
        sequence_metrics = []
        
        for i in range(num_squats):
            print(f"\n--- SQUAT {i+1}/{num_squats} ---")
            
            # Ejecutar sentadilla
            metrics = self.execute_single_squat()
            sequence_metrics.append(metrics)
            
            # Mostrar progreso
            print(f"Squat {i+1}: {metrics.quality.value} "
                  f"({metrics.completion_rate:.1%} complete, "
                  f"{metrics.duration:.1f}s)")
            
            # Descanso entre sentadillas (excepto la última)
            if i < num_squats - 1:
                print(f"😴 Resting for {rest_time} seconds...")
                self._rest_period(rest_time)
        
        # Finalizar sesión
        self.stop_squat_session()
        
        # Mostrar resumen
        self._print_sequence_summary(sequence_metrics)
        
        return sequence_metrics
    
    def _get_squat_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Obtener acción para la sentadilla usando el modelo RL
        
        Modifica ligeramente la predicción del modelo según la fase actual
        """
        
        try:
            # Predicción base del modelo RL
            base_action, self.lstm_states = self.rl_model.predict(
                observation,
                state=self.lstm_states,
                deterministic=True  # Política determinística para consistencia
            )
            
            # Modificaciones leves según la fase actual
            modified_action = self._apply_phase_modifications(base_action)
            
            return modified_action
            
        except Exception as e:
            print(f"⚠️ RL model error: {e}")
            # Fallback a acción neutral de balance
            return self._get_balance_action(observation)
    
    def _get_balance_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Obtener acción de balance usando el modelo RL
        
        Similar a _get_squat_action pero optimizada para mantener balance
        """
        
        try:
            action, self.lstm_states = self.rl_model.predict(
                observation,
                state=self.lstm_states,
                deterministic=True
            )
            
            # Suavizar la acción para balance estable
            # Reducir variabilidad para movimientos más conservadores
            balanced_action = action * 0.8 + np.array([0.3, 0.4, 0.3, 0.4, 0.1, 0.1]) * 0.2
            
            return np.clip(balanced_action, 0.0, 1.0)
            
        except Exception as e:
            print(f"⚠️ RL model error during balance: {e}")
            # Acción completamente neutral como último recurso
            return np.array([0.3, 0.4, 0.3, 0.4, 0.1, 0.1])
    
    def _apply_phase_modifications(self, base_action: np.ndarray) -> np.ndarray:
        """
        Aplicar modificaciones leves a la acción según la fase actual
        """
        
        modified_action = base_action.copy()
        
        # Modificaciones muy leves para guiar el comportamiento
        if self.current_phase == SquatPhase.DESCENT:
            # Favorecer ligeramente flexión en caderas
            modified_action[0] *= 1.1  # left hip flexor
            modified_action[2] *= 1.1  # right hip flexor
            
        elif self.current_phase == SquatPhase.ASCENT:
            # Favorecer ligeramente extensión en caderas
            modified_action[1] *= 1.1  # left hip extensor
            modified_action[3] *= 1.1  # right hip extensor
            
        elif self.current_phase == SquatPhase.BOTTOM_HOLD:
            # Mantener estabilidad, reducir variabilidad
            modified_action = modified_action * 0.9 + base_action * 0.1
        
        return np.clip(modified_action, 0.0, 1.0)
    
    def _update_squat_phase(self):
        """Actualizar la fase actual de la sentadilla basada en métricas"""
        
        current_time = time.time()
        phase_duration = current_time - self.phase_start_time
        total_duration = current_time - self.squat_start_time
        
        # Extraer altura actual de la observación
        robot_height = 1.0  # Default
        if self.last_observation is not None and len(self.last_observation) > 1:
            robot_height = self.last_observation[1]  # pos[2] del robot
        
        old_phase = self.current_phase
        
        # Lógica de transición entre fases basada en altura y tiempo
        if self.current_phase == SquatPhase.PREPARATION and phase_duration > 1.0:
            self.current_phase = SquatPhase.DESCENT
            
        elif self.current_phase == SquatPhase.DESCENT and robot_height < self.config.target_depth:
            self.current_phase = SquatPhase.BOTTOM_HOLD
            
        elif self.current_phase == SquatPhase.BOTTOM_HOLD and phase_duration > self.config.hold_duration:
            self.current_phase = SquatPhase.ASCENT
            
        elif self.current_phase == SquatPhase.ASCENT and robot_height > 0.9:
            self.current_phase = SquatPhase.RECOVERY
            
        elif self.current_phase == SquatPhase.RECOVERY and phase_duration > 1.5:
            self.current_phase = SquatPhase.COMPLETED
        
        # Si cambió la fase, actualizar tiempo
        if old_phase != self.current_phase:
            self.phase_start_time = current_time
            print(f"   Phase transition: {old_phase.value} → {self.current_phase.value}")
    
    def _update_metrics(self, observation: np.ndarray, info: Dict, reward: float):
        """Actualizar métricas en tiempo real"""
        
        # Extraer información de la observación (basado en Simple_BalanceSquat_BipedEnv)
        if len(observation) >= 16:  # Tu observation space tiene 16 elementos
            robot_height = observation[1]     # z position
            orientation = observation[2:4]    # roll, pitch
            # velocities = observation[4:8]   # linear and angular velocities
        else:
            robot_height = 1.0
            orientation = [0.0, 0.0]
        
        # Actualizar métricas
        self.current_metrics['max_depth_reached'] = min(
            self.current_metrics['max_depth_reached'], 
            robot_height
        )
        
        # Calcular estabilidad basada en orientación
        stability = 1.0 - (abs(orientation[0]) + abs(orientation[1])) / (2 * self.config.stability_threshold)
        self.current_metrics['current_stability'] = max(0.0, min(1.0, stability))
        
        # Actualizar historias
        self.height_history.append(robot_height)
        self.stability_history.append(stability)
        self.reward_history.append(reward)
        
        # Calcular recompensa promedio
        if len(self.reward_history) > 0:
            self.current_metrics['average_reward'] = sum(self.reward_history) / len(self.reward_history)
    
    def _is_squat_completed(self) -> bool:
        """Verificar si la sentadilla se completó exitosamente"""
        return self.current_phase == SquatPhase.COMPLETED
    
    def _safety_check(self, observation: np.ndarray) -> bool:
        """Verificar condiciones de seguridad"""
        
        if len(observation) < 4:
            return True  # No suficiente información, asumir seguro
        
        robot_height = observation[1]
        orientation = observation[2:4]
        
        # Verificar altura mínima
        if robot_height < 0.3:
            return False
        
        # Verificar inclinación excesiva
        if abs(orientation[0]) > 0.5 or abs(orientation[1]) > 0.5:  # ~30 grados
            return False
        
        return True
    
    def _calculate_final_metrics(self, completed: bool, duration: float) -> SquatMetrics:
        """Calcular métricas finales de la sentadilla"""
        
        # Calcular profundidad alcanzada (qué tan bajo llegó)
        depth_score = max(0.0, (1.2 - self.current_metrics['max_depth_reached']) / 0.5)
        depth_score = min(1.0, depth_score)
        
        # Estabilidad promedio
        stability_score = np.mean(list(self.stability_history)) if self.stability_history else 0.0
        stability_score = max(0.0, min(1.0, stability_score))
        
        # Suavidad basada en variabilidad de altura
        if len(self.height_history) > 10:
            height_changes = np.diff(list(self.height_history))
            smoothness_score = max(0.0, 1.0 - np.std(height_changes) * 10)
        else:
            smoothness_score = 0.5
        
        # Tasa de completitud
        completion_rate = 1.0 if completed else max(0.0, min(1.0, depth_score))
        
        # Determinar calidad general
        overall_score = (depth_score + stability_score + smoothness_score + completion_rate) / 4
        
        if overall_score >= 0.8:
            quality = SquatQuality.EXCELLENT
        elif overall_score >= 0.6:
            quality = SquatQuality.GOOD
        elif overall_score >= 0.4:
            quality = SquatQuality.FAIR
        else:
            quality = SquatQuality.POOR
        
        return SquatMetrics(
            duration=duration,
            max_depth=self.current_metrics['max_depth_reached'],
            stability_score=stability_score,
            smoothness_score=smoothness_score,
            completion_rate=completion_rate,
            quality=quality,
            phase_times={}  # Simplificado
        )
    
    def _rest_period(self, duration: float):
        """Período de descanso entre sentadillas manteniendo balance"""
        
        rest_start = time.time()
        step_count = 0
        
        while time.time() - rest_start < duration:
            # Mantener balance durante el descanso usando modelo RL
            action = self._get_balance_action(self.last_observation)
            
            obs, reward, done, truncated, info = self.env.step(action)
            self.last_observation = obs
            step_count += 1
            
            # Mostrar progreso cada segundo aprox
            if step_count % 300 == 0:
                remaining = duration - (time.time() - rest_start)
                print(f"   Resting... {remaining:.1f}s remaining")
            
            if done:
                print("⚠️ Robot fell during rest period")
                break
    
    def _reset_squat_state(self):
        """Resetear estado del controlador para nueva sesión"""
        self.current_phase = SquatPhase.PREPARATION
        self.squat_start_time = 0.0
        self.phase_start_time = 0.0
        self.phase_history.clear()
        self.stability_history.clear()
        self.height_history.clear()
        self.reward_history.clear()
        self.lstm_states = None
        self.last_observation = None
    
    def _reset_squat_metrics(self):
        """Resetear métricas para nueva sentadilla"""
        self.current_metrics = {
            'max_depth_reached': 1.2,
            'current_stability': 1.0,
            'phase_duration': 0.0,
            'total_duration': 0.0,
            'average_reward': 0.0
        }
        self.stability_history.clear()
        self.height_history.clear()
        self.reward_history.clear()
    
    def _print_sequence_summary(self, metrics_list: List[SquatMetrics]):
        """Imprimir resumen de la secuencia de sentadillas"""
        
        print(f"\n🏆 SQUAT SEQUENCE SUMMARY")
        print("=" * 40)
        
        if not metrics_list:
            print("No squats completed")
            return
        
        # Estadísticas generales
        total_squats = len(metrics_list)
        completed_squats = sum(1 for m in metrics_list if m.completion_rate >= 1.0)
        avg_duration = np.mean([m.duration for m in metrics_list])
        avg_stability = np.mean([m.stability_score for m in metrics_list])
        avg_depth = np.mean([m.max_depth for m in metrics_list])
        
        print(f"Total squats: {total_squats}")
        print(f"Completed: {completed_squats}/{total_squats} ({completed_squats/total_squats:.1%})")
        print(f"Average duration: {avg_duration:.1f}s")
        print(f"Average stability: {avg_stability:.2f}")
        print(f"Average depth reached: {avg_depth:.2f}m")
        
        # Distribución de calidad
        quality_counts = {}
        for quality in SquatQuality:
            count = sum(1 for m in metrics_list if m.quality == quality)
            if count > 0:
                quality_counts[quality.value] = count
        
        print("\nQuality distribution:")
        for quality, count in quality_counts.items():
            print(f"  {quality.capitalize()}: {count}")
        
        # Mejor sentadilla
        if metrics_list:
            best_squat = max(metrics_list, 
                           key=lambda m: m.stability_score + m.smoothness_score + m.completion_rate)
            print(f"\nBest squat: #{metrics_list.index(best_squat) + 1}")
            print(f"  Quality: {best_squat.quality.value}")
            print(f"  Depth: {best_squat.max_depth:.2f}m")
            print(f"  Stability: {best_squat.stability_score:.2f}")
            print(f"  Duration: {best_squat.duration:.1f}s")
    
    def get_performance_stats(self) -> Dict:
        """Obtener estadísticas de rendimiento"""
        
        if not self.squat_metrics_history:
            return {"total_squats": 0, "message": "No squats executed yet"}
        
        metrics = self.squat_metrics_history
        
        return {
            "total_squats": len(metrics),
            "completed_squats": sum(1 for m in metrics if m.completion_rate >= 1.0),
            "average_duration": np.mean([m.duration for m in metrics]),
            "average_stability": np.mean([m.stability_score for m in metrics]),
            "average_depth": np.mean([m.max_depth for m in metrics]),
            "average_reward": self.current_metrics.get('average_reward', 0.0),
            "success_rate": sum(1 for m in metrics if m.completion_rate >= 1.0) / len(metrics),
            "quality_distribution": {
                q.value: sum(1 for m in metrics if m.quality == q) 
                for q in SquatQuality
            }
        }
    
    def stop_squat_session(self):
        """Detener la sesión de sentadillas"""
        self.is_active = False
        self.lstm_states = None
        print("🛑 Squat session stopped")
    
    def close(self):
        """Cerrar el controlador y limpiar recursos"""
        if hasattr(self, 'env'):
            self.env.close()
        print("🔚 SquatController closed")


# ===== FUNCIONES DE UTILIDAD =====

def create_squat_controller(
    model_path: str = "./models_balance_squat/best_model.zip",
    config: Optional[SquatConfig] = None,
    render_mode: str = 'human'
) -> SquatController:
    """
    Función de conveniencia para crear un controlador de sentadillas
    """
    
    normalize_path = model_path.replace('.zip', '_normalize.pkl')
    if not os.path.exists(normalize_path):
        normalize_path = None
    
    return SquatController(
        model_path=model_path,
        normalize_path=normalize_path,
        config=config,
        render_mode=render_mode
    )

def demo_squat_controller():
    """Demostración del controlador de sentadillas simplificado"""
    
    print("🏋️ DEMO: Controlador de Sentadillas (Simplificado)")
    print("=" * 50)
    
    # Verificar que el modelo existe
    model_path = "./models_balance_squat/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   Ejecuta primero el entrenamiento para generar el modelo")
        return
    
    # Configuración personalizada
    config = SquatConfig(
        target_depth=0.7,           # Sentadillas moderadas
        hold_duration=2.0,          # 2 segundos en posición baja
        use_balance_action=True     # Usar acciones de balance
    )
    
    try:
        # Crear controlador
        controller = create_squat_controller(config=config)
        
        # Ejecutar secuencia de 3 sentadillas
        metrics = controller.execute_squat_sequence(
            num_squats=3,
            rest_time=3.0
        )
        
        # Mostrar estadísticas finales
        stats = controller.get_performance_stats()
        print(f"\n📊 Final Stats:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        
        # Cerrar
        controller.close()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_squat_controller()