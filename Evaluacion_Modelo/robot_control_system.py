#!/usr/bin/env python3
"""
SISTEMA DE CONTROL PARA ROBOT BÍPEDO ENTRENADO
Control completo del robot usando el modelo RL entrenado con integración de controladores expertos
"""

import os
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import json

# Importar tus módulos específicos
from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
from Controlador.discrete_action_controller import ActionType

class ControlMode(Enum):
    """Modos de control disponibles"""
    RL_TRAINED = "rl_trained"           # Usar modelo entrenado
    EXPERT_BALANCE = "expert_balance"   # Controlador experto balance
    EXPERT_SQUAT = "expert_squat"       # Controlador experto sentadillas
    HYBRID = "hybrid"                   # Combinación RL + Expert

class RobotState(Enum):
    """Estados del robot"""
    INITIALIZING = "initializing"
    BALANCING = "balancing" 
    SQUATTING = "squatting"
    FALLEN = "fallen"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ControllerConfig:
    """Configuración del controlador"""
    model_path: str
    normalize_path: Optional[str] = None
    render_mode: str = 'human'
    max_episode_steps: int = 10000
    safety_limits: Dict[str, float] = None
    
    def __post_init__(self):
        if self.safety_limits is None:
            self.safety_limits = {
                'max_tilt_angle': 0.5,      # Máximo ángulo de inclinación (rad)
                'min_height': 0.3,           # Altura mínima del robot (m) 
                'max_joint_velocity': 10.0,  # Velocidad máxima articulaciones
                'emergency_fall_threshold': 0.2  # Altura para emergency stop
            }

class TrainedRobotController:
    """
    Controlador principal del robot bípedo entrenado.
    
    Integra:
    - Modelo RL entrenado
    - Controladores expertos de respaldo
    - Sistema de seguridad
    - Monitoreo en tiempo real
    """
    
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.current_mode = ControlMode.RL_TRAINED
        self.current_state = RobotState.INITIALIZING
        
        # Inicializar componentes
        self._load_trained_model()
        self._setup_environment()
        self._setup_expert_controllers()
        self._setup_safety_system()
        
        # Variables de estado
        self.episode_step = 0
        self.total_reward = 0
        self.last_observation = None
        self.lstm_states = None
        
        print(f"🤖 Robot Controller initialized successfully")
        print(f"   Model: {config.model_path}")
        print(f"   Mode: {self.current_mode.value}")
    
    def _load_trained_model(self):
        """Cargar modelo RL entrenado"""
        try:
            print(f"📚 Loading trained model from: {self.config.model_path}")
            
            # Cargar modelo RecurrentPPO
            self.trained_model = RecurrentPPO.load(self.config.model_path)
            
            # Cargar normalización si existe
            if self.config.normalize_path and os.path.exists(self.config.normalize_path):
                print(f"🔧 Loading normalization from: {self.config.normalize_path}")
                # Aquí cargarías la normalización específica
            
            print(f"   ✅ Model loaded successfully")
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            raise
    
    def _setup_environment(self):
        """Configurar entorno de control"""
        print(f"🌍 Setting up control environment...")
        
        # Crear entorno principal
        self.env = create_simple_balance_squat_env(
            render_mode=self.config.render_mode,
            max_episode_steps=self.config.max_episode_steps
        )
        
        # Si hay normalización, aplicarla
        if hasattr(self, 'vec_normalize'):
            self.env = self.vec_normalize
        
        print(f"   ✅ Environment ready")
    
    #def _setup_expert_controllers(self):
        """Configurar controladores expertos de respaldo"""
    #    print(f"🧠 Setting up expert controllers...")
        
        # Controlador experto principal
    #    self.expert_controller = create_balance_squat_controller(self.env)
        
    #    print(f"   ✅ Expert controllers ready")
    
    def _setup_safety_system(self):
        """Configurar sistema de seguridad"""
        print(f"🛡️ Setting up safety system...")
        
        self.safety_active = True
        self.safety_violations = 0
        self.max_safety_violations = 3
        
        print(f"   ✅ Safety system active")
    
    def reset_robot(self) -> np.ndarray:
        """Resetear robot a posición inicial"""
        print(f"🔄 Resetting robot to initial state...")
        
        # Reset environment
        observation, info = self.env.reset()
        
        # Reset internal states
        self.episode_step = 0
        self.total_reward = 0
        self.last_observation = observation
        self.lstm_states = None  # Reset LSTM states for RecurrentPPO
        self.current_state = RobotState.BALANCING
        
        # Reset expert controller
        self.expert_controller.set_action(ActionType.BALANCE_STANDING)
        
        print(f"   ✅ Robot reset complete")
        return observation
    
    def switch_control_mode(self, new_mode: ControlMode):
        """Cambiar modo de control"""
        print(f"🔀 Switching control mode: {self.current_mode.value} → {new_mode.value}")
        
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        # Configurar controlador según nuevo modo
        if new_mode == ControlMode.EXPERT_BALANCE:
            self.expert_controller.set_action(ActionType.BALANCE_STANDING)
        elif new_mode == ControlMode.EXPERT_SQUAT:
            self.expert_controller.set_action(ActionType.SQUAT)
        
        print(f"   ✅ Control mode changed successfully")
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Obtener acción según el modo de control actual
        
        Returns:
            action: Acción para el robot (6 valores PAM)
        """
        
        if self.current_mode == ControlMode.RL_TRAINED:
            return self._get_rl_action(observation)
        
        elif self.current_mode == ControlMode.EXPERT_BALANCE:
            return self._get_expert_action(ActionType.BALANCE_STANDING)
        
        elif self.current_mode == ControlMode.EXPERT_SQUAT:
            return self._get_expert_action(ActionType.SQUAT)
        
        elif self.current_mode == ControlMode.HYBRID:
            return self._get_hybrid_action(observation)
        
        else:
            raise ValueError(f"Unknown control mode: {self.current_mode}")
    
    def _get_rl_action(self, observation: np.ndarray) -> np.ndarray:
        """Obtener acción del modelo RL entrenado"""
        
        # Preparar observación para el modelo
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        # Predicción con RecurrentPPO (incluye LSTM states)
        with torch.no_grad():
            action, self.lstm_states = self.trained_model.predict(
                observation,
                state=self.lstm_states,
                deterministic=True  # Usar política determinística para control
            )
        
        return action
    
    def _get_expert_action(self, action_type: ActionType) -> np.ndarray:
        """Obtener acción del controlador experto"""
        
        return self.expert_controller.get_expert_action(self.env.time_step)
    
    def _get_hybrid_action(self, observation: np.ndarray) -> np.ndarray:
        """Combinar RL y expert según situación"""
        
        # Análisis de situación
        robot_height = observation[2] if len(observation) > 2 else 0.5
        
        # Si robot muy bajo, usar expert para recovery
        if robot_height < self.config.safety_limits['min_height']:
            return self._get_expert_action(ActionType.BALANCE_STANDING)
        
        # Sino, usar RL entrenado
        return self._get_rl_action(observation)
    
    def check_safety(self, observation: np.ndarray) -> bool:
        """
        Verificar condiciones de seguridad
        
        Returns:
            True si es seguro continuar, False si requiere intervención
        """
        
        if not self.safety_active:
            return True
        
        # Extraer información de la observación
        try:
            # Asumiendo estructura de observación del entorno
            robot_height = observation[2] if len(observation) > 2 else 0.5
            robot_pitch = observation[3] if len(observation) > 3 else 0.0
            
            # Check altura mínima
            if robot_height < self.config.safety_limits['emergency_fall_threshold']:
                print(f"⚠️ SAFETY: Robot too low (h={robot_height:.3f})")
                self.safety_violations += 1
                return False
            
            # Check ángulo de inclinación
            if abs(robot_pitch) > self.config.safety_limits['max_tilt_angle']:
                print(f"⚠️ SAFETY: Robot tilted too much (pitch={robot_pitch:.3f})")
                self.safety_violations += 1
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ SAFETY: Error checking safety: {e}")
            return False
    
    def execute_control_step(self) -> Dict[str, Any]:
        """
        Ejecutar un paso completo de control
        
        Returns:
            Dict con información del paso ejecutado
        """
        
        if self.last_observation is None:
            raise RuntimeError("Robot not initialized. Call reset_robot() first.")
        
        # 1. Verificar seguridad
        if not self.check_safety(self.last_observation):
            # Activar modo de emergencia
            if self.safety_violations >= self.max_safety_violations:
                self.current_state = RobotState.EMERGENCY_STOP
                return {
                    'action': np.zeros(6),
                    'reward': 0,
                    'done': True,
                    'info': {'safety_stop': True}
                }
        
        # 2. Obtener acción
        action = self.get_action(self.last_observation)
        
        # 3. Ejecutar acción en entorno
        observation, reward, done, truncated, info = self.env.step(action)
        
        # 4. Actualizar estado interno
        self.last_observation = observation
        self.episode_step += 1
        self.total_reward += reward
        
        # 5. Retornar información del paso
        step_info = {
            'action': action,
            'observation': observation,
            'reward': reward,
            'done': done,
            'truncated': truncated,
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'control_mode': self.current_mode.value,
            'robot_state': self.current_state.value,
            'safety_violations': self.safety_violations,
            **info
        }
        
        return step_info
    
    def run_episode(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Ejecutar episodio completo de control
        
        Args:
            max_steps: Máximo número de pasos (usa config si None)
            
        Returns:
            Estadísticas del episodio
        """
        
        if max_steps is None:
            max_steps = self.config.max_episode_steps
        
        print(f"🚀 Starting control episode (max_steps={max_steps})")
        
        # Reset para episodio nuevo
        observation = self.reset_robot()
        
        # Ejecutar pasos
        episode_rewards = []
        episode_actions = []
        
        for step in range(max_steps):
            
            # Ejecutar paso
            step_info = self.execute_control_step()
            
            episode_rewards.append(step_info['reward'])
            episode_actions.append(step_info['action'])
            
            # Mostrar progreso cada 5 segundos aprox
            if step % 200 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                print(f"   Step {step:4d} | Reward: {step_info['reward']:6.2f} | Avg: {avg_reward:6.2f} | Mode: {step_info['control_mode']}")
            
            # Terminar si done o emergency
            if step_info['done'] or self.current_state == RobotState.EMERGENCY_STOP:
                break
        
        # Estadísticas finales
        episode_stats = {
            'total_steps': self.episode_step,
            'total_reward': self.total_reward,
            'average_reward': np.mean(episode_rewards),
            'max_reward': np.max(episode_rewards) if episode_rewards else 0,
            'min_reward': np.min(episode_rewards) if episode_rewards else 0,
            'final_state': self.current_state.value,
            'safety_violations': self.safety_violations,
            'control_mode_used': self.current_mode.value
        }
        
        print(f"📊 Episode completed:")
        print(f"   Steps: {episode_stats['total_steps']}")
        print(f"   Total reward: {episode_stats['total_reward']:.2f}")
        print(f"   Average reward: {episode_stats['average_reward']:.2f}")
        print(f"   Final state: {episode_stats['final_state']}")
        
        return episode_stats
    
    def save_controller_state(self, filepath: str):
        """Guardar estado actual del controlador"""
        
        state_data = {
            'config': {
                'model_path': self.config.model_path,
                'normalize_path': self.config.normalize_path,
                'max_episode_steps': self.config.max_episode_steps,
                'safety_limits': self.config.safety_limits
            },
            'current_mode': self.current_mode.value,
            'current_state': self.current_state.value,
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'safety_violations': self.safety_violations
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        print(f"💾 Controller state saved to: {filepath}")


# ===== FUNCIONES DE UTILIDAD =====

def create_robot_controller(
    model_path: str,
    normalize_path: Optional[str] = None,
    render_mode: str = 'human',
    max_episode_steps: int = 10000
) -> TrainedRobotController:
    """
    Función de conveniencia para crear controlador del robot
    
    Args:
        model_path: Path al modelo RL entrenado
        normalize_path: Path a normalización (opcional)
        render_mode: Modo de renderizado ('human', 'rgb_array', 'direct')
        max_episode_steps: Máximo pasos por episodio
    
    Returns:
        Controlador del robot inicializado
    """
    
    config = ControllerConfig(
        model_path=model_path,
        normalize_path=normalize_path,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps
    )
    
    return TrainedRobotController(config)


def demo_robot_control():
    """Función de demostración del sistema de control"""
    
    print("🎯 DEMO: Sistema de Control de Robot Bípedo")
    print("="*50)
    
    # Configurar paths (ajusta según tu estructura)
    model_path = "./models_balance_squat/best_model.zip"
    normalize_path = "./models_balance_squat/vec_normalize.pkl"
    
    # Verificar que existen los archivos
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train a model first using start_training_script.py")
        return
    
    try:
        # Crear controlador
        controller = create_robot_controller(
            model_path=model_path,
            normalize_path=normalize_path if os.path.exists(normalize_path) else None,
            render_mode='human',  # Visualización
            max_episode_steps=5000
        )
        
        # Ejecutar demo
        print(f"\n🤖 Running demo episode...")
        
        # Episodio con RL
        controller.switch_control_mode(ControlMode.RL_TRAINED)
        stats_rl = controller.run_episode(max_steps=2000)
        
        time.sleep(2)
        
        # Episodio con controlador experto
        controller.switch_control_mode(ControlMode.EXPERT_BALANCE)
        stats_expert = controller.run_episode(max_steps=1000)
        
        print(f"\n📊 COMPARISON:")
        print(f"   RL Model - Avg Reward: {stats_rl['average_reward']:.2f}")
        print(f"   Expert   - Avg Reward: {stats_expert['average_reward']:.2f}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_robot_control()