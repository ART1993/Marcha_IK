# train_discrete_actions.py
"""
Script principal para entrenar el robot bípedo con acciones discretas
usando currículo experto y sistema de 6 PAMs antagónicos
"""

import os
import time
import json
import numpy as np
from datetime import datetime
from training_config import DiscreteActionsTrainingConfig, get_phase_config
from Gymnasium_Start.Enhanced_UnifiedBipedTrainer import Enhanced_UnifiedBipedTrainer
from Controlador.discrete_action_controller import DiscreteActionController, ActionType

from Gymnasium_Start.Enhanced_PAMIKBipedEnv import Enhanced_PAMIKBipedEnv
from sb3_contrib import RecurrentPPO

def train_discrete_actions_biped(
    resume=True,
    test_mode=False,
    start_phase=None,
    visualize=False
):
    """
    Función principal de entrenamiento para acciones discretas
    
    Args:
        resume: Si continuar desde checkpoint anterior
        test_mode: Modo de prueba con menos timesteps
        start_phase: Fase específica desde la que empezar (None = desde el principio)
        visualize: Si mostrar visualización durante entrenamiento
    """
    
    print("🤖 ENTRENAMIENTO DE ROBOT BÍPEDO CON ACCIONES DISCRETAS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # ===== CONFIGURACIÓN =====
    
    config = DiscreteActionsTrainingConfig()
    
    if test_mode:
        print("⚠️ MODO DE PRUEBA ACTIVADO - Timesteps reducidos")
        total_timesteps = 50000  # Solo 50k para prueba rápida
        n_envs = 2
    else:
        total_timesteps = config.TOTAL_TIMESTEPS
        n_envs = config.N_ENVS
    
    # ===== IMPORTAR COMPONENTES =====
    
    try:
        
        
        print("✅ Componentes importados correctamente")
        
    except ImportError as e:
        print(f"❌ Error importando componentes: {e}")
        print("   Asegúrate de tener todos los archivos necesarios")
        return False
    
    # ===== CREAR TRAINER =====
    
    print(f"\n📦 Creando Enhanced Trainer...")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Entornos paralelos: {n_envs}")
    print(f"   Action space: PAM (6 músculos antagónicos)")
    
    trainer = Enhanced_UnifiedBipedTrainer(
        env_type='enhanced_pam',
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=config.LEARNING_RATES['phase_0_2'],  # Se ajustará por fase
        use_wandb=False,  # Activar si quieres logging con Weights & Biases
        action_space="pam",
        enable_expert_curriculum=True,
        system_version="enhanced",
    )
    
    print("✅ Trainer creado exitosamente")
    
    # ===== CONFIGURAR ENTORNO CON ACCIONES DISCRETAS =====
    
    print("\n🌍 Configurando entorno con acciones discretas...")
    
    # Modificar la creación del entorno para usar acciones discretas
    original_create_env = trainer.create_training_env
    
    def create_discrete_env():
        """Wrapper para crear entorno con acciones discretas"""
        # Primero crear el entorno normal
        env = original_create_env()
        
        # Luego configurar para acciones discretas
        if hasattr(env, 'envs'):
            for vec_env in env.envs:
                base_env = vec_env.env if hasattr(vec_env, 'env') else vec_env
                if hasattr(base_env, 'use_discrete_actions'):
                    base_env.use_discrete_actions = True
                    print("   ✅ Acciones discretas habilitadas en entorno de entrenamiento")
        
        return env
    
    # Reemplazar el método
    trainer.create_training_env = create_discrete_env
    
    # ===== CONFIGURAR MÉTRICAS Y LOGGING =====
    
    metrics = {
        'training_start': datetime.now().isoformat(),
        'phase_metrics': {},
        'global_metrics': {
            'total_episodes': 0,
            'total_falls': 0,
            'best_reward': -float('inf'),
            'training_time_hours': 0
        }
    }
    
    # ===== CONFIGURAR CALLBACKS PERSONALIZADOS =====
    
    print("\n📊 Configurando callbacks y métricas...")
    
    class DiscreteActionsCallback:
        """Callback personalizado para monitorear acciones discretas"""
        
        def __init__(self, phase_id):
            self.phase_id = phase_id
            self.phase_start_time = time.time()
            self.episode_rewards = []
            self.episode_lengths = []
            self.falls_count = 0
            self.success_count = 0
            
        def on_step(self):
            return True
        
        def on_episode_end(self, reward, length, did_fall):
            """Llamado al final de cada episodio"""
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            if did_fall:
                self.falls_count += 1
            
            # Determinar si fue exitoso
            phase_config = get_phase_config(self.phase_id)
            if reward >= phase_config['success_threshold']:
                self.success_count += 1
            
            # Log cada 10 episodios
            if len(self.episode_rewards) % 10 == 0:
                self._log_progress()
        
        def _log_progress(self):
            """Muestra progreso de la fase actual"""
            mean_reward = np.mean(self.episode_rewards[-10:])
            mean_length = np.mean(self.episode_lengths[-10:])
            success_rate = self.success_count / len(self.episode_rewards)
            fall_rate = self.falls_count / len(self.episode_rewards)
            
            elapsed_time = (time.time() - self.phase_start_time) / 60  # minutos
            
            print(f"\n📈 Phase {self.phase_id} Progress:")
            print(f"   Episodes: {len(self.episode_rewards)}")
            print(f"   Mean Reward (last 10): {mean_reward:.2f}")
            print(f"   Mean Length (last 10): {mean_length:.1f}")
            print(f"   Success Rate: {success_rate:.2%}")
            print(f"   Fall Rate: {fall_rate:.2%}")
            print(f"   Time Elapsed: {elapsed_time:.1f} min")
        
        def get_phase_summary(self):
            """Retorna resumen de la fase"""
            return {
                'total_episodes': len(self.episode_rewards),
                'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
                'success_rate': self.success_count / max(1, len(self.episode_rewards)),
                'fall_rate': self.falls_count / max(1, len(self.episode_rewards)),
                'training_time_minutes': (time.time() - self.phase_start_time) / 60
            }
    
    # ===== OVERRIDE DE FASES SI SE ESPECIFICA =====
    
    if start_phase is not None:
        print(f"\n⚠️ Comenzando desde fase {start_phase}")
        # Ajustar el curriculum manager para saltar fases
        if hasattr(trainer, 'curriculum_manager'):
            trainer.curriculum_manager.current_phase = start_phase
    
    # ===== COMENZAR ENTRENAMIENTO =====
    
    print("\n" + "=" * 70)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    
    try:
        start_time = time.time()
        
        # Ejecutar entrenamiento
        trainer.train(resume=resume)
        
        # Calcular tiempo total
        total_time = (time.time() - start_time) / 3600  # horas
        metrics['global_metrics']['training_time_hours'] = total_time
        
        print(f"\n✅ Entrenamiento completado exitosamente")
        print(f"   Tiempo total: {total_time:.2f} horas")
        
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ===== GUARDAR MÉTRICAS FINALES =====
    
    print("\n💾 Guardando métricas finales...")
    
    metrics_file = f"discrete_actions_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"   Métricas guardadas en: {metrics_file}")
    
    # ===== EVALUACIÓN FINAL =====
    
    if not test_mode:
        print("\n🏆 EVALUACIÓN FINAL")
        print("=" * 60)
        
        # Aquí podrías ejecutar una evaluación completa del modelo final
        # evaluate_final_model(trainer, visualize=True)
    
    return True


def evaluate_trained_model(model_path=None, num_episodes=10, action_sequence=None):
    """
    Evalúa un modelo entrenado con visualización
    
    Args:
        model_path: Ruta al modelo guardado (None = buscar el más reciente)
        num_episodes: Número de episodios a evaluar
        action_sequence: Secuencia específica de acciones a ejecutar
    """
    
    print("\n🎮 EVALUACIÓN DE MODELO ENTRENADO")
    print("-" * 50)
    
    # Crear entorno con visualización
    env = Enhanced_PAMIKBipedEnv(
        render_mode='human',
        action_space="pam",
        use_discrete_actions=True
    )
    
    # Cargar modelo
    if model_path is None:
        # Buscar el modelo más reciente
        model_files = [f for f in os.listdir('./models') if f.endswith('_final.zip')]
        if model_files:
            model_path = os.path.join('./models', sorted(model_files)[-1])
            print(f"   Cargando modelo: {model_path}")
        else:
            print("   ❌ No se encontró modelo entrenado")
            return
    
    model = RecurrentPPO.load(model_path)
    
    # Secuencia de acciones por defecto
    if action_sequence is None:
        action_sequence = [
            ActionType.BALANCE_STANDING,
            ActionType.SQUAT,
            ActionType.BALANCE_STANDING,
            ActionType.LIFT_LEFT_LEG,
            ActionType.BALANCE_STANDING,
            ActionType.LIFT_RIGHT_LEG,
            ActionType.BALANCE_STANDING,
            ActionType.STEP_LEFT,
            ActionType.BALANCE_STANDING,
            ActionType.STEP_RIGHT,
        ]
    
    print(f"   Ejecutando {num_episodes} episodios de evaluación...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Seleccionar acción de la secuencia
        action_idx = episode % len(action_sequence)
        env.walking_controller.set_action(action_sequence[action_idx])
        
        print(f"\n   Episodio {episode + 1}: {action_sequence[action_idx].value}")
        
        episode_reward = 0
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        for step in range(1000):  # Máximo 1000 pasos por episodio
            # Predecir acción con el modelo entrenado
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_start = np.array([False])
            
            if done:
                print(f"      Reward total: {episode_reward:.2f}")
                print(f"      Pasos: {step}")
                break
        
        time.sleep(1)  # Pausa entre episodios
    
    env.close()
    print("\n✅ Evaluación completada")


def main(mode='train', resume=False, visualize=False, start_phase=None, episodes=10):
    """    
        Función principal para ejecutar el script según el modo
        Args:
            mode: Modo de ejecución ('train', 'test', 'eval')
            resume: Si continuar desde checkpoint anterior
            visualize: Si mostrar visualización durante entrenamiento
            start_phase: Fase específica desde la que empezar (0-12)
            episodes: Número de episodios para evaluación"""
    if mode == 'train':
        # Entrenamiento completo
        success = train_discrete_actions_biped(
            resume=resume,
            test_mode=False,
            start_phase=start_phase,
            visualize=visualize
        )
        
        if success:
            print("\n🎉 ¡Entrenamiento exitoso!")
            print("   Ejecuta con --mode eval para ver el resultado")
    
    elif mode == 'test':
        # Modo de prueba rápida
        print("🧪 Ejecutando entrenamiento de prueba (50k timesteps)...")
        train_discrete_actions_biped(
            resume=False,
            test_mode=True,
            start_phase=start_phase,
            visualize=visualize
        )
    
    elif mode == 'eval':
        # Evaluación de modelo entrenado
        evaluate_trained_model(num_episodes=episodes)
    
    print("\n✨ Proceso completado")

# Script principal
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento de Robot Bípedo con Acciones Discretas')
    parser.add_argument('--mode', choices=['train', 'test', 'eval'], default='train',
                      help='Modo de ejecución: train, test, o eval')
    parser.add_argument('--resume', action='store_true',
                      help='Continuar desde checkpoint anterior')
    parser.add_argument('--visualize', action='store_true',
                      help='Mostrar visualización durante entrenamiento')
    parser.add_argument('--start-phase', type=int, default=None,
                      help='Fase específica desde la que empezar (0-12)')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Número de episodios para evaluación')
    
    args = parser.parse_args()
