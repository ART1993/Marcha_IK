# training_config.py
"""
Configuración optimizada para entrenamiento con acciones discretas
basada en los resultados del test de integración
"""

class DiscreteActionsTrainingConfig:
    """Configuración optimizada para el entrenamiento del robot bípedo con acciones discretas"""
    
    # ===== CONFIGURACIÓN GENERAL =====
    
    # Timesteps totales y distribución
    TOTAL_TIMESTEPS = 5_000_000  # 5M para dominar todas las acciones
    
    # Configuración de entornos paralelos
    N_ENVS = 6  # Reducido de 8 para mejor estabilidad con acciones discretas
    
    # Learning rate adaptativo por fase
    LEARNING_RATES = {
        'phase_0_2': 5e-4,   # Equilibrio: learning rate más alto
        'phase_3_4': 3e-4,   # Sentadillas: moderado
        'phase_5_8': 3e-4,   # Levantar piernas: moderado
        'phase_9_12': 2e-4,  # Pasos: más conservador
    }
    
    # ===== CONFIGURACIÓN POR FASE =====
    
    # Duración mínima de éxito antes de avanzar (en episodios)
    MIN_SUCCESS_EPISODES = {
        0: 50,   # Equilibrio básico
        1: 100,  # Equilibrio con imitación
        2: 100,  # Equilibrio con exploración
        3: 150,  # Sentadilla parcial
        4: 200,  # Sentadilla completa
        5: 150,  # Levantar pierna izquierda
        6: 150,  # Exploración pierna izquierda
        7: 150,  # Levantar pierna derecha
        8: 150,  # Exploración pierna derecha
        9: 200,  # Paso izquierdo
        10: 200, # Exploración paso izquierdo
        11: 200, # Paso derecho
        12: 250, # Maestría bilateral
    }
    
    # Criterios de éxito por fase (reward promedio mínimo)
    SUCCESS_THRESHOLDS = {
        0: -5.0,   # Fase 0: Solo necesita no caerse mucho
        1: 3.0,    # Fase 1-2: Equilibrio estable
        2: 3.5,
        3: 4.0,    # Fase 3-4: Sentadillas controladas
        4: 4.5,
        5: 4.0,    # Fase 5-8: Levantar piernas
        6: 4.5,
        7: 4.0,
        8: 4.5,
        9: 3.5,    # Fase 9-12: Pasos (más difícil)
        10: 4.0,
        11: 3.5,
        12: 4.0,
    }
    
    # ===== CALLBACKS Y MONITOREO =====
    
    # Frecuencias de callbacks
    CHECKPOINT_FREQ = 5000  # Guardar modelo cada 5000 pasos
    EVAL_FREQ = 2500        # Evaluar cada 2500 pasos
    LOG_FREQ = 100          # Log cada 100 pasos
    
    # Evaluación
    N_EVAL_EPISODES = 10    # Episodios para evaluación
    
    # ===== AJUSTES ESPECÍFICOS PARA ACCIONES DISCRETAS =====
    
    # Noise para exploración (por fase)
    EXPLORATION_NOISE = {
        0: 0.15,   # Más exploración en equilibrio básico
        1: 0.05,   # Mínima con imitación alta
        2: 0.10,   # Moderada en exploración
        3: 0.08,   # Sentadillas
        4: 0.12,
        5: 0.08,   # Levantar piernas
        6: 0.12,
        7: 0.08,
        8: 0.12,
        9: 0.10,   # Pasos
        10: 0.15,
        11: 0.10,
        12: 0.20,  # Máxima exploración en maestría
    }
    
    # Duración de acciones (segundos por acción completa)
    ACTION_DURATIONS = {
        'balance_standing': 3.0,
        'squat': 3.0,
        'lift_left_leg': 2.5,
        'lift_right_leg': 2.5,
        'step_left': 2.0,
        'step_right': 2.0,
    }
    
    # ===== CONFIGURACIÓN DE RECOMPENSAS =====
    
    # Multiplicadores de recompensa por tipo de acción
    REWARD_MULTIPLIERS = {
        'balance_standing': 1.0,
        'squat': 1.2,           # Bonus por dificultad
        'lift_left_leg': 1.1,
        'lift_right_leg': 1.1,
        'step_left': 1.3,       # Mayor bonus por complejidad
        'step_right': 1.3,
    }
    
    # Penalizaciones adicionales
    FALL_PENALTY = -20.0
    EXCESSIVE_TILT_PENALTY = -5.0
    ASYMMETRY_PENALTY = -2.0  # Para acciones que deben ser simétricas
    
    # ===== CONFIGURACIÓN DE EARLY STOPPING =====
    
    # Si el robot domina una fase muy rápido, avanzar
    EARLY_ADVANCE_MULTIPLIER = 0.5  # Avanzar si completa 50% de timesteps con éxito alto
    EARLY_ADVANCE_REWARD = 1.5      # Reward 50% superior al threshold
    
    # Si una fase es muy difícil, extender el entrenamiento
    STUCK_DETECTION_EPISODES = 500   # Si no mejora en 500 episodios
    STUCK_EXTENSION_MULTIPLIER = 1.5 # Extender la fase 50% más
    
    # ===== CONFIGURACIÓN DE LSTM =====
    
    # Arquitectura LSTM optimizada para acciones discretas
    LSTM_CONFIG = {
        'lstm_hidden_size': 128,  # Tamaño moderado para acciones discretas
        'n_lstm_layers': 2,       # 2 capas para capturar dependencias temporales
        'shared_lstm': False,     # LSTMs separados para policy y value
    }
    
    # ===== HIPERPARÁMETROS PPO =====
    
    PPO_CONFIG = {
        'n_steps': 512,          # Reducido para acciones discretas
        'batch_size': 128,       # Batches más pequeños
        'n_epochs': 8,           # Más épocas por update
        'gamma': 0.99,           # Descuento estándar
        'gae_lambda': 0.95,      # GAE estándar
        'clip_range': 0.2,       # Clipping estándar
        'vf_coef': 0.5,          # Coeficiente de value function
        'ent_coef': 0.01,        # Entropía para exploración
        'max_grad_norm': 0.5,    # Gradient clipping
    }


def get_phase_config(phase_id):
    """
    Obtiene la configuración específica para una fase del currículo
    
    Args:
        phase_id: ID de la fase (0-12)
    
    Returns:
        dict: Configuración para esa fase
    """
    config = DiscreteActionsTrainingConfig()
    
    # Determinar learning rate
    if phase_id <= 2:
        lr = config.LEARNING_RATES['phase_0_2']
    elif phase_id <= 4:
        lr = config.LEARNING_RATES['phase_3_4']
    elif phase_id <= 8:
        lr = config.LEARNING_RATES['phase_5_8']
    else:
        lr = config.LEARNING_RATES['phase_9_12']
    
    return {
        'learning_rate': lr,
        'min_success_episodes': config.MIN_SUCCESS_EPISODES.get(phase_id, 100),
        'success_threshold': config.SUCCESS_THRESHOLDS.get(phase_id, 3.0),
        'exploration_noise': config.EXPLORATION_NOISE.get(phase_id, 0.1),
        'checkpoint_freq': config.CHECKPOINT_FREQ,
        'eval_freq': config.EVAL_FREQ,
        'n_eval_episodes': config.N_EVAL_EPISODES,
    }


def should_advance_phase(phase_metrics, phase_id, current_timesteps, allocated_timesteps):
    """
    Determina si es momento de avanzar a la siguiente fase
    
    Args:
        phase_metrics: Métricas de la fase actual
        phase_id: ID de la fase actual
        current_timesteps: Timesteps completados en la fase
        allocated_timesteps: Timesteps asignados a la fase
    
    Returns:
        bool: True si debe avanzar
    """
    config = DiscreteActionsTrainingConfig()
    phase_config = get_phase_config(phase_id)
    
    # Verificar si ha completado suficientes episodios exitosos
    if 'successful_episodes' in phase_metrics:
        if phase_metrics['successful_episodes'] < phase_config['min_success_episodes']:
            return False
    
    # Verificar reward promedio
    if 'mean_reward' in phase_metrics:
        if phase_metrics['mean_reward'] < phase_config['success_threshold']:
            # Verificar si está atascado
            if current_timesteps > allocated_timesteps * config.STUCK_EXTENSION_MULTIPLIER:
                print(f"⚠️ Phase {phase_id} extended but still struggling. Advancing anyway.")
                return True
            return False
    
    # Early stopping si lo está haciendo muy bien
    if 'mean_reward' in phase_metrics:
        early_threshold = phase_config['success_threshold'] * config.EARLY_ADVANCE_REWARD
        if phase_metrics['mean_reward'] > early_threshold:
            if current_timesteps > allocated_timesteps * config.EARLY_ADVANCE_MULTIPLIER:
                print(f"🌟 Phase {phase_id} mastered early! Advancing.")
                return True
    
    # Avanzar si completó los timesteps asignados
    return current_timesteps >= allocated_timesteps


# Función de inicialización del entrenamiento
def initialize_training():
    """
    Inicializa el entrenamiento con la configuración optimizada
    
    Returns:
        dict: Configuración completa para el entrenamiento
    """
    config = DiscreteActionsTrainingConfig()
    
    print("🎯 Configuración de Entrenamiento con Acciones Discretas")
    print("=" * 60)
    print(f"📊 Total timesteps: {config.TOTAL_TIMESTEPS:,}")
    print(f"🖥️ Entornos paralelos: {config.N_ENVS}")
    print(f"💾 Checkpoint cada: {config.CHECKPOINT_FREQ} pasos")
    print(f"📈 Evaluación cada: {config.EVAL_FREQ} pasos")
    print("=" * 60)
    
    return {
        'total_timesteps': config.TOTAL_TIMESTEPS,
        'n_envs': config.N_ENVS,
        'ppo_config': config.PPO_CONFIG,
        'lstm_config': config.LSTM_CONFIG,
        'callbacks': {
            'checkpoint_freq': config.CHECKPOINT_FREQ,
            'eval_freq': config.EVAL_FREQ,
            'n_eval_episodes': config.N_EVAL_EPISODES,
        }
    }


if __name__ == "__main__":
    # Ejemplo de uso
    training_config = initialize_training()
    
    # Mostrar configuración para cada fase
    print("\n📋 Configuración por Fase:")
    for phase_id in range(13):
        phase_cfg = get_phase_config(phase_id)
        print(f"\nFase {phase_id}:")
        print(f"  LR: {phase_cfg['learning_rate']}")
        print(f"  Success threshold: {phase_cfg['success_threshold']}")
        print(f"  Min episodes: {phase_cfg['min_success_episodes']}")
        print(f"  Exploration: {phase_cfg['exploration_noise']}")
