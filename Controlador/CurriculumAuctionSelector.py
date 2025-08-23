from collections import deque
import numpy as np


class CurriculumActionSelector:
    """
    Sistema que decide cuándo usar acciones expertas vs acciones de la red neuronal.
    Implementa curriculum learning progresivo basado en rendimiento.
    """
    
    def __init__(self):
        # Parámetros de curriculum
        self.episode_count = 0
        self.recent_rewards = deque(maxlen=100)  # Últimos 100 episodios
        self.expert_ratio = 1.0  # Empezar 100% experto
        
        # Umbrales de transición
        self.balance_reward_threshold = 5.0     # Reward para considerar balance exitoso
        self.squat_reward_threshold = 8.0       # Reward para permitir sentadillas
        self.min_expert_ratio = 0.1             # Mínimo 10% experto siempre
        
        # Estados del curriculum
        self.current_phase = "BALANCE_LEARNING"  # BALANCE_LEARNING, SQUAT_LEARNING, ADVANCED
        self.phase_episodes = 0
        
        print(f"🎓 Curriculum Action Selector initialized")
        print(f"   Starting phase: {self.current_phase}")
        print(f"   Expert ratio: {self.expert_ratio:.1%}")
    
    def should_use_expert_action(self, step_count_in_episode):
        """
        Decide si usar acción experta en este step específico.
        
        Args:
            step_count_in_episode: Step actual dentro del episodio
            
        Returns:
            bool: True si usar experto, False si usar RL agent
        """
        
        # Factor aleatorio basado en expert_ratio actual
        random_factor = np.random.random()
        
        # En los primeros steps del episodio, usar más experto para estabilización
        if step_count_in_episode < 50:  # Primeros ~0.33s
            early_bonus = 0.2
        else:
            early_bonus = 0.0
        
        effective_expert_ratio = min(1.0, self.expert_ratio + early_bonus)
        
        return random_factor < effective_expert_ratio
    
    def should_transition_to_squat(self):
        """
        Decide si es momento de cambiar de balance a sentadillas.
        
        Returns:
            bool: True si debe cambiar a sentadilla
        """
        
        if self.current_phase == "BALANCE_LEARNING":
            # Solo permitir sentadillas si balance está dominado
            recent_avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else -10
            
            if (recent_avg_reward >= self.balance_reward_threshold and 
                self.phase_episodes >= 100):  # Mínimo 100 episodios de balance
                return True
        
        elif self.current_phase in ["SQUAT_LEARNING", "ADVANCED"]:
            # En fase avanzada, alternar según performance
            if len(self.recent_rewards) >= 10:
                last_10_avg = np.mean(list(self.recent_rewards)[-10:])
                return last_10_avg >= self.squat_reward_threshold
        
        return False
    
    def update_after_episode(self, total_episode_reward, episode_length):
        """
        Actualizar curriculum después de cada episodio.
        
        Args:
            total_episode_reward: Reward total del episodio
            episode_length: Duración del episodio en steps
        """
        
        self.episode_count += 1
        self.phase_episodes += 1
        self.recent_rewards.append(total_episode_reward)
        
        # Calcular métricas de rendimiento
        recent_avg = np.mean(self.recent_rewards) if len(self.recent_rewards) >= 10 else total_episode_reward
        
        # ===== ACTUALIZAR EXPERT RATIO =====
        
        old_ratio = self.expert_ratio
        
        if recent_avg >= self.balance_reward_threshold:
            # Rendimiento bueno → Reducir dependencia del experto
            self.expert_ratio *= 0.995  # Reducción gradual 0.5% por episodio
        else:
            # Rendimiento malo → Aumentar ayuda del experto
            self.expert_ratio = min(1.0, self.expert_ratio * 1.01)  # Aumento 1%
        
        # Mantener límites
        self.expert_ratio = max(self.min_expert_ratio, min(1.0, self.expert_ratio))
        
        # ===== ACTUALIZAR FASE DEL CURRICULUM =====
        
        old_phase = self.current_phase
        
        if self.current_phase == "BALANCE_LEARNING":
            if (recent_avg >= self.balance_reward_threshold and 
                self.phase_episodes >= 150 and
                self.expert_ratio < 0.5):
                self.current_phase = "SQUAT_LEARNING"
                self.phase_episodes = 0
                
        elif self.current_phase == "SQUAT_LEARNING":
            if (recent_avg >= self.squat_reward_threshold and 
                self.phase_episodes >= 200 and
                self.expert_ratio < 0.3):
                self.current_phase = "ADVANCED"
                self.phase_episodes = 0
        
        # ===== LOGGING =====
        
        if self.episode_count % 50 == 0 or old_phase != self.current_phase:
            print(f"\n📚 Curriculum Update (Episode {self.episode_count}):")
            print(f"   Phase: {self.current_phase} (episode {self.phase_episodes})")
            print(f"   Expert ratio: {old_ratio:.1%} → {self.expert_ratio:.1%}")
            print(f"   Recent avg reward: {recent_avg:.2f}")
            print(f"   Episode reward: {total_episode_reward:.2f}")
            
            if old_phase != self.current_phase:
                print(f"   🎉 PHASE TRANSITION: {old_phase} → {self.current_phase}")
    
    def get_curriculum_info(self):
        """Info del estado actual del curriculum para debugging"""
        return {
            'episode_count': self.episode_count,
            'current_phase': self.current_phase,
            'phase_episodes': self.phase_episodes,
            'expert_ratio': self.expert_ratio,
            'recent_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0,
            'balance_threshold_met': np.mean(self.recent_rewards) >= self.balance_reward_threshold if self.recent_rewards else False
        }