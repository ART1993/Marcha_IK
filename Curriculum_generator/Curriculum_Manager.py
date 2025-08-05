from datetime import datetime

from Curriculum_generator.Curriculum_phase import CurriculumPhase

class ExpertCurriculumManager:
    """Gestor del currículo experto para entrenamiento PAM"""
    
    def __init__(self, total_timesteps):
        self.total_timesteps = total_timesteps
        self.current_phase = 0
        self.phases = self._define_curriculum_phases()
        self.phase_transitions = []  # Registro de transiciones
        
    def _define_curriculum_phases(self):
        """Define las fases del currículo experto optimizado para PAM"""
        
        phases = [
        # ==========================================
        # GRUPO 1: EQUILIBRIO BÁSICO (20% total)
        # ==========================================
        
        # FASE 0: Equilibrio estático puro
        CurriculumPhase(
            phase_id=0,
            name="Static Balance Foundation",
            description="Robot learns to maintain upright posture without movement",
            duration_ratio=0.08,  # 8%
            expert_weight=0.0,    # Sin acciones expertas - RL puro para equilibrio
            exploration_factor=0.3,
            control_mode="pam",
            difficulty_level=1
        ),
        
        # FASE 1: Imitación de patrones de equilibrio
        CurriculumPhase(
            phase_id=1,
            name="Balance Imitation",
            description="Imitate expert PAM pressures for stable standing",
            duration_ratio=0.07,  # 7%
            expert_weight=0.9,    # Alta imitación de presiones PAM para equilibrio
            exploration_factor=0.1,
            control_mode="pam",
            difficulty_level=1
        ),
        
        # FASE 2: Exploración guiada de equilibrio
        CurriculumPhase(
            phase_id=2,
            name="Balance Exploration", 
            description="Explore variations in balance while maintaining stability",
            duration_ratio=0.05,  # 5%
            expert_weight=0.6,    # Moderada guía experta
            exploration_factor=0.4,
            control_mode="pam",
            difficulty_level=2
        ),
        
        # ==========================================
        # GRUPO 2: LEVANTAMIENTO PIERNA IZQUIERDA (18% total)
        # ==========================================
        
        # FASE 3: Imitación pura - levantar pierna izquierda
        CurriculumPhase(
            phase_id=3,
            name="Left Leg Lift Imitation",
            description="Direct imitation of expert actions for left leg lifting",
            duration_ratio=0.10,  # 10%
            expert_weight=0.9,    # Alta imitación para nueva habilidad
            exploration_factor=0.1,
            control_mode="pam",   # Control directo PAM
            difficulty_level=2
        ),
        
        # FASE 4: Exploración guiada - pierna izquierda
        CurriculumPhase(
            phase_id=4,
            name="Left Leg Lift Exploration",
            description="Guided exploration of left leg lifting with balance maintenance",
            duration_ratio=0.08,  # 8%
            expert_weight=0.6,    # Reducir guía, aumentar exploración
            exploration_factor=0.4,
            control_mode="pam",
            difficulty_level=2
        ),
        
        # ==========================================
        # GRUPO 3: LEVANTAMIENTO PIERNA DERECHA (18% total)
        # ==========================================
        
        # FASE 5: Imitación pura - levantar pierna derecha
        CurriculumPhase(
            phase_id=5,
            name="Right Leg Lift Imitation",
            description="Direct imitation of expert actions for right leg lifting",
            duration_ratio=0.10,  # 10%
            expert_weight=0.9,    # Alta imitación (transferencia desde pierna izquierda)
            exploration_factor=0.1,
            control_mode="pam",
            difficulty_level=2
        ),
        
        # FASE 6: Exploración guiada - pierna derecha  
        CurriculumPhase(
            phase_id=6,
            name="Right Leg Lift Exploration",
            description="Guided exploration of right leg lifting with balance maintenance",
            duration_ratio=0.08,  # 8%
            expert_weight=0.6,    # Misma progresión que pierna izquierda
            exploration_factor=0.4,
            control_mode="pam",
            difficulty_level=2
        ),
        
        # ==========================================
        # GRUPO 4: PASO CON PIERNA IZQUIERDA (22% total)
        # ==========================================
        
        # FASE 7: Imitación de paso con pierna izquierda
        CurriculumPhase(
            phase_id=7,
            name="Left Step Imitation",
            description="Imitate expert single forward step with left leg",
            duration_ratio=0.12,  # 12% - más tiempo para habilidad compleja
            expert_weight=0.8,    # Ligeramente menos que levantamiento (más complejo)
            exploration_factor=0.2,
            control_mode="hybrid", # Introducir IK para precisión de paso
            difficulty_level=3
        ),
        
        # FASE 8: Exploración de paso izquierdo
        CurriculumPhase(
            phase_id=8,
            name="Left Step Exploration",
            description="Explore variations in left leg stepping while maintaining balance",
            duration_ratio=0.10,  # 10%
            expert_weight=0.5,    # Balance entre guía y exploración
            exploration_factor=0.5,
            control_mode="hybrid",
            difficulty_level=3
        ),
        
        # ==========================================
        # GRUPO 5: PASO CON PIERNA DERECHA (22% total)
        # ==========================================
        
        # FASE 9: Imitación de paso con pierna derecha
        CurriculumPhase(
            phase_id=9,
            name="Right Step Imitation", 
            description="Imitate expert single forward step with right leg",
            duration_ratio=0.12,  # 12% - misma complejidad que paso izquierdo
            expert_weight=0.8,    # Misma configuración que paso izquierdo
            exploration_factor=0.2,
            control_mode="hybrid", # Mantener IK para precisión
            difficulty_level=3
        ),
        
        # FASE 10: Maestría y integración de ambos pasos
        CurriculumPhase(
            phase_id=10,
            name="Bilateral Step Mastery",
            description="Master both left and right stepping with minimal guidance",
            duration_ratio=0.10,  # 10% - consolidación final
            expert_weight=0.2,    # Mínima guía experta
            exploration_factor=0.8,
            control_mode="pam",   # Volver a control PAM puro
            difficulty_level=4    # Alta dificultad para maestría
        )
        ]

        # Verificación de integridad del currículo
        total_duration = sum(phase.duration_ratio for phase in phases)
        assert abs(total_duration - 1.0) < 1e-6, f"Duration ratios must sum to 1.0, got {total_duration}"

        # Logging del currículo para debugging
        print("\n🎓 Simplified Task Curriculum Summary:")
        print("=" * 50)
        for phase in phases:
            print(f"Phase {phase.phase_id:2d}: {phase.name:25s} ({phase.duration_ratio*100:4.1f}%) "
                f"Expert:{phase.expert_weight:.1f} Mode:{phase.control_mode:7s} Diff:{phase.difficulty_level}")
        print(f"Total: {total_duration*100:.1f}%")
        print("=" * 50)
        
        return phases
    
    def get_phase_timesteps(self, phase_id):
        """Calcula los timesteps para una fase específica"""
        if phase_id >= len(self.phases):
            return 0
        return int(self.total_timesteps * self.phases[phase_id].duration_ratio)
    
    def get_current_phase(self):
        """Obtiene la fase actual"""
        return self.phases[self.current_phase] if self.current_phase < len(self.phases) else None
    
    def advance_phase(self):
        """Avanza a la siguiente fase"""
        if self.current_phase < len(self.phases) - 1:
            self.current_phase += 1
            self.phase_transitions.append({
                'phase': self.current_phase,
                'timestamp': datetime.now().isoformat()
            })
            return True
        return False
    
    def get_phase_info(self, phase_id=None):
        """Obtiene información detallada de una fase"""
        if phase_id is None:
            phase_id = self.current_phase
            
        if phase_id >= len(self.phases):
            return None
            
        phase = self.phases[phase_id]
        return {
            'phase_id': phase.phase_id,
            'name': phase.name,
            'description': phase.description,
            'timesteps': self.get_phase_timesteps(phase_id),
            'expert_weight': phase.expert_weight,
            'exploration_factor': phase.exploration_factor,
            'control_mode': phase.control_mode,
            'difficulty_level': phase.difficulty_level
        }