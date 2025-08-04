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
            # FASE 0: Estabilización básica
            CurriculumPhase(
                phase_id=0,
                name="Stabilization",
                description="Robot learns basic balance and posture",
                duration_ratio=0.08,  # 8% del tiempo total
                expert_weight=0.0,    # Sin acciones expertas, solo equilibrio
                exploration_factor=0.3,
                control_mode="pam",
                difficulty_level=1
            ),
            
            # FASE 1: Imitación pura de acciones expertas
            CurriculumPhase(
                phase_id=1,
                name="Pure Imitation",
                description="Direct imitation of expert PAM pressures",
                duration_ratio=0.15,
                expert_weight=0.9,    # Alta imitación
                exploration_factor=0.1,
                control_mode="pam",
                difficulty_level=1
            ),
            
            # FASE 2: Imitación con exploración moderada
            CurriculumPhase(
                phase_id=2,
                name="Guided Exploration",
                description="Expert guidance with moderate exploration",
                duration_ratio=0.12,
                expert_weight=0.7,
                exploration_factor=0.3,
                control_mode="pam",
                difficulty_level=2
            ),
            
            # FASE 3: Control híbrido IK + PAM con imitación
            CurriculumPhase(
                phase_id=3,
                name="Hybrid Control Introduction",
                description="Introduction of IK control with PAM fine-tuning",
                duration_ratio=0.15,
                expert_weight=0.6,
                exploration_factor=0.4,
                control_mode="hybrid",
                difficulty_level=2
            ),
            
            # FASE 4: Transición progresiva a RL
            CurriculumPhase(
                phase_id=4,
                name="Progressive RL Transition",
                description="Gradual reduction of expert influence",
                duration_ratio=0.20,
                expert_weight=0.3,    # Reducción significativa de imitación
                exploration_factor=0.7,
                control_mode="hybrid",
                difficulty_level=3
            ),
            
            # FASE 5: RL con guía mínima
            CurriculumPhase(
                phase_id=5,
                name="Minimal Guidance RL",
                description="RL learning with minimal expert guidance",
                duration_ratio=0.15,
                expert_weight=0.1,
                exploration_factor=0.9,
                control_mode="pam",
                difficulty_level=4
            ),
            
            # FASE 6: RL puro con desafíos adicionales
            CurriculumPhase(
                phase_id=6,
                name="Pure RL Mastery",
                description="Pure RL learning with environmental challenges",
                duration_ratio=0.15,
                expert_weight=0.0,
                exploration_factor=1.0,
                control_mode="pam",
                difficulty_level=5
            )
        ]
        
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