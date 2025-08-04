class CurriculumPhase:
    """Representa una fase individual del currículo experto"""
    
    def __init__(self, phase_id, name, description, duration_ratio, 
                 expert_weight=1.0, exploration_factor=0.0, 
                 control_mode="hybrid", difficulty_level=1):
        self.phase_id = phase_id
        self.name = name
        self.description = description
        self.duration_ratio = duration_ratio  # Porcentaje del tiempo total de entrenamiento
        self.expert_weight = expert_weight    # Peso de las acciones expertas (0-1)
        self.exploration_factor = exploration_factor  # Factor de exploración añadido
        self.control_mode = control_mode      # Modo de control: "ik", "pam", "hybrid"
        self.difficulty_level = difficulty_level  # Nivel de dificultad (1-5)
        
    def __str__(self):
        return f"Phase {self.phase_id}: {self.name} ({self.duration_ratio*100:.1f}% duration)"
    
