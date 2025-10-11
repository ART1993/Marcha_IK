"""
    Aquí se crearán los métodos por los cuales se producen las distintas recompensas en RewardSystemSimple
    El objetivo es que se puedan reutilizar para distintos movimientos y así no tener que reescribir todo el código
    cada vez que se quiera crear una nueva recompensa.
"""

def height_reward_method(height):
    """
        Recompensa por altura ganada
    """
    if height > 0.8:
        return 1.0  # Buena altura
    elif height > 0.7:
        return 0.8  # Altura mínima
    elif height <= 0.7:
        return -1.0  # Caída
    elif height<= 0.5:       # and self.last_done_reason == self.bad_ending[0]:
        return -10
    

def contacto_pies_reward(pie_izquierdo_contacto, pie_derecho_contacto):
    """
        Recompensa por contacto de los pies con el suelo
    """
    
    if pie_izquierdo_contacto is False and pie_derecho_contacto:
        return 2.0
    elif pie_izquierdo_contacto and pie_derecho_contacto:
        return 0.1
    else:
        return -2.0
    
def knee_reward_method(self, target_knee, support_knee):
        
        """
            Metodo que recompensa tener una rodilla doblada u otra rigida
            target_knee: pierna que se doblara.
            support_knee: pierna que se mantendra como apoyo.
        """
        
        if 0.1<target_knee<0.2:
            reward_knee_left=1
        elif 0.2<= target_knee < 0.4:
            reward_knee_left=2
        else:
            reward_knee_left=-2

        if 0.1<support_knee<0.2:
            reward_knee_right=1
        elif 0<=support_knee<=0.1:
            reward_knee_right=0.5
        else:
            reward_knee_right=-2

        self.reawrd_step['reward_knee_right'] =  reward_knee_right
        self.reawrd_step['reward_knee_left'] =  reward_knee_left
        
        return reward_knee_right+ reward_knee_left

def pitch_stability_rewards(self, pitch):
    if pitch < 0.2:
            2.5  # Muy estable
    elif pitch < 0.4:
        0.5  # Moderadamente estable
    elif pitch < self.max_tilt_by_level[self.level]:
        return -2.0  # Inestable
    elif pitch >= self.max_tilt_by_level[self.level]:# self.last_done_reason == self.bad_ending[1]:
        return  -25  # Inestable