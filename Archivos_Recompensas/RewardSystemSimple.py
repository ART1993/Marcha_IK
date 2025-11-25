# OBJETIVO: Eliminar complejidades innecesarias y hacer el código más mantenible

import pybullet as p
import numpy as np
from enum import Enum

class RewardMode(Enum):
    PROGRESSIVE = "progressive"      # curriculum por niveles (modo actual por defecto)
    WALK3D = "walk3d"                # caminar en 3D (avance +X)
    

# =============================================================================
# SISTEMA DE RECOMPENSAS PROGRESIVO SIMPLE
# Solo 3 niveles, fácil de entender y modificar
# =============================================================================
    
class SimpleProgressiveReward:
    """
    Sistema súper simple: 3 niveles que van aumentando la dificultad y las recompensas
    
    NIVEL 1: Solo mantenerse de pie (recompensas pequeñas 0-3) (0-15 episodios)
    NIVEL 2: Balance estable (recompensas medias 0-5)  (15-40 episodios)
    NIVEL 3: Levantar piernas (recompensas altas 0-8) (40+ episodios)
    """
    
    def __init__(self, env):
        self.env=env
        self.frequency_simulation = env.frequency_simulation
        self.robot_id = env.robot_id
        self.single_support_ticks = 0
        # === Modo de recompensa seleccionable desde el env (si no existe, progressive) ===
        mode_str = getattr(env, "simple_reward_mode", "progressive")
        try:
            self.mode = RewardMode(mode_str)
        except Exception:
            self.mode = RewardMode.PROGRESSIVE
        # Parametrización útil para modos nuevos
        self.allow_hops = bool(getattr(env, "allow_hops", False))
        
        # Debug para confirmar configuración
        self.reawrd_step=self.env.reawrd_step
        # --- Effort weight scheduler ---
        self.action_previous = None  # para suavidad (du^2)
        # mean reward running, switching mean value and
        self._vx_target=self.env.vx_target
        
        if self.env.logger:
            self.env.logger.log("main",f"🎯 Progressive System initialized:")
            self.env.logger.log("main",f"   Frequency: {self.frequency_simulation} Hz")

    def reset(self):
        self.action_previous = None

    def calculate_reward(self, action):
        """
        Método principal: calcula reward según el nivel actual
        """
        # Decido usar este método para crear varias opciones de creación de recompensas. Else, curriculo clásico
        if getattr(self, "mode", RewardMode.PROGRESSIVE).value == RewardMode.WALK3D.value:
            return self.calculate_reward_walk3d(action)
        else:
            raise Exception ("Solo se acepta caminar ahora")

    
    def is_episode_done(self, step_count):
        """Criterios simples de terminación"""
        self.com_x,self.com_y,self.com_z=self.env.com_x,self.env.com_y,self.env.com_z
        self.zmp_x, self.zmp_y=self.env.zmp_x, self.env.zmp_y
        self.vel_COM=self.env.vel_COM
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        # Penalizo deriva frente y lateral
        self.dx = float(pos[0])
        self.dy = float(pos[1])
        # Caída
        if pos[2] <= self.env.init_com_z/2:
            self.last_done_reason = "fall"
            if self.env.logger:
                self.env.logger.log("main",f"❌ Episode done: Robot fell {pos[2]}")
            return True
        
        if self.dx < -0.6 or abs(self.dy)>1.0:
            self.last_done_reason = "drift"
            if self.env.logger:
                self.env.logger.log("main","❌ Episode done: Excessive longitudinal drift")
            return True
        
        max_tilt = 0.8
        #Inclinación extrema
        if abs(euler[0]) > max_tilt or abs(euler[1]) > max_tilt:
            self.last_done_reason = "tilt"
            if self.env.logger:
                self.env.logger.log("main",f"❌ Episode done: Robot tilted too much {euler[0]}, {euler[1]}, {euler[2]}")
            return True

        # Tiempo máximo (crece con nivel)
        max_steps = 2000
        if step_count >= max_steps:
            self.last_done_reason = "time"
            if self.env.logger:
                self.env.logger.log("main","⏰ Episode done: Max time reached")
            return True
        
        self.last_done_reason = None
        
        return False
    
    # ============================================================================================================================================= #
    # ================================================= Nuevos metodos de recompensa para nuevas acciones ========================================= #
    # ============================================================================================================================================= #

    # ===================== NUEVO: Caminar 3D =====================
    def calculate_reward_walk3d(self, action):
        
        vx = float(self.vel_COM[0])
        vy = float(self.vel_COM[1])
        z_star = getattr(self, "init_com_z", 0.89)
        vcmd = float(getattr(self, "_vx_target",0.6))
        #self.env.torque_max_generation(torque_mapping=torque_mapping)
        w_velocidad=0.8
        w_altura=0.3

        w_lateral=0.2
        w_smooth=0.3
        # Para indicar al modelo que más tiempo igual a más recompensa
        supervivencia=0.8

        #Recompensas de ciclo del pie

        # Recompensa velocidad
        if 0<=vx<vcmd:
            reward_speed= np.exp(-(vx-vcmd)**2)
        elif vx<0:
            reward_speed=0
        else:
            reward_speed = 1
        
        #SI com_z esta fuera de la altura objetivo
        castigo_altura = ((self.com_z-z_star)/0.1)**2
        castigo_posicion = (self.com_y/0.1)**2
        castigo_velocidad_lateral=(vy)**2

        castigo_esfuerzo = self.castigo_effort(action, w_smooth)

        reward= ((supervivencia + w_velocidad*reward_speed)
                  -(w_altura*castigo_altura+ w_lateral*castigo_posicion+ 
                    w_lateral*castigo_velocidad_lateral+ castigo_esfuerzo)) 
                    
        self.reawrd_step['reward_speed']   = w_velocidad*reward_speed
        self.reawrd_step['castigo_altura']  = w_altura*castigo_altura
        self.reawrd_step['castigo_posicion_y'] = w_lateral*castigo_posicion
        self.reawrd_step['castigo_velocidad_y'] =  w_lateral*castigo_velocidad_lateral
        self.reawrd_step['castigo_esfuerzo']  = castigo_esfuerzo
        
        self.action_previous=action
        return float(reward)
    
    def castigo_effort(self,action, w_smooth):
        # Suavidad en presiones (acciones en [0,1])
        accion_previa = self.action_previous if self.action_previous is not None else np.zeros_like(action)
        # Evita que el torque pase de +1 a -1 instantaneamente
        delta_p = np.asarray(action) - np.asarray(accion_previa)
        smooth_efectivo=float(np.mean(delta_p**2))
        # Cuenta cuantos actuadores están activos
        
        return w_smooth*smooth_efectivo 
        
        