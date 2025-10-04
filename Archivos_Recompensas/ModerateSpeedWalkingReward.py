# === NUEVA RECOMPENSA: CAMINAR A VELOCIDAD MODERADA ===========================
import numpy as np
import pybullet as p

class ModerateSpeedWalkingReward:
    """
    Objetivo: caminar estable a una velocidad objetivo v* (por defecto 0.30 m/s),
    con clearance en el pie en swing, poca zarpada (toe-drag), y ZMP dentro.
    
    Currículum muy simple:
      L1: mantenerse de pie y moverse un poco hacia +x, tolerante (sin exigir single-support).
      L2: empezar pasos: recompensa por clearance del swing y por velocidad hacia +x.
      L3: marcha continua: tracking estricto de velocidad con shaping por single-support.
    """
    def __init__(self, env):
        self.env = env
        self.target_speed = self.env.target_speed
        self.enable_curriculum = self.env.enable_curriculum
        self.level = 1 if self.enable_curriculum else 3

        # Parámetros comunes
        self.min_F = 30.0            # umbral de “contacto con carga”
        self.clearance_target = 0.08 # 8 cm
        self.speed_tol_L1 = 0.30     # tolerancia amplia al inicio
        self.speed_tol_L2 = 0.20
        self.speed_tol_L3 = 0.12

        # Puertas suaves (no un “todo o nada”)
        self.support_ratio_gate_L1 = 0.45
        self.support_ratio_gate_L2 = 0.55
        self.support_ratio_gate_L3 = 0.65

        # Límites de inclinación (roll/pitch) por nivel
        self.max_tilt = {
            1: 0.8,   # muy permisivo
            2: 0.6,
            3: 0.5,
        }

        # Contadores simples
        self.episodes = 0
        self.success_streak = 0

    # ------------- API paralela a SimpleProgressiveReward ---------------------
    def calculate_reward(self, action, step_count):
        pos, orn = p.getBasePositionAndOrientation(self.env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, _ = p.getBaseVelocity(self.env.robot_id)
        vx = float(lin_vel[0])

        # Estabilidad básica (altura y tilt)
        base = self._posture_term(pos, euler)

        # Tracking de velocidad hacia +x
        track = self._speed_tracking_term(vx)

        # Contactos y “single-support” (pero en puertas suaves)
        shape = self._gait_shaping_term()

        # ZMP si está disponible (usa self.zmp_calculator o self.zmp)
        zmp_term = self._zmp_term()

        # Suavidad de acciones/pares si está disponible
        smooth = self._smoothness_term()

        # Penaliza deriva lateral y retrocesos marcados
        drift = self._drift_term(pos)

        return base + track + shape + zmp_term + smooth + drift

    def calculate_reward_2d_walking(self, action, step_count):
        # Si usas el modo 2D, mantenemos la misma función para integrarte con el env
        return self.calculate_reward(action, step_count)

    def is_episode_done(self, step_count):
        # Termina si cae mucho o si pasa demasiado tiempo (igual mentalidad que tu sistema actual)
        pos, orn = p.getBasePositionAndOrientation(self.env.robot_id)
        euler = p.getEulerFromQuaternion(orn)
        if pos[2] < 0.7:
            self.last_done_reason = "fall"
            return True
        if abs(euler[0]) > self.max_tilt[self.level] or abs(euler[1]) > self.max_tilt[self.level]:
            self.last_done_reason = "tilt"
            return True
        # duración máxima de episodio (p. ej. 10 s)
        max_steps = int(10.0 * self.env.frequency_simulation)
        if step_count >= max_steps:
            self.last_done_reason = "time"
            return True
        return False

    def get_info(self):
        return {
            "level": self.level,
            "episodes": self.episodes,
            "target_speed": self.target_speed
        }

    # ----------------------------- helpers ------------------------------------
    def _posture_term(self, pos, euler):
        # Altura
        h = pos[2]
        if h > 0.9:
            h_r = 0.8
        elif h > 0.8:
            h_r = 0.5
        else:
            h_r = -1.5

        # Tilt suave
        roll, pitch = abs(euler[0]), abs(euler[1])
        tilt = roll + pitch
        tilt_pen = - np.clip(tilt - 0.10, 0.0, self.max_tilt[self.level]) * 1.0
        return h_r + tilt_pen

    def _speed_tracking_term(self, vx):
        # Tolerancia depende del nivel
        tol = {1: self.speed_tol_L1, 2: self.speed_tol_L2, 3: self.speed_tol_L3}[self.level]

        # Recompensa cónica centrada en target_speed (máx ≈ +2.0)
        err = abs(vx - self.target_speed)
        track = 2.0 * max(0.0, 1.0 - (err / tol))

        # Penaliza ir hacia atrás de forma notable
        back_pen = -1.0 if vx < -0.05 else 0.0
        return track + back_pen

    def _gait_shaping_term(self):
        # Fuerzas de contacto en ambos pies
        # Nota: en tu env ya existen helpers para contar fuerza/contactos
        #       (p.ej., contact_normal_force):contentReference[oaicite:0]{index=0}
        if self.env.plane_mode2D:
            # índices 2 y 5 son tobillos en 2D; en 3D usa left/right_anckle_link_id
            left_a = self.env.dict_joints["left_anckle_joint"]
            right_a = self.env.dict_joints["right_anckle_joint"]
        else:
            left_a = self.env.left_foot_link_id
            right_a = self.env.right_foot_link_id

        nL, FL = self.env.contact_normal_force(left_a)
        nR, FR = self.env.contact_normal_force(right_a)
        Fsum = max(FL + FR, 1e-9)
        ratio_sup = max(FL, FR) / Fsum if Fsum > 0 else 0.0
        swing_force = min(FL, FR)

        # Puerta suave de single-support según nivel
        gate = {1: self.support_ratio_gate_L1, 2: self.support_ratio_gate_L2, 3: self.support_ratio_gate_L3}[self.level]
        ss_bonus = np.clip((ratio_sup - gate) / (1.0 - gate), 0.0, 1.0) * 0.8

        # Toe-drag (penaliza si el pie “ligero” tiene fuerza 0–min_F)
        toe_drag = -0.6 if (0.0 < swing_force < self.min_F) else 0.0

        # Clearance del pie más “ligero” (swing) si está realmente al aire
        # Elegimos el que tenga menor fuerza
        swing_id = left_a if FL < FR else right_a
        swing_down = (min(FL, FR) >= self.min_F)
        clearance = 0.0
        if not swing_down:
            foot_z = p.getLinkState(self.env.robot_id, swing_id)[0][2]
            clearance = 0.6 * np.clip(foot_z / self.clearance_target, 0.0, 1.0)

        # Bonus por pie de SOPORTE plano (planta paralela al suelo)
        flat = 0.0
        stance_id = right_a if FR >= FL else left_a
        if max(FL, FR) >= self.min_F:
            flat = self._foot_flat_reward(stance_id, only_if_contact=True)

        return ss_bonus + clearance + flat + toe_drag

    def _zmp_term(self):
        try:
            zmp_obj = getattr(self.env, "zmp_calculator", None) or getattr(self.env, "zmp", None)
            if zmp_obj and hasattr(zmp_obj, "stability_margin_distance"):
                margin = float(zmp_obj.stability_margin_distance())  # + dentro, − fuera:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}
                return 0.5 * np.clip(margin / 0.05, -1.0, 1.0)
        except Exception:
            pass
        return 0.0

    def _smoothness_term(self):
        # Igual que tu shaping actual: penaliza vibraciones fuertes de presiones/pares (si disponibles)
        pen = 0.0
        try:
            ps_prev = getattr(self.env, "pam_states_prev", {}).get("pressures", None)
            ps_curr = getattr(self.env, "pam_states", {}).get("pressures", None)
            if ps_prev is not None and ps_curr is not None:
                d = np.linalg.norm(np.array(ps_curr) - np.array(ps_prev), ord=1)
                pen -= 0.05 * d
        except Exception:
            pass
        return pen

    def _drift_term(self, pos):
        # Penaliza deriva lateral grande |y| y zig-zag; premio leve por avanzar en +x
        x, y = pos[0], pos[1]
        lateral = - np.clip(abs(y) - 0.05, 0.0, 0.30) * 2.0
        forward_hint = +0.1 * np.clip(x, 0.0, 1.0)  # premio muy leve por avance global
        return lateral + forward_hint

    # ---------------------------- utilidades ----------------------------------
    def _foot_flat_reward(self, foot_link_id, only_if_contact=True):
        """
        Bonus pequeño si el pie de soporte está “plano”.
        Implementación simple: el vector z del pie aproximadamente paralelo a +z mundo.
        """
        # Estado del link: devuelve orientación
        st = p.getLinkState(self.env.robot_id, foot_link_id, computeForwardKinematics=True)
        orn = st[1]
        # Matriz de rotación → eje z del pie en mundo
        R = p.getMatrixFromQuaternion(orn)
        # z_local en mundo = (R[2], R[5], R[8])
        z_world = np.array([R[2], R[5], R[8]])
        # Queremos z_world ≈ +z (0,0,1): usa componente |z|
        flatness = abs(z_world[2])
        bonus = 0.3 * np.clip(flatness, 0.0, 1.0)
        if only_if_contact:
            cps = p.getContactPoints(self.env.robot_id, self.env.plane_id, linkIndexA=foot_link_id, linkIndexB=-1)
            if not cps:
                return 0.0
        return float(bonus)