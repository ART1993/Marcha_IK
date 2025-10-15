import numpy as np
import pybullet as p


class ZMPCalculator:
    """
        Calculador ZMP SIMPLIFICADO para balance y sentadillas.
        
        OBJETIVO ESPECÍFICO:
        - Calcular ZMP básico usando ecuaciones físicas
        - Determinar estabilidad simple (dentro/fuera de polígono de soporte)
        - Calcular margen de estabilidad básico
        
        ELIMINADO:
        - Historia compleja COM para aceleraciones
        - Cálculos avanzados de recompensa ZMP
        - Múltiples parámetros de inicialización
        - Dimensiones complejas del robot
    """
    
    def __init__(self, robot_id, left_foot_id, 
                 right_foot_id, 
                 frequency_simulation=1500,
                 robot_data=None,
                 ground_id=None,
                 contact_state_fn=None):
        
        # ===== CONFIGURACIÓN BÁSICA =====
        # ----- IDs -----
        self.robot_id = robot_id
        self.left_foot_id = left_foot_id  
        self.right_foot_id = right_foot_id
        self.frequency_simulation=frequency_simulation
        self.dt=1.0/frequency_simulation  # Paso de tiempo
        #robot_id y floor_id
        self.robot_data = robot_data
        self.ground_id=ground_id
        self.contact_state_fn=contact_state_fn

        # ----- Constantes de contacto/ZMP (robustas) -----
        self.MIN_CONTACT_FORCE = 8.0   # N (ignorar micro-contactos)
        self.EPS_FZ = 30.0            # N (suma mínima para ZMP fiable)
        # Geometría aproximada del pie (rectángulo en el frame del link)
        self.FOOT_LENGTH = 0.3       # m (talón→punta)
        self.FOOT_WIDTH  = 0.15       # m (medial↔lateral)
        # Historial de último ZMP válido (para “hold” si ΣFz baja)
        
        # ===== PARÁMETROS FÍSICOS SIMPLIFICADOS =====
        
        self.g = 9.81  # Gravedad
        self.l = 1.0   # Altura aproximada COM (simplificada)
        
        # ===== PARÁMETROS DE ESTABILIDAD =====
        
        self.stability_margin = 0.15  # Margen de seguridad (15cm)
        
        # ===== HISTORIA MÍNIMA PARA ACELERACIÓN =====
        
        self.com_history = []
        self.max_history = 3  # Solo 3 puntos para cálculo básico
        self._zmp_last = None
        self._zmp_lp   = None
        self.ZMP_ALPHA = 0.3  # low-pass (0..1)
        self.STATE_NONE    = 0
        self.STATE_TOUCH   = 1
        self.STATE_PLANTED = 2
    
    def calculate_zmp(self):
        """
            Calcula ZMP:
            1) Con fuerzas de contacto si ΣFz >= EPS_FZ
            2) Fallback LIPM (COM-accel) si no hay soporte; hace 'hold' del último ZMP válido.
            Aplica también un low-pass simple para suavizar ruido.
        """
        # --- 0) Estados de pies (si hay callback)
        # Regla: calculamos ZMP por fuerzas si al menos 1 pie está PLANTED.
        left_state = right_state = None
        if self.contact_state_fn is not None:
            try:
                left_state, _, _  = self.contact_state_fn(self.left_foot_id)
                right_state, _, _ = self.contact_state_fn(self.right_foot_id)
            except Exception:
                left_state = right_state = None

        # --- 1) Contactos por fuerzas (si hay al menos un pie PLANTED, o no hay callback)
        samples = self._collect_contact_samples()
        wsum = sum(Fz for (_, Fz) in samples)
        planted_ok = (
            (left_state is None and right_state is None) or
            (left_state  is not None and left_state  >= self.STATE_PLANTED) or
            (right_state is not None and right_state >= self.STATE_PLANTED)
        )
        if planted_ok and wsum >= self.EPS_FZ:
            zx = sum(px*Fz for ((px, _), Fz) in samples) / wsum
            zy = sum(py*Fz for ((_, py), Fz) in samples) / wsum
            zmp = np.array([zx, zy], dtype=float)
            # Low-pass
            self._zmp_lp = zmp if self._zmp_lp is None else (1-self.ZMP_ALPHA)*self._zmp_lp + self.ZMP_ALPHA*zmp
            self._zmp_last = np.array(self._zmp_lp, dtype=float)
            # Actualiza historia COM igualmente (para debugging/telemetría)
            if self.robot_data:
                com_pos, _ = self.robot_data.get_center_of_mass()
                self.update_com_history(np.array(com_pos))
            return self._zmp_last.copy()

        # --- 2) Fallback LIPM (sin soporte suficiente o sin PLANTED)
        # Obtener posición del centro de masa
        if self.robot_data:
            try:
                com_pos, _ = self.robot_data.get_center_of_mass()
                com_pos = np.array(com_pos)
                l_dynamic = com_pos[2]  # Altura Z actual del COM
            except:
                # Fallback: usar posición de la base
                com_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                com_pos = np.array(com_pos)
                l_dynamic = self.l
        else:
            com_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            com_pos = np.array(com_pos)
        
        # Actualizar historia para cálculo de aceleración
        self.update_com_history(com_pos)
        
        # Calcular aceleración básica
        com_acceleration = self.calculate_simple_acceleration()
        
        # ===== ECUACIONES ZMP SIMPLIFICADAS =====
        
        # X_zmp = X_com - (l/g) * X_com_accel
        # Y_zmp = Y_com - (l/g) * Y_com_accel
        
        zmp_x = com_pos[0] - (l_dynamic / self.g) * com_acceleration[0]
        zmp_y = com_pos[1] - (l_dynamic / self.g) * com_acceleration[1]
        
        zmp = np.array([zmp_x, zmp_y], dtype=float)
        # Mantén último ZMP bueno si existe (hold) cuando NO hay pies PLANTED
        no_planted = (
            (left_state is not None and left_state  < self.STATE_PLANTED) and
            (right_state is not None and right_state < self.STATE_PLANTED)
        ) if (left_state is not None and right_state is not None) else (wsum < self.EPS_FZ)
        if no_planted and (self._zmp_last is not None):
            return self._zmp_last.copy()

        self._zmp_lp = zmp.copy()
        self._zmp_last = zmp.copy()
        return zmp
    
    def update_com_history(self, com_position):
        """Actualizar historia del COM (simplificada)"""
        self.com_history.append(com_position[:3])  # Solo x, y, z
        
        # Mantener solo los últimos puntos necesarios
        if len(self.com_history) > self.max_history:
            self.com_history.pop(0)
    
    def calculate_simple_acceleration(self):
        """
        Calcular aceleración COM usando diferencias finitas SIMPLES.
        
        Si no hay suficiente historia, asumir aceleración cero.
        """
        
        if len(self.com_history) < 3:
            return np.array([0.0, 0.0, 0.0])
        
        # Diferencias finitas de segundo orden: a = (pos[t] - 2*pos[t-1] + pos[t-2]) / dt^2
        pos_current = np.array(self.com_history[-1])
        pos_prev = np.array(self.com_history[-2])
        pos_prev2 = np.array(self.com_history[-3])
        
        acceleration = (pos_current - 2*pos_prev + pos_prev2) / (self.dt**2)
        
        # Limitar valores extremos (filtrado básico)
        acceleration = np.clip(acceleration, -20.0, 20.0)
        
        return acceleration
    
    def get_support_polygon(self):
        """Polígono de soporte (x,y) en MUNDO a partir de la geometría de los pies."""
        poly = self._support_polygon_world()
        return np.asarray(poly, dtype=float) if poly else np.array([])
    
    def is_stable(self, zmp_point=None):
        """
        Verificar si el ZMP está dentro del polígono de soporte.
        
        SIMPLIFICADO: Para balance de pie, solo verificar que ZMP esté entre los pies.
        """
        
        if zmp_point is None:
            zmp_point = self.calculate_zmp()
        
        rect = self._support_polygon_world()
        if not rect:
            return False
        d = self._point_in_rect_margin(tuple(np.asarray(zmp_point)[:2]), rect)
        return d >= 0.0
    
    def stability_margin_distance(self, zmp_point=None):
        """
        Calcular distancia del ZMP al borde del polígono de soporte.
        
        Returns:
            float: Positivo = dentro (estable), Negativo = fuera (inestable)
        """
        
        if zmp_point is None:
            zmp_point = self.calculate_zmp()
        
        rect = self._support_polygon_world()
        if not rect:
            return float(-self.stability_margin)  # sin soporte: inestable
        return float(self._point_in_rect_margin(tuple(np.asarray(zmp_point)[:2]), rect))
    
    def get_stability_info(self):
        """
        Obtener información completa de estabilidad para debugging.
        
        Returns:
            dict: Información de ZMP, estabilidad, margen, etc.
        """
        
        zmp_point = self.calculate_zmp()
        is_stable = self.is_stable(zmp_point)
        margin = self.stability_margin_distance(zmp_point)
        support_polygon = self.get_support_polygon()
        # KPI de cargas
        try:
            sFz, fzL, fzR = self.vertical_loads()
        except Exception:
            sFz = fzL = fzR = 0.0

        info = {
            'zmp_position': zmp_point.tolist(),
            'is_stable': is_stable,
            'stability_margin': margin,
            'support_polygon': support_polygon.tolist(),
            'support_polygon_vertices': len(support_polygon),
            'num_contact_samples': len(self._collect_contact_samples()),
            'sumFz_total': sFz, 'sumFz_left': fzL, 'sumFz_right': fzR,
            'com_history_length': len(self.com_history)
        }

        if self.contact_state_fn is not None:
            try:
                ls, _, _ = self.contact_state_fn(self.left_foot_id)
                rs, _, _ = self.contact_state_fn(self.right_foot_id)
                info['left_state'] = int(ls)
                info['right_state'] = int(rs)
            except Exception:
                pass
        
        return info
    
    def reset(self):
        """Reset del calculador ZMP"""
        self.com_history.clear()
        self._zmp_last = None
        self._zmp_lp   = None
        print(f"🔄 ZMP Calculator reset")

    def get_stability_analysis(self):
        """
        Análisis completo usando tanto COM como ZMP
        """
        # COM para análisis básico
        com_pos, total_mass = self.robot_data.get_center_of_mass() if self.robot_data else ([0,0,1], 25.0)
        
        # ZMP para análisis dinámico  
        zmp_point_info = self.get_stability_info()
        
        # Diferencia entre ZMP y COM (indica actividad dinámica)
        com_zmp_difference = np.linalg.norm(zmp_point_info["zmp_position"] - com_pos[:2])

        info_stability={"zmp_info":zmp_point_info, 
                        "com_info":{'com_position': com_pos,
                                    'static_stable': self.is_stable(com_pos[:2]),  # Basado en COM
                                    'com_height_current': com_pos[2]},
                        'dynamic_activity': com_zmp_difference,  # >0.1 indica movimiento significativo
                        }
        
        return info_stability
    

     # ---------------------------
     #  Helpers de contacto/soporte
     # ---------------------------
    def _foot_has_contact(self, foot_id) -> bool:
        ground = self.ground_id if self.ground_id is not None else 0
        if self.contact_state_fn is not None:
            try:
                state, n, F = self.contact_state_fn(foot_id)
                return (state >= self.STATE_PLANTED) or (F >= self.EPS_FZ)
            except Exception:
                pass
        # Fallback por PyBullet
        cps = p.getContactPoints(self.robot_id, ground, linkIndexA=foot_id, linkIndexB=-1)
        if not cps: 
            return False
        # Cuenta “contactos buenos” con Fz >= MIN_CONTACT_FORCE y también mira ΣFz
        good = [cp for cp in cps if max(0.0, cp[9]) >= self.MIN_CONTACT_FORCE]
        sum_fz = sum(max(0.0, cp[9]) for cp in good)
        return (len(good) >= 2) or (sum_fz >= self.EPS_FZ * 0.5)

    def _collect_contact_samples(self):
        """Devuelve lista [(pos_world(x,y), Fz), ...] filtrada por MIN_CONTACT_FORCE."""
        ground = self.ground_id if self.ground_id is not None else 0
        samples = []
        for foot_id in (self.left_foot_id, self.right_foot_id):
            # Si hay callback y el pie está NONE, ignóralo del todo; si TOUCH, acepta pero será filtrado por Fz.
            if self.contact_state_fn is not None:
                try:
                    state, _, _ = self.contact_state_fn(foot_id)
                    if state == self.STATE_NONE:
                        continue
                except Exception:
                    pass
            cps = p.getContactPoints(self.robot_id, ground, linkIndexA=foot_id, linkIndexB=-1)
            for cp in cps or []:
                Fz = float(max(0.0, cp[9]))      # normalForce
                if Fz >= self.MIN_CONTACT_FORCE:
                    # cp[5] = contact position on link A in world coordinates
                    px, py = float(cp[5][0]), float(cp[5][1])
                    samples.append(((px, py), Fz))
        return samples

    def _foot_rectangle_world(self, foot_id):
        """4 vértices (x,y) del rectángulo del pie en mundo a partir de su pose."""
        L = self.FOOT_LENGTH * 0.5
        W = self.FOOT_WIDTH  * 0.5
        local = [(+L, +W, 0.0), (+L, -W, 0.0), (-L, -W, 0.0), (-L, +W, 0.0)]
        pos, orn = p.getLinkState(self.robot_id, foot_id, computeForwardKinematics=1)[:2]
        verts = []
        for vx, vy, vz in local:
            wx, wy, _ = p.multiplyTransforms(pos, orn, [vx, vy, vz],[0,0,0,1])[0]
            verts.append((float(wx), float(wy)))
        return verts

    def _support_polygon_world(self):
        """
        Polígono de soporte en MUNDO como lista de vértices (x,y).
        Un pie: su rectángulo. Dos pies: unión aproximada (envolvente min/max).
        """
        L_on = self._foot_has_contact(self.left_foot_id)
        R_on = self._foot_has_contact(self.right_foot_id)
        if not (L_on or R_on):
            return []  # sin soporte real

        polys = []
        if L_on: polys += self._foot_rectangle_world(self.left_foot_id)
        if R_on: polys += self._foot_rectangle_world(self.right_foot_id)
        if len(polys) <= 4:
            return polys  # un solo pie
        # Envolvente axis-aligned (rápida y suficiente para margen)
        xs = [v[0] for v in polys]; ys = [v[1] for v in polys]
        rect = [(min(xs), min(ys)), (min(xs), max(ys)),
                (max(xs), max(ys)), (max(xs), min(ys))]
        return rect

    def _point_in_rect_margin(self, pt, rect):
        """Devuelve distancia con signo al borde del rectángulo (≫0 dentro)."""
        if len(rect) < 4:
            # Tratarlo como círculo alrededor del centro si algo raro
            cx = float(np.mean([v[0] for v in rect])) if rect else 0.0
            cy = float(np.mean([v[1] for v in rect])) if rect else 0.0
            r = self.stability_margin
            d = r - np.hypot(pt[0]-cx, pt[1]-cy)
            return d
        xs = [v[0] for v in rect]; ys = [v[1] for v in rect]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # distancia al borde más cercano (positivo si dentro)
        dx = min(pt[0] - min_x, max_x - pt[0])
        dy = min(pt[1] - min_y, max_y - pt[1])
        return min(dx, dy)
    
    # KPIs de carga (debug/telemetría)
    def vertical_loads(self):
        """Devuelve (ΣFz_total, ΣFz_left, ΣFz_right) tras filtrado por MIN_CONTACT_FORCE."""
        ground = self.ground_id if self.ground_id is not None else 0
        def sum_fz(foot):
            cps = p.getContactPoints(self.robot_id, ground, linkIndexA=foot, linkIndexB=-1)
            return sum(max(0.0, cp[9]) for cp in (cps or []) if max(0.0, cp[9]) >= self.MIN_CONTACT_FORCE)
        fzL = sum_fz(self.left_foot_id)
        fzR = sum_fz(self.right_foot_id)
        return (fzL + fzR, fzL, fzR)
    
    def _soft_saturate_to_rect(self, pt):
        """(Opcional) proyecta suavemente el ZMP al rectángulo de soporte cuando reaparece soporte."""
        rect = self._support_polygon_world()
        if not rect:
            return np.asarray(pt, dtype=float)
        xs = [v[0] for v in rect]; ys = [v[1] for v in rect]
        x = float(np.clip(pt[0], min(xs), max(xs)))
        y = float(np.clip(pt[1], min(ys), max(ys)))
        return np.array([x, y], dtype=float)