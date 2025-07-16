import os
from stable_baselines3.common.vec_env import VecNormalize


def cargar_posible_normalizacion(model_dir, resume_path, config, train_env):
        """Load normalization statistics if they exist"""
        if resume_path and isinstance(train_env, VecNormalize):
            norm_path = os.path.join(model_dir, f"{config['model_prefix']}_normalize.pkl")
            if os.path.exists(norm_path):
                print(f"📊 Loading normalization statistics from: {norm_path}")
                try:
                    # Load normalization statistics
                    train_env = VecNormalize.load(norm_path, train_env)
                    # Keep normalization training active
                    train_env.training = True
                    train_env.norm_reward = True
                except Exception as e:
                    print(f"⚠️ Warning: Could not load normalization stats: {e}")
                    print("   Continuing with fresh normalization...")
        return train_env

def set_env_phase(env_wrapper, phase, phase_timesteps):
    if hasattr(env_wrapper, 'envs'):
        for env in env_wrapper.envs:
            base_env = env.env if hasattr(env, 'env') else env
            if hasattr(base_env, 'set_training_phase'):
                base_env.set_training_phase(phase, phase_timesteps)
    else:
        if hasattr(env_wrapper, 'set_training_phase'):
            env_wrapper.set_training_phase(phase, phase_timesteps)

def phase_trainig_preparations(model_dir, remaining_timesteps, train_env, eval_env, current_timesteps,
                                model, callbacks, phase_timesteps, config, num_phase:int):
    # Configurar entornos para usar ciclo base
    set_env_phase(train_env, num_phase, phase_timesteps)
    set_env_phase(eval_env, num_phase, phase_timesteps)
    if num_phase > 1:
        reset_timestep=False
        model.learn(
            total_timesteps=phase_timesteps,
            callback=callbacks,
            tb_log_name=f"{config['model_prefix']}_phase{num_phase}",
            reset_num_timesteps=reset_timestep
        )
    else:
        model.learn(
            total_timesteps=phase_timesteps,
            callback=callbacks,
            tb_log_name=f"{config['model_prefix']}_phase{num_phase}"
        )
    current_timesteps += phase_timesteps

    # Guardar modelo de fase i
    phase_path = os.path.join(model_dir, f"{config['model_prefix']}_phase1")
    model.save(phase_path)
    print(f"✅ Phase {num_phase} model saved at: {phase_path}")

    return model, current_timesteps, phase_timesteps



def swing_pierna_anticuado_SimpleWalkingCycle(self, alpha, right_start, foot_bezier_parabola,
                                                          left_start, step_length, step_height):
        import pybullet as p
        """
        Método para calcular las posiciones articulares deseadas en un ciclo de marcha simple.
        Requiere índices de links de pies izquierdo y derecho.
        Returns: array de posiciones articulares
        """
        if alpha < 0.5:
            print(alpha, "right foot in swing")
            # Pierna derecha en swing
            swing_alpha = alpha / 0.5
            start=right_start
            end = [right_start[0] + step_length, right_start[1], right_start[2]]
            ctrl1 = [start[0] + 0.1, start[1], start[2] + step_height]
            ctrl2 = [end[0] - 0.1, end[1], end[2] + step_height]
            target_pos = foot_bezier_parabola(
                start=start,end=end,
                ctrl1=ctrl1,ctrl2=ctrl2,
                alpha=swing_alpha,height=step_height
            )
            joint_positions = p.calculateInverseKinematics(self.robot_id, self.right_foot_index, target_pos)
        else:
            print(alpha, "left foot in swing")
            # Pierna izquierda en swing
            swing_alpha = (alpha - 0.5) / 0.5
            start=left_start
            end = [left_start[0] + step_length, left_start[1], left_start[2]]
            ctrl1 = [start[0] + 0.1, start[1], start[2] + step_height]
            ctrl2 = [end[0] - 0.1, end[1], end[2] + step_height]
            target_pos = foot_bezier_parabola(
                start=start,end=end,
                ctrl1=ctrl1,ctrl2=ctrl2,
                alpha=swing_alpha,height=step_height
            )
            joint_positions = p.calculateInverseKinematics(self.robot_id, self.left_foot_index, target_pos)

def obtener_posicion_inicial_robot(self, robot_id_temp, urdf_path, joint_positions, 
                                       left_foot_id, right_foot_id, random_friction=1.0):
    import pybullet as p
    # 2. Poner posiciones iniciales
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id_temp, i, pos)
    p.stepSimulation()
    
    # 3. Obtener altura del pie más bajo
    left_foot_z = p.getLinkState(robot_id_temp, left_foot_id)[0][2]
    right_foot_z = p.getLinkState(robot_id_temp, right_foot_id)[0][2]
    min_foot_z = min(left_foot_z, right_foot_z)
    
    # 5. Volver a cargar robot con la base ajustada
    p.resetSimulation()
    adjusted_base_z = self.previous_position[2] - min_foot_z
    p.removeBody(robot_id_temp)
    self.plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(self.plane_id, -1, lateralFriction=random_friction)
    robot_id = p.loadURDF(urdf_path, [0, 0, adjusted_base_z], useFixedBase=False)
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    p.stepSimulation()
    return robot_id