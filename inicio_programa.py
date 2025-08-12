import multiprocessing as mp
import os
from collections import deque
import numpy as np
import pybullet as p
from Gymnasium_Start.biped_pam_IK_train import UnifiedBipedTrainer
from Gymnasium_Start.Enhanced_PAMIKBipedEnv import Enhanced_PAMIKBipedEnv, test_complete_integration
from Gymnasium_Start.Enhanced_UnifiedBipedTrainer import Enhanced_UnifiedBipedTrainer

def train_pam_biped(total_timesteps=3000000, n_envs=4, learning_rate=3e-4, resume=True, action_space="hybrid"):
    """Inicio de Entrenamiento PAM + IK."""
    _setup_multiprocessing()
    
    trainer = UnifiedBipedTrainer(
        env_type='pam',
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=learning_rate,
        action_space=action_space,
        seleccion="simple"
    )
    
    print("🚀 Starting PAM biped training...")
    trainer.train(resume=resume)
    return trainer

def enhanced_train_pam_biped(total_timesteps=3000000, n_envs=4, learning_rate=3e-4, 
                             resume=True, action_space="pam"):
    """Inicio de Entrenamiento PAM + IK."""
    _setup_multiprocessing()
    
    trainer = UnifiedBipedTrainer(
        env_type='pam',
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        learning_rate=learning_rate,
        action_space=action_space,
        seleccion="enhanced"
    )
    
    print("🚀 Starting PAM biped training...")
    trainer.train(resume=resume)
    return trainer


def test_pam_biped(model_path=None, episodes=10, action_space="hybrid"):
    """Test trained PAM biped model."""
    trainer = UnifiedBipedTrainer(env_type='pam', action_space=action_space)
    
    if model_path is None:
        # Find latest model
        final_model = os.path.join(trainer.model_dir, "biped_pam_final.zip")
        if os.path.exists(final_model):
            model_path = final_model
        else:
            # Try to find latest checkpoint
            latest_checkpoint, _ = trainer.find_latest_checkpoint()
            if latest_checkpoint:
                model_path = latest_checkpoint
            else:
                print("❌ No trained model found!")
                return
    
    # Check for normalization file
    norm_path = os.path.join(trainer.model_dir, "biped_pam_normalize.pkl")
    if not os.path.exists(norm_path):
        norm_path = None
    
    trainer.test_model(model_path, episodes=episodes, normalization_path=norm_path)


def start_training():
    """Función optimizada para iniciar entrenamiento"""
    
    # Crear trainer con configuración óptima
    trainer = Enhanced_UnifiedBipedTrainer(
        env_type='enhanced_pam',
        system_version='enhanced', 
        total_timesteps=5_000_000,
        n_envs=6,
        learning_rate=3e-4,
        enable_expert_curriculum=True
    )
    
    # Entrenar con resume automático
    trainer.train(resume=True)
    
    return trainer




def _setup_multiprocessing():
    # Prepara el multiprocesado para el caso de n_env>1
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        print("⚠️ Multiprocessing already initialized with different method")


# ===================================================================================================================================================== #
# =================================================Testeo de la calidad de PAMBipedENV================================================================= #
# ===================================================================================================================================================== #

def test_enhanced_6pam_system():
    """Script de prueba para verificar el sistema de 6 PAMs"""
    
    print("🔧 Testing Enhanced PAM System (6 actuators)")
    
    env = Enhanced_PAMIKBipedEnv(render_mode='human', action_space="pam")
    
    obs, info = env.reset()
    print(f"✅ Environment created successfully")
    print(f"   - Action space: {env.action_space.shape}")
    print(f"   - Observation space: {env.observation_space.shape}")
    print(f"   - Active PAMs: {env.num_active_pams}")
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"   Step {step}: Reward = {reward:.2f}")
            if 'pam_pressures' in info:
                print(f"      PAM pressures = {info['pam_pressures']}")
        
        if done:
            print(f"   Episode ended at step {step}")
            obs, info = env.reset()
    
    env.close()
    print("🎉 Test completed successfully!")

def test_complete_integration_inicio():
    # Ejecutar test de integración si se ejecuta directamente
    success = test_complete_integration()
    
    if success:
        print("\n🚀 Ready to begin training with Enhanced PAM system!")
    else:
        print("\n🔧 Please fix integration issues before proceeding.")


    # ===== PROPIEDADES Y MÉTODOS DE CONFIGURACIÓN =====
    
    def variables_seguimiento(self):
        """Variables para seguimiento de rendimiento"""
        self.step_count = 0
        self.total_reward = 0
        self.previous_position = None
        self.previous_velocity = None
        
        self.history_length = 5
        self.observation_history = deque(maxlen=self.history_length)
        
        self.previous_contacts = [False, False]
        self.previous_action = None
        
        self.robot_id = None
        self.plane_id = None
        
        self.zmp_calculator = None
        self.zmp_history = []
        self.max_zmp_history = 20
        
        self.zmp_reward_weight = 0.2
        self.stability_bonus = 20.0
        self.instability_penalty = -15.0
        
        self.walking_controller = None
        self.use_walking_cycle = True

    @property
    def setup_reset_simulation(self):
        """Configuración inicial del robot y simulación"""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSubSteps=4)
        
        # Cargar entorno
        random_friction = np.random.uniform(0.8, 1.1)
        correction_quaternion = p.getQuaternionFromEuler([0, 0, 0])
        self.previous_position = [0, 0, 1.2]
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=random_friction)
        self.robot_id = p.loadURDF(
            self.urdf_path,
            self.previous_position,
            correction_quaternion,
            useFixedBase=False
        )
        self.setup_physics_properties()
        
        # Estabilizar
        for _ in range(20):
            p.stepSimulation()
        
        self.repeat = 0
        
        # Randomización de masa
        for link_id in range(p.getNumJoints(self.robot_id)):
            orig_mass = p.getDynamicsInfo(self.robot_id, link_id)[0]
            rand_mass = orig_mass * np.random.uniform(0.8, 1.1)
            p.changeDynamics(self.robot_id, link_id, mass=rand_mass)

    def setup_physics_properties(self):
        """Configurar propiedades físicas"""
        p.changeDynamics(self.robot_id, self.left_foot_id,
                        lateralFriction=1.0, restitution=0.0)
        p.changeDynamics(self.robot_id, self.right_foot_id,
                        lateralFriction=1.0, restitution=0.0)
        
    def set_walking_cycle(self, enabled=True):
        """Activar/desactivar el ciclo de paso"""
        self.use_walking_cycle = enabled
        if not enabled:
            self.walking_controller = None

    def _apply_joint_forces(self, forces):
        """Aplica fuerzas tensoriales a las articulaciones"""
        active_joints = [0, 1, 3, 4]  # caderas y rodillas
        for i, force in enumerate(forces):
            if i < len(active_joints):
                p.setJointMotorControl2(
                    self.robot_id,
                    active_joints[i],
                    p.TORQUE_CONTROL,
                    force=force
                )

    def _apply_ankle_spring(self, k_spring=8.0):
        """Aplica torques de resorte pasivo en los tobillos"""
        for ankle_joint in [2, 5]:  # left_ankle=2, right_ankle=5
            theta = p.getJointState(self.robot_id, ankle_joint)[0]
            torque = -k_spring * theta
            p.setJointMotorControl2(
                self.robot_id,
                ankle_joint,
                controlMode=p.TORQUE_CONTROL,
                force=torque
            )








# Main execution
#if __name__ == "__main__":
    # Choose training type
#    training_type = 'pam'  # Change to 'standard' for standard biped
    
#    if training_type == 'pam':
#        train_pam_biped()
#    else:
#        test_pam_biped()