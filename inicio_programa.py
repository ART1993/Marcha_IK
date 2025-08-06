import multiprocessing as mp
import os
from biped_pam_IK_train import UnifiedBipedTrainer

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




def _setup_multiprocessing():
    # Prepara el multiprocesado para el caso de n_env>1
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    elif mp.get_start_method() != 'spawn':
        print("⚠️ Multiprocessing already initialized with different method")





# Main execution
if __name__ == "__main__":
    # Choose training type
    training_type = 'pam'  # Change to 'standard' for standard biped
    
    if training_type == 'pam':
        train_pam_biped()
    else:
        test_pam_biped()