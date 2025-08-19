#!/usr/bin/env python3
"""
SETUP TESTING ENVIRONMENT

This script creates all the simplified class files needed for testing.
Run this BEFORE running the comprehensive tests.
"""

import os

def create_file_structure():
    """Create the necessary directory structure"""
    
    directories = [
        "Simplified_Project",
        "Simplified_Project/simplified_components",
        "Simplified_Project/tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def save_simplified_classes():
    """Save all the simplified classes we created"""
    
    # Note: The actual content would be copied from the artifacts we created earlier
    files_to_create = [
        {
            'name': 'Simplified_Project/Simplified_BalanceSquat_Trainer.py',
            'description': 'Simplified trainer for balance and squats'
        },
        {
            'name': 'Simplified_Project/Simplified_BalanceSquat_PAMEnv.py', 
            'description': 'Simplified environment for balance and squats'
        },
        {
            'name': 'Simplified_Project/Simplified_BalanceSquat_Controller.py',
            'description': 'Simplified controller for balance and squats'
        },
        {
            'name': 'Simplified_Project/Simplified_BalanceSquat_RewardSystem.py',
            'description': 'Simplified reward system for balance and squats'
        },
        {
            'name': 'Simplified_Project/Simplified_ZMP_Calculator.py',
            'description': 'Simplified ZMP calculator'
        }
    ]
    
    print("📝 FILES TO CREATE MANUALLY:")
    print("="*60)
    print("You need to copy the content from the artifacts we created earlier")
    print("into these files:")
    print()
    
    for file_info in files_to_create:
        print(f"   📄 {file_info['name']}")
        print(f"      {file_info['description']}")
        print()
    
    print("💡 TIP: Copy the content from each artifact in our conversation")
    print("   into the corresponding .py file")

def create_main_test_script():
    """Create the main testing script"""
    
    print("🧪 COPY TEST SCRIPT:")
    print("="*60)
    print("Copy the 'test_simplified_balance_squat_system.py' content")
    print("to: Simplified_Project/run_tests.py")

def create_quick_start_script():
    """Create a quick start script"""
    
    quick_start_content = '''#!/usr/bin/env python3
"""
QUICK START SCRIPT for Simplified Balance & Squat System

This script helps you get started quickly with the simplified system.
"""

import sys
import os

# Add paths
sys.path.append('.')
sys.path.append('./simplified_components')

def main():
    print("🎯 SIMPLIFIED BALANCE & SQUAT SYSTEM - QUICK START")
    print("="*60)
    
    print("\\n📋 Available options:")
    print("1. Run component tests")
    print("2. Run integration tests") 
    print("3. Run full system test")
    print("4. Start mini training")
    print("5. Test balance behavior")
    print("6. Test squat behavior")
    
    choice = input("\\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        print("🧪 Running component tests...")
        from run_tests import test_components_only
        test_components_only()
        
    elif choice == "2":
        print("🔗 Running integration tests...")
        from run_tests import test_integration_only
        test_integration_only()
        
    elif choice == "3":
        print("🚀 Running full system test...")
        from run_tests import test_system_only
        test_system_only()
        
    elif choice == "4":
        print("🏋️ Starting mini training...")
        try:
            from Simplified_BalanceSquat_Trainer import train_balance_and_squats
            trainer, model = train_balance_and_squats(
                total_timesteps=5000,
                n_envs=1,
                resume=False
            )
            print("✅ Mini training completed!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            
    elif choice == "5":
        print("⚖️ Testing balance behavior...")
        test_balance_only()
        
    elif choice == "6":
        print("🏋️ Testing squat behavior...")
        test_squat_only()
    
    else:
        print("❌ Invalid choice")

def test_balance_only():
    """Test only balance behavior"""
    try:
        from Simplified_BalanceSquat_PAMEnv import create_simple_balance_squat_env
        from Simplified_BalanceSquat_Controller import create_balance_squat_controller, ActionType
        
        env = create_simple_balance_squat_env(render_mode='human')  # Visual
        controller = create_balance_squat_controller(env)
        controller.set_action(ActionType.BALANCE_STANDING)
        
        obs, info = env.reset()
        print("🎯 Balance test running... (Press Ctrl+C to stop)")
        
        for step in range(1000):
            action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(action)
            
            if step % 100 == 0:
                print(f"   Step {step}: Reward = {reward:.2f}")
            
            if done:
                print(f"   Episode ended at step {step}")
                break
        
        env.close()
        print("✅ Balance test completed")
        
    except Exception as e:
        print(f"❌ Balance test failed: {e}")

def test_squat_only():
    """Test only squat behavior"""
    try:
        from Simplified_BalanceSquat_PAMEnv import create_simple_balance_squat_env
        from Simplified_BalanceSquat_Controller import create_balance_squat_controller, ActionType
        
        env = create_simple_balance_squat_env(render_mode='human')  # Visual
        controller = create_balance_squat_controller(env)
        controller.set_action(ActionType.SQUAT)
        
        obs, info = env.reset()
        print("🏋️ Squat test running... (Press Ctrl+C to stop)")
        
        for step in range(2000):  # Longer for full squat cycles
            action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(action)
            
            if step % 200 == 0:
                controller_info = controller.get_current_action_info()
                print(f"   Step {step}: Reward = {reward:.2f}, Phase = {controller_info['current_phase']}")
            
            if done:
                print(f"   Episode ended at step {step}")
                break
        
        env.close()
        print("✅ Squat test completed")
        
    except Exception as e:
        print(f"❌ Squat test failed: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("Simplified_Project/quick_start.py", "w") as f:
        f.write(quick_start_content)
    
    print("📄 Created: Simplified_Project/quick_start.py")

def main():
    print("🎯 SETTING UP TESTING ENVIRONMENT")
    print("="*60)
    print("This script will help you prepare for comprehensive testing")
    print()
    
    create_file_structure()
    print("Fin create file structure")
    save_simplified_classes()
    print("Fin save fimplified classes")
    create_main_test_script()
    print("Fin create test script")
    create_quick_start_script()
    
    print("\\n✅ SETUP INSTRUCTIONS COMPLETE!")
    print("="*60)
    print("📋 NEXT STEPS:")
    print("1. Copy all artifact content to the files listed above")
    print("2. Copy test_simplified_balance_squat_system.py to run_tests.py")
    print("3. Run: python quick_start.py")
    print("4. Or run: python run_tests.py")
    print()
    print("💡 TIP: Start with component tests (option 1) before full system")

if __name__ == "__main__":
    main()
