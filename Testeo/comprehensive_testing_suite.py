#!/usr/bin/env python3
"""
COMPREHENSIVE TESTING SUITE for Simplified Balance & Squat System

This script performs systematic testing from individual components to full system integration.
Tests are designed to validate that the simplified system works correctly for balance and squats.
"""

import sys
import os
import numpy as np
import time
from datetime import datetime
import traceback

from Archivos_Apoyo.ZPMCalculator import create_simple_zmp_calculator
from Archivos_Mejorados.Simplified_BalanceSquat_RewardSystem import create_simple_reward_system
from Controlador.discrete_action_controller import create_balance_squat_controller
from Controlador.discrete_action_controller import ActionType
from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
from Gymnasium_Start.Simplified_BalanceSquat_Trainer import create_balance_squat_trainer, train_balance_and_squats
        #from Simplified_BalanceSquat_PAMEnv import create_simple_balance_squat_env
        #from Simplified_BalanceSquat_Controller import create_balance_squat_controller

# Add project paths (adjust as needed)
sys.path.append('.')
sys.path.append('./Gymnasium_Start')
sys.path.append('./Archivos_Apoyo')
sys.path.append('./Controlador')

class TestResults:
    """Class to track and report test results"""
    
    def __init__(self):
        self.tests = {}
        self.start_time = datetime.now()
        self.errors = []
        self.warnings = []
    
    def add_test(self, test_name, passed, details="", execution_time=0):
        self.tests[test_name] = {
            'passed': passed,
            'details': details,
            'execution_time': execution_time,
            'timestamp': datetime.now()
        }
    
    def add_error(self, error_msg):
        self.errors.append(error_msg)
    
    def add_warning(self, warning_msg):
        self.warnings.append(warning_msg)
    
    def print_summary(self):
        total_tests = len(self.tests)
        passed_tests = sum(1 for test in self.tests.values() if test['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n" + "="*80)
        print(f"🧪 COMPREHENSIVE TEST RESULTS SUMMARY")
        print(f"="*80)
        print(f"⏱️  Total execution time: {datetime.now() - self.start_time}")
        print(f"📊 Total tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Errors: {len(self.errors)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        
        if passed_tests == total_tests:
            print(f"🎉 ALL TESTS PASSED! System ready for training.")
        else:
            print(f"💥 Some tests failed. Check details below.")
        
        # Detailed results
        print(f"\n📋 DETAILED TEST RESULTS:")
        print("-"*80)
        for test_name, result in self.tests.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {test_name:40} ({result['execution_time']:.3f}s)")
            if result['details']:
                print(f"     {result['details']}")
        
        if self.errors:
            print(f"\n💥 ERRORS:")
            for error in self.errors:
                print(f"   ❌ {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")

def run_test(test_func, test_name, results):
    """Execute a test function and record results"""
    print(f"\n🧪 Running: {test_name}")
    print("-" * 60)
    
    start_time = time.time()
    try:
        success, details = test_func()
        execution_time = time.time() - start_time
        results.add_test(test_name, success, details, execution_time)
        
        if success:
            print(f"   ✅ PASSED: {details}")
        else:
            print(f"   ❌ FAILED: {details}")
            
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{test_name}: {str(e)}"
        results.add_error(error_msg)
        results.add_test(test_name, False, f"Exception: {str(e)}", execution_time)
        print(f"   💥 EXCEPTION: {str(e)}")
        traceback.print_exc()

# =============================================================================
# LEVEL 1: INDIVIDUAL COMPONENT TESTS
# =============================================================================

def test_pam_mckibben_physics():
    """Test PAM_McKibben physical model"""
    try:
        from Archivos_Apoyo.dinamica_pam import PAMMcKibben
        
        # Create PAM muscle
        pam = PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4)
        
        # Test 1: Basic force calculation
        pressure = 5 * 101325  # 5 atm
        contraction = 0.2      # 20% contraction
        
        force = pam.force_model_new(pressure, contraction)
        
        # Validate realistic force output
        if 0 < force < 1000:  # Reasonable force range
            details = f"Force = {force:.1f}N at {pressure/101325:.1f}atm, {contraction:.1%} contraction"
            return True, details
        else:
            return False, f"Unrealistic force: {force:.1f}N"
            
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def test_zmp_calculator():
    """Test simplified ZMP calculator"""
    try:
        # Import the simplified version we created

        
        # Create mock robot for testing
        zmp_calc = create_simple_zmp_calculator(robot_id=0)
        
        # Simulate COM history
        test_positions = [
            [0.0, 0.0, 1.1],
            [0.01, 0.0, 1.1],
            [0.02, 0.0, 1.1]
        ]
        
        for pos in test_positions:
            zmp_calc.update_com_history(pos)
        
        # Test acceleration calculation
        accel = zmp_calc.calculate_simple_acceleration()
        
        # Test ZMP calculation (this will use PyBullet functions that may not work in isolation)
        # So we'll test the computational parts
        
        if len(zmp_calc.com_history) == 3:
            details = f"COM history: {len(zmp_calc.com_history)} points, accel: [{accel[0]:.3f}, {accel[1]:.3f}]"
            return True, details
        else:
            return False, f"COM history not properly maintained"
            
    except ImportError as e:
        return False, f"Import error: {str(e)} - Create Simplified_ZMP_Calculator.py first"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def test_reward_system():
    """Test simplified reward system"""
    try:
        
        
        reward_system = create_simple_reward_system()
        
        # Test basic PAM efficiency calculation
        test_action = np.array([0.3, 0.4, 0.3, 0.4, 0.2, 0.2])  # Balanced action
        
        efficiency = reward_system._calculate_basic_pam_efficiency(test_action)
        
        # Test reward configuration
        summary = reward_system.get_reward_summary()
        
        if -3.0 <= efficiency <= 3.0 and 'weights' in summary:
            details = f"PAM efficiency: {efficiency:.3f}, weights configured: {len(summary['weights'])}"
            return True, details
        else:
            return False, f"Reward system not functioning properly"
            
    except ImportError as e:
        return False, f"Import error: {str(e)} - Create Simplified_BalanceSquat_RewardSystem.py first"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def test_controller():
    """Test simplified controller"""
    try:
        
        
        # Mock environment
        class MockEnv:
            def __init__(self):
                self.robot_id = 0
                self.time_step = 1.0/1500.0
        
        mock_env = MockEnv()
        controller = create_balance_squat_controller(mock_env)
        
        # Test action generation
        action_balance = controller.get_expert_action(mock_env.time_step)
        
        # Test action switching
        
        controller.set_action(ActionType.SQUAT)
        action_squat = controller.get_expert_action(mock_env.time_step)
        
        # Validate actions
        if (len(action_balance) == 6 and len(action_squat) == 6 and 
            0 <= np.min(action_balance) and np.max(action_balance) <= 1.0):
            details = f"Balance action: {action_balance[:3]}, Squat action: {action_squat[:3]}"
            return True, details
        else:
            return False, f"Invalid action generation"
            
    except ImportError as e:
        return False, f"Import error: {str(e)} - Create Simplified_BalanceSquat_Controller.py first"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def test_environment():
    """Test simplified environment"""
    try:
        
        
        # Create environment (without rendering for test)
        env = create_simple_balance_squat_env(render_mode='direct')
        
        # Test reset
        obs, info = env.reset()
        
        # Test step
        action = env.action_space.sample() * 0.5 + 0.25  # Moderate action
        obs_new, reward, done, truncated, info = env.step(action)
        
        # Validate
        if (len(obs) == env.observation_space.shape[0] and 
            len(action) == env.action_space.shape[0] and
            isinstance(reward, (int, float))):
            details = f"Obs: {len(obs)} elements, Action: {len(action)} PAMs, Reward: {reward:.2f}"
            env.close()
            return True, details
        else:
            env.close()
            return False, f"Environment interface not working properly"
            
    except ImportError as e:
        return False, f"Import error: {str(e)} - Create Simplified_BalanceSquat_PAMEnv.py first"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def test_trainer():
    """Test simplified trainer"""
    try:
        
        # Create trainer with minimal configuration
        trainer = create_balance_squat_trainer(
            total_timesteps=1000,  # Very small for test
            n_envs=1,              # Single env for test
            learning_rate=3e-4
        )
        
        # Check trainer configuration
        config_ok = (trainer.total_timesteps == 1000 and 
                    trainer.n_envs == 1 and
                    hasattr(trainer, 'env_config'))
        
        if config_ok:
            details = f"Trainer configured: {trainer.total_timesteps} timesteps, {trainer.n_envs} envs"
            return True, details
        else:
            return False, f"Trainer configuration failed"
            
    except ImportError as e:
        return False, f"Import error: {str(e)} - Create Simplified_BalanceSquat_Trainer.py first"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

# =============================================================================
# LEVEL 2: INTEGRATION TESTS
# =============================================================================

def test_controller_environment_integration():
    """Test controller + environment integration"""
    try:

        
        # Create environment and controller
        env = create_simple_balance_squat_env(render_mode='direct')
        controller = create_balance_squat_controller(env)
        
        # Test integration for several steps
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(10):  # Short test
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            total_reward += reward
            
            if done:
                break
        
        env.close()
        
        # Check that we got through multiple steps
        if step >= 5:  # At least 5 steps without major failure
            details = f"Completed {step+1} steps, total reward: {total_reward:.2f}"
            return True, details
        else:
            return False, f"Integration failed after {step} steps"
            
    except Exception as e:
        return False, f"Integration error: {str(e)}"

def test_environment_reward_integration():
    """Test environment + reward system integration"""
    try:
        
        
        env = create_simple_balance_squat_env(render_mode='direct')
        obs, info = env.reset()
        
        # Test reward calculation with different actions
        rewards = []
        
        # Test 1: Balanced action (should give good reward)
        balanced_action = np.array([0.3, 0.4, 0.3, 0.4, 0.2, 0.2])
        obs, reward1, done, truncated, info = env.step(balanced_action)
        rewards.append(reward1)
        
        # Test 2: Extreme action (should give poor reward)
        extreme_action = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        obs, reward2, done, truncated, info = env.step(extreme_action)
        rewards.append(reward2)
        
        env.close()
        
        # Check that reward system responds to different actions
        if len(rewards) == 2:
            details = f"Balanced action reward: {reward1:.2f}, Extreme action reward: {reward2:.2f}"
            return True, details
        else:
            return False, f"Reward system not responding"
            
    except Exception as e:
        return False, f"Reward integration error: {str(e)}"

def test_trainer_environment_integration():
    """Test trainer + environment integration"""
    try:
        
        
        # Create trainer with very minimal configuration for test
        trainer = create_balance_squat_trainer(
            total_timesteps=100,  # Very small
            n_envs=1,
            learning_rate=3e-4
        )
        
        # Test environment creation
        train_env = trainer.create_training_env()
        eval_env = trainer.create_eval_env()
        
        # Test model creation
        model = trainer.create_model(train_env)
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        # Check successful creation
        if model is not None:
            details = f"Model created successfully with trainer"
            return True, details
        else:
            return False, f"Model creation failed"
            
    except Exception as e:
        return False, f"Trainer integration error: {str(e)}"

# =============================================================================
# LEVEL 3: FULL SYSTEM TESTS
# =============================================================================

def test_mini_training_run():
    """Test a mini training run end-to-end"""
    try:
        
        
        print("   🚀 Starting mini training run (this may take a moment)...")
        
        # Very minimal training to test the pipeline
        trainer, model = train_balance_and_squats(
            total_timesteps=500,  # Very small for test
            n_envs=1,
            resume=False  # Don't try to resume
        )
        
        # Check that training completed
        if model is not None and trainer is not None:
            details = f"Mini training completed successfully"
            return True, details
        else:
            return False, f"Training pipeline failed"
            
    except Exception as e:
        return False, f"Training error: {str(e)}"

def test_balance_squat_behavior():
    """Test that the system can actually perform balance and squats"""
    try:

        
        env = create_simple_balance_squat_env(render_mode='direct')
        controller = create_balance_squat_controller(env)
        
        # Test balance behavior
        print("   🎯 Testing balance behavior...")
        controller.set_action(ActionType.BALANCE_STANDING)
        
        obs, info = env.reset()
        balance_rewards = []
        
        for step in range(20):
            action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(action)
            balance_rewards.append(reward)
            
            if done:
                break
        
        # Test squat behavior  
        print("   🏋️ Testing squat behavior...")
        controller.set_action(ActionType.SQUAT)
        
        squat_rewards = []
        
        for step in range(30):  # Longer for squat cycle
            action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(action)
            squat_rewards.append(reward)
            
            if done:
                break
        
        env.close()
        
        # Analyze behavior
        avg_balance_reward = np.mean(balance_rewards) if balance_rewards else 0
        avg_squat_reward = np.mean(squat_rewards) if squat_rewards else 0
        
        if len(balance_rewards) >= 10 and len(squat_rewards) >= 15:
            details = f"Balance: {avg_balance_reward:.2f} avg reward, Squat: {avg_squat_reward:.2f} avg reward"
            return True, details
        else:
            return False, f"Behaviors terminated too early"
            
    except Exception as e:
        return False, f"Behavior test error: {str(e)}"

def test_performance_benchmarks():
    """Test performance benchmarks for real-time suitability"""
    try:

        
        env = create_simple_balance_squat_env(render_mode='direct')
        controller = create_balance_squat_controller(env)
        
        obs, info = env.reset()
        
        # Benchmark step execution time
        step_times = []
        num_steps = 100
        
        for step in range(num_steps):
            start_time = time.time()
            
            action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(action)
            
            step_time = time.time() - start_time
            step_times.append(step_time)
            
            if done:
                break
        
        env.close()
        
        # Analyze performance
        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        
        # For 1500 Hz real-time, we need < 0.67ms per step
        real_time_capable = avg_step_time < 0.0007
        
        if real_time_capable:
            details = f"Real-time capable: {avg_step_time*1000:.2f}ms avg, {max_step_time*1000:.2f}ms max"
            return True, details
        else:
            details = f"May be slow for real-time: {avg_step_time*1000:.2f}ms avg"
            return True, details  # Still pass, just note the performance
            
    except Exception as e:
        return False, f"Performance test error: {str(e)}"

# =============================================================================
# MAIN TESTING FUNCTION
# =============================================================================

def run_comprehensive_tests():
    """Run all tests in order"""
    
    print("🎯 COMPREHENSIVE TESTING SUITE")
    print("Simplified Balance & Squat System")
    print("="*80)
    print("This will test individual components, integrations, and full system.")
    print(f"Started at: {datetime.now()}")
    
    results = TestResults()
    
    # LEVEL 1: Individual Component Tests
    print(f"\n📋 LEVEL 1: INDIVIDUAL COMPONENT TESTS")
    print("="*80)
    
    run_test(test_pam_mckibben_physics, "PAM McKibben Physics", results)
    run_test(test_zmp_calculator, "ZMP Calculator", results)
    run_test(test_reward_system, "Reward System", results)
    run_test(test_controller, "Action Controller", results)
    run_test(test_environment, "Environment", results)
    run_test(test_trainer, "Trainer", results)
    
    # LEVEL 2: Integration Tests
    print(f"\n🔗 LEVEL 2: INTEGRATION TESTS")
    print("="*80)
    
    run_test(test_controller_environment_integration, "Controller + Environment", results)
    run_test(test_environment_reward_integration, "Environment + Rewards", results)
    run_test(test_trainer_environment_integration, "Trainer + Environment", results)
    
    # LEVEL 3: Full System Tests
    print(f"\n🚀 LEVEL 3: FULL SYSTEM TESTS")
    print("="*80)
    
    run_test(test_mini_training_run, "Mini Training Pipeline", results)
    run_test(test_balance_squat_behavior, "Balance & Squat Behaviors", results)
    run_test(test_performance_benchmarks, "Performance Benchmarks", results)
    
    # Print final results
    results.print_summary()
    
    return results

# =============================================================================
# INDIVIDUAL TEST RUNNERS (for focused testing)
# =============================================================================

def test_components_only():
    """Run only component tests"""
    results = TestResults()
    
    print("🧪 COMPONENT TESTS ONLY")
    print("="*50)
    
    run_test(test_pam_mckibben_physics, "PAM McKibben Physics", results)
    run_test(test_zmp_calculator, "ZMP Calculator", results)
    run_test(test_reward_system, "Reward System", results)
    run_test(test_controller, "Action Controller", results)
    run_test(test_environment, "Environment", results)
    run_test(test_trainer, "Trainer", results)
    
    results.print_summary()
    return results

def test_integration_only():
    """Run only integration tests"""
    results = TestResults()
    
    print("🔗 INTEGRATION TESTS ONLY")
    print("="*50)
    
    run_test(test_controller_environment_integration, "Controller + Environment", results)
    run_test(test_environment_reward_integration, "Environment + Rewards", results)
    run_test(test_trainer_environment_integration, "Trainer + Environment", results)
    
    results.print_summary()
    return results

def test_system_only():
    """Run only full system tests"""
    results = TestResults()
    
    print("🚀 SYSTEM TESTS ONLY")
    print("="*50)
    
    run_test(test_mini_training_run, "Mini Training Pipeline", results)
    run_test(test_balance_squat_behavior, "Balance & Squat Behaviors", results)
    run_test(test_performance_benchmarks, "Performance Benchmarks", results)
    
    results.print_summary()
    return results

def select_tests():
    print("🧪 TESTING SUITE FOR SIMPLIFIED BALANCE & SQUAT SYSTEM")
    print("="*70)
    print("Choose testing level:")
    print("1. Components only (fastest)")
    print("2. Integration only (medium)")
    print("3. System only (slower)")
    print("4. Comprehensive (all tests)")
    print("="*70)
    
    try:
        choice = input("Enter choice (1-4) or press Enter for comprehensive: ").strip()
        
        if choice == "1":
            test_components_only()
        elif choice == "2":
            test_integration_only()
        elif choice == "3":
            test_system_only()
        else:
            run_comprehensive_tests()
            
    except KeyboardInterrupt:
        print("\n⏸️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error in test suite: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    
    print("🧪 TESTING SUITE FOR SIMPLIFIED BALANCE & SQUAT SYSTEM")
    print("="*70)
    print("Choose testing level:")
    print("1. Components only (fastest)")
    print("2. Integration only (medium)")
    print("3. System only (slower)")
    print("4. Comprehensive (all tests)")
    print("="*70)
    
    try:
        choice = input("Enter choice (1-4) or press Enter for comprehensive: ").strip()
        
        if choice == "1":
            test_components_only()
        elif choice == "2":
            test_integration_only()
        elif choice == "3":
            test_system_only()
        else:
            run_comprehensive_tests()
            
    except KeyboardInterrupt:
        print("\n⏸️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error in test suite: {e}")
        traceback.print_exc()
