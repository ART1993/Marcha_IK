#!/usr/bin/env python3
"""
TEST SUITE CORREGIDO para el Sistema Simplificado de Balance & Sentadillas

Este script corrige los problemas detectados:
1. ❌ PAM McKibben Physics - Fuerza 0.0N → ✅ Corrección en inicialización
2. ❌ Trainer Configuration Failed → ✅ Corrección en atributos

OBJETIVO: Validar que el sistema funciona correctamente para balance y sentadillas.
"""

import sys
import os
import numpy as np
import time
from datetime import datetime
import traceback

# Configurar paths
sys.path.append('.')
sys.path.append('./Archivos_Apoyo')
sys.path.append('./Controlador')
sys.path.append('./Gymnasium_Start')

class CorrectedTestResults:
    """Clase para trackear resultados con mejor diagnóstico"""
    
    def __init__(self):
        self.tests = {}
        self.start_time = datetime.now()
        self.errors = []
        self.warnings = []
        self.debug_info = []
    
    def add_test(self, test_name, passed, details="", debug_info=""):
        self.tests[test_name] = {
            'passed': passed,
            'details': details,
            'debug_info': debug_info,
            'timestamp': datetime.now()
        }
        
        if debug_info:
            self.debug_info.append(f"{test_name}: {debug_info}")
    
    def print_summary(self):
        total_tests = len(self.tests)
        passed_tests = sum(1 for test in self.tests.values() if test['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n" + "="*80)
        print(f"🧪 CORRECTED TEST RESULTS SUMMARY")
        print(f"="*80)
        print(f"⏱️  Total execution time: {datetime.now() - self.start_time}")
        print(f"📊 Total tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"🐛 Debug entries: {len(self.debug_info)}")
        
        if passed_tests == total_tests:
            print(f"🎉 ALL TESTS PASSED! Sistema corregido y listo.")
        else:
            print(f"⚠️ Some tests still failing. Check corrections below.")
        
        # Resultados detallados
        print(f"\n📋 DETAILED TEST RESULTS:")
        print("-"*80)
        for test_name, result in self.tests.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} {test_name}")
            if result['details']:
                print(f"     {result['details']}")
            if result['debug_info']:
                print(f"     🐛 Debug: {result['debug_info']}")

def test_corrected_pam_mckibben():
    """Test 1 CORREGIDO: PAM_McKibben con inicialización arreglada"""
    
    try:
        # Usar la clase corregida
        from Archivos_Apoyo.dinamica_pam import PAMMcKibben
        
        # Crear PAM con parámetros válidos
        pam = PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4)
        
        # ✅ CORRECCIÓN: Verificar que los límites se inicializaron
        debug_info = f"epsilon_max={pam.epsilon_max:.3f}, a={pam.a:.3f}, b={pam.b:.3f}"
        
        if not hasattr(pam, 'epsilon_max') or pam.epsilon_max <= 0:
            return False, "epsilon_max no inicializado correctamente", debug_info
        
        # Test con parámetros realistas
        pressure = 3 * 101325  # 3 atm en Pascales
        contraction = 0.15     # 15% de contracción (seguro dentro de límites)
        
        force = pam.force_model_new(pressure, contraction)
        
        # ✅ CORRECCIÓN: Verificar que la fuerza sea realista
        if force > 0 and force < 2000:  # Fuerza entre 0 y 2000N es realista
            details = f"Force = {force:.1f}N at {pressure/101325:.1f}atm, {contraction:.1%} contraction"
            return True, details, debug_info
        else:
            return False, f"Fuerza fuera de rango: {force:.1f}N", debug_info
            
    except Exception as e:
        return False, f"Exception: {str(e)}", f"Error en importación o cálculo"

def test_corrected_simplified_trainer():
    """Test 2 CORREGIDO: Trainer con atributos arreglados"""
    
    try:
        # Usar la clase corregida
        from Gymnasium_Start.Simplified_BalanceSquat_Trainer import Simplified_BalanceSquat_Trainer
        
        # Crear trainer con configuración mínima
        trainer = Simplified_BalanceSquat_Trainer(
            total_timesteps=1000,
            n_envs=1,
            learning_rate=3e-4
        )
        
        # ✅ CORRECCIÓN: Verificar atributos específicamente
        checks = {
            'total_timesteps': trainer.total_timesteps == 1000,
            'n_envs': trainer.n_envs == 1,
            'model_dir': hasattr(trainer, 'model_dir'),
            'env_config': hasattr(trainer, 'env_config'),  # Singular para tests
            'env_configs': hasattr(trainer, 'env_configs'), # Plural para interno
            'dir_exists': os.path.exists(trainer.model_dir)
        }
        
        failed_checks = [k for k, v in checks.items() if not v]
        
        if not failed_checks:
            debug_info = f"All attributes OK: {list(checks.keys())}"
            details = f"Trainer configured: {trainer.total_timesteps} timesteps, {trainer.n_envs} envs"
            return True, details, debug_info
        else:
            debug_info = f"Failed checks: {failed_checks}"
            return False, f"Missing attributes: {failed_checks}", debug_info
            
    except Exception as e:
        return False, f"Exception: {str(e)}", f"Error en creación del trainer"

def test_simplified_reward_system():
    """Test 3: Sistema de recompensas simplificado"""
    
    try:
        from Archivos_Mejorados.Simplified_BalanceSquat_RewardSystem import create_simple_reward_system
        
        reward_system = create_simple_reward_system()
        
        # Test de eficiencia PAM básica
        test_action = np.array([0.3, 0.4, 0.3, 0.4, 0.2, 0.2])
        efficiency = reward_system._calculate_basic_pam_efficiency(test_action)
        
        # Verificar configuración
        summary = reward_system.get_reward_summary()
        
        debug_info = f"Weights: {len(summary.get('weights', {}))}, System: {summary.get('system', 'Unknown')}"
        
        if 'weights' in summary and -5.0 <= efficiency <= 5.0:
            details = f"PAM efficiency: {efficiency:.3f}, weights configured: {len(summary['weights'])}"
            return True, details, debug_info
        else:
            return False, f"Reward system malfunction", debug_info
            
    except Exception as e:
        return False, f"Exception: {str(e)}", f"Error en reward system"

def test_simplified_controller():
    """Test 4: Controlador de acciones discretas"""
    
    try:
        from Controlador.discrete_action_controller import create_balance_squat_controller, ActionType
        
        # Mock environment
        class MockEnv:
            def __init__(self):
                self.robot_id = 0
                self.time_step = 1.0 / 150.0
        
        mock_env = MockEnv()
        controller = create_balance_squat_controller(mock_env)
        
        # Test de generación de acciones
        controller.set_action(ActionType.BALANCE_STANDING)
        balance_action = controller.get_expert_action(mock_env.time_step)
        
        controller.set_action(ActionType.SQUAT)
        squat_action = controller.get_expert_action(mock_env.time_step)
        
        debug_info = f"Balance: {balance_action[:3]}, Squat: {squat_action[:3]}"
        
        if (len(balance_action) == 6 and len(squat_action) == 6 and
            0 <= np.min(balance_action) and np.max(balance_action) <= 1.0):
            details = f"Actions OK: Balance={balance_action[:3]}, Squat={squat_action[:3]}"
            return True, details, debug_info
        else:
            return False, f"Invalid action generation", debug_info
            
    except Exception as e:
        return False, f"Exception: {str(e)}", f"Error en controller"

def test_simplified_environment():
    """Test 5: Entorno simplificado"""
    
    try:
        from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
        
        # Crear entorno sin rendering para test
        env = create_simple_balance_squat_env(render_mode='direct')
        
        # Test reset
        obs, info = env.reset()
        
        # Test step
        action = env.action_space.sample() * 0.5 + 0.25  # Acción moderada
        obs_new, reward, done, truncated, info = env.step(action)
        
        debug_info = f"Obs shape: {obs.shape}, Action shape: {action.shape}, Reward: {reward:.2f}"
        
        env.close()
        
        if (len(obs) == env.observation_space.shape[0] and 
            len(action) == env.action_space.shape[0] and
            isinstance(reward, (int, float))):
            details = f"Obs: {len(obs)} elements, Action: {len(action)} PAMs, Reward: {reward:.2f}"
            return True, details, debug_info
        else:
            return False, f"Environment interface error", debug_info
            
    except Exception as e:
        return False, f"Exception: {str(e)}", f"Error en environment"

def test_system_integration():
    """Test 6: Integración del sistema completo"""
    
    try:
        from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
        from Controlador.discrete_action_controller import create_balance_squat_controller, ActionType
        
        # Crear componentes
        env = create_simple_balance_squat_env(render_mode='direct')
        controller = create_balance_squat_controller(env)
        
        # Test de integración básica
        obs, info = env.reset()
        controller.set_action(ActionType.BALANCE_STANDING)
        
        total_reward = 0
        steps_completed = 0
        
        for step in range(50):  # Test corto
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            total_reward += reward
            steps_completed = step + 1
            
            if done:
                break
        
        env.close()
        
        debug_info = f"Steps: {steps_completed}, Total reward: {total_reward:.2f}"
        
        if steps_completed >= 10:  # Al menos 10 steps sin fallar
            details = f"Integration OK: {steps_completed} steps, reward: {total_reward:.2f}"
            return True, details, debug_info
        else:
            return False, f"Integration failed after {steps_completed} steps", debug_info
            
    except Exception as e:
        return False, f"Exception: {str(e)}", f"Error en integración"

def run_corrected_tests():
    """Ejecutar todos los tests corregidos"""
    
    print("🛠️ CORRECTED TEST SUITE FOR BALANCE & SQUAT SYSTEM")
    print("="*70)
    print("Este test suite incluye las correcciones para:")
    print("  ✅ PAM McKibben Physics - Inicialización corregida") 
    print("  ✅ Trainer Configuration - Atributos arreglados")
    print("  🧪 Tests adicionales de integración")
    print("="*70)
    
    results = CorrectedTestResults()
    
    # Lista de tests corregidos
    tests = [
        ("PAM McKibben Physics (CORRECTED)", test_corrected_pam_mckibben),
        ("Trainer Configuration (CORRECTED)", test_corrected_simplified_trainer),
        ("Reward System", test_simplified_reward_system),
        ("Action Controller", test_simplified_controller),
        ("Environment", test_simplified_environment),
        ("System Integration", test_system_integration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 60)
        
        try:
            passed, details, debug_info = test_func()
            results.add_test(test_name, passed, details, debug_info)
            
            if passed:
                print(f"   ✅ PASSED: {details}")
                if debug_info:
                    print(f"   🐛 Debug: {debug_info}")
            else:
                print(f"   ❌ FAILED: {details}")
                if debug_info:
                    print(f"   🐛 Debug: {debug_info}")
                    
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            results.add_test(test_name, False, error_msg, f"Exception in test execution")
            print(f"   💥 EXCEPTION: {str(e)}")
            traceback.print_exc()
    
    # Imprimir resumen final
    results.print_summary()
    
    # Análisis de correcciones
    print(f"\n🔧 CORRECTION ANALYSIS:")
    print("-"*50)
    
    pam_test = results.tests.get("PAM McKibben Physics (CORRECTED)", {})
    trainer_test = results.tests.get("Trainer Configuration (CORRECTED)", {})
    
    if pam_test.get('passed', False):
        print("✅ PAM McKibben correction SUCCESSFUL")
        print("   - Initialization fixed")
        print("   - Force calculation working")
    else:
        print("❌ PAM McKibben correction FAILED")
        print("   - Need further debugging")
    
    if trainer_test.get('passed', False):
        print("✅ Trainer correction SUCCESSFUL")
        print("   - Attribute naming fixed")
        print("   - Configuration working")
    else:
        print("❌ Trainer correction FAILED")
        print("   - Need further debugging")
    
    # Recomendaciones
    total_passed = sum(1 for test in results.tests.values() if test['passed'])
    total_tests = len(results.tests)
    
    if total_passed == total_tests:
        print(f"\n🎉 ALL CORRECTIONS SUCCESSFUL!")
        print(f"   Sistema listo para entrenamiento")
        print(f"   Puedes proceder con train_balance_and_squats()")
    elif total_passed >= total_tests * 0.8:
        print(f"\n✅ MOST CORRECTIONS SUCCESSFUL")
        print(f"   {total_passed}/{total_tests} tests passing")
        print(f"   Sistema mayormente funcional")
    else:
        print(f"\n⚠️ NEED MORE CORRECTIONS")
        print(f"   Only {total_passed}/{total_tests} tests passing")
        print(f"   Review debug info above")
    
    return results

def quick_pam_test():
    """Test rápido específico para PAM con debug"""
    
    print("🔬 QUICK PAM DEBUG TEST")
    print("-"*30)
    
    try:
        from Archivos_Apoyo.dinamica_pam import PAMMcKibben
        
        print("Creating PAM with debug info...")
        pam = PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4)
        
        print(f"✅ PAM created")
        print(f"   L0: {pam.L0}")
        print(f"   r0: {pam.r0}")
        print(f"   alpha0: {pam.alpha0}")
        print(f"   a: {pam.a:.3f}")
        print(f"   b: {pam.b:.3f}")
        
        if hasattr(pam, 'epsilon_max'):
            print(f"   epsilon_max: {pam.epsilon_max:.3f}")
        else:
            print("   ❌ epsilon_max not found!")
        
        # Test force calculation
        pressure = 3 * 101325
        contraction = 0.15
        
        print(f"\nTesting force calculation:")
        print(f"   Pressure: {pressure/101325:.1f} atm")
        print(f"   Contraction: {contraction:.1%}")
        
        force = pam.force_model_new(pressure, contraction)
        print(f"   Force: {force:.1f} N")
        
        if force > 0:
            print("✅ Force calculation working!")
        else:
            print("❌ Force calculation failed!")
            
    except Exception as e:
        print(f"❌ Quick PAM test failed: {e}")
        traceback.print_exc()


def main():
    print("🎯 CORRECTED BALANCE & SQUAT TEST SUITE")
    print("="*50)
    print("Choose testing option:")
    print("1. Run corrected full test suite")
    print("2. Quick PAM debug test only")
    print("3. Help with corrections")
    print("="*50)
    
    try:
        choice = input("Enter choice (1-3) or press Enter for full suite: ").strip()
        
        if choice == "2":
            quick_pam_test()
        elif choice == "3":
            print("\n📋 CORRECTIONS APPLIED:")
            print("1. PAM McKibben: Fixed initialization of epsilon_max")
            print("2. Trainer: Fixed env_config attribute naming")
            print("3. Added debug info to all tests")
            print("\n💡 If tests still fail, check:")
            print("- File paths and imports")
            print("- PyBullet installation")
            print("- Dependencies (numpy, stable-baselines3, etc.)")
        else:
            run_corrected_tests()
            
    except KeyboardInterrupt:
        print("\n⏸️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error in test suite: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    
    print("🎯 CORRECTED BALANCE & SQUAT TEST SUITE")
    print("="*50)
    print("Choose testing option:")
    print("1. Run corrected full test suite")
    print("2. Quick PAM debug test only")
    print("3. Help with corrections")
    print("="*50)
    
    try:
        choice = input("Enter choice (1-3) or press Enter for full suite: ").strip()
        
        if choice == "2":
            quick_pam_test()
        elif choice == "3":
            print("\n📋 CORRECTIONS APPLIED:")
            print("1. PAM McKibben: Fixed initialization of epsilon_max")
            print("2. Trainer: Fixed env_config attribute naming")
            print("3. Added debug info to all tests")
            print("\n💡 If tests still fail, check:")
            print("- File paths and imports")
            print("- PyBullet installation")
            print("- Dependencies (numpy, stable-baselines3, etc.)")
        else:
            run_corrected_tests()
            
    except KeyboardInterrupt:
        print("\n⏸️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error in test suite: {e}")
        traceback.print_exc()
