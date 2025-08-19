#!/usr/bin/env python3
"""
SISTEMA DE TESTING COMPLETO para el proyecto simplificado de Balance & Sentadillas

Este script ejecuta una batería completa de tests que van desde lo más básico
hasta el entrenamiento completo, identificando problemas paso a paso.

ORDEN DE TESTING:
1. Tests individuales por clase
2. Tests de integración por pares  
3. Test del sistema completo
4. Test de entrenamiento corto
"""

import os
import sys
import numpy as np
import traceback
from datetime import datetime
import json

from Archivos_Apoyo.ZPMCalculator import ZMPCalculator
from Archivos_Mejorados.Simplified_BalanceSquat_RewardSystem import Simplified_BalanceSquat_RewardSystem
from Controlador.discrete_action_controller import ActionType, DiscreteActionController, create_balance_squat_controller
from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import Simple_BalanceSquat_BipedEnv, create_simple_balance_squat_env
from Gymnasium_Start.Simplified_BalanceSquat_Trainer import Simplified_BalanceSquat_Trainer, create_balance_squat_trainer

def print_header(title):
    """Imprimir header bonito para secciones"""
    print("\n" + "="*70)
    print(f"🧪 {title}")
    print("="*70)

def print_test_result(test_name, passed, details=""):
    """Imprimir resultado de test con formato"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"     {details}")

class TestResults:
    """Clase para trackear resultados de todos los tests"""
    def __init__(self):
        self.results = {
            'individual_tests': {},
            'integration_tests': {},
            'system_tests': {},
            'training_tests': {},
            'errors': [],
            'warnings': []
        }
        self.start_time = datetime.now()
    
    def add_result(self, category, test_name, passed, details="", error=None):
        self.results[category][test_name] = {
            'passed': passed,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        
        if error:
            self.results['errors'].append(f"{test_name}: {str(error)}")
    
    def get_summary(self):
        total_tests = sum(len(cat) for cat in self.results.values() if isinstance(cat, dict))
        passed_tests = sum(
            sum(1 for test in cat.values() if test.get('passed', False))
            for cat in self.results.values() if isinstance(cat, dict)
        )
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'duration': (datetime.now() - self.start_time).total_seconds(),
            'errors_count': len(self.results['errors']),
            'warnings_count': len(self.results['warnings'])
        }

# =====================================
# TESTS INDIVIDUALES POR CLASE
# =====================================

def test_pam_mckibben():
    """Test 1: PAM_McKibben - El corazón del control de actuadores"""
    try:
        from Archivos_Apoyo.dinamica_pam import PAMMcKibben
        
        # Crear PAM de prueba
        pam = PAMMcKibben(L0=0.5, r0=0.02, alpha0=np.pi/4)
        
        # Test de parámetros básicos
        assert hasattr(pam, 'L0'), "PAM debe tener longitud inicial L0"
        assert hasattr(pam, 'r0'), "PAM debe tener radio inicial r0"
        assert hasattr(pam, 'F_max_factor'), "PAM debe calcular factor de fuerza máxima"
        
        # Test de cálculo de fuerza
        pressure = 3 * 101325  # 3 atm
        contraction = 0.2      # 20% de contracción
        
        force = pam.force_model_new(pressure, contraction)
        assert force >= 0, f"Fuerza debe ser positiva, obtenido: {force}"
        assert force < 1000, f"Fuerza parece excesiva: {force}N"
        
        # Test de límites
        force_zero = pam.force_model_new(pressure, 0.9)  # Contracción excesiva
        assert force_zero == 0, "Fuerza debe ser 0 para contracción excesiva"
        
        return True, f"PAM genera {force:.1f}N a {pressure/101325:.1f}atm con {contraction*100}% contracción"
        
    except Exception as e:
        return False, f"Error en PAM_McKibben: {str(e)}"

def test_simplified_zmp_calculator():
    """Test 2: ZMP Calculator simplificado"""
    try:
        # Importar la clase simplificada
        exec(open('test_artifacts.py').read())  # Simular que tenemos el archivo
        
        # Crear calculador de prueba
        zmp_calc = ZMPCalculator(
            robot_id=0, left_foot_id=2, right_foot_id=5
        )
        
        # Test de parámetros básicos
        assert zmp_calc.g == 9.81, "Gravedad debe ser 9.81"
        assert zmp_calc.stability_margin > 0, "Margen de estabilidad debe ser positivo"
        
        # Simular historia COM para test
        fake_positions = [
            [0.0, 0.0, 1.1],
            [0.01, 0.0, 1.1], 
            [0.02, 0.0, 1.1]
        ]
        
        for pos in fake_positions:
            zmp_calc.update_com_history(pos)
        
        # Test de cálculo de aceleración
        accel = zmp_calc.calculate_simple_acceleration()
        assert len(accel) == 3, "Aceleración debe tener 3 componentes"
        assert not np.any(np.isnan(accel)), "Aceleración no debe contener NaN"
        
        return True, f"ZMP Calculator: aceleración = {accel[:2]}"
        
    except Exception as e:
        return False, f"Error en ZMP Calculator: {str(e)}"

def test_simplified_reward_system():
    """Test 3: Sistema de recompensas simplificado"""
    try:
        exec(open('test_artifacts.py').read())  # Simular que tenemos el archivo
        
        # Crear sistema de recompensas
        reward_system = Simplified_BalanceSquat_RewardSystem()
        
        # Test de configuración
        assert 'height' in reward_system.weights, "Debe incluir recompensa de altura"
        assert 'orientation' in reward_system.weights, "Debe incluir recompensa de orientación"
        assert reward_system.target_height > 0, "Altura objetivo debe ser positiva"
        
        # Test de eficiencia PAM básica
        test_action = np.array([0.3, 0.4, 0.3, 0.4, 0.2, 0.2])
        efficiency = reward_system._calculate_basic_pam_efficiency(test_action)
        
        assert -5.0 <= efficiency <= 5.0, f"Eficiencia fuera de rango: {efficiency}"
        
        return True, f"Reward System: eficiencia PAM = {efficiency:.3f}"
        
    except Exception as e:
        return False, f"Error en Reward System: {str(e)}"

def test_simplified_controller():
    """Test 4: Controlador simplificado de balance/sentadillas"""
    try:
        exec(open('test_artifacts.py').read())  # Simular que tenemos el archivo
        
        # Mock environment
        class MockEnv:
            def __init__(self):
                self.robot_id = 0
                self.time_step = 1.0 / 150.0
        
        mock_env = MockEnv()
        controller = DiscreteActionController(mock_env)
        
        # Test de acciones disponibles
        assert ActionType.BALANCE_STANDING in controller.action_patterns
        assert ActionType.SQUAT in controller.action_patterns
        assert len(controller.action_patterns) == 2, "Debe tener exactamente 2 acciones"
        
        # Test de generación de acciones
        controller.set_action(ActionType.BALANCE_STANDING)
        balance_action = controller.get_expert_action(mock_env.time_step)
        
        assert len(balance_action) == 6, "Acción debe tener 6 elementos (6 PAMs)"
        assert np.all(balance_action >= 0), "Presiones PAM deben ser positivas"
        assert np.all(balance_action <= 1), "Presiones PAM deben estar normalizadas"
        
        return True, f"Controller: acción balance = {balance_action[:3]}"
        
    except Exception as e:
        return False, f"Error en Controller: {str(e)}"

def test_simplified_trainer():
    """Test 5: Trainer simplificado"""
    try:
        exec(open('test_artifacts.py').read())  # Simular que tenemos el archivo
        
        # Crear trainer con configuración mínima
        trainer = Simplified_BalanceSquat_Trainer(
            total_timesteps=1000,  # Muy pocos para test
            n_envs=1,              # Solo un entorno
            learning_rate=3e-4
        )
        
        # Test de configuración
        assert trainer.total_timesteps == 1000
        assert trainer.n_envs == 1
        assert hasattr(trainer, 'model_dir'), "Debe tener directorio de modelos"
        assert hasattr(trainer, 'env_config'), "Debe tener configuración del entorno"
        
        # Test de directorios
        assert os.path.exists(trainer.model_dir), "Directorio de modelos debe existir"
        assert os.path.exists(trainer.logs_dir), "Directorio de logs debe existir"
        
        return True, f"Trainer: {trainer.total_timesteps} timesteps, {trainer.n_envs} envs"
        
    except Exception as e:
        return False, f"Error en Trainer: {str(e)}"

# =====================================
# TESTS DE INTEGRACIÓN POR PARES
# =====================================

def test_controller_reward_integration():
    """Test 6: Integración Controller + Reward System"""
    try:
        # Simular importaciones
        exec(open('test_artifacts.py').read())

        
        # Crear componentes
        class MockEnv:
            def __init__(self):
                self.robot_id = 0
                self.time_step = 1.0 / 150.0
        
        mock_env = MockEnv()
        controller = DiscreteActionController(mock_env)
        reward_system = Simplified_BalanceSquat_RewardSystem()
        
        # Test de flujo: Controller -> Reward System
        controller.set_action(ActionType.BALANCE_STANDING)
        action = controller.get_expert_action(mock_env.time_step)
        
        # Configurar estados PAM en reward system
        reward_system.pam_states = {
            'pressures': action * reward_system.max_pressure
        }
        
        # Calcular eficiencia PAM
        efficiency = reward_system._calculate_basic_pam_efficiency(action)
        
        assert not np.isnan(efficiency), "Eficiencia no debe ser NaN"
        
        return True, f"Integración exitosa: eficiencia = {efficiency:.3f}"
        
    except Exception as e:
        return False, f"Error en integración Controller-Reward: {str(e)}"

def test_env_zmp_integration():
    """Test 7: Integración Environment + ZMP Calculator"""
    try:
        # Test básico de que el ZMP calculator puede funcionar con el entorno
        exec(open('test_artifacts.py').read())
        
        # Simular estados del entorno
        robot_id = 0
        left_foot_id = 2
        right_foot_id = 5
        
        zmp_calc = ZMPCalculator(robot_id, left_foot_id, right_foot_id)
        
        # Simular posiciones COM
        for i in range(5):
            fake_com = [0.01 * i, 0.0, 1.1]
            zmp_calc.update_com_history(fake_com)
        
        # Test de cálculo ZMP
        zmp_point = zmp_calc.calculate_zmp()
        stability_info = zmp_calc.get_stability_info()
        
        assert 'zmp_position' in stability_info
        assert 'is_stable' in stability_info
        
        return True, f"ZMP integración: posición = {zmp_point[:2]}"
        
    except Exception as e:
        return False, f"Error en integración Env-ZMP: {str(e)}"

# =====================================
# TEST DEL SISTEMA COMPLETO
# =====================================

def test_complete_system():
    """Test 8: Sistema completo sin entrenamiento"""
    try:
        print("   🔧 Intentando crear el sistema completo...")
        
        # Simular la creación del sistema completo
        exec(open('test_artifacts.py').read())
        
        # Test de que todos los componentes pueden crearse juntos
        components = {}
        
        # 1. Environment
        print("   📦 Creando entorno...")
        env = create_simple_balance_squat_env(render_mode='direct')
        components['environment'] = True
        
        # 2. Controller
        print("   🎮 Creando controlador...")
        controller = create_balance_squat_controller(env)
        components['controller'] = True
        
        # 3. Test de reset
        print("   🔄 Testing reset...")
        obs, info = env.reset()
        components['reset'] = True
        
        # 4. Test de step básico
        print("   👟 Testing step...")
        action = controller.get_expert_action(env.time_step)
        obs, reward, done, truncated, info = env.step(action)
        components['step'] = True
        
        # 5. Cerrar entorno
        env.close()
        components['cleanup'] = True
        
        success_count = sum(components.values())
        return True, f"Sistema completo: {success_count}/5 componentes funcionando"
        
    except Exception as e:
        return False, f"Error en sistema completo: {str(e)}"

def test_training_setup():
    """Test 9: Setup de entrenamiento (sin entrenar)"""
    try:
        print("   🧠 Configurando entrenamiento...")
        
        exec(open('test_artifacts.py').read())
        
        # Crear trainer con configuración mínima
        trainer = create_balance_squat_trainer(
            total_timesteps=100,  # Muy pocos para test
            n_envs=1,
            learning_rate=3e-4
        )
        
        # Test de preparación de entrenamiento (sin entrenar)
        print("   📁 Verificando directorios...")
        assert os.path.exists(trainer.model_dir)
        assert os.path.exists(trainer.logs_dir)
        
        print("   ⚙️ Verificando configuración...")
        assert trainer.total_timesteps == 100
        assert trainer.n_envs == 1
        
        return True, "Training setup completado exitosamente"
        
    except Exception as e:
        return False, f"Error en training setup: {str(e)}"

# =====================================
# TEST DE ENTRENAMIENTO CORTO
# =====================================

def test_short_training():
    """Test 10: Entrenamiento muy corto para verificar que RecurrentPPO funciona"""
    try:
        print("   🚀 Iniciando entrenamiento corto...")
        
        exec(open('test_artifacts.py').read())
        
        # Crear trainer con configuración ultra-mínima
        trainer = create_balance_squat_trainer(
            total_timesteps=50,   # Ultra-corto
            n_envs=1,
            learning_rate=1e-3    # Learning rate alto para convergencia rápida
        )
        
        print("   📚 Intentando entrenar por 50 timesteps...")
        
        # IMPORTANTE: Configurar para que no haga resume
        trainer.resume_from = None
        
        # Simular entrenamiento ultra-corto
        model = trainer.train(resume=False)
        
        if model is not None:
            print("   ✅ Modelo creado exitosamente")
            
            # Verificar que el modelo tiene la estructura correcta
            assert hasattr(model, 'policy'), "Modelo debe tener policy"
            assert hasattr(model, 'learn'), "Modelo debe tener método learn"
            
            return True, "Entrenamiento corto exitoso - RecurrentPPO funciona"
        else:
            return False, "Entrenamiento retornó None"
        
    except Exception as e:
        return False, f"Error en entrenamiento corto: {str(e)}"

# =====================================
# FUNCIÓN PRINCIPAL DE TESTING
# =====================================

def run_all_tests():
    """Ejecutar todos los tests en orden"""
    
    print_header("SISTEMA DE TESTING COMPLETO - BALANCE & SENTADILLAS")
    print("Este script verificará que todas las clases simplificadas funcionan")
    print("correctamente de forma individual y en conjunto.")
    
    results = TestResults()
    
    # ===== TESTS INDIVIDUALES =====
    print_header("TESTS INDIVIDUALES POR CLASE")
    
    individual_tests = [
        ("PAM_McKibben", test_pam_mckibben),
        ("ZMP_Calculator", test_simplified_zmp_calculator), 
        ("Reward_System", test_simplified_reward_system),
        ("Controller", test_simplified_controller),
        ("Trainer", test_simplified_trainer)
    ]
    
    for test_name, test_func in individual_tests:
        try:
            passed, details = test_func()
            print_test_result(test_name, passed, details)
            results.add_result('individual_tests', test_name, passed, details)
        except Exception as e:
            print_test_result(test_name, False, f"Exception: {str(e)}")
            results.add_result('individual_tests', test_name, False, error=e)
    
    # ===== TESTS DE INTEGRACIÓN =====
    print_header("TESTS DE INTEGRACIÓN POR PARES")
    
    integration_tests = [
        ("Controller_Reward_Integration", test_controller_reward_integration),
        ("Environment_ZMP_Integration", test_env_zmp_integration)
    ]
    
    for test_name, test_func in integration_tests:
        try:
            passed, details = test_func()
            print_test_result(test_name, passed, details)
            results.add_result('integration_tests', test_name, passed, details)
        except Exception as e:
            print_test_result(test_name, False, f"Exception: {str(e)}")
            results.add_result('integration_tests', test_name, False, error=e)
    
    # ===== TESTS DEL SISTEMA COMPLETO =====
    print_header("TESTS DEL SISTEMA COMPLETO")
    
    system_tests = [
        ("Complete_System", test_complete_system),
        ("Training_Setup", test_training_setup)
    ]
    
    for test_name, test_func in system_tests:
        try:
            passed, details = test_func()
            print_test_result(test_name, passed, details)
            results.add_result('system_tests', test_name, passed, details)
        except Exception as e:
            print_test_result(test_name, False, f"Exception: {str(e)}")
            results.add_result('system_tests', test_name, False, error=e)
    
    # ===== TEST DE ENTRENAMIENTO =====
    print_header("TEST DE ENTRENAMIENTO CORTO")
    
    training_tests = [
        ("Short_Training", test_short_training)
    ]
    
    for test_name, test_func in training_tests:
        try:
            passed, details = test_func()
            print_test_result(test_name, passed, details)
            results.add_result('training_tests', test_name, passed, details)
        except Exception as e:
            print_test_result(test_name, False, f"Exception: {str(e)}")
            results.add_result('training_tests', test_name, False, error=e)
    
    # ===== RESUMEN FINAL =====
    print_header("RESUMEN FINAL")
    
    summary = results.get_summary()
    
    print(f"📊 ESTADÍSTICAS DE TESTING:")
    print(f"   Total tests: {summary['total_tests']}")
    print(f"   Tests exitosos: {summary['passed_tests']}")
    print(f"   Tests fallidos: {summary['failed_tests']}")
    print(f"   Tasa de éxito: {summary['success_rate']*100:.1f}%")
    print(f"   Duración: {summary['duration']:.1f} segundos")
    print(f"   Errores: {summary['errors_count']}")
    
    # Determinar estado general
    if summary['success_rate'] >= 0.9:
        status = "🎉 EXCELENTE"
        message = "El sistema está listo para entrenamiento completo"
    elif summary['success_rate'] >= 0.7:
        status = "✅ BUENO"
        message = "El sistema funciona con algunos problemas menores"
    elif summary['success_rate'] >= 0.5:
        status = "⚠️ PARCIAL"
        message = "El sistema tiene problemas significativos que requieren atención"
    else:
        status = "❌ CRÍTICO"
        message = "El sistema tiene problemas graves que impiden el funcionamiento"
    
    print(f"\n{status} | {message}")
    
    # Mostrar errores si los hay
    if results.results['errors']:
        print(f"\n🐛 ERRORES ENCONTRADOS:")
        for error in results.results['errors'][:5]:  # Mostrar solo los primeros 5
            print(f"   - {error}")
        
        if len(results.results['errors']) > 5:
            print(f"   ... y {len(results.results['errors']) - 5} errores más")
    
    # Guardar resultados detallados
    save_test_results(results)
    
    return results

def save_test_results(results):
    """Guardar resultados detallados en archivo JSON"""
    try:
        os.makedirs("test_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results.results, f, indent=2)
        
        print(f"\n💾 Resultados detallados guardados en: {filename}")
        
    except Exception as e:
        print(f"⚠️ No se pudieron guardar los resultados: {e}")

# =====================================
# FUNCIONES DE UTILIDAD
# =====================================

def quick_test():
    """Test rápido solo de componentes básicos"""
    print_header("QUICK TEST - Solo componentes básicos")
    
    basic_tests = [
        ("PAM_McKibben", test_pam_mckibben),
        ("ZMP_Calculator", test_simplified_zmp_calculator),
        ("Reward_System", test_simplified_reward_system)
    ]
    
    for test_name, test_func in basic_tests:
        try:
            passed, details = test_func()
            print_test_result(test_name, passed, details)
        except Exception as e:
            print_test_result(test_name, False, f"Exception: {str(e)}")

def debug_test(test_name):
    """Ejecutar un test específico con debug detallado"""
    print_header(f"DEBUG TEST - {test_name}")
    
    test_map = {
        'pam': test_pam_mckibben,
        'zmp': test_simplified_zmp_calculator,
        'reward': test_simplified_reward_system,
        'controller': test_simplified_controller,
        'trainer': test_simplified_trainer,
        'integration': test_controller_reward_integration,
        'system': test_complete_system,
        'training': test_short_training
    }
    
    if test_name in test_map:
        try:
            passed, details = test_map[test_name]()
            print_test_result(test_name, passed, details)
        except Exception as e:
            print_test_result(test_name, False, f"Exception: {str(e)}")
            print(f"\n🐛 Traceback detallado:")
            traceback.print_exc()
    else:
        print(f"❌ Test '{test_name}' no encontrado")
        print(f"Tests disponibles: {list(test_map.keys())}")

# =====================================
# MAIN
# =====================================

def main_test(command=None):
    if command == "quick":
        quick_test()
    elif command == "debug" and len(sys.argv) > 2:
        debug_test(sys.argv[2])
    elif command == "help":
        print("Uso:")
        print("  python test_simplified_system.py           # Test completo")
        print("  python test_simplified_system.py quick     # Test rápido")
        print("  python test_simplified_system.py debug <test_name>  # Debug específico")
        print("  python test_simplified_system.py help      # Esta ayuda")
    else:
        print(f"Comando desconocido: {command}")
        print("Usa 'help' para ver opciones disponibles")
        run_all_tests()

if __name__ == "__main__":
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            quick_test()
        elif command == "debug" and len(sys.argv) > 2:
            debug_test(sys.argv[2])
        elif command == "help":
            print("Uso:")
            print("  python test_simplified_system.py           # Test completo")
            print("  python test_simplified_system.py quick     # Test rápido")
            print("  python test_simplified_system.py debug <test_name>  # Debug específico")
            print("  python test_simplified_system.py help      # Esta ayuda")
        else:
            print(f"Comando desconocido: {command}")
            print("Usa 'help' para ver opciones disponibles")
    else:
        # Ejecutar test completo
        run_all_tests()
