import numpy as np
import time
import pybullet as p
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

from Controlador.discrete_action_controller import DiscreteActionController, ActionType
from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import Enhanced_PAMIKBipedEnv

def test_discrete_actions_integration():
    """
    Test completo de integración para el sistema de acciones discretas con PAMs.
    
    Este test valida:
    1. Creación correcta del DiscreteActionController
    2. Generación de patrones PAM para cada acción
    3. Integración con Enhanced_PAMIKBipedEnv
    4. Transiciones entre fases del currículo
    5. Sistema de recompensas para cada acción
    6. Visualización y métricas de rendimiento
    
    Returns:
        dict: Reporte completo del test con métricas y resultados
    """
    
    print("🧪 TEST COMPLETO DEL SISTEMA DE ACCIONES DISCRETAS PAM")
    print("=" * 70)
    print("Este test validará la integración completa del nuevo sistema")
    print("de acciones discretas con el robot bípedo de 6 PAMs antagónicos.")
    print("=" * 70)
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'UNKNOWN',
        'controller_tests': {},
        'environment_tests': {},
        'action_tests': {},
        'curriculum_tests': {},
        'metrics': {},
        'errors': [],
        'warnings': []
    }
    
    try:
        # ===== TEST 1: CREACIÓN DEL CONTROLADOR DE ACCIONES DISCRETAS =====
        
        print("\n📋 TEST 1: DiscreteActionController")
        print("-" * 50)
        
        try:
            # Importar el nuevo controlador
            
            # Crear un entorno mock para el controlador
            class MockEnv:
                def __init__(self):
                    self.robot_id = 0
                    self.time_step = 1.0 / 150.0
            
            mock_env = MockEnv()
            controller = DiscreteActionController(mock_env)
            
            print("   ✅ DiscreteActionController creado exitosamente")
            
            # Verificar que todas las acciones están definidas
            expected_actions = [
                ActionType.BALANCE_STANDING,
                ActionType.LIFT_LEFT_LEG,
                ActionType.LIFT_RIGHT_LEG,
                ActionType.SQUAT,
                ActionType.STEP_LEFT,
                ActionType.STEP_RIGHT
            ]
            
            for action_type in expected_actions:
                if action_type in controller.action_patterns:
                    print(f"   ✅ Acción definida: {action_type.value}")
                    test_results['controller_tests'][action_type.value] = 'DEFINED'
                else:
                    print(f"   ❌ Acción faltante: {action_type.value}")
                    test_results['errors'].append(f"Missing action: {action_type.value}")
            
        except ImportError as e:
            print(f"   ❌ Error importando DiscreteActionController: {e}")
            test_results['errors'].append(f"Import error: {str(e)}")
            test_results['controller_tests']['status'] = 'FAILED'
            return test_results
        except Exception as e:
            print(f"   ❌ Error creando controlador: {e}")
            test_results['errors'].append(f"Controller creation error: {str(e)}")
            return test_results
        
        # ===== TEST 2: GENERACIÓN DE PATRONES PAM =====
        
        print("\n📊 TEST 2: Generación de Patrones PAM")
        print("-" * 50)
        
        patterns_data = {}
        
        for action_type in expected_actions:
            print(f"\n   Testing {action_type.value}...")
            controller.set_action(action_type)
            controller.action_progress = 0.0
            
            # Simular la acción completa
            action_sequence = []
            pressures_timeline = []
            
            steps = int(controller.action_duration / mock_env.time_step)
            for step in range(steps):
                expert_action = controller.get_expert_action(mock_env.time_step)
                action_sequence.append(expert_action.copy())
                
                # Convertir de [-1,1] a presiones [0,1]
                pressures = (expert_action + 1.0) / 2.0
                pressures_timeline.append(pressures)
            
            # Analizar el patrón generado
            action_array = np.array(action_sequence)
            
            # Métricas del patrón
            metrics = {
                'mean_activation': np.mean(action_array, axis=0),
                'std_activation': np.std(action_array, axis=0),
                'max_activation': np.max(action_array, axis=0),
                'min_activation': np.min(action_array, axis=0),
                'smoothness': np.mean(np.abs(np.diff(action_array, axis=0)))  # Cambios promedio
            }
            
            patterns_data[action_type.value] = {
                'sequence': action_array,
                'metrics': metrics,
                'duration': controller.action_duration
            }
            
            print(f"      Duration: {controller.action_duration:.2f}s")
            print(f"      Smoothness: {metrics['smoothness']:.4f}")
            print(f"      Mean activations: {metrics['mean_activation']}")
            
            # Verificar que los patrones son razonables
            if metrics['smoothness'] > 0.5:
                test_results['warnings'].append(
                    f"High smoothness value for {action_type.value}: {metrics['smoothness']:.4f}"
                )
            
            test_results['action_tests'][action_type.value] = 'GENERATED'
        
        # ===== TEST 3: INTEGRACIÓN CON ENHANCED_PAMIKBipedEnv =====
        
        print("\n🤖 TEST 3: Integración con Enhanced_PAMIKBipedEnv")
        print("-" * 50)
        
        try:
            from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import Enhanced_PAMIKBipedEnv
            
            # Crear entorno con acciones discretas habilitadas
            env = Enhanced_PAMIKBipedEnv(
                render_mode='direct',  # Sin visualización para test rápido
                action_space="pam",
                use_discrete_actions=True  # IMPORTANTE: habilitar acciones discretas
            )
            
            print("   ✅ Entorno creado con soporte para acciones discretas")
            
            # Reset y verificar inicialización
            obs, info = env.reset()
            
            if hasattr(env, 'walking_controller'):
                if isinstance(env.walking_controller, DiscreteActionController):
                    print("   ✅ DiscreteActionController integrado correctamente")
                    test_results['environment_tests']['controller_integration'] = 'SUCCESS'
                else:
                    print("   ⚠️ Walking controller no es DiscreteActionController")
                    test_results['warnings'].append("Wrong controller type in environment")
            else:
                print("   ❌ No walking controller found in environment")
                test_results['errors'].append("No walking controller in environment")
            
            # Verificar dimensiones
            print(f"   📏 Observation space: {env.observation_space.shape}")
            print(f"   🎮 Action space: {env.action_space.shape}")
            
            if env.action_space.shape[0] != 6:
                test_results['warnings'].append(
                    f"Unexpected action space dimension: {env.action_space.shape[0]}"
                )
            
        except Exception as e:
            print(f"   ❌ Error en integración con entorno: {e}")
            test_results['errors'].append(f"Environment integration error: {str(e)}")
            test_results['environment_tests']['status'] = 'FAILED'
        
        # ===== TEST 4: EJECUCIÓN DE CADA ACCIÓN =====
        
        print("\n🎯 TEST 4: Ejecución de Acciones en el Entorno")
        print("-" * 50)
        
        action_performance = {}
        
        for phase_id, action_type in enumerate(expected_actions):
            print(f"\n   Testing Phase {phase_id}: {action_type.value}")
            
            try:
                # Configurar fase del currículo
                env.set_training_phase(phase_id, 1000)
                
                # Verificar que se estableció la acción correcta
                if hasattr(env.walking_controller, 'current_action'):
                    current = env.walking_controller.current_action
                    expected = env.walking_controller.get_action_for_phase(phase_id)
                    if current == expected:
                        print(f"      ✅ Acción configurada correctamente: {current.value}")
                    else:
                        print(f"      ⚠️ Acción incorrecta: {current.value} (esperada: {expected.value})")
                        test_results['warnings'].append(
                            f"Action mismatch in phase {phase_id}"
                        )
                
                # Ejecutar algunos pasos para esta acción
                action_metrics = {
                    'rewards': [],
                    'positions': [],
                    'falls': 0,
                    'stability_scores': []
                }
                
                obs, info = env.reset()
                
                for step in range(50):  # 50 pasos de prueba
                    # Obtener acción experta
                    if env.use_walking_cycle and env.walking_controller:
                        expert_action = env.walking_controller.get_expert_action(env.time_step)
                    else:
                        expert_action = env.action_space.sample()
                    
                    # Añadir algo de ruido para simular exploración
                    noise = np.random.normal(0, 0.05, size=expert_action.shape)
                    action = np.clip(expert_action + noise, -1.0, 1.0)
                    
                    # Step
                    obs, reward, done, truncated, info = env.step(action)
                    
                    action_metrics['rewards'].append(reward)
                    
                    # Obtener posición del robot
                    if hasattr(env, 'robot_id') and env.robot_id is not None:
                        pos, _ = p.getBasePositionAndOrientation(env.robot_id)
                        action_metrics['positions'].append(pos)
                        
                        # Verificar estabilidad
                        if 'zmp_stable' in info:
                            action_metrics['stability_scores'].append(
                                1.0 if info['zmp_stable'] else 0.0
                            )
                    
                    if done:
                        action_metrics['falls'] += 1
                        obs, info = env.reset()
                        print(f"      ⚠️ Caída detectada en step {step}")
                
                # Calcular métricas de rendimiento
                mean_reward = np.mean(action_metrics['rewards'])
                stability_rate = np.mean(action_metrics['stability_scores']) if action_metrics['stability_scores'] else 0
                
                action_performance[action_type.value] = {
                    'mean_reward': mean_reward,
                    'total_falls': action_metrics['falls'],
                    'stability_rate': stability_rate
                }
                
                print(f"      📊 Mean reward: {mean_reward:.3f}")
                print(f"      💔 Falls: {action_metrics['falls']}")
                print(f"      ⚖️ Stability rate: {stability_rate:.2%}")
                
                test_results['action_tests'][f"{action_type.value}_performance"] = {
                    'mean_reward': mean_reward,
                    'falls': action_metrics['falls'],
                    'stability': stability_rate
                }
                
            except Exception as e:
                print(f"      ❌ Error ejecutando {action_type.value}: {e}")
                test_results['errors'].append(f"Action execution error ({action_type.value}): {str(e)}")
        
        # ===== TEST 5: VALIDACIÓN DEL CURRÍCULO =====
        
        print("\n📚 TEST 5: Validación del Currículo Experto")
        print("-" * 50)
        
        try:
            from Curriculum_generator.Curriculum_Manager import ExpertCurriculumManager
            
            # Crear gestor de currículo
            curriculum = ExpertCurriculumManager(total_timesteps=100000)
            
            print(f"   📖 Total de fases en el currículo: {len(curriculum.phases)}")
            
            # Verificar que las fases mapean correctamente a acciones
            phase_action_mapping = {
                    0: ActionType.BALANCE_STANDING,
                    1: ActionType.BALANCE_STANDING,
                    2: ActionType.BALANCE_STANDING,
                    # Añadir fases para sentadillas si las agregas al currículo
                    3: ActionType.SQUAT,
                    4: ActionType.SQUAT,
                    # Resto de fases como antes

                
            }
            
            for phase_id, expected_action in phase_action_mapping.items():
                if phase_id < len(curriculum.phases):
                    phase = curriculum.phases[phase_id]
                    print(f"   Phase {phase_id}: {phase.name}")
                    
                    # Verificar que el controlador puede manejar esta fase
                    if hasattr(controller, 'get_action_for_phase'):
                        mapped_action = controller.get_action_for_phase(phase_id)
                        if mapped_action == expected_action:
                            print(f"      ✅ Mapeo correcto a {mapped_action.value}")
                        else:
                            print(f"      ⚠️ Mapeo incorrecto: {mapped_action.value} (esperado: {expected_action.value})")
                            test_results['warnings'].append(
                                f"Phase {phase_id} mapping mismatch"
                            )
                    
                    test_results['curriculum_tests'][f"phase_{phase_id}"] = 'VALIDATED'
            
        except Exception as e:
            print(f"   ❌ Error validando currículo: {e}")
            test_results['errors'].append(f"Curriculum validation error: {str(e)}")
        
        # ===== TEST 6: ANÁLISIS DE PATRONES BIOMECÁNICOS =====
        
        print("\n🦴 TEST 6: Análisis de Patrones Biomecánicos")
        print("-" * 50)
        
        for action_name, data in patterns_data.items():
            print(f"\n   Analizando {action_name}...")
            
            sequence = data['sequence']
            
            # Analizar antagonismo (músculos opuestos)
            left_hip_flexor = sequence[:, 0]
            left_hip_extensor = sequence[:, 1]
            right_hip_flexor = sequence[:, 2]
            right_hip_extensor = sequence[:, 3]
            
            # Calcular co-activación promedio
            left_coactivation = np.mean(np.minimum(
                (left_hip_flexor + 1) / 2,
                (left_hip_extensor + 1) / 2
            ))
            right_coactivation = np.mean(np.minimum(
                (right_hip_flexor + 1) / 2,
                (right_hip_extensor + 1) / 2
            ))
            
            print(f"      Co-activación cadera izq: {left_coactivation:.3f}")
            print(f"      Co-activación cadera der: {right_coactivation:.3f}")
            
            # Verificar que la co-activación está en rangos razonables
            if left_coactivation > 0.4 or right_coactivation > 0.4:
                test_results['warnings'].append(
                    f"High co-activation in {action_name}: L={left_coactivation:.3f}, R={right_coactivation:.3f}"
                )
            
            # Analizar simetría bilateral (para acciones simétricas como sentadillas)
            if action_name in ['SQUAT', 'BALANCE_STANDING']:
                left_activation = np.mean(sequence[:, [0, 1, 4]], axis=1)
                right_activation = np.mean(sequence[:, [2, 3, 5]], axis=1)
                
                symmetry = 1.0 - np.mean(np.abs(left_activation - right_activation))
                print(f"      Simetría bilateral: {symmetry:.3f}")
                
                if symmetry < 0.7:
                    test_results['warnings'].append(
                        f"Low symmetry in {action_name}: {symmetry:.3f}"
                    )
            
            test_results['metrics'][f"{action_name}_biomechanics"] = {
                'left_coactivation': left_coactivation,
                'right_coactivation': right_coactivation
            }
        
        # ===== TEST 7: VISUALIZACIÓN DE PATRONES (OPCIONAL) =====
        
        print("\n📈 TEST 7: Generación de Visualizaciones")
        print("-" * 50)
        
        try:
            # Crear figura con subplots para cada acción
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('Patrones de Activación PAM para Acciones Discretas', fontsize=16)
            
            muscle_names = ['L_Hip_Flex', 'L_Hip_Ext', 'R_Hip_Flex', 'R_Hip_Ext', 'L_Knee', 'R_Knee']
            colors = ['red', 'darkred', 'blue', 'darkblue', 'green', 'darkgreen']
            
            for idx, (action_name, data) in enumerate(patterns_data.items()):
                ax = axes[idx // 2, idx % 2]
                sequence = data['sequence']
                
                # Convertir a presiones [0,1]
                pressures = (sequence + 1.0) / 2.0
                
                # Plot cada músculo
                time_points = np.arange(len(sequence)) * mock_env.time_step
                
                for muscle_idx in range(6):
                    ax.plot(time_points, pressures[:, muscle_idx], 
                           label=muscle_names[muscle_idx],
                           color=colors[muscle_idx],
                           linewidth=2)
                
                ax.set_title(f'{action_name}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Pressure (normalized)')
                ax.set_ylim([0, 1])
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Guardar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"discrete_actions_patterns_{timestamp}.png"
            plt.savefig(filename, dpi=150)
            print(f"   ✅ Visualización guardada como: {filename}")
            
            test_results['metrics']['visualization'] = filename
            
        except Exception as e:
            print(f"   ⚠️ Error generando visualización: {e}")
            test_results['warnings'].append(f"Visualization error: {str(e)}")
        
        # ===== EVALUACIÓN FINAL =====
        
        print("\n" + "=" * 70)
        print("📋 EVALUACIÓN FINAL DEL TEST")
        print("=" * 70)
        
        # Contar errores y warnings
        error_count = len(test_results['errors'])
        warning_count = len(test_results['warnings'])
        
        # Determinar estado general
        if error_count == 0:
            if warning_count == 0:
                test_results['overall_status'] = '🎉 PERFECT'
                print("🎉 Estado: PERFECTO - Sistema completamente funcional")
            elif warning_count <= 3:
                test_results['overall_status'] = '✅ EXCELLENT'
                print("✅ Estado: EXCELENTE - Sistema funcional con advertencias menores")
            else:
                test_results['overall_status'] = '⚠️ GOOD'
                print("⚠️ Estado: BUENO - Sistema funcional pero requiere atención")
        elif error_count <= 2:
            test_results['overall_status'] = '⚠️ FUNCTIONAL'
            print("⚠️ Estado: FUNCIONAL - Sistema operativo con algunos problemas")
        else:
            test_results['overall_status'] = '❌ NEEDS_WORK'
            print("❌ Estado: REQUIERE TRABAJO - Problemas críticos detectados")
        
        print(f"\n📊 Resumen:")
        print(f"   Errores encontrados: {error_count}")
        print(f"   Advertencias: {warning_count}")
        print(f"   Acciones validadas: {len([k for k in test_results['action_tests'] if 'GENERATED' in str(k)])}/6")
        
        # Mostrar errores si existen
        if test_results['errors']:
            print("\n❌ ERRORES CRÍTICOS:")
            for i, error in enumerate(test_results['errors'], 1):
                print(f"   {i}. {error}")
        
        # Mostrar warnings importantes
        if test_results['warnings']:
            print("\n⚠️ ADVERTENCIAS:")
            for i, warning in enumerate(test_results['warnings'][:5], 1):  # Mostrar solo las primeras 5
                print(f"   {i}. {warning}")
            if len(test_results['warnings']) > 5:
                print(f"   ... y {len(test_results['warnings']) - 5} advertencias más")
        
        # Recomendaciones basadas en los resultados
        print("\n💡 RECOMENDACIONES:")
        
        if error_count == 0:
            print("   ✅ El sistema está listo para comenzar el entrenamiento")
            print("   🎯 Considera comenzar con las fases de equilibrio (0-2)")
            print("   📈 Monitorea las métricas de co-activación durante el entrenamiento")
        else:
            print("   🔧 Corrige los errores antes de iniciar el entrenamiento")
            print("   📝 Revisa la integración del DiscreteActionController")
            print("   🤖 Verifica que Enhanced_PAMIKBipedEnv tenga el parámetro use_discrete_actions")
        
        if warning_count > 5:
            print("   ⚠️ Alto número de advertencias - revisa los patrones de activación")
            print("   🔍 Verifica especialmente la co-activación y simetría")
        
        # Cleanup
        try:
            if 'env' in locals():
                env.close()
            if 'fig' in locals():
                plt.close(fig)
        except:
            pass
        
        print("\n" + "=" * 70)
        print("Test de integración completado")
        print("=" * 70)
        
        return test_results
        
    except Exception as e:
        print(f"\n💥 Error crítico en el test: {e}")
        test_results['errors'].append(f"Critical test failure: {str(e)}")
        test_results['overall_status'] = '💥 CRITICAL_FAILURE'
        
        import traceback
        traceback.print_exc()
        
        return test_results


def run_quick_action_test(action_type_str='BALANCE_STANDING', duration_seconds=5):
    """
    Test rápido para una acción específica con visualización en tiempo real.
    
    Args:
        action_type_str: Nombre de la acción a probar
        duration_seconds: Duración del test en segundos
    """
    print(f"\n🚀 TEST RÁPIDO: {action_type_str}")
    print("-" * 50)
    
    try:
        
        
        # Crear entorno con visualización
        env = Enhanced_PAMIKBipedEnv(
            render_mode='human',  # Visualización activada
            action_space="pam",
            use_discrete_actions=True
        )
        
        # Configurar acción específica
        action_type = ActionType[action_type_str]
        # Se ha ejecutado antes de iniciar 
        env.generate_walking_controller()
        env.walking_controller.set_action(action_type)
        
        print(f"   Ejecutando {action_type_str} por {duration_seconds} segundos...")
        
        obs, info = env.reset()
        
        steps = int(duration_seconds / env.time_step)
        rewards = []
        
        for step in range(steps):
            # Obtener acción experta
            expert_action = env.walking_controller.get_expert_action(env.time_step)
            
            # Ejecutar
            obs, reward, done, truncated, info = env.step(expert_action)
            rewards.append(reward)
            
            # Mostrar progreso cada segundo
            if step % 150 == 0:
                progress = env.walking_controller.action_progress
                mean_reward = np.mean(rewards[-150:]) if len(rewards) >= 150 else np.mean(rewards)
                print(f"      t={step/150:.1f}s | Progress: {progress:.2%} | Reward: {mean_reward:.3f}")
            
            if done:
                print(f"      ⚠️ Episode terminado en step {step}")
                obs, info = env.reset()
                env.walking_controller.set_action(action_type)
        
        print(f"\n   📊 Resultados:")
        print(f"      Reward promedio: {np.mean(rewards):.3f}")
        print(f"      Reward máximo: {np.max(rewards):.3f}")
        print(f"      Reward mínimo: {np.min(rewards):.3f}")
        
        env.close()
        
    except Exception as e:
        print(f"   ❌ Error en test rápido: {e}")
        import traceback
        traceback.print_exc()


# Función principal para ejecutar el test completo
if __name__ == "__main__":
    print("🎯 Iniciando validación del sistema de acciones discretas...")
    
    # Ejecutar test completo
    results = test_discrete_actions_integration()
    
    # Si todo está bien, opcionalmente ejecutar un test visual rápido
    if results['overall_status'] in ['🎉 PERFECT', '✅ EXCELLENT']:
        print("\n¿Deseas ver una demostración visual? (s/n): ", end="")
        response = input().lower()
        
        if response == 's':
            print("\nSelecciona una acción para demostrar:")
            print("1. BALANCE_STANDING")
            print("2. SQUAT")
            print("3. LIFT_LEFT_LEG")
            print("4. LIFT_RIGHT_LEG")
            print("5. STEP_LEFT")
            print("6. STEP_RIGHT")
            
            choice = input("Opción (1-6): ")
            
            action_map = {
                '1': 'BALANCE_STANDING',
                '2': 'SQUAT',
                '3': 'LIFT_LEFT_LEG',
                '4': 'LIFT_RIGHT_LEG',
                '5': 'STEP_LEFT',
                '6': 'STEP_RIGHT'
            }
            
            if choice in action_map:
                run_quick_action_test(action_map[choice], duration_seconds=10)
            else:
                print("Opción inválida")
    
    print("\n✅ Proceso de validación completado")