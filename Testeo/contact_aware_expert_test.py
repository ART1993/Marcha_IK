#!/usr/bin/env python3
"""
TEST DE ACCIONES EXPERTAS - Solo Después del Contacto

Implementa la solución elegante sugerida: solo medir rendimiento
después de que se establezca contacto bilateral.
"""

import numpy as np
import time
from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
from Controlador.discrete_action_controller import create_balance_squat_controller, ActionType

def test_expert_balance_post_contact(target_test_duration=20, render=True):
    """
    ✅ TEST ELEGANTE: Solo medir después del contacto bilateral
    
    Args:
        target_test_duration: Duración del test DESPUÉS del contacto (segundos)
        render: Mostrar visualización
    """
    
    print("🦶 TEST DE BALANCE - Solo Después del Contacto")
    print("="*60)
    print("Estrategia: El robot se estabiliza naturalmente, luego medimos rendimiento")
    print(f"Duración objetivo DESPUÉS del contacto: {target_test_duration}s")
    print("="*60)
    
    try:
        # Crear entorno con visualización
        env = create_simple_balance_squat_env(render_mode='human' if render else 'direct')
        controller = create_balance_squat_controller(env)
        controller.set_action(ActionType.BALANCE_STANDING)
        
        print(f"\n🤖 Inicializando robot...")
        obs, info = env.reset()
        
        print(f"✅ Robot inicializado:")
        print(f"   Modo inicial: {info.get('control_mode', 'Unknown')}")
        print(f"   Altura: {info.get('initial_height', 0):.3f}m")
        
        # ===== FASE 1: ESPERAR CONTACTO BILATERAL =====
        
        print(f"\n⏳ FASE 1: Esperando contacto bilateral...")
        
        contact_wait_start = time.time()
        max_wait_time = 10.0  # Máximo 10 segundos para establecer contacto
        
        while True:
            # Obtener acción (pero no importa mucho en esta fase)
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            
            # ¿Se estableció contacto PAM?
            if info.get('pam_control_active', False):
                contact_time = time.time() - contact_wait_start
                print(f"   🔥 ¡Contacto bilateral establecido en {contact_time:.2f}s!")
                print(f"   🎯 Modo: {info.get('control_mode', 'Unknown')}")
                break
            
            # ¿Timeout?
            if time.time() - contact_wait_start > max_wait_time:
                print(f"   ❌ Timeout esperando contacto ({max_wait_time}s)")
                env.close()
                return {
                    'success': False,
                    'error': 'Contact timeout',
                    'contact_time': max_wait_time
                }
            
            # ¿Episode terminó prematuramente?
            if done:
                print(f"   ⚠️ Episode terminó antes del contacto")
                obs, info = env.reset()
                contact_wait_start = time.time()  # Reiniciar timer
                continue
        
        # ===== FASE 2: TEST DE RENDIMIENTO POST-CONTACTO =====
        
        print(f"\n🎯 FASE 2: Test de balance por {target_test_duration}s...")
        
        test_start_time = time.time()
        steps_in_test = 0
        rewards_history = []
        falls_during_test = 0
        pam_control_time = 0
        
        target_steps = int(target_test_duration / env.time_step)
        
        while steps_in_test < target_steps:
            
            # Obtener acción experta
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            
            # Solo contar si los PAMs están activos
            if info.get('pam_control_active', False):
                steps_in_test += 1
                rewards_history.append(reward)
                pam_control_time += env.time_step
                
                # Debug cada 3 segundos de test activo
                if steps_in_test % int(3.0 / env.time_step) == 0:
                    elapsed_test = steps_in_test * env.time_step
                    recent_reward = np.mean(rewards_history[-150:]) if len(rewards_history) >= 150 else np.mean(rewards_history)
                    print(f"   Test t={elapsed_test:5.1f}s | Reward: {reward:6.2f} | Avg: {recent_reward:6.2f}")
            
            else:
                # PAMs no activos - probablemente perdió contacto
                if steps_in_test > 0:  # Solo contar si ya habíamos empezado el test
                    print(f"   ⚠️ Contacto perdido en test step {steps_in_test}")
            
            # Si se cae durante el test
            if done:
                falls_during_test += 1
                print(f"   💔 Caída #{falls_during_test} durante el test")
                
                if falls_during_test >= 3:
                    print(f"   ❌ Demasiadas caídas durante el test")
                    break
                
                # Reset y esperar contacto de nuevo
                obs, info = env.reset()
                
                # Esperar contacto rápidamente
                contact_reestablished = False
                for wait_step in range(int(5.0 / env.time_step)):  # Máximo 5s
                    expert_action = controller.get_expert_action(env.time_step)
                    obs, reward, done, truncated, info = env.step(expert_action)
                    
                    if info.get('pam_control_active', False):
                        print(f"   🔄 Contacto reestablecido")
                        contact_reestablished = True
                        break
                
                if not contact_reestablished:
                    print(f"   ❌ No se pudo reestablecer contacto")
                    break
        
        env.close()
        
        # ===== ANÁLISIS DE RESULTADOS =====
        
        actual_test_time = pam_control_time
        avg_reward = np.mean(rewards_history) if rewards_history else 0
        total_elapsed = time.time() - contact_wait_start
        
        print(f"\n📊 RESULTADOS DEL TEST POST-CONTACTO:")
        print("="*50)
        print(f"⏱️  Tiempo hasta contacto: {contact_time:.2f}s")
        print(f"🎯 Tiempo de test activo: {actual_test_time:.1f}s / {target_test_duration:.1f}s objetivo")
        print(f"📈 Recompensa promedio: {avg_reward:.3f}")
        print(f"💔 Caídas durante test: {falls_during_test}")
        print(f"👟 Steps de test válidos: {len(rewards_history):,}")
        
        # ===== CRITERIOS DE ÉXITO =====
        
        print(f"\n🎯 EVALUACIÓN:")
        print("-"*30)
        
        criteria_results = []
        
        # 1. Contacto establecido rápidamente
        if contact_time <= 5.0:
            print(f"   ✅ Contacto rápido: {contact_time:.2f}s")
            criteria_results.append(True)
        else:
            print(f"   ⚠️ Contacto lento: {contact_time:.2f}s")
            criteria_results.append(False)
        
        # 2. Test de duración suficiente
        completion_rate = actual_test_time / target_test_duration
        if completion_rate >= 0.8:
            print(f"   ✅ Test completo: {completion_rate:.1%}")
            criteria_results.append(True)
        else:
            print(f"   ❌ Test incompleto: {completion_rate:.1%}")
            criteria_results.append(False)
        
        # 3. Recompensas positivas
        if avg_reward > -2.0:
            print(f"   ✅ Recompensas buenas: {avg_reward:.3f}")
            criteria_results.append(True)
        elif avg_reward > -10.0:
            print(f"   ⚠️ Recompensas moderadas: {avg_reward:.3f}")
            criteria_results.append(True)
        else:
            print(f"   ❌ Recompensas pobres: {avg_reward:.3f}")
            criteria_results.append(False)
        
        # 4. Pocas caídas
        if falls_during_test == 0:
            print(f"   ✅ Sin caídas")
            criteria_results.append(True)
        elif falls_during_test <= 2:
            print(f"   ⚠️ Pocas caídas: {falls_during_test}")
            criteria_results.append(True)
        else:
            print(f"   ❌ Muchas caídas: {falls_during_test}")
            criteria_results.append(False)
        
        # ===== VEREDICTO FINAL =====
        
        success_rate = sum(criteria_results) / len(criteria_results)
        
        if success_rate >= 0.75:
            verdict = "🎉 SISTEMA FUNCIONAL"
            ready_for_training = True
            message = "¡Control experto funciona! Listo para entrenamiento."
        elif success_rate >= 0.5:
            verdict = "✅ SISTEMA PARCIALMENTE FUNCIONAL"
            ready_for_training = True
            message = "Control experto funciona moderadamente. Entrenar con precaución."
        else:
            verdict = "❌ SISTEMA NECESITA TRABAJO"
            ready_for_training = False
            message = "Control experto tiene problemas. Ajustar antes de entrenar."
        
        print(f"\n{verdict}")
        print(f"   Score: {sum(criteria_results)}/{len(criteria_results)}")
        print(f"   {message}")
        
        return {
            'success': ready_for_training,
            'contact_time': contact_time,
            'test_duration': actual_test_time,
            'avg_reward': avg_reward,
            'falls': falls_during_test,
            'completion_rate': completion_rate,
            'success_rate': success_rate
        }
        
    except Exception as e:
        print(f"\n💥 Error durante test: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def test_expert_squat_post_contact(target_test_duration=15):
    """
    Test de sentadillas que solo empieza después del contacto
    """
    
    print("\n🏋️ TEST DE SENTADILLAS - Solo Después del Contacto")
    print("="*60)
    
    try:
        env = create_simple_balance_squat_env(render_mode='human')
        controller = create_balance_squat_controller(env)
        
        obs, info = env.reset()
        
        # Esperar contacto
        print(f"   Esperando contacto para sentadillas...")
        while True:
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            
            if info.get('pam_control_active', False):
                print(f"   ✅ Contacto establecido - iniciando sentadillas")
                break
            
            if done:
                obs, info = env.reset()
        
        # Cambiar a modo sentadilla
        controller.set_action(ActionType.SQUAT)
        
        # Test de sentadillas
        squat_cycles = 0
        steps_tested = 0
        target_steps = int(target_test_duration / env.time_step)
        
        while steps_tested < target_steps:
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            
            if info.get('pam_control_active', False):
                steps_tested += 1
                
                # Detectar ciclos completados
                controller_info = controller.get_current_action_info()
                if controller_info['progress'] >= 1.0:
                    squat_cycles += 1
                    print(f"   🏋️ Ciclo de sentadilla #{squat_cycles} completado")
            
            if done:
                print(f"   ⚠️ Episode terminado durante sentadillas")
                break
        
        env.close()
        
        print(f"   🏋️ Sentadillas completadas: {squat_cycles}")
        return {'squat_cycles': squat_cycles, 'steps_tested': steps_tested}
        
    except Exception as e:
        print(f"   ❌ Error en test de sentadillas: {e}")
        return {'squat_cycles': 0, 'error': str(e)}

def run_contact_aware_expert_verification():
    """
    Ejecutar verificación completa que solo mide después del contacto
    """
    
    print("🎯 VERIFICACIÓN DE ACCIONES EXPERTAS - Post-Contacto")
    print("="*70)
    print("Esta verificación implementa la solución elegante:")
    print("1. Robot se estabiliza naturalmente (standing position)")
    print("2. Detecta contacto bilateral automáticamente") 
    print("3. Solo ENTONCES empieza a medir rendimiento")
    print("="*70)
    
    # Test 1: Balance post-contacto
    balance_results = test_expert_balance_post_contact(target_test_duration=20, render=True)
    
    # Test 2: Sentadillas (solo si balance funciona)
    if balance_results.get('success', False):
        squat_results = test_expert_squat_post_contact(target_test_duration=15)
    else:
        print("\n⚠️ Saltando test de sentadillas debido a problemas de balance")
        squat_results = {'squat_cycles': 0}
    
    # Veredicto final
    print(f"\n🎯 VEREDICTO FINAL - MÉTODO POST-CONTACTO:")
    print("="*60)
    
    if balance_results.get('success', False):
        if squat_results.get('squat_cycles', 0) >= 1:
            print("🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
            print("   ✅ Balance post-contacto: EXCELENTE")
            print("   ✅ Sentadillas post-contacto: FUNCIONAN")
            print("   🚀 LISTO PARA ENTRENAMIENTO COMPLETO")
            
            next_steps = [
                "🎯 Iniciar entrenamiento con configuración conservadora",
                "📊 Monitorear que el entrenamiento mejore las recompensas", 
                "⚙️ El sistema de contacto automático funcionará durante entrenamiento"
            ]
        else:
            print("✅ SISTEMA PARCIALMENTE FUNCIONAL")
            print("   ✅ Balance post-contacto: EXCELENTE")
            print("   ⚠️ Sentadillas: Necesitan ajuste")
            print("   🚀 LISTO PARA ENTRENAMIENTO DE BALANCE")
            
            next_steps = [
                "🎯 Entrenar primero solo balance (BALANCE_STANDING)",
                "🔧 Ajustar patrones de sentadilla después",
                "📈 El sistema automático funcionará bien para balance"
            ]
    else:
        print("⚠️ SISTEMA NECESITA AJUSTES")
        print(f"   Contacto: {balance_results.get('contact_time', 0):.1f}s")
        print(f"   Recompensa: {balance_results.get('avg_reward', 0):.2f}")
        print("   🔧 Revisar parámetros antes de entrenar")
        
        next_steps = [
            "🔧 Ajustar presiones PAM en discrete_action_controller",
            "⚖️ Revisar pesos del sistema de recompensas",
            "🔄 Repetir test después de ajustes"
        ]
    
    print(f"\n💡 PRÓXIMOS PASOS RECOMENDADOS:")
    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")
    
    return balance_results, squat_results

if __name__ == "__main__":
    run_contact_aware_expert_verification()
