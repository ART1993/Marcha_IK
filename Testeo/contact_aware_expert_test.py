#!/usr/bin/env python3
"""
TEST DE ACCIONES EXPERTAS - Contacto + Buffer de Estabilización

Mejora del test existente añadiendo buffer post-contacto para evaluar
solo cuando el robot está realmente estabilizado.
"""

import numpy as np
import time
import pybullet as p

from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
from Controlador.discrete_action_controller import create_balance_squat_controller, ActionType

def test_expert_balance_with_stabilization_buffer(target_test_duration=20, 
                                                 stabilization_frames=20, 
                                                 render=True):
    """
    ✅ TEST MEJORADO: Contacto + Buffer de estabilización + Evaluación
    
    Fases:
    1. Esperar contacto bilateral (caída natural)
    2. Buffer de estabilización (X frames para que se asiente)
    3. Evaluación de rendimiento (solo entonces medir)
    
    Args:
        target_test_duration: Duración del test DESPUÉS del buffer (segundos)
        stabilization_frames: Frames de buffer después del contacto
        render: Mostrar visualización
    """
    
    print("🦶 TEST DE BALANCE - Contacto + Buffer de Estabilización")
    print("="*70)
    print("Estrategia mejorada:")
    print("1. ⏳ Esperar contacto bilateral (robot cae naturalmente)")
    print(f"2. 🛠️ Buffer de estabilización ({stabilization_frames} frames)")
    print(f"3. 📊 Evaluación de rendimiento ({target_test_duration}s)")
    print("="*70)
    
    try:
        # Crear entorno
        env = create_simple_balance_squat_env(render_mode='human' if render else 'direct')
        controller = create_balance_squat_controller(env)
        controller.set_action(ActionType.BALANCE_STANDING)
        
        obs, info = env.reset()
        
        print(f"🤖 Robot inicializado:")
        print(f"   Altura inicial: {info.get('initial_height', 'Unknown')}")
        print(f"   Time step: {env.time_step:.4f}s")
        print(f"   Buffer: {stabilization_frames} frames = {stabilization_frames * env.time_step:.2f}s")
        
        # ===== FASE 1: ESPERAR CONTACTO BILATERAL =====
        
        print(f"\n⏳ FASE 1: Esperando contacto bilateral...")
        
        contact_wait_start = time.time()
        max_wait_time = 10.0
        contact_frame = None
        
        frame_count = 0
        while True:
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            frame_count += 1
            
            # ¿Contacto bilateral establecido?
            left_contacts = len(p.getContactPoints(env.robot_id, env.plane_id, env.left_foot_id, -1)) > 0
            right_contacts = len(p.getContactPoints(env.robot_id, env.plane_id, env.right_foot_id, -1)) > 0
            
            if left_contacts and right_contacts:
                contact_time = time.time() - contact_wait_start
                contact_frame = frame_count
                print(f"   🔥 ¡Contacto bilateral en frame {contact_frame} ({contact_time:.2f}s)!")
                break
            
            # Timeout o episode terminado
            if time.time() - contact_wait_start > max_wait_time:
                print(f"   ❌ Timeout esperando contacto")
                env.close()
                return {'success': False, 'error': 'Contact timeout'}
            
            if done:
                print(f"   ⚠️ Episode terminó antes del contacto")
                obs, info = env.reset()
                contact_wait_start = time.time()
                frame_count = 0
                continue
        
        # ===== FASE 2: BUFFER DE ESTABILIZACIÓN =====
        
        print(f"\n🛠️ FASE 2: Buffer de estabilización ({stabilization_frames} frames)...")
        
        stabilization_start = time.time()
        frames_stabilized = 0
        stabilization_rewards = []
        
        while frames_stabilized < stabilization_frames:
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            
            frames_stabilized += 1
            stabilization_rewards.append(reward)
            
            # Debug cada 10 frames durante estabilización
            if frames_stabilized % 10 == 0:
                elapsed_stab = frames_stabilized * env.time_step
                avg_reward_stab = np.mean(stabilization_rewards[-10:])
                print(f"   Estabilización {frames_stabilized}/{stabilization_frames} | "
                      f"t={elapsed_stab:.2f}s | Reward: {avg_reward_stab:.2f}")
            
            if done:
                print(f"   💔 Robot se cayó durante estabilización")
                env.close()
                return {'success': False, 'error': 'Fall during stabilization'}
        
        stabilization_time = time.time() - stabilization_start
        stabilization_avg_reward = np.mean(stabilization_rewards)
        print(f"   ✅ Estabilización completada en {stabilization_time:.2f}s")
        print(f"   📈 Recompensa promedio durante estabilización: {stabilization_avg_reward:.3f}")
        
        # ===== FASE 3: EVALUACIÓN DE RENDIMIENTO =====
        
        print(f"\n📊 FASE 3: Evaluación de rendimiento ({target_test_duration}s)...")
        
        evaluation_start = time.time()
        evaluation_frames = 0
        evaluation_rewards = []
        falls_during_eval = 0
        
        target_eval_frames = int(target_test_duration / env.time_step)
        
        while evaluation_frames < target_eval_frames:
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            
            evaluation_frames += 1
            evaluation_rewards.append(reward)
            
            # Debug cada 3 segundos
            if evaluation_frames % int(3.0 / env.time_step) == 0:
                elapsed_eval = evaluation_frames * env.time_step
                recent_reward = np.mean(evaluation_rewards[-150:]) if len(evaluation_rewards) >= 150 else np.mean(evaluation_rewards)
                print(f"   Eval t={elapsed_eval:5.1f}s | Reward: {reward:6.2f} | Avg: {recent_reward:6.2f}")
            
            if done:
                falls_during_eval += 1
                print(f"   💔 Caída #{falls_during_eval} durante evaluación")
                
                if falls_during_eval >= 2:
                    print(f"   ❌ Demasiadas caídas durante evaluación")
                    break
                
                # Reset y re-estabilizar
                obs, info = env.reset()
                # Saltar directamente a modo evaluación (asumiendo que se re-estabilizará rápido)
                
        env.close()
        
        # ===== ANÁLISIS DE RESULTADOS =====
        
        actual_eval_time = evaluation_frames * env.time_step
        eval_avg_reward = np.mean(evaluation_rewards) if evaluation_rewards else 0
        total_time = time.time() - contact_wait_start
        
        print(f"\n📊 RESULTADOS CON BUFFER DE ESTABILIZACIÓN:")
        print("="*60)
        print(f"⏱️  Tiempo hasta contacto: {contact_time:.2f}s")
        print(f"🛠️  Tiempo de estabilización: {stabilization_time:.2f}s ({stabilization_frames} frames)")
        print(f"📊 Tiempo de evaluación: {actual_eval_time:.1f}s / {target_test_duration:.1f}s")
        print(f"📈 Recompensa durante estabilización: {stabilization_avg_reward:.3f}")
        print(f"📈 Recompensa durante evaluación: {eval_avg_reward:.3f}")
        print(f"💔 Caídas durante evaluación: {falls_during_eval}")
        print(f"🎯 Frames totales evaluados: {len(evaluation_rewards):,}")
        
        # ===== CRITERIOS DE ÉXITO AJUSTADOS =====
        
        print(f"\n🎯 EVALUACIÓN CON BUFFER:")
        print("-"*40)
        
        criteria_results = []
        
        # 1. Contacto + estabilización exitosa
        if contact_time <= 5.0 and stabilization_avg_reward > -5.0:
            print(f"   ✅ Estabilización exitosa")
            criteria_results.append(True)
        else:
            print(f"   ❌ Problemas en estabilización")
            criteria_results.append(False)
        
        # 2. Evaluación completa
        completion_rate = actual_eval_time / target_test_duration
        if completion_rate >= 0.8:
            print(f"   ✅ Evaluación completa: {completion_rate:.1%}")
            criteria_results.append(True)
        else:
            print(f"   ⚠️ Evaluación incompleta: {completion_rate:.1%}")
            criteria_results.append(False)
        
        # 3. Rendimiento durante evaluación
        if eval_avg_reward > -2.0:
            print(f"   ✅ Excelente rendimiento: {eval_avg_reward:.3f}")
            criteria_results.append(True)
        elif eval_avg_reward > -5.0:
            print(f"   ✅ Buen rendimiento: {eval_avg_reward:.3f}")
            criteria_results.append(True)
        else:
            print(f"   ❌ Rendimiento pobre: {eval_avg_reward:.3f}")
            criteria_results.append(False)
        
        # 4. Estabilidad durante evaluación
        if falls_during_eval == 0:
            print(f"   ✅ Sin caídas durante evaluación")
            criteria_results.append(True)
        elif falls_during_eval <= 1:
            print(f"   ⚠️ Caídas mínimas: {falls_during_eval}")
            criteria_results.append(True)
        else:
            print(f"   ❌ Múltiples caídas: {falls_during_eval}")
            criteria_results.append(False)
        
        # ===== VEREDICTO FINAL =====
        
        success_rate = sum(criteria_results) / len(criteria_results)
        
        if success_rate >= 0.75:
            verdict = "🎉 SISTEMA FUNCIONAL CON ESTABILIZACIÓN"
            ready = True
            message = "¡Buffer de estabilización funciona! Sistema listo para entrenamiento."
        elif success_rate >= 0.5:
            verdict = "✅ SISTEMA PARCIALMENTE FUNCIONAL"
            ready = True
            message = "Buffer mejora resultados. Listo para entrenamiento cauteloso."
        else:
            verdict = "❌ SISTEMA NECESITA AJUSTES"
            ready = False
            message = "Buffer insuficiente. Revisar parámetros."
        
        print(f"\n{verdict}")
        print(f"   Score: {sum(criteria_results)}/{len(criteria_results)}")
        print(f"   {message}")
        
        return {
            'success': ready,
            'contact_time': contact_time,
            'stabilization_time': stabilization_time,
            'stabilization_avg_reward': stabilization_avg_reward,
            'evaluation_time': actual_eval_time,
            'evaluation_avg_reward': eval_avg_reward,
            'falls': falls_during_eval,
            'completion_rate': completion_rate,
            'success_rate': success_rate,
            'buffer_frames': stabilization_frames
        }
        
    except Exception as e:
        print(f"\n💥 Error durante test con buffer: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_enhanced_expert_verification(stabilization_frames=20):
    """
    Ejecutar verificación mejorada con buffer de estabilización
    """
    
    print("🎯 VERIFICACIÓN MEJORADA - Buffer Post-Contacto")
    print("="*70)
    print(f"Mejora clave: {stabilization_frames} frames de buffer después del contacto")
    print("Esto permite que el robot se asiente antes de evaluar rendimiento")
    print("="*70)
    
    # Test con buffer
    balance_results = test_expert_balance_with_stabilization_buffer(
        target_test_duration=20, 
        stabilization_frames=stabilization_frames,
        render=True
    )
    
    # Veredicto
    print(f"\n🎯 VEREDICTO FINAL - MÉTODO CON BUFFER:")
    print("="*60)
    
    if balance_results.get('success', False):
        print("🎉 ¡BUFFER DE ESTABILIZACIÓN EXITOSO!")
        print(f"   ✅ Contacto + Estabilización: {balance_results.get('stabilization_time', 0):.2f}s")
        print(f"   ✅ Rendimiento post-buffer: {balance_results.get('evaluation_avg_reward', 0):.3f}")
        print("   🚀 LISTO PARA ENTRENAMIENTO")
        
        recommendations = [
            f"✅ Usar {stabilization_frames} frames como buffer estándar",
            "📈 El sistema ahora evalúa solo cuando el robot está estable",
            "🎯 Proceder con entrenamiento completo"
        ]
    else:
        print("⚠️ BUFFER AYUDA PERO NECESITA AJUSTES")
        print(f"   Buffer usado: {stabilization_frames} frames")
        print(f"   Resultado: {balance_results.get('success_rate', 0):.2f}")
        
        recommendations = [
            "🔧 Probar con más frames de buffer (30-40)",
            "⚙️ Ajustar parámetros de control experto",
            "📊 Revisar sistema de recompensas"
        ]
    
    print(f"\n💡 RECOMENDACIONES:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return balance_results

if __name__ == "__main__":
    # Importar PyBullet aquí para evitar problemas
    
    
    # Test con 20 frames de buffer (aproximadamente 0.13s a 1500Hz)
    run_enhanced_expert_verification(stabilization_frames=20)