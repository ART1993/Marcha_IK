#!/usr/bin/env python3
"""
TEST DE ACCIONES EXPERTAS
Verificar que el robot puede balancearse usando el controlador experto antes del entrenamiento.
"""

import numpy as np
import time
from Gymnasium_Start.Simple_BalanceSquat_BipedEnv import create_simple_balance_squat_env
from Controlador.discrete_action_controller import create_balance_squat_controller, ActionType

def test_expert_balance(duration_seconds=30, render=True):
    """
    Test del comportamiento experto de balance por 30 segundos
    """
    
    print("🎯 TESTING EXPERT BALANCE BEHAVIOR")
    print("="*50)
    print("Objetivo: Verificar que el robot puede mantener balance usando acciones expertas")
    print(f"Duración: {duration_seconds} segundos")
    print("="*50)
    
    # Crear entorno con visualización si se desea
    env = create_simple_balance_squat_env(render_mode='human' if render else 'direct')
    controller = create_balance_squat_controller(env)
    
    # Configurar para balance estático
    controller.set_action(ActionType.BALANCE_STANDING)
    
    # Reset
    obs, info = env.reset()
    
    # Métricas de seguimiento
    total_reward = 0
    step_count = 0
    rewards_history = []
    falls = 0
    
    steps_total = int(duration_seconds / env.time_step)
    
    print(f"\n🤖 Iniciando test de balance experto...")
    print(f"   Time step: {env.time_step:.4f}s")
    print(f"   Total steps: {steps_total}")
    
    try:
        for step in range(steps_total):
            # Obtener acción experta del controlador
            expert_action = controller.get_expert_action(env.time_step)
            
            # Ejecutar acción
            obs, reward, done, truncated, info = env.step(expert_action)
            
            # Trackear métricas
            total_reward += reward
            step_count += 1
            rewards_history.append(reward)
            
            # Mostrar progreso cada 3 segundos (aprox)
            if step % (int(3.0 / env.time_step)) == 0:
                current_time = step * env.time_step
                avg_reward = np.mean(rewards_history[-150:]) if len(rewards_history) >= 150 else np.mean(rewards_history)
                print(f"   t={current_time:5.1f}s | Steps: {step:4d} | Reward: {reward:6.2f} | Avg: {avg_reward:6.2f}")
            
            # Si el robot se cae, resetear
            if done:
                falls += 1
                print(f"   ⚠️ Fall detected at t={step * env.time_step:.1f}s (Fall #{falls})")
                obs, info = env.reset()
                controller.set_action(ActionType.BALANCE_STANDING)  # Reconfigurar acción
                
                # Si se cae demasiado, terminar early
                if falls >= 3:
                    print(f"   ❌ Too many falls ({falls}). Terminating test early.")
                    break
        
        env.close()
        
        # ===== ANÁLISIS DE RESULTADOS =====
        
        print(f"\n📊 EXPERT BALANCE TEST RESULTS:")
        print("="*50)
        
        # Métricas básicas
        actual_duration = step_count * env.time_step
        avg_reward = total_reward / step_count if step_count > 0 else 0
        
        print(f"⏱️  Duration: {actual_duration:.1f}s / {duration_seconds:.1f}s target")
        print(f"👟 Steps completed: {step_count:,}")
        print(f"🎯 Total reward: {total_reward:.2f}")
        print(f"📈 Average reward: {avg_reward:.3f}")
        print(f"💔 Falls: {falls}")
        
        # Análisis de estabilidad
        if len(rewards_history) > 100:
            recent_rewards = rewards_history[-100:]
            stability_score = np.mean(recent_rewards)
            print(f"🔄 Recent stability (last 100 steps): {stability_score:.3f}")
        
        # ===== EVALUACIÓN DEL DESEMPEÑO =====
        
        print(f"\n🎯 PERFORMANCE EVALUATION:")
        print("-"*30)
        
        # Criterios de éxito
        success_criteria = {
            'duration': actual_duration >= duration_seconds * 0.8,  # Al menos 80% del tiempo
            'avg_reward': avg_reward > -5.0,  # Recompensa promedio razonable
            'falls': falls <= 1,  # Máximo 1 caída
            'completion': step_count >= steps_total * 0.8  # Al menos 80% de steps
        }
        
        passed_criteria = sum(success_criteria.values())
        total_criteria = len(success_criteria)
        
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
        
        # Veredicto final
        if passed_criteria == total_criteria:
            verdict = "🎉 EXCELLENT"
            message = "Expert controller works perfectly! Ready for training."
        elif passed_criteria >= 3:
            verdict = "✅ GOOD"
            message = "Expert controller works well. Training should be successful."
        elif passed_criteria >= 2:
            verdict = "⚠️ MODERATE"
            message = "Expert controller needs tuning. Training may be challenging."
        else:
            verdict = "❌ POOR"
            message = "Expert controller failing. Fix before training."
        
        print(f"\n{verdict}")
        print(f"   Score: {passed_criteria}/{total_criteria}")
        print(f"   {message}")
        
        # Recomendaciones
        print(f"\n💡 RECOMMENDATIONS:")
        if passed_criteria >= 3:
            print("   🚀 Proceed with full training")
            print("   📈 Use current expert actions for imitation learning")
            print("   ⚙️ Consider total_timesteps=500k-1M for first training")
        else:
            print("   🔧 Tune expert action patterns before training")
            print("   📊 Check reward system weights")
            print("   🎯 Verify robot URDF and physics parameters")
        
        return {
            'success': passed_criteria >= 3,
            'score': passed_criteria / total_criteria,
            'avg_reward': avg_reward,
            'falls': falls,
            'duration': actual_duration,
            'verdict': verdict
        }
        
    except KeyboardInterrupt:
        print("\n⏸️ Test interrupted by user")
        env.close()
        return {'success': False, 'score': 0, 'interrupted': True}
    
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        env.close()
        return {'success': False, 'score': 0, 'error': str(e)}

def test_expert_squat(duration_seconds=20, render=True):
    """
    Test del comportamiento experto de sentadillas
    """
    
    print("\n🏋️ TESTING EXPERT SQUAT BEHAVIOR")
    print("="*50)
    print("Objetivo: Verificar que el robot puede hacer sentadillas usando acciones expertas")
    print(f"Duración: {duration_seconds} segundos")
    print("="*50)
    
    env = create_simple_balance_squat_env(render_mode='human' if render else 'direct')
    controller = create_balance_squat_controller(env)
    
    # Configurar para sentadillas
    controller.set_action(ActionType.SQUAT)
    
    obs, info = env.reset()
    
    total_reward = 0
    step_count = 0
    squat_cycles = 0
    
    steps_total = int(duration_seconds / env.time_step)
    
    print(f"\n🏋️ Iniciando test de sentadillas expertas...")
    
    try:
        for step in range(steps_total):
            expert_action = controller.get_expert_action(env.time_step)
            obs, reward, done, truncated, info = env.step(expert_action)
            
            total_reward += reward
            step_count += 1
            
            # Detectar ciclos de sentadilla completados
            controller_info = controller.get_current_action_info()
            if controller_info['progress'] >= 1.0:
                squat_cycles += 1
                print(f"   🏋️ Squat cycle #{squat_cycles} completed at t={step * env.time_step:.1f}s")
            
            # Mostrar progreso
            if step % (int(4.0 / env.time_step)) == 0:
                current_time = step * env.time_step
                progress = controller_info['progress']
                phase = controller_info['current_phase']
                print(f"   t={current_time:5.1f}s | Phase: {phase}/3 | Progress: {progress:.1%} | Reward: {reward:6.2f}")
            
            if done:
                print(f"   ❌ Episode ended early at t={step * env.time_step:.1f}s")
                break
        
        env.close()
        
        print(f"\n📊 EXPERT SQUAT TEST RESULTS:")
        print(f"   🏋️ Squat cycles completed: {squat_cycles}")
        print(f"   🎯 Average reward: {total_reward / step_count:.3f}")
        print(f"   ⏱️ Duration: {step_count * env.time_step:.1f}s")
        
        if squat_cycles >= 1:
            print(f"   ✅ Squat controller working!")
        else:
            print(f"   ⚠️ No complete squat cycles detected")
        
        return {'squat_cycles': squat_cycles, 'avg_reward': total_reward / step_count}
        
    except KeyboardInterrupt:
        print("\n⏸️ Squat test interrupted")
        env.close()
        return {'squat_cycles': 0, 'interrupted': True}

def run_expert_verification():
    """Ejecutar verificación completa de acciones expertas"""
    
    print("🎯 EXPERT ACTION VERIFICATION SUITE")
    print("="*60)
    print("Este test verifica que las acciones expertas funcionan antes del entrenamiento")
    print("="*60)
    
    # Test 1: Balance
    balance_results = test_expert_balance(duration_seconds=20, render=True)
    
    # Test 2: Sentadillas (solo si balance funciona bien)
    if balance_results.get('success', False):
        squat_results = test_expert_squat(duration_seconds=15, render=True)
    else:
        print("\n⚠️ Skipping squat test due to balance issues")
        squat_results = {'squat_cycles': 0}
    
    # Veredicto final
    print(f"\n🎯 FINAL EXPERT VERIFICATION VERDICT:")
    print("="*50)
    
    if balance_results.get('success', False):
        if squat_results.get('squat_cycles', 0) >= 1:
            print("🎉 READY FOR TRAINING!")
            print("   ✅ Balance works")
            print("   ✅ Squats work")
            print("   🚀 Proceed with train_balance_and_squats()")
        else:
            print("✅ MOSTLY READY")
            print("   ✅ Balance works")
            print("   ⚠️ Squats need tuning")
            print("   🚀 Can start with balance-focused training")
    else:
        print("⚠️ NEEDS WORK")
        print("   ❌ Balance issues detected")
        print("   🔧 Fix expert controller before training")
    
    return balance_results, squat_results

if __name__ == "__main__":
    run_expert_verification()
