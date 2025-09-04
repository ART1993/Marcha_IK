#!/usr/bin/env python3
"""
SCRIPT DE PRUEBA: Control de Una Pierna usando tu modelo entrenado
Versión paso-a-paso para aprendizaje y debugging
"""

import numpy as np
import time
import os

# Importar SOLO tu entorno existente (sin modificaciones)
from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv

def verificar_prerequisites():
    """
    Verificar que todo está en su lugar antes de empezar
    """
    print("🔍 PASO 1: Verificando prerequisites...")
    
    # Verificar modelo entrenado
    model_path = "./models_lift_leg/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Modelo no encontrado en {model_path}")
        print(f"   Solución: Ejecuta primero el entrenamiento con 'python inicio_programa.py'")
        return False
    else:
        print(f"✅ Modelo encontrado: {model_path}")
    
    # Verificar URDF del robot
    urdf_path = "./2_legged_human_like_robot.urdf"
    if not os.path.exists(urdf_path):
        print(f"❌ ERROR: URDF del robot no encontrado en {urdf_path}")
        return False
    else:
        print(f"✅ URDF del robot encontrado: {urdf_path}")
    
    # Verificar que podemos importar el entorno
    try:
        env_test = Simple_Lift_Leg_BipedEnv(render_mode='direct', action_space="pam")
        env_test.close()
        print("✅ Entorno importado y probado correctamente")
    except Exception as e:
        print(f"❌ ERROR importando entorno: {e}")
        return False
    
    print("🎉 Todos los prerequisites verificados correctamente\n")
    return True

def explicar_patrones_pam():
    """
    Explicar cómo funcionan los patrones PAM en tu sistema
    """
    print("📚 PASO 2: Entendiendo los patrones PAM...")
    print()
    print("Tu robot tiene 6 músculos PAM (Pneumatic Artificial Muscles):")
    print("  PAM 0: Flexor cadera izquierda  - Levanta el muslo izquierdo")
    print("  PAM 1: Extensor cadera izquierda - Empuja el muslo izquierdo hacia atrás")
    print("  PAM 2: Flexor cadera derecha    - Levanta el muslo derecho") 
    print("  PAM 3: Extensor cadera derecha  - Empuja el muslo derecho hacia atrás")
    print("  PAM 4: Flexor rodilla izquierda - Dobla la rodilla izquierda")
    print("  PAM 5: Flexor rodilla derecha   - Dobla la rodilla derecha")
    print()
    print("Los valores van de 0.0 (músculo relajado) a 1.0 (músculo totalmente contraído)")
    print()
    
    # Mostrar patrones que usaremos
    patterns = {
        'Balance Normal': [0.4, 0.5, 0.4, 0.5, 0.1, 0.1],
        'Pierna Izq. Arriba': [0.8, 0.2, 0.3, 0.7, 0.6, 0.1],
        'Pierna Der. Arriba': [0.3, 0.7, 0.8, 0.2, 0.1, 0.6]
    }
    
    print("Patrones que probaremos:")
    for name, pattern in patterns.items():
        print(f"  {name:20}: {pattern}")
    print()
    
    print("💡 Lógica de los patrones:")
    print("  Para levantar pierna IZQUIERDA:")
    print("    - Aumentar flexor cadera izq (PAM 0: 0.8) → levanta muslo")
    print("    - Reducir extensor cadera izq (PAM 1: 0.2) → permite flexión")
    print("    - Fortalecer extensor cadera der (PAM 3: 0.7) → soporte en pierna derecha")
    print("    - Doblar rodilla izq (PAM 4: 0.6) → completa el levantamiento")
    print()

def test_balance_inicial(env):
    """
    Probar primero solo el balance para verificar estabilidad básica
    """
    print("🧪 PASO 3: Test de balance básico (10 segundos)...")
    
    # Patrón de balance (tu patrón existente)
    balance_pattern = [0.4, 0.5, 0.4, 0.5, 0.1, 0.1]
    print(f"   Usando patrón: {balance_pattern}")
    
    # Reset del entorno usando tu método
    obs, info = env.reset()
    
    print("✅ Entorno reseteado correctamente")
    
    # Test de balance por 10 segundos
    start_time = time.time()
    step_count = 0
    max_height = 0
    min_height = 10
    
    while time.time() - start_time < 10.0:
        # Usar tu método step exactamente como está implementado
        obs, reward, done, truncated, info = env.step(balance_pattern)
        step_count += 1
        
        # Extraer altura usando tu estructura de observación
        current_height = obs[1] if len(obs) > 1 else 1.0
        max_height = max(max_height, current_height)
        min_height = min(min_height, current_height)
        
        # Mostrar progreso cada 2 segundos
        if step_count % 600 == 0:  # 600 steps ≈ 2 segundos a tu frecuencia
            elapsed = time.time() - start_time
            print(f"   ⏱️  {elapsed:.1f}s - Altura: {current_height:.2f}m - Reward: {reward:.2f}")
        
        # Verificar si el episodio terminó (usando tu lógica)
        if done:
            print(f"   ⚠️  Test de balance falló en {time.time() - start_time:.1f}s")
            print(f"   Causa: Robot perdió estabilidad")
            return False
    
    print(f"✅ Test de balance exitoso!")
    print(f"   Duración: 10.0s")
    print(f"   Pasos ejecutados: {step_count}")
    print(f"   Altura máxima: {max_height:.2f}m")
    print(f"   Altura mínima: {min_height:.2f}m")
    print(f"   Variación de altura: {max_height - min_height:.3f}m (menor es mejor)")
    print()
    
    return True

def test_secuencia_piernas(env):
    """
    Test principal: secuencia completa de levantamiento de piernas
    """
    print("🦵 PASO 4: Test de secuencia completa de piernas...")
    
    # Definir la secuencia de prueba
    sequence = [
        ('balance', [0.4, 0.5, 0.4, 0.5, 0.1, 0.1], 3.0, "🟢 Balance inicial"),
        ('left_up', [0.8, 0.2, 0.3, 0.7, 0.6, 0.1], 5.0, "🦵 Pierna IZQUIERDA arriba"),
        ('balance', [0.4, 0.5, 0.4, 0.5, 0.1, 0.1], 2.0, "🟢 Volver a balance"),
        ('right_up', [0.3, 0.7, 0.8, 0.2, 0.1, 0.6], 5.0, "🦵 Pierna DERECHA arriba"),
        ('balance', [0.4, 0.5, 0.4, 0.5, 0.1, 0.1], 3.0, "🟢 Balance final")
    ]
    
    print(f"Secuencia planificada: {len(sequence)} fases, {sum(s[2] for s in sequence):.1f}s total")
    print()
    
    # Variables para tracking de resultados
    results = []
    total_steps = 0
    
    for phase_num, (phase_name, pam_pattern, duration, description) in enumerate(sequence, 1):
        
        print(f"FASE {phase_num}/{len(sequence)}: {description}")
        print(f"   Patrón PAM: {pam_pattern}")
        print(f"   Duración objetivo: {duration}s")
        
        # Ejecutar esta fase
        start_time = time.time()
        phase_steps = 0
        phase_success = True
        heights_recorded = []
        
        while time.time() - start_time < duration:
            
            # Ejecutar un paso usando tu entorno
            obs, reward, done, truncated, info = env.step(pam_pattern)
            phase_steps += 1
            total_steps += 1
            
            # Registrar datos para análisis
            current_height = obs[1] if len(obs) > 1 else 1.0
            heights_recorded.append(current_height)
            
            # Mostrar progreso cada 1.5 segundos
            if phase_steps % 450 == 0:  # 450 steps ≈ 1.5s
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                print(f"   ⏱️  +{elapsed:.1f}s - Altura: {current_height:.2f}m - Reward: {reward:.2f} - Restante: {remaining:.1f}s")
            
            # Verificar fallo del episodio
            if done:
                actual_duration = time.time() - start_time
                print(f"   ❌ FASE FALLIDA en {actual_duration:.1f}s")
                print(f"   Motivo: Robot perdió estabilidad (done=True)")
                phase_success = False
                break
        
        # Resumen de la fase
        actual_duration = time.time() - start_time
        if phase_success:
            avg_height = np.mean(heights_recorded) if heights_recorded else 0
            height_stability = np.std(heights_recorded) if len(heights_recorded) > 1 else 0
            print(f"   ✅ FASE COMPLETADA en {actual_duration:.1f}s")
            print(f"   Estadísticas: Altura promedio: {avg_height:.2f}m, Estabilidad: {height_stability:.3f}")
        
        # Guardar resultado de la fase
        results.append({
            'phase': phase_name,
            'description': description,
            'success': phase_success,
            'duration_planned': duration,
            'duration_actual': actual_duration,
            'steps': phase_steps,
            'avg_height': np.mean(heights_recorded) if heights_recorded else 0,
            'height_stability': np.std(heights_recorded) if len(heights_recorded) > 1 else 0
        })
        
        print()
        
        # Si una fase falla, terminar la secuencia
        if not phase_success:
            print("🛑 Secuencia interrumpida debido a falla en fase")
            break
    
    return results, total_steps

def analizar_resultados(results, total_steps):
    """
    Analizar y mostrar los resultados de la prueba
    """
    print("📊 PASO 5: Análisis de resultados...")
    print()
    
    # Resumen general
    successful_phases = sum(1 for r in results if r['success'])
    total_phases = len(results)
    total_planned_time = sum(r['duration_planned'] for r in results)
    total_actual_time = sum(r['duration_actual'] for r in results)
    
    print("🏆 RESUMEN GENERAL:")
    print(f"   Fases completadas: {successful_phases}/{total_phases}")
    print(f"   Tasa de éxito: {successful_phases/total_phases:.1%}")
    print(f"   Tiempo planificado: {total_planned_time:.1f}s")
    print(f"   Tiempo real ejecutado: {total_actual_time:.1f}s")
    print(f"   Total de pasos simulados: {total_steps}")
    print()
    
    # Análisis por fase
    print("📋 ANÁLISIS DETALLADO POR FASE:")
    for i, result in enumerate(results, 1):
        status_emoji = "✅" if result['success'] else "❌"
        print(f"   {i}. {status_emoji} {result['description']}")
        print(f"      Duración: {result['duration_actual']:.1f}s/{result['duration_planned']:.1f}s")
        if result['success']:
            print(f"      Altura promedio: {result['avg_height']:.2f}m")
            print(f"      Estabilidad altura: ±{result['height_stability']:.3f}m")
        print()
    
    # Evaluación de calidad
    if successful_phases == total_phases:
        print("🎉 ¡ÉXITO COMPLETO! Tu modelo puede controlar ambas piernas exitosamente")
        if all(r['height_stability'] < 0.05 for r in results if r['success']):
            print("🌟 Calidad EXCELENTE: Movimientos muy estables")
        elif all(r['height_stability'] < 0.1 for r in results if r['success']):
            print("👍 Calidad BUENA: Movimientos estables")
        else:
            print("👌 Calidad ACEPTABLE: Algunos movimientos inestables")
    elif successful_phases >= total_phases * 0.6:
        print("⚠️ ÉXITO PARCIAL: Algunas fases fallaron, pero el control básico funciona")
    else:
        print("❌ NECESITA MEJORAS: Muchas fases fallaron")
    
    print()
    
    # Sugerencias basadas en resultados
    print("💡 SUGERENCIAS:")
    if successful_phases < total_phases:
        print("   - Considera entrenar el modelo por más tiempo")
        print("   - Verifica que los patrones PAM sean apropiados para tu robot")
        print("   - Revisa si hay problemas de estabilidad en el URDF del robot")
    
    balance_phases = [r for r in results if 'balance' in r['phase'].lower() and r['success']]
    if balance_phases:
        avg_balance_stability = np.mean([r['height_stability'] for r in balance_phases])
        if avg_balance_stability > 0.1:
            print("   - El balance base necesita mejora - considera más entrenamiento")
    
    leg_phases = [r for r in results if 'up' in r['phase'] and r['success']]
    if len(leg_phases) < 2:
        print("   - El control de piernas individuales necesita trabajo")
        print("   - Podrías ajustar los patrones PAM para movimientos más suaves")

def main():
    """
    Función principal que ejecuta todo el test paso a paso
    """
    print("🎯 TEST DE CONTROL DE UNA PIERNA")
    print("Usando tu modelo entrenado sin modificaciones")
    print("=" * 60)
    print()
    
    # Paso 1: Verificar prerequisites
    if not verificar_prerequisites():
        print("❌ No se pueden ejecutar los tests. Corrige los problemas anteriores.")
        return
    
    # Paso 2: Explicar el sistema
    explicar_patrones_pam()
    
    # Crear entorno (tu clase sin modificaciones)
    print("🏗️ Creando entorno de simulación...")
    env = Simple_Lift_Leg_BipedEnv(
        render_mode='human',    # Para que puedas ver la simulación
        action_space="pam",     # Usando tus 6 PAMs
        enable_curriculum=False # Sin curriculum para control directo
    )
    print("✅ Entorno creado exitosamente")
    print()
    
    try:
        # Paso 3: Test de balance básico
        if not test_balance_inicial(env):
            print("❌ El test de balance falló. El modelo necesita más entrenamiento.")
            return
        
        # Paso 4: Test de secuencia de piernas
        results, total_steps = test_secuencia_piernas(env)
        
        # Paso 5: Análisis de resultados
        analizar_resultados(results, total_steps)
        
    except KeyboardInterrupt:
        print("\n⏸️ Test interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔚 Cerrando entorno...")
        env.close()
        print("Test completado.")

if __name__ == "__main__":
    main()
