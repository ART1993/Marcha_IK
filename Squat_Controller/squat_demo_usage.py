#!/usr/bin/env python3
"""
EJEMPLO DE USO - CONTROLADOR DE SENTADILLAS
Ejemplos prácticos de cómo usar el SquatController con tu modelo entrenado
"""

import os
import time
from Squat_Controller.squat_controller import (
    SquatController, 
    SquatConfig, 
    SquatQuality, 
    create_squat_controller,
    demo_squat_controller
)

def ejemplo_basico():
    """Ejemplo básico: Una sola sentadilla"""
    
    print("📝 EJEMPLO 1: Sentadilla Básica")
    print("=" * 40)
    
    # Crear controlador con configuración por defecto
    controller = create_squat_controller()
    
    try:
        # Iniciar sesión
        if controller.start_squat_session():
            
            # Ejecutar una sentadilla
            metrics = controller.execute_single_squat()
            
            # Mostrar resultados
            print(f"\n🏆 Resultados:")
            print(f"  Calidad: {metrics.quality.value}")
            print(f"  Duración: {metrics.duration:.1f}s")
            print(f"  Profundidad máxima: {metrics.max_depth:.2f}m")
            print(f"  Estabilidad: {metrics.stability_score:.2f}")
            print(f"  Completitud: {metrics.completion_rate:.1%}")
        
        # Detener sesión
        controller.stop_squat_session()
        
    finally:
        controller.close()

def ejemplo_secuencia_personalizada():
    """Ejemplo avanzado: Secuencia personalizada de sentadillas"""
    
    print("\n📝 EJEMPLO 2: Secuencia Personalizada")
    print("=" * 40)
    
    # Configuración personalizada
    config = SquatConfig(
        target_depth=0.6,           # Sentadillas más profundas
        hold_duration=3.0,          # 3 segundos en posición baja
        descent_speed=0.2,          # Descenso más lento
        ascent_speed=0.3,           # Ascenso controlado
        use_expert_backup=True,     # Con respaldo experto
        max_squat_duration=20.0     # Más tiempo por sentadilla
    )
    
    # Crear controlador
    controller = create_squat_controller(config=config)
    
    try:
        # Ejecutar secuencia de 5 sentadillas con descanso de 4 segundos
        print("🏋️ Ejecutando 5 sentadillas con configuración personalizada...")
        
        metrics_list = controller.execute_squat_sequence(
            num_squats=5,
            rest_time=4.0
        )
        
        # Análisis detallado de resultados
        analizar_resultados_detallados(metrics_list)
        
    except Exception as e:
        print(f"❌ Error durante ejecución: {e}")
        
    finally:
        controller.close()

def ejemplo_entrenamiento_progresivo():
    """Ejemplo: Entrenamiento progresivo con diferentes dificultades"""
    
    print("\n📝 EJEMPLO 3: Entrenamiento Progresivo")
    print("=" * 40)
    
    # Diferentes configuraciones de dificultad
    configuraciones = [
        ("🟢 Fácil", SquatConfig(target_depth=0.8, hold_duration=1.0, use_expert_backup=True)),
        ("🟡 Moderado", SquatConfig(target_depth=0.7, hold_duration=2.0, use_expert_backup=True)),
        ("🔴 Difícil", SquatConfig(target_depth=0.6, hold_duration=3.0, use_expert_backup=False)),
    ]
    
    resultados_progresion = {}
    
    for nivel, config in configuraciones:
        print(f"\n--- {nivel} ---")
        
        controller = create_squat_controller(config=config)
        
        try:
            # 3 sentadillas por nivel
            metrics_list = controller.execute_squat_sequence(
                num_squats=3,
                rest_time=2.0
            )
            
            # Guardar resultados
            resultados_progresion[nivel] = {
                'metrics': metrics_list,
                'promedio_calidad': calcular_promedio_calidad(metrics_list),
                'tasa_exito': sum(1 for m in metrics_list if m.completion_rate >= 1.0) / len(metrics_list)
            }
            
            print(f"  Tasa de éxito: {resultados_progresion[nivel]['tasa_exito']:.1%}")
            
        except Exception as e:
            print(f"  ❌ Error en nivel {nivel}: {e}")
            resultados_progresion[nivel] = None
            
        finally:
            controller.close()
    
    # Mostrar resumen de progresión
    mostrar_resumen_progresion(resultados_progresion)

def ejemplo_monitoreo_tiempo_real():
    """Ejemplo: Monitoreo y análisis en tiempo real"""
    
    print("\n📝 EJEMPLO 4: Monitoreo en Tiempo Real")
    print("=" * 40)
    
    # Configuración con feedback detallado
    config = SquatConfig(
        target_depth=0.7,
        hold_duration=2.0,
        use_expert_backup=True
    )
    
    controller = create_squat_controller(config=config)
    
    try:
        if not controller.start_squat_session():
            return
        
        print("🔍 Iniciando monitoreo en tiempo real...")
        print("📊 Métricas se actualizarán cada segundo durante la sentadilla")
        
        # Hook personalizado para monitoreo
        monitor_squat_en_tiempo_real(controller)
        
    finally:
        controller.close()

def monitor_squat_en_tiempo_real(controller):
    """Función auxiliar para monitoreo en tiempo real"""
    
    # Resetear para nueva sentadilla
    controller._reset_squat_metrics()
    controller.current_phase = controller.current_phase.PREPARATION
    controller.squat_start_time = time.time()
    
    # Cambiar a modo sentadilla si hay controlador experto
    if controller.expert_controller:
        controller.expert_controller.set_action(controller.env.ActionType.SQUAT)
    
    print("\n🏋️ Sentadilla iniciada - Monitoreando...")
    print("Fase | Tiempo | Altura | Estabilidad | Estado")
    print("-" * 50)
    
    last_update = time.time()
    start_time = time.time()
    
    while time.time() - start_time < 15.0:  # Máximo 15 segundos
        
        # Obtener observación
        current_obs = controller.env.last_observation if hasattr(controller.env, 'last_observation') else None
        if current_obs is None:
            break
        
        # Obtener acción
        action = controller._get_squat_action(current_obs)
        
        # Ejecutar paso
        obs, reward, done, truncated, info = controller.env.step(action)
        
        # Actualizar métricas
        controller._update_metrics(obs, info)
        controller._update_squat_phase()
        
        # Mostrar actualización cada segundo
        if time.time() - last_update >= 1.0:
            altura = obs[1] if len(obs) > 1 else 1.0
            estabilidad = controller.current_metrics['current_stability']
            fase = controller.current_phase.value[:8]  # Primeros 8 caracteres
            tiempo = int(time.time() - start_time)
            
            status = "✅" if estabilidad > 0.7 else "⚠️" if estabilidad > 0.4 else "❌"
            
            print(f"{fase:<8} | {tiempo:4}s | {altura:5.2f}m | {estabilidad:8.2f} | {status}")
            last_update = time.time()
        
        # Verificar terminación
        if done or controller._is_squat_completed():
            print(f"🏁 Sentadilla completada en {time.time() - start_time:.1f}s")
            break
    
    # Mostrar métricas finales
    print(f"\n📊 Métricas Finales:")
    print(f"  Profundidad máxima: {controller.current_metrics['max_depth_reached']:.2f}m")
    print(f"  Estabilidad promedio: {controller.current_metrics['current_stability']:.2f}")

def analizar_resultados_detallados(metrics_list):
    """Análisis detallado de una lista de métricas"""
    
    if not metrics_list:
        print("❌ No hay métricas para analizar")
        return
    
    print(f"\n🔍 ANÁLISIS DETALLADO")
    print("=" * 30)
    
    # Estadísticas por sentadilla
    for i, metrics in enumerate(metrics_list, 1):
        print(f"\nSentadilla #{i}:")
        print(f"  📊 Calidad: {metrics.quality.value}")
        print(f"  ⏱️ Duración: {metrics.duration:.1f}s")
        print(f"  📏 Profundidad: {metrics.max_depth:.2f}m")
        print(f"  ⚖️ Estabilidad: {metrics.stability_score:.2f}")
        print(f"  🎯 Suavidad: {metrics.smoothness_score:.2f}")
        print(f"  ✅ Completitud: {metrics.completion_rate:.1%}")
    
    # Estadísticas globales
    duraciones = [m.duration for m in metrics_list]
    estabilidades = [m.stability_score for m in metrics_list]
    profundidades = [m.max_depth for m in metrics_list]
    
    print(f"\n📈 ESTADÍSTICAS GLOBALES:")
    print(f"  Duración promedio: {sum(duraciones)/len(duraciones):.1f}s")
    print(f"  Estabilidad promedio: {sum(estabilidades)/len(estabilidades):.2f}")
    print(f"  Profundidad promedio: {sum(profundidades)/len(profundidades):.2f}m")
    print(f"  Tasa de éxito: {sum(1 for m in metrics_list if m.completion_rate >= 1.0)/len(metrics_list):.1%}")

def calcular_promedio_calidad(metrics_list):
    """Calcular calidad promedio de una lista de métricas"""
    
    if not metrics_list:
        return 0.0
    
    # Mapear calidades a valores numéricos
    calidad_valores = {
        SquatQuality.EXCELLENT: 4,
        SquatQuality.GOOD: 3,
        SquatQuality.FAIR: 2,
        SquatQuality.POOR: 1
    }
    
    valores = [calidad_valores[m.quality] for m in metrics_list]
    return sum(valores) / len(valores)

def mostrar_resumen_progresion(resultados):
    """Mostrar resumen de la progresión de dificultad"""
    
    print(f"\n🏆 RESUMEN DE PROGRESIÓN")
    print("=" * 40)
    
    for nivel, datos in resultados.items():
        if datos is None:
            print(f"{nivel}: ❌ Falló")
            continue
        
        calidad_promedio = datos['promedio_calidad']
        tasa_exito = datos['tasa_exito']
        
        # Emoji según rendimiento
        if tasa_exito >= 0.8 and calidad_promedio >= 3.0:
            emoji = "🌟"
        elif tasa_exito >= 0.6:
            emoji = "👍"
        elif tasa_exito >= 0.3:
            emoji = "👌"
        else:
            emoji = "📈"
        
        print(f"{nivel}: {emoji} Éxito {tasa_exito:.1%}, Calidad {calidad_promedio:.1f}/4")
    
    print(f"\n💡 Recomendación: Basándote en estos resultados, ajusta la configuración")
    print(f"   para encontrar el nivel óptimo de dificultad para tu robot.")

def ejemplo_completo_con_guardado():
    """Ejemplo completo que guarda resultados en archivo"""
    
    print("\n📝 EJEMPLO 5: Sesión Completa con Guardado")
    print("=" * 40)
    
    # Ejecutar sesión completa
    config = SquatConfig(target_depth=0.7, hold_duration=2.0, use_expert_backup=True)
    controller = create_squat_controller(config=config)
    
    try:
        # Ejecutar 10 sentadillas
        metrics_list = controller.execute_squat_sequence(num_squats=10, rest_time=2.0)
        
        # Obtener estadísticas
        stats = controller.get_performance_stats()
        
        # Guardar resultados
        import json
        from datetime import datetime
        
        resultados = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'target_depth': config.target_depth,
                'hold_duration': config.hold_duration,
                'use_expert_backup': config.use_expert_backup
            },
            'statistics': stats,
            'individual_metrics': [
                {
                    'quality': m.quality.value,
                    'duration': m.duration,
                    'max_depth': m.max_depth,
                    'stability_score': m.stability_score,
                    'completion_rate': m.completion_rate
                }
                for m in metrics_list
            ]
        }
        
        # Guardar en archivo
        filename = f"squat_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(resultados, f, indent=2)
        
        print(f"💾 Resultados guardados en: {filename}")
        
    finally:
        controller.close()

if __name__ == "__main__":
    """Ejecutar todos los ejemplos"""
    
    print("🏋️ EJEMPLOS DE USO - CONTROLADOR DE SENTADILLAS")
    print("=" * 60)
    
    # Verificar que el modelo existe
    model_path = "./models_balance_squat/best_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   Ejecuta primero el entrenamiento para generar el modelo")
        exit(1)
    
    try:
        # Ejemplo 1: Básico
        ejemplo_basico()
        
        input("\nPresiona Enter para continuar al siguiente ejemplo...")
        
        # Ejemplo 2: Secuencia personalizada
        ejemplo_secuencia_personalizada()
        
        input("\nPresiona Enter para continuar al siguiente ejemplo...")
        
        # Ejemplo 3: Entrenamiento progresivo
        ejemplo_entrenamiento_progresivo()
        
        input("\nPresiona Enter para continuar al siguiente ejemplo...")
        
        # Ejemplo 4: Monitoreo tiempo real
        ejemplo_monitoreo_tiempo_real()
        
        input("\nPresiona Enter para el ejemplo final...")
        
        # Ejemplo 5: Sesión completa
        ejemplo_completo_con_guardado()
        
        print(f"\n🎉 ¡Todos los ejemplos completados exitosamente!")
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Ejemplos interrumpidos por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante ejemplos: {e}")
        import traceback
        traceback.print_exc()