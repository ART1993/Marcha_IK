#!/usr/bin/env python3
"""
SISTEMA COMPLETO: Entrena el modelo y luego lo prueba
Integra entrenamiento y testing en un solo script
"""

import os
import sys
import time

from Testeo_Modelo.quick_test_model import quick_test_model

def main():
    """Sistema completo de entrenamiento y testing"""
    
    print("🤖 SISTEMA COMPLETO DE CONTROL DE PIERNAS")
    print("Robot bípedo con músculos PAM - Equilibrio en una pierna")
    print("=" * 60)
    
    print("\n📋 OPCIONES DISPONIBLES:")
    print("1. Solo entrenar modelo (2M timesteps)")
    print("2. Solo entrenar modelo (500K timesteps - rápido)")
    print("3. Solo probar modelo existente")
    print("4. Entrenar y probar (completo)")
    print("5. Entrenar rápido y probar")
    print("6. Evaluación detallada del modelo")
    
    try:
        choice = input("\nElige una opción (1-6) [Enter = 4]: ").strip()
        if not choice:
            choice = "4"
    except:
        choice = "4"
    
    print(f"\n🎯 Ejecutando opción {choice}...")
    
    # Función para verificar si hay modelo entrenado
    def model_exists():
        model_paths = [
            "./models_lift_leg/best_model.zip",
            "./models_lift_leg/single_leg_balance_pam_final.zip"
        ]
        return any(os.path.exists(path) for path in model_paths)
    
    try:
        if choice in ["1", "2", "4", "5"]:
            # FASE DE ENTRENAMIENTO
            print("\n" + "="*60)
            print("🏋️ FASE 1: ENTRENAMIENTO")
            print("="*60)
            
            # Determinar timesteps
            if choice in ["2", "5"]:
                timesteps = 500000
                print("⚡ Entrenamiento RÁPIDO (500K timesteps)")
            else:
                timesteps = 2000000
                print("🎯 Entrenamiento COMPLETO (2M timesteps)")
            
            # Importar y ejecutar entrenamiento
            from inicio_programa import train_balance_pure_rl
            
            print(f"🚀 Iniciando entrenamiento con {timesteps:,} timesteps...")
            start_time = time.time()
            
            trainer, model = train_balance_pure_rl(
                total_timesteps=timesteps,
                n_envs=4,
                resume=True
            )
            
            train_time = time.time() - start_time
            print(f"\n⏱️ Entrenamiento completado en {train_time/60:.1f} minutos")
            
            if model:
                print("✅ Modelo entrenado exitosamente")
            else:
                print("❌ Error en el entrenamiento")
                return
        
        if choice in ["3", "4", "5", "6"]:
            # Verificar que existe modelo
            if not model_exists():
                print("\n❌ No se encontró modelo entrenado")
                print("💡 Ejecuta primero el entrenamiento")
                return
            
            # FASE DE TESTING
            print("\n" + "="*60)
            print("🧪 FASE 2: TESTING DEL MODELO")
            print("="*60)
            
            if choice == "6":
                # Evaluación detallada
                print("🔬 Ejecutando evaluación detallada...")
                os.system("python test_trained_model_leg_control.py")
            
            elif choice == "3":
                # Solo testing rápido
                print("⚡ Ejecutando test rápido (30s)...")
                
                quick_test_model(30)
            
            else:
                # Testing estándar después de entrenamiento
                print("🎮 Ejecutando test estándar...")
                
                # Pequeña pausa para que el usuario vea el mensaje
                print("   (Ejecutando en 3 segundos...)")
                time.sleep(3)
                
                quick_test_model(45)
                
                print(f"\n💡 Para evaluación más detallada, ejecuta:")
                print(f"   python test_trained_model_leg_control.py")
        
        print(f"\n🎉 ¡SISTEMA COMPLETADO EXITOSAMENTE!")
        
        # Mostrar ubicaciones importantes
        print(f"\n📁 ARCHIVOS IMPORTANTES:")
        if os.path.exists("./models_lift_leg"):
            print(f"   Modelos: ./models_lift_leg/")
        if os.path.exists("./logs_lift_leg"):
            print(f"   Logs: ./logs_lift_leg/")
        
        print(f"\n🎮 COMANDOS ÚTILES:")
        print(f"   Test rápido: python quick_test.py")
        print(f"   Test completo: python test_trained_model_leg_control.py")
        print(f"   Solo entrenar: python inicio_programa.py")
        print(f"   Tensorboard: tensorboard --logdir=logs_lift_leg")
        
    except KeyboardInterrupt:
        print(f"\n⏸️ Sistema interrumpido por el usuario")
    
    except Exception as e:
        print(f"\n❌ Error en el sistema: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n🔚 Sistema finalizado")

if __name__ == "__main__":
    main()
