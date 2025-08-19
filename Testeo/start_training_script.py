#!/usr/bin/env python3
"""
SCRIPT DE ENTRENAMIENTO INICIAL
Balance & Squats con RecurrentPPO

Este script inicia el entrenamiento con configuración optimizada para aprendizaje rápido.
"""

from Gymnasium_Start.Simplified_BalanceSquat_Trainer import train_balance_and_squats, create_balance_squat_trainer
import os

def start_initial_training(timesteps=500000, n_envs=4):
    """
    Iniciar entrenamiento inicial con configuración conservadora
    """
    
    print("🚀 INICIANDO ENTRENAMIENTO BALANCE & SQUATS")
    print("="*60)
    print(f"🎯 Objetivo: Balance de pie + Sentadillas")
    print(f"🧠 Algoritmo: RecurrentPPO")
    print(f"⏱️ Timesteps: {timesteps:,}")
    print(f"🔄 Entornos paralelos: {n_envs}")
    print("="*60)
    
    # Configuración de entrenamiento inicial
    trainer, model = train_balance_and_squats(
        total_timesteps=timesteps,
        n_envs=n_envs,
        resume=True  # Permitir resume automático
    )
    
    if model is not None:
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"📁 Modelo guardado en: {trainer.model_dir}")
        print(f"📊 Logs en: {trainer.logs_dir}")
        
        # Mostrar información del modelo final
        print(f"\n📋 INFORMACIÓN DEL MODELO:")
        print(f"   Timesteps completados: {trainer.training_info.get('completed_timesteps', 'Unknown')}")
        print(f"   Learning rate: {trainer.learning_rate}")
        print(f"   LSTM hidden size: {trainer.policy_kwargs_lstm['lstm_hidden_size']}")
        
        return trainer, model
    else:
        print(f"\n❌ Entrenamiento falló o fue interrumpido")
        return None, None

def start_conservative_training():
    """Entrenamiento ultra-conservador para verificar que funciona"""
    
    print("🧪 ENTRENAMIENTO CONSERVADOR (PROOF OF CONCEPT)")
    print("="*50)
    print("Configuración mínima para verificar que el entrenamiento funciona")
    
    return start_initial_training(timesteps=100000, n_envs=1)

def start_production_training():
    """Entrenamiento de producción para resultados reales"""
    
    print("🏭 ENTRENAMIENTO DE PRODUCCIÓN")
    print("="*50)
    print("Configuración completa para entrenamiento serio")
    
    return start_initial_training(timesteps=2000000, n_envs=6)

def resume_training(timesteps_additional=500000):
    """Continuar entrenamiento existente"""
    
    print("🔄 CONTINUANDO ENTRENAMIENTO EXISTENTE")
    print("="*50)
    
    # Buscar entrenamiento existente
    model_dir = "./models_balance_squat"
    if os.path.exists(model_dir):
        print(f"📁 Directorio de modelos encontrado: {model_dir}")
        
        trainer = create_balance_squat_trainer(
            total_timesteps=timesteps_additional,
            n_envs=4
        )
        
        model = trainer.train(resume=True)
        return trainer, model
    else:
        print("❌ No se encontró entrenamiento previo")
        print("🔄 Iniciando entrenamiento nuevo...")
        return start_initial_training()

def main():
    """Función principal con menú de opciones"""
    
    print("🎯 SISTEMA DE ENTRENAMIENTO BALANCE & SQUATS")
    print("="*60)
    print("Opciones de entrenamiento:")
    print("1. 🧪 Entrenamiento conservador (100k steps, 1 env)")
    print("2. 🚀 Entrenamiento inicial (500k steps, 4 envs)")
    print("3. 🏭 Entrenamiento de producción (2M steps, 6 envs)")
    print("4. 🔄 Continuar entrenamiento existente")
    print("5. ❓ Ayuda y recomendaciones")
    print("="*60)
    
    try:
        choice = input("Elige opción (1-5) o Enter para entrenamiento inicial: ").strip()
        
        if choice == "1":
            trainer, model = start_conservative_training()
        elif choice == "3":
            trainer, model = start_production_training()
        elif choice == "4":
            trainer, model = resume_training()
        elif choice == "5":
            print_training_help()
            return
        else:  # Default o "2"
            trainer, model = start_initial_training()
        
        # Mostrar resultados
        if trainer and model:
            print(f"\n🎊 ¡ENTRENAMIENTO EXITOSO!")
            print(f"📁 Archivos guardados en: {trainer.model_dir}")
            
            # Sugerir próximos pasos
            print(f"\n💡 PRÓXIMOS PASOS SUGERIDOS:")
            print(f"   1. Evaluar modelo entrenado")
            print(f"   2. Visualizar comportamiento aprendido")
            print(f"   3. Ajustar parámetros si es necesario")
            print(f"   4. Entrenar por más tiempo si el robot no converge")
        
    except KeyboardInterrupt:
        print("\n⏸️ Entrenamiento interrumpido por el usuario")
    except Exception as e:
        print(f"\n💥 Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

def print_training_help():
    """Mostrar ayuda y recomendaciones de entrenamiento"""
    
    print("\n📚 GUÍA DE ENTRENAMIENTO BALANCE & SQUATS")
    print("="*60)
    
    print("🎯 OBJETIVOS DEL ENTRENAMIENTO:")
    print("   1. El robot debe mantener equilibrio de pie por 30+ segundos")
    print("   2. El robot debe realizar sentadillas controladas y volver a balance")
    print("   3. Los músculos PAM deben trabajar eficientemente")
    
    print("\n⚙️ CONFIGURACIONES RECOMENDADAS:")
    print("   🧪 Conservador: Para verificar que funciona (30 min)")
    print("   🚀 Inicial: Para primeros resultados (2-3 horas)")
    print("   🏭 Producción: Para resultados finales (8-12 horas)")
    
    print("\n📊 MÉTRICAS DE ÉXITO:")
    print("   ✅ Recompensa promedio > 5.0")
    print("   ✅ Duración de episodio > 1000 steps")
    print("   ✅ Robot no se cae por 30+ segundos")
    
    print("\n🔧 SOLUCIÓN DE PROBLEMAS:")
    print("   Si recompensa muy negativa → Ajustar pesos de recompensa")
    print("   Si robot se cae mucho → Reducir learning rate")
    print("   Si no aprende → Aumentar timesteps o envs paralelos")
    print("   Si entrenamiento muy lento → Reducir n_envs")
    
    print("\n💡 CONSEJOS:")
    print("   1. Empezar siempre con entrenamiento conservador")
    print("   2. Monitorear logs en TensorBoard")
    print("   3. Guardar checkpoints frecuentemente")
    print("   4. Evaluar modelo cada 50k-100k steps")

if __name__ == "__main__":
    main()
