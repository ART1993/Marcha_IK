#!/usr/bin/env python3
"""
SCRIPT RÁPIDO: Prueba simple del modelo entrenado
Ejecuta el modelo y muestra si puede levantar las piernas
"""

import numpy as np
import time
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv

def quick_test_model(model_path= "./models_lift_leg/best_model.zip", duration=30):
    """
    Prueba rápida del modelo entrenado
    
    Args:
        duration: Duración en segundos del test
    """
    
    print("🚀 PRUEBA RÁPIDA DEL MODELO ENTRENADO")
    print("=" * 50)
    
    # Buscar modelo
    if not os.path.exists(model_path):
        # Buscar checkpoint más reciente
        import glob
        checkpoints = glob.glob("./models_lift_leg/checkpoints/*_checkpoint_*.zip")
        if checkpoints:
            # Obtener el más reciente por fecha de modificación
            model_path = max(checkpoints, key=os.path.getmtime)
        else:
            print("❌ No se encontró modelo entrenado")
            print("💡 Ejecuta primero: python inicio_programa.py")
            return
    
    print(f"📂 Usando modelo: {os.path.basename(model_path)}")
    
    # Crear entorno
    print("🏗️ Creando entorno...")
    env = Simple_Lift_Leg_BipedEnv(
        render_mode='human',
        action_space="pam", 
        enable_curriculum=False
    )
    
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, training=False)
    
    # Cargar normalización si existe
    norm_path = "./models_lift_leg/single_leg_balance_pam_normalize.pkl"
    if os.path.exists(norm_path):
        print("📊 Cargando normalización...")
        try:
            vec_env = VecNormalize.load(norm_path, vec_env)
            vec_env.training = False
        except:
            print("⚠️ Error con normalización, usando básica")
    
    # Cargar modelo
    print("🧠 Cargando modelo...")
    try:
        model = RecurrentPPO.load(model_path, env=vec_env)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        vec_env.close()
        return
    
    # Ejecutar test
    print(f"🎮 Ejecutando test de {duration}s...")
    print("   Observa si el robot:")
    print("   • Mantiene equilibrio")
    print("   • Levanta piernas alternadamente")
    print("   • Se mantiene estable")
    print()
    
    obs = vec_env.reset()
    lstm_states = None
    
    start_time = time.time()
    step_count = 0
    rewards = []
    leg_lifts_detected = 0
    last_contacts = (True, True)
    
    try:
        while time.time() - start_time < duration:
            # Predecir acción
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            
            # Ejecutar
            obs, reward, done, info = vec_env.step(action)
            
            step_count += 1
            rewards.append(reward[0])
            
            # Detectar levantamiento de piernas
            current_contacts = env.contacto_pies
            if last_contacts != current_contacts:
                # Cambió el patrón de contacto
                if (last_contacts[0] and last_contacts[1]) and not (current_contacts[0] and current_contacts[1]):
                    leg_lifts_detected += 1
                    print(f"   🦵 Pierna levantada detectada! (#{leg_lifts_detected})")
            last_contacts = current_contacts
            
            # Mostrar progreso cada 5s
            if step_count % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"   ⏱️ {elapsed:.0f}s - Altura: {env.pos[2]:.2f}m - Reward: {reward[0]:.1f}")
            
            if done[0]:
                print("   🛑 Episodio terminado (robot se cayó)")
                break
                
    except KeyboardInterrupt:
        print("\n⏸️ Test interrumpido")
    
    # Resultados
    elapsed = time.time() - start_time
    avg_reward = np.mean(rewards) if rewards else 0
    
    print(f"\n📊 RESULTADOS:")
    print(f"   Duración: {elapsed:.1f}s / {duration}s")
    print(f"   Pasos ejecutados: {step_count}")
    print(f"   Reward promedio: {avg_reward:.2f}")
    print(f"   Levantamientos detectados: {leg_lifts_detected}")
    
    # Evaluación simple
    success = elapsed >= duration * 0.9 and avg_reward > 1.0
    
    if success:
        print("🎉 ¡ÉXITO! El modelo puede controlar las piernas")
        if leg_lifts_detected >= 2:
            print("   ✅ Detectó múltiples levantamientos de piernas")
        else:
            print("   ⚠️ Pocos levantamientos detectados")
    else:
        print("⚠️ El modelo necesita más entrenamiento")
        if elapsed < duration * 0.5:
            print("   • Robot se cayó muy pronto")
        if avg_reward < 1.0:
            print("   • Reward muy bajo")
    
    vec_env.close()
    print("\n✅ Test completado")

if __name__ == "__main__":
    import sys
    
    # Permitir especificar duración como argumento
    duration = 30
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            print("⚠️ Duración inválida, usando 30s")
    
    quick_test_model(duration)
