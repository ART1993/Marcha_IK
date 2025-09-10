#!/usr/bin/env python3
"""
SCRIPT DE PRUEBA: Ejecutar modelo entrenado para control de una pierna
Versión mejorada que carga y usa tu modelo RecurrentPPO entrenado
"""

import numpy as np
import time
import os
import glob
from collections import deque

# Importar stable_baselines3 para cargar el modelo
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Importar tu entorno
from Gymnasium_Start.Simple_Lift_Leg_BipedEnv import Simple_Lift_Leg_BipedEnv

class TrainedModelTester:
    """
    Clase para probar tu modelo entrenado de equilibrio en una pierna
    """
    
    def __init__(self):
        self.model = None
        self.env = None
        self.vec_env = None
        self.model_path = None
        self.normalize_path = None
        
    def find_best_model(self):
        """Encontrar el mejor modelo entrenado"""
        
        print("🔍 Buscando modelo entrenado...")
        
        # Buscar en diferentes ubicaciones posibles
        search_paths = [
            "./models_lift_leg/best_model.zip",
            "./models_lift_leg/single_leg_balance_pam_final.zip", 
            "./models_lift_leg/checkpoints/single_leg_balance_pam_checkpoint_*.zip"
        ]
        
        found_models = []
        
        for pattern in search_paths:
            if "*" in pattern:
                # Usar glob para patrones con wildcards
                matches = glob.glob(pattern)
                found_models.extend(matches)
            else:
                # Verificar archivo específico
                if os.path.exists(pattern):
                    found_models.append(pattern)
        
        if not found_models:
            print("❌ ERROR: No se encontró ningún modelo entrenado")
            print("   Ubicaciones buscadas:")
            for path in search_paths:
                print(f"   - {path}")
            print("\n💡 Solución: Ejecuta primero el entrenamiento:")
            print("   python inicio_programa.py")
            return False
        
        # Elegir el mejor modelo (el más reciente o best_model si existe)
        if any("best_model" in path for path in found_models):
            self.model_path = next(path for path in found_models if "best_model" in path)
        else:
            # Elegir el checkpoint más reciente
            checkpoint_models = [p for p in found_models if "checkpoint" in p]
            if checkpoint_models:
                # Extraer números de steps y elegir el mayor
                def extract_steps(path):
                    import re
                    match = re.search(r'(\d+)_steps', path)
                    return int(match.group(1)) if match else 0
                
                self.model_path = max(checkpoint_models, key=extract_steps)
            else:
                self.model_path = found_models[0]
        
        print(f"✅ Modelo encontrado: {self.model_path}")
        
        # Buscar archivo de normalización
        model_dir = os.path.dirname(self.model_path)
        norm_patterns = [
            os.path.join(model_dir, "single_leg_balance_pam_normalize.pkl"),
            os.path.join(model_dir, "*_normalize.pkl")
        ]
        
        for pattern in norm_patterns:
            if "*" in pattern:
                matches = glob.glob(pattern)
                if matches:
                    self.normalize_path = matches[0]
                    break
            else:
                if os.path.exists(pattern):
                    self.normalize_path = pattern
                    break
        
        if self.normalize_path:
            print(f"✅ Normalización encontrada: {self.normalize_path}")
        else:
            print("⚠️ No se encontró archivo de normalización (continuando sin él)")
        
        return True
    
    def load_model_and_env(self):
        """Cargar modelo y crear entorno"""
        
        print("🏗️ Creando entorno...")
        
        # Crear entorno base (igual que en entrenamiento)
        base_env = Simple_Lift_Leg_BipedEnv(
            render_mode='human',  # Modo visual para ver la simulación
            action_space="pam",
            enable_curriculum=False  # Sin curriculum para testing
        )
        
        # Envolver en VecEnv (como en entrenamiento)
        self.vec_env = DummyVecEnv([lambda: base_env])
        
        # Aplicar normalización si existe
        if self.normalize_path and os.path.exists(self.normalize_path):
            print(f"📊 Cargando normalización...")
            try:
                self.vec_env = VecNormalize.load(self.normalize_path, self.vec_env)
                self.vec_env.training = False  # Modo evaluación
                self.vec_env.norm_reward = False  # No normalizar rewards en test
                print("✅ Normalización cargada correctamente")
            except Exception as e:
                print(f"⚠️ Error cargando normalización: {e}")
                print("   Continuando sin normalización...")
                self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=False, training=False)
        else:
            # Crear normalización básica
            self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=False, training=False)
        
        print("🧠 Cargando modelo entrenado...")
        
        try:
            # Cargar modelo RecurrentPPO
            self.model = RecurrentPPO.load(self.model_path, env=self.vec_env)
            print("✅ Modelo cargado correctamente")
            
            # Información del modelo
            print(f"📋 Información del modelo:")
            print(f"   Algoritmo: {self.model.__class__.__name__}")
            print(f"   Observaciones: {self.model.observation_space}")
            print(f"   Acciones: {self.model.action_space}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def run_single_episode(self, episode_duration=30.0, show_details=True):
        """
        Ejecutar un episodio completo usando el modelo entrenado
        
        Args:
            episode_duration: Duración en segundos
            show_details: Mostrar detalles durante la ejecución
        """
        
        print(f"\n🚀 Iniciando episodio de {episode_duration}s...")
        
        # Reset del entorno
        obs = self.vec_env.reset()
        
        # Para modelos recurrentes, necesitamos estados LSTM
        lstm_states = None
        
        # Variables de tracking
        step_count = 0
        start_time = time.time()
        rewards = []
        heights = []
        leg_switches = 0
        last_target_leg = None
        
        # Información de contactos
        contact_history = deque(maxlen=100)
        
        while time.time() - start_time < episode_duration:
            
            # Predecir acción usando el modelo entrenado
            action, lstm_states = self.model.predict(
                obs, 
                state=lstm_states,
                deterministic=True  # Usar política determinística para testing
            )
            
            # Ejecutar acción en el entorno
            obs, reward, done, info = self.vec_env.step(action)
            
            step_count += 1
            rewards.append(reward[0])
            
            # Extraer información del entorno base
            base_env = self.vec_env.envs[0]
            robot_pos, robot_orn = base_env.pos, base_env.euler
            heights.append(robot_pos[2])
            
            # Detectar cambios de pierna objetivo
            if hasattr(base_env, 'simple_reward_system') and base_env.simple_reward_system:
                current_target = base_env.simple_reward_system.target_leg
                if last_target_leg and current_target != last_target_leg:
                    leg_switches += 1
                last_target_leg = current_target
            
            # Tracking de contactos
            left_contact, right_contact = base_env.contacto_pies
            contact_history.append((left_contact, right_contact))
            
            # Mostrar progreso cada 2 segundos
            if show_details and step_count % 800 == 0:  # ~2s a 400Hz
                elapsed = time.time() - start_time
                remaining = episode_duration - elapsed
                
                print(f"   ⏱️  {elapsed:.1f}s - Altura: {robot_pos[2]:.2f}m - "
                      f"Reward: {reward[0]:.2f} - Restante: {remaining:.1f}s")
                
                # Mostrar información de equilibrio
                if hasattr(base_env, 'simple_reward_system') and base_env.simple_reward_system:
                    target = base_env.simple_reward_system.target_leg
                    print(f"           Pierna objetivo: {target} - "
                          f"Contactos: L={left_contact}, R={right_contact}")
            
            # Verificar si el episodio terminó
            if done[0]:
                print(f"   🛑 Episodio terminado en {time.time() - start_time:.1f}s")
                break
        
        # Estadísticas finales
        total_time = time.time() - start_time
        avg_reward = np.mean(rewards) if rewards else 0
        avg_height = np.mean(heights) if heights else 0
        height_stability = np.std(heights) if len(heights) > 1 else 0
        
        # Análisis de contactos
        contact_changes = 0
        balance_quality = 0
        
        if len(contact_history) > 1:
            for i in range(1, len(contact_history)):
                if contact_history[i] != contact_history[i-1]:
                    contact_changes += 1
            
            # Calidad del equilibrio: proporción de tiempo con un solo pie
            single_foot_frames = sum(1 for l, r in contact_history if (l and not r) or (r and not l))
            balance_quality = single_foot_frames / len(contact_history) if contact_history else 0
        
        results = {
            'duration': total_time,
            'steps': step_count,
            'avg_reward': avg_reward,
            'total_reward': sum(rewards),
            'avg_height': avg_height,
            'height_stability': height_stability,
            'leg_switches': leg_switches,
            'contact_changes': contact_changes,
            'balance_quality': balance_quality,
            'success': not done[0] or total_time >= episode_duration * 0.95
        }
        
        return results
    
    def run_multiple_episodes(self, num_episodes=5, episode_duration=20.0):
        """Ejecutar múltiples episodios y analizar resultados"""
        
        print(f"\n🎯 EVALUACIÓN COMPLETA: {num_episodes} episodios de {episode_duration}s")
        print("=" * 70)
        
        all_results = []
        
        for episode in range(num_episodes):
            print(f"\n📍 EPISODIO {episode + 1}/{num_episodes}")
            print("-" * 40)
            
            try:
                results = self.run_single_episode(episode_duration, show_details=True)
                all_results.append(results)
                
                # Resumen del episodio
                status = "✅ ÉXITO" if results['success'] else "❌ FALLO"
                print(f"\n   {status}")
                print(f"   Duración: {results['duration']:.1f}s")
                print(f"   Reward promedio: {results['avg_reward']:.2f}")
                print(f"   Altura promedio: {results['avg_height']:.2f}m")
                print(f"   Estabilidad: ±{results['height_stability']:.3f}m")
                print(f"   Cambios de pierna: {results['leg_switches']}")
                print(f"   Calidad equilibrio: {results['balance_quality']:.1%}")
                
            except KeyboardInterrupt:
                print("\n⏸️ Evaluación interrumpida por el usuario")
                break
            except Exception as e:
                print(f"\n❌ Error en episodio {episode + 1}: {e}")
                continue
        
        if all_results:
            self._analyze_overall_results(all_results)
        
        return all_results
    
    def _analyze_overall_results(self, results):
        """Analizar resultados globales"""
        
        print(f"\n📊 ANÁLISIS GLOBAL")
        print("=" * 70)
        
        # Estadísticas básicas
        successful_episodes = sum(1 for r in results if r['success'])
        total_episodes = len(results)
        success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0
        
        avg_duration = np.mean([r['duration'] for r in results])
        avg_reward = np.mean([r['avg_reward'] for r in results])
        avg_height = np.mean([r['avg_height'] for r in results])
        avg_stability = np.mean([r['height_stability'] for r in results])
        avg_leg_switches = np.mean([r['leg_switches'] for r in results])
        avg_balance_quality = np.mean([r['balance_quality'] for r in results])
        
        print(f"🏆 RESUMEN GENERAL:")
        print(f"   Episodios exitosos: {successful_episodes}/{total_episodes} ({success_rate:.1%})")
        print(f"   Duración promedio: {avg_duration:.1f}s")
        print(f"   Reward promedio: {avg_reward:.2f}")
        print(f"   Altura promedio: {avg_height:.2f}m")
        print(f"   Estabilidad promedio: ±{avg_stability:.3f}m")
        print(f"   Cambios de pierna promedio: {avg_leg_switches:.1f}")
        print(f"   Calidad de equilibrio: {avg_balance_quality:.1%}")
        
        # Evaluación de calidad
        print(f"\n🎖️ EVALUACIÓN DE CALIDAD:")
        
        if success_rate >= 0.8:
            print("   🌟 EXCELENTE: El modelo funciona muy bien")
        elif success_rate >= 0.6:
            print("   👍 BUENO: El modelo funciona bien con algunas fallas")
        elif success_rate >= 0.4:
            print("   👌 ACEPTABLE: El modelo funciona pero necesita mejoras")
        else:
            print("   ⚠️ NECESITA TRABAJO: El modelo necesita más entrenamiento")
        
        if avg_stability < 0.05:
            print("   🎯 Movimientos muy estables")
        elif avg_stability < 0.1:
            print("   ✅ Movimientos estables")
        else:
            print("   ⚠️ Movimientos algo inestables")
        
        if avg_balance_quality > 0.7:
            print("   🦵 Excelente control de equilibrio en una pierna")
        elif avg_balance_quality > 0.5:
            print("   🦵 Buen control de equilibrio en una pierna")
        else:
            print("   🦵 Control de equilibrio necesita mejoras")
        
        # Sugerencias
        print(f"\n💡 SUGERENCIAS:")
        if success_rate < 0.6:
            print("   - Considera entrenar el modelo por más tiempo")
            print("   - Verifica los hiperparámetros del entrenamiento")
        
        if avg_stability > 0.1:
            print("   - El balance base podría mejorar con más entrenamiento")
        
        if avg_balance_quality < 0.5:
            print("   - El control específico de una pierna necesita trabajo")
            print("   - Considera ajustar el curriculum de entrenamiento")
        
        if avg_leg_switches < 1:
            print("   - El modelo no está alternando entre piernas efectivamente")
    
    def cleanup(self):
        """Limpiar recursos"""
        if self.vec_env:
            self.vec_env.close()

def main():
    """Función principal"""
    
    print("🎯 EVALUACIÓN DEL MODELO ENTRENADO")
    print("Control de equilibrio en una pierna con músculos PAM")
    print("=" * 70)
    
    tester = TrainedModelTester()
    
    try:
        # Paso 1: Encontrar modelo
        if not tester.find_best_model():
            return
        
        # Paso 2: Cargar modelo y entorno
        if not tester.load_model_and_env():
            return
        
        print("\n" + "=" * 70)
        print("🎮 OPCIONES DE EVALUACIÓN:")
        print("1. Episodio único largo (60s)")
        print("2. Evaluación estándar (5 episodios de 20s)")
        print("3. Evaluación intensiva (10 episodios de 15s)")
        print("4. Demo rápido (1 episodio de 30s)")
        
        try:
            choice = input("\nElige una opción (1-4) [Enter = 2]: ").strip()
            if not choice:
                choice = "2"
        except:
            choice = "2"
        
        # Ejecutar evaluación según elección
        if choice == "1":
            print("\n🎯 Ejecutando episodio único largo...")
            tester.run_single_episode(episode_duration=60.0, show_details=True)
        
        elif choice == "3":
            print("\n🎯 Ejecutando evaluación intensiva...")
            tester.run_multiple_episodes(num_episodes=10, episode_duration=15.0)
        
        elif choice == "4":
            print("\n🎯 Ejecutando demo rápido...")
            tester.run_single_episode(episode_duration=30.0, show_details=True)
        
        else:  # choice == "2" o cualquier otra cosa
            print("\n🎯 Ejecutando evaluación estándar...")
            tester.run_multiple_episodes(num_episodes=5, episode_duration=20.0)
        
        print(f"\n🎉 Evaluación completada!")
        print(f"   Modelo usado: {tester.model_path}")
        
    except KeyboardInterrupt:
        print("\n⏸️ Evaluación interrumpida por el usuario")
    
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔚 Cerrando entorno...")
        tester.cleanup()
        print("Evaluación completada.")

if __name__ == "__main__":
    main()
