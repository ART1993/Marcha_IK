import multiprocessing as mp
import os
import sys
import platform
import logging
from typing import Optional

def _setup_multiprocessing(force_method: Optional[str] = None, 
                          verbose: bool = True,
                          safety_checks: bool = True) -> bool:
    """
    Configuración robusta de multiprocessing optimizada para entornos de simulación física.
    
    Esta función no solo establece el método de multiprocessing, sino que también
    configura el entorno para manejar de manera segura recursos como PyBullet,
    OpenGL, y otros componentes que pueden ser problemáticos en configuraciones paralelas.
    
    Args:
        force_method: Forzar un método específico ('spawn', 'fork', 'forkserver')
        verbose: Mostrar información detallada sobre la configuración
        safety_checks: Ejecutar verificaciones de compatibilidad
    
    Returns:
        bool: True si la configuración fue exitosa, False si hubo problemas
    """
    
    if verbose:
        print("🔧 Configurando multiprocessing para entornos PAM...")
        print(f"   Sistema operativo: {platform.system()}")
        print(f"   Versión de Python: {sys.version}")
        print(f"   Procesos disponibles: {mp.cpu_count()}")
    
    # ===== PASO 1: Determinar el mejor método según el sistema =====
    
    current_method = mp.get_start_method(allow_none=True)
    
    if force_method:
        target_method = force_method
        if verbose:
            print(f"   Método forzado por usuario: {target_method}")
    else:
        # Lógica inteligente para seleccionar el mejor método
        if platform.system() == "Windows":
            # En Windows, 'spawn' es generalmente más estable para PyBullet
            target_method = 'spawn'
            reason = "Recomendado para Windows + PyBullet"
        elif platform.system() == "Darwin":  # macOS
            # En macOS, 'spawn' también es más seguro
            target_method = 'spawn' 
            reason = "Recomendado para macOS + simulaciones gráficas"
        else:  # Linux
            # En Linux, podemos usar 'forkserver' que es un compromiso entre velocidad y seguridad
            target_method = 'forkserver' if 'forkserver' in mp.get_all_start_methods() else 'spawn'
            reason = "Optimizado para Linux + estabilidad"
        
        if verbose:
            print(f"   Método seleccionado automáticamente: {target_method} ({reason})")
    
    # ===== PASO 2: Configurar el método de manera segura =====
    
    try:
        if current_method is None:
            # Primera configuración
            mp.set_start_method(target_method)
            if verbose:
                print(f"   ✅ Método establecido: {target_method}")
        elif current_method != target_method:
            # Ya hay un método configurado pero es diferente
            if verbose:
                print(f"   ⚠️ Multiprocessing ya inicializado con método: {current_method}")
                print(f"   📝 Método deseado: {target_method}")
            
            # Intentar cambiar solo si es absolutamente necesario y seguro
            if current_method in ['fork'] and target_method in ['spawn', 'forkserver']:
                print(f"   🔄 Intentando cambiar de {current_method} a {target_method}...")
                # Esta operación puede fallar si ya hay procesos activos
                try:
                    mp.set_start_method(target_method, force=True)
                    print(f"   ✅ Método cambiado exitosamente a: {target_method}")
                except RuntimeError as e:
                    print(f"   ⚠️ No se pudo cambiar el método: {e}")
                    print(f"   📌 Continuando con método actual: {current_method}")
                    return False
            else:
                if verbose:
                    print(f"   📌 Manteniendo método actual: {current_method}")
        else:
            if verbose:
                print(f"   ✅ Método ya configurado correctamente: {current_method}")
    
    except Exception as e:
        if verbose:
            print(f"   ❌ Error configurando multiprocessing: {e}")
        return False
    
    # ===== PASO 3: Configuraciones adicionales para estabilidad =====
    
    if safety_checks:
        _configure_process_environment(verbose)
    
    # ===== PASO 4: Verificaciones de compatibilidad =====
    
    if safety_checks:
        compatibility_score = _verify_multiprocessing_compatibility(verbose)
        if compatibility_score < 0.7:  # Umbral de compatibilidad
            if verbose:
                print(f"   ⚠️ Puntuación de compatibilidad baja: {compatibility_score:.2f}")
                print(f"   💡 Considera usar n_envs=1 para mayor estabilidad")
    
    if verbose:
        print(f"   🎯 Configuración de multiprocessing completada")
        print(f"   📊 Método final: {mp.get_start_method()}")
    
    return True

def _configure_process_environment(verbose: bool = True):
    """
    Configurar variables de entorno y ajustes específicos para procesos hijo.
    
    Esta función establece configuraciones que ayudan a que PyBullet y otros
    componentes funcionen mejor en entornos multiproceso.
    """
    
    if verbose:
        print("   🛠️ Configurando entorno para procesos hijo...")
    
    # Configuraciones para PyBullet en multiprocessing
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # Evitar conflictos de GPU
    os.environ.setdefault('DISPLAY', '')  # Desactivar display en procesos hijo
    
    # Configuraciones para OpenGL/rendering
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')  # Forzar renderizado por software
    os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')  # Versión de OpenGL compatible
    
    # Configuraciones para estabilidad de memoria
    os.environ.setdefault('MALLOC_TRIM_THRESHOLD_', '100000')  # Liberar memoria más agresivamente
    
    # Configuraciones específicas para Windows
    if platform.system() == "Windows":
        os.environ.setdefault('PYTHONHASHSEED', '0')  # Hacer hashing reproducible
        os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')  # Limitar threads en procesos hijo
    
    if verbose:
        print("   ✅ Variables de entorno configuradas")

def _verify_multiprocessing_compatibility(verbose: bool = True) -> float:
    """
    Verificar la compatibilidad del sistema con multiprocessing para simulaciones.
    
    Esta función ejecuta pruebas básicas para determinar qué tan bien
    funcionará el multiprocessing en este sistema específico.
    
    Returns:
        float: Puntuación de compatibilidad (0.0 a 1.0, donde 1.0 es perfecto)
    """
    
    if verbose:
        print("   🧪 Verificando compatibilidad de multiprocessing...")
    
    compatibility_score = 1.0
    issues_found = []
    
    # Test 1: Verificar que el método elegido está disponible
    try:
        available_methods = mp.get_all_start_methods()
        current_method = mp.get_start_method()
        
        if current_method not in available_methods:
            compatibility_score -= 0.3
            issues_found.append(f"Método {current_method} no disponible")
        
    except Exception as e:
        compatibility_score -= 0.2
        issues_found.append(f"Error verificando métodos: {e}")
    
    # Test 2: Verificar disponibilidad de recursos del sistema
    try:
        cpu_count = mp.cpu_count()
        if cpu_count < 2:
            compatibility_score -= 0.2
            issues_found.append("Pocos CPUs disponibles para paralelización")
        
        # Verificar memoria disponible (estimación básica)
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4:  # Menos de 4GB disponibles
            compatibility_score -= 0.1
            issues_found.append("Memoria limitada para múltiples entornos")
            
    except ImportError:
        # psutil no disponible, no penalizar
        pass
    except Exception as e:
        compatibility_score -= 0.1
        issues_found.append(f"Error verificando recursos: {e}")
    
    # Test 3: Verificar compatibilidad con PyBullet
    try:
        import pybullet as p
        # PyBullet está disponible, pero verificar si funciona en modo DIRECT
        temp_client = p.connect(p.DIRECT)
        p.disconnect(temp_client)
    except Exception as e:
        compatibility_score -= 0.2
        issues_found.append(f"Problemas con PyBullet: {e}")
    
    if verbose:
        if compatibility_score >= 0.9:
            print(f"   ✅ Compatibilidad excelente: {compatibility_score:.2f}")
        elif compatibility_score >= 0.7:
            print(f"   ⚠️ Compatibilidad aceptable: {compatibility_score:.2f}")
        else:
            print(f"   ❌ Compatibilidad problemática: {compatibility_score:.2f}")
        
        if issues_found:
            print("   📋 Problemas detectados:")
            for issue in issues_found:
                print(f"      - {issue}")
    
    return compatibility_score

def setup_multiprocessing_for_training(n_envs: int, verbose: bool = True) -> bool:
    """
    Función de conveniencia específicamente para configurar multiprocessing
    antes del entrenamiento de sistemas PAM.
    
    Args:
        n_envs: Número de entornos paralelos que se van a usar
        verbose: Mostrar información detallada
    
    Returns:
        bool: True si la configuración es adecuada para el entrenamiento
    """
    
    if verbose:
        print(f"🚀 Preparando multiprocessing para {n_envs} entornos paralelos...")
    
    # Configurar multiprocessing básico
    success = _setup_multiprocessing(verbose=verbose)
    
    if not success:
        if verbose:
            print("   ❌ Configuración básica falló")
        return False
    
    # Verificaciones específicas para el número de entornos
    if n_envs > 1:
        cpu_count = mp.cpu_count()
        
        if n_envs > cpu_count:
            if verbose:
                print(f"   ⚠️ Advertencia: {n_envs} entornos > {cpu_count} CPUs")
                print("   💡 Considera reducir n_envs para mejor rendimiento")
        
        # Recomendaciones específicas según el número de entornos
        if n_envs > 8:
            if verbose:
                print("   ⚠️ Muchos entornos paralelos pueden causar inestabilidad")
                print("   🎯 Recomendación: empezar con 4-6 entornos")
    
    if verbose:
        print("   ✅ Sistema preparado para entrenamiento paralelo")
    
    return True