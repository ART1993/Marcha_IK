# ===== SOLUCIÓN SIMPLE: UN SOLO ARCHIVO DE MODIFICACIÓN =====

import sys
import os
from datetime import datetime

class SimpleLogRedirect:
    """
    Redirección súper simple de prints personalizados.
    
    USO:
    1. Inicializar al inicio del script
    2. Usar print_to_log() en lugar de print() para logs
    3. Usar print() normal para mantener en consola
    """
    
    def __init__(self, log_file=None):
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"./logs_lift_leg/training_{timestamp}.txt"
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        self.log_file = log_file
        self.original_stdout = sys.stdout
        
        # Abrir archivo para append
        self.log_handle = open(log_file, 'w', encoding='utf-8')
        
        print(f"📝 Logs personalizados → {log_file}")
        print(f"🖥️  SB3 verbose → consola")
    
    def print_to_log(self, *args, **kwargs):
        """Print que va al archivo de log"""
        print(*args, **kwargs, file=self.log_handle)
        self.log_handle.flush()  # Asegurar que se escriba inmediatamente
    
    def print_to_both(self, *args, **kwargs):
        """Print que va tanto a consola como a log"""
        print(*args, **kwargs)  # Consola
        print(*args, **kwargs, file=self.log_handle)  # Log
        self.log_handle.flush()
    
    def close(self):
        """Cerrar el archivo de log"""
        if hasattr(self, 'log_handle'):
            self.log_handle.close()
    
    def __del__(self):
        """Cerrar automáticamente al destruir"""
        self.close()

# ===== INSTANCIA GLOBAL PARA USAR EN TODO EL PROYECTO =====
_log_redirect = None

def init_simple_logging(log_file=None):
    """Inicializar logging simple"""
    global _log_redirect
    _log_redirect = SimpleLogRedirect(log_file)
    return _log_redirect

def log_print(*args, **kwargs):
    """Función global para print a log"""
    if _log_redirect:
        _log_redirect.print_to_log(*args, **kwargs)
    else:
        print("⚠️ Logging not initialized, using regular print:", *args, **kwargs)

def both_print(*args, **kwargs):
    """Función global para print a ambos lugares"""
    if _log_redirect:
        _log_redirect.print_to_both(*args, **kwargs)
    else:
        print(*args, **kwargs)

class MultiLogRedirect:
    """
    Permite tener múltiples logs (por categorías) además del log global.
    Ejemplo:
        logger = MultiLogRedirect()
        logger.log("reward", "Reward=2.3", step=512)
        logger.log("pressure", "PAM1=2.1, PAM2=1.8", step=512)
        logger.console("Entrenamiento activo")
    """

    def __init__(self, base_dir="./logs_lift_leg"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # Archivo principal
        self.global_file = os.path.join(base_dir, f"training_{timestamp}.txt")
        self._handles = {"main": open(self.global_file, "w", encoding="utf-8")}
        print(f"📝 Logging directory: {base_dir}")

    def _get_handle(self, name: str):
        """Obtiene o crea un handle de log específico"""
        if name not in self._handles:
            file_path = os.path.join(self.base_dir, f"{name}_log.txt")
            self._handles[name] = open(file_path, "w", encoding="utf-8")
        return self._handles[name]

    def log(self, category: str, *args, step=None, **kwargs):
        """Escribe solo en el archivo de categoría"""
        handle = self._get_handle(category)
        prefix = f"[{step:06d}] " if step is not None else ""
        print(prefix, *args, **kwargs, file=handle)
        handle.flush()

    def both(self, category: str, *args, step=None, **kwargs):
        """Escribe en consola y en el archivo de categoría"""
        handle = self._get_handle(category)
        prefix = f"[{step:06d}] " if step is not None else ""
        print(prefix, *args, **kwargs)
        print(prefix, *args, **kwargs, file=handle)
        handle.flush()

    def console(self, *args, **kwargs):
        """Print normal en consola"""
        print(*args, **kwargs)

    def close(self):
        for h in self._handles.values():
            h.close()
        self._handles.clear()

# ===== EJEMPLO DE USO EN TUS ARCHIVOS =====

if __name__ == "__main__":
    # Inicializar logging simple
    logger1 = init_simple_logging()
    
    # Prints que VAN A CONSOLA (SB3 verbose, mensajes críticos)
    print("🚀 Training started")  # CONSOLA
    print("Progress: 25%")        # CONSOLA
    
    # Prints que VAN AL LOG (detalles, debug, info)
    log_print("🤖 Environment initialized")
    log_print("   Action space: 6 PAM pressures") 
    log_print("   PAM 0 (left_hip_flexor): P=2.1atm, θ=0.12rad")
    
    # Prints que VAN A AMBOS (eventos importantes)
    both_print("🏋️ Transitioning to lift_leg mode")
    both_print("🎉 Training completed!")
    
    # Cerrar al final
    logger1.close()
