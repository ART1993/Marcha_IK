from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class BiomechanicalMetricsCallback(BaseCallback):
    """
    Callback especializado para monitorear métricas biomecánicas durante el entrenamiento.
    
    Este callback se conecta al loop de entrenamiento de Stable Baselines3 y extrae
    información específica sobre la coordinación muscular, eficiencia energética,
    y otros indicadores de calidad de movimiento biomecánico.
    
    Es como tener un fisiólogo deportivo observando constantemente al atleta
    y tomando notas sobre la calidad de sus movimientos.
    """
    
    def __init__(self, trainer, log_freq=1000, verbose=0):
        """
        Inicializar el callback biomecánico.
        
        Args:
            trainer: Referencia al trainer que contiene las métricas biomecánicas
            log_freq: Frecuencia de logging (cada cuántos pasos registrar métricas)
            verbose: Nivel de detalle en los logs
        """
        super(BiomechanicalMetricsCallback, self).__init__(verbose)
        
        self.trainer = trainer
        self.log_freq = log_freq
        
        # Contadores internos para tracking
        self.step_count = 0
        self.last_log_step = 0
        
        # Métricas acumuladas
        self.accumulated_metrics = {
            'pam_efficiency': [],
            'coordination_scores': [],
            'energy_consumption': [],
            'stability_measures': []
        }
    
    def _init_callback(self) -> None:
        """
        Método de inicialización requerido por Stable Baselines3.
        
        Este método es llamado automáticamente por SB3 cuando el callback
        se registra. Es como el momento en que el especialista se presenta
        oficialmente al equipo de entrenamiento y establece su protocolo.
        """
        
        # Verificar que tenemos acceso a la información que necesitamos
        if not hasattr(self.trainer, 'biomechanical_metrics'):
            self.trainer.biomechanical_metrics = {
                'coordination_scores': [],
                'energy_efficiency': [],
                'muscle_activation_patterns': [],
                'temporal_coordination': []
            }
            
        # Inicializar logging en TensorBoard si está disponible
        if hasattr(self.logger, 'record'):
            self.logger.record("biomechanical/callback_initialized", 1)
        
        if self.verbose > 0:
            print("🧠 Biomechanical metrics callback initialized successfully")
            print(f"   Logging frequency: every {self.log_freq} steps")
            print(f"   Monitoring: PAM efficiency, coordination, energy consumption")
    
    def _on_training_start(self) -> None:
        """
        Método llamado al inicio del entrenamiento.
        
        Aquí podemos realizar cualquier configuración adicional que necesitemos
        ahora que sabemos que el entrenamiento está comenzando oficialmente.
        """
        
        if self.verbose > 0:
            print("📊 Starting biomechanical monitoring for training session")
        
        # Reiniciar contadores para esta sesión de entrenamiento
        self.step_count = 0
        self.last_log_step = 0
        
        # Limpiar métricas acumuladas de sesiones anteriores
        for key in self.accumulated_metrics:
            self.accumulated_metrics[key].clear()
    
    def _on_step(self) -> bool:
        """
        Método llamado en cada paso del entrenamiento.
        
        Este es el "corazón" del callback - se ejecuta constantemente durante
        el entrenamiento y es donde extraemos y procesamos las métricas biomecánicas.
        
        Returns:
            bool: True para continuar entrenamiento, False para detenerlo
        """
        
        self.step_count += 1
        
        # Extraer métricas del entorno si están disponibles
        self._extract_biomechanical_metrics()
        
        # Log métricas periódicamente
        if self.step_count - self.last_log_step >= self.log_freq:
            self._log_biomechanical_metrics()
            self.last_log_step = self.step_count
        
        # Siempre retornar True para continuar entrenamiento
        # (Solo retornarías False si quisieras detener el entrenamiento por alguna razón)
        return True
    
    def _extract_biomechanical_metrics(self):
        """
        Extraer métricas biomecánicas de la información del entorno.
        
        Este método busca en la información que viene del entorno para encontrar
        datos específicos sobre la coordinación muscular y otros indicadores
        biomecánicos que queremos monitorear.
        """
        
        try:
            # Acceder a la información del último paso
            if hasattr(self.training_env, 'get_attr'):
                # Para entornos vectorizados
                infos = self.training_env.get_attr('info_buffer')
                
                for env_info in infos:
                    if env_info and len(env_info) > 0:
                        latest_info = env_info[-1]  # Información más reciente
                        self._process_environment_info(latest_info)
            
        except Exception as e:
            # No fallar el entrenamiento por problemas de extracción de métricas
            if self.verbose > 1:
                print(f"Warning: Could not extract biomechanical metrics: {e}")
    
    def _process_environment_info(self, info):
        """
        Procesar información específica del entorno para extraer métricas biomecánicas.
        
        Args:
            info: Diccionario de información del entorno
        """
        
        if not isinstance(info, dict):
            return
        
        # Extraer eficiencia PAM si está disponible
        if 'reward_components' in info:
            components = info['reward_components']
            
            if 'pam_efficiency' in components:
                efficiency = components['pam_efficiency']
                self.accumulated_metrics['pam_efficiency'].append(efficiency)
                
                # Guardar en el trainer para análisis posterior
                self.trainer.biomechanical_metrics['energy_efficiency'].append(efficiency)
        
        # Extraer métricas de coordinación
        if 'coordination_metrics' in info:
            coord_metrics = info['coordination_metrics']
            
            if isinstance(coord_metrics, dict):
                for metric_name, value in coord_metrics.items():
                    if metric_name not in self.accumulated_metrics:
                        self.accumulated_metrics[metric_name] = []
                    self.accumulated_metrics[metric_name].append(value)
        
        # Extraer información de estabilidad
        if 'zmp_stable' in info:
            stability = 1.0 if info['zmp_stable'] else 0.0
            self.accumulated_metrics['stability_measures'].append(stability)
    
    def _log_biomechanical_metrics(self):
        """
        Registrar métricas biomecánicas en TensorBoard y consola.
        
        Este método calcula estadísticas sobre las métricas acumuladas
        y las registra tanto en TensorBoard como en la consola para monitoreo.
        """
        
        try:
            # Calcular estadísticas recientes
            recent_window = 100  # Últimas 100 observaciones
            
            if len(self.accumulated_metrics['pam_efficiency']) > 0:
                recent_efficiency = self.accumulated_metrics['pam_efficiency'][-recent_window:]
                avg_efficiency = np.mean(recent_efficiency)
                
                # Log en TensorBoard
                if hasattr(self.logger, 'record'):
                    self.logger.record("biomechanical/pam_efficiency", avg_efficiency)
                
                # Log en consola si es verboso
                if self.verbose > 0:
                    print(f"   📊 Recent PAM efficiency: {avg_efficiency:.3f}")
            
            if len(self.accumulated_metrics['stability_measures']) > 0:
                recent_stability = self.accumulated_metrics['stability_measures'][-recent_window:]
                avg_stability = np.mean(recent_stability)
                
                if hasattr(self.logger, 'record'):
                    self.logger.record("biomechanical/stability", avg_stability)
                
                if self.verbose > 0:
                    print(f"   ⚖️ Recent stability: {avg_stability:.3f}")
            
            # Log métricas de coordinación específicas
            for metric_name, values in self.accumulated_metrics.items():
                if metric_name.startswith('coordination_') and len(values) > 0:
                    recent_values = values[-recent_window:]
                    avg_value = np.mean(recent_values)
                    
                    if hasattr(self.logger, 'record'):
                        self.logger.record(f"biomechanical/{metric_name}", avg_value)
        
        except Exception as e:
            # No interrumpir el entrenamiento por errores de logging
            if self.verbose > 1:
                print(f"Warning: Error logging biomechanical metrics: {e}")
    
    def _on_training_end(self) -> None:
        """
        Método llamado al final del entrenamiento.
        
        Aquí podemos generar reportes finales o guardar métricas acumuladas
        para análisis posterior.
        """
        
        if self.verbose > 0:
            print("📈 Biomechanical monitoring session completed")
            
            # Mostrar resumen final
            total_steps = self.step_count
            
            if len(self.accumulated_metrics['pam_efficiency']) > 0:
                final_avg_efficiency = np.mean(self.accumulated_metrics['pam_efficiency'])
                print(f"   Final average PAM efficiency: {final_avg_efficiency:.3f}")
            
            print(f"   Total steps monitored: {total_steps:,}")
        
        # Guardar métricas finales en el trainer
        if hasattr(self.trainer, 'final_biomechanical_summary'):
            self.trainer.final_biomechanical_summary = {
                'total_monitored_steps': self.step_count,
                'final_metrics': {
                    metric_name: np.mean(values) if values else 0.0
                    for metric_name, values in self.accumulated_metrics.items()
                }
            }