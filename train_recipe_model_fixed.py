# Fixed version with just the key fix - move import to top and remove duplicate definitions

# Add this after all other imports but before any class definitions:
try:
    from transformers import TrainerCallback
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    class TrainerCallback:
        pass

# Then add these classes at module level (not inside train method):
if TRANSFORMERS_AVAILABLE:
    class TrainingMetricsCallback(TrainerCallback):
        def __init__(self, training_logger):
            self.training_logger = training_logger
        
        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if logs and self.training_logger:
                metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                if metrics:
                    self.training_logger.set_step(state.global_step)
                    self.training_logger.log_metrics(metrics, step=state.global_step)
    
    class DataLoadingMonitorCallback(TrainerCallback):
        def __init__(self, system_monitor, enable_profiling=False, profile_schedule="wait=1;warmup=1;active=3;repeat=2"):
            self.system_monitor = system_monitor
            self.batch_start_time = None
            
        def on_step_begin(self, args, state, control, **kwargs):
            self.batch_start_time = time.time()
            
        def on_step_end(self, args, state, control, **kwargs):
            if self.batch_start_time:
                batch_time = time.time() - self.batch_start_time
                self.system_monitor.log_data_loading_metrics(
                    batch_time, 
                    args.per_device_train_batch_size, 
                    step=state.global_step
                )
else:
    class TrainingMetricsCallback:
        def __init__(self, *args, **kwargs):
            pass
    
    class DataLoadingMonitorCallback:
        def __init__(self, *args, **kwargs):
            pass