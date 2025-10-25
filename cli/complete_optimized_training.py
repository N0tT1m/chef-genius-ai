#!/usr/bin/env python3
"""
Complete Optimized Training Pipeline with ALL Monitoring Features
Combines Ryzen 3900X + RTX 4090 optimizations with your existing:
- W&B logging and monitoring
- Discord webhook notifications  
- SMS alerts
- Data pipeline monitoring
- System monitoring
- All datasets unified
"""

import os
import sys
import time
import torch
import threading
import requests
import json
import traceback
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup

# Import your existing monitoring classes
from ryzen_4090_optimized_training import Ryzen4090OptimizedConfig
# Note: Removed AllDatasetsTrainingIntegration - now using clean JSONL dataloader

# Import checkpoint utilities for proper state management
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from checkpoint_utils import CheckpointManager

# Import all monitoring dependencies 
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    pass

try:
    import psutil
    import GPUtil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False
    pass

# Global crash handler
class CrashHandler:
    """Global crash handler to send Discord notifications before script exits."""
    
    def __init__(self, alerter=None):
        self.alerter = alerter
        self.original_excepthook = sys.excepthook
        sys.excepthook = self.handle_exception
        
        # Handle SIGTERM and SIGINT 
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions and send Discord notification."""
        if self.alerter and self.alerter.enabled:
            error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self.alerter.training_crashed(error_msg)
        
        # Call the original exception handler
        self.original_excepthook(exc_type, exc_value, exc_traceback)
    
    def handle_signal(self, signum, frame):
        """Handle process termination signals."""
        if self.alerter and self.alerter.enabled:
            signal_name = signal.Signals(signum).name
            self.alerter.training_crashed(f"Training interrupted by signal: {signal_name}")
        
        # Exit gracefully
        sys.exit(1)

class DiscordAlerter:
    """Discord webhook and SMS notifications for training events."""
    
    def __init__(self, webhook_url: str = None, phone_number: str = None):
        self.webhook_url = webhook_url
        self.phone_number = phone_number
        self.discord_enabled = webhook_url is not None
        self.sms_enabled = phone_number is not None
        self.enabled = self.discord_enabled or self.sms_enabled
        
    def send_sms(self, message: str):
        """Send SMS notification using multiple services as fallbacks."""
        if not self.sms_enabled:
            return
            
        phone = ''.join(c for c in self.phone_number if c.isdigit() or c == '+')
        
        # Try TextBelt (free service)
        try:
            data = {
                'phone': phone,
                'message': f"Chef Genius: {message}",
                'key': 'textbelt'
            }
            
            response = requests.post('https://textbelt.com/text', data=data, timeout=10)
            if response.json().get('success'):
                return True
        except Exception as e:
            pass
            
        return False
    
    def send_notification(self, title: str, description: str, color: int = 0x00ff00, fields: list = None, sms_message: str = None):
        """Send both Discord and SMS notifications."""
        if not self.enabled:
            return
        
        # Send Discord notification
        if self.discord_enabled:
            try:
                embed = {
                    "title": title,
                    "description": description,
                    "color": color,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "footer": {"text": "Chef Genius Training Bot"}
                }
                
                if fields:
                    embed["fields"] = fields
                
                payload = {
                    "embeds": [embed],
                    "username": "Chef Genius Training"
                }
                
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
            except Exception as e:
                pass
        
        # Send SMS notification
        if self.sms_enabled:
            if sms_message is None:
                sms_text = f"{title}: {description}"
                if fields and len(fields) > 0:
                    key_info = ", ".join([f"{f['name']}: {f['value']}" for f in fields[:2]])
                    sms_text += f" ({key_info})"
            else:
                sms_text = sms_message
                
            if len(sms_text) > 160:
                sms_text = sms_text[:157] + "..."
                
            self.send_sms(sms_text)
    
    def training_started(self, model_type: str, epochs: int, batch_size: int, dataset_info: str):
        """Notify training start."""
        fields = [
            {"name": "Model", "value": model_type, "inline": True},
            {"name": "Epochs", "value": str(epochs), "inline": True},
            {"name": "Batch Size", "value": str(batch_size), "inline": True},
            {"name": "Dataset", "value": dataset_info, "inline": False}
        ]
        
        sms_msg = f"Training started: {model_type}, {epochs} epochs, batch size {batch_size}"
        
        self.send_notification(
            title="🚀 Training Started",
            description="Model training has begun with ALL datasets!",
            color=0x0099ff,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_progress(self, epoch: int, total_epochs: int, loss: float, lr: float, samples_per_sec: float = None):
        """Notify training progress."""
        progress = (epoch / total_epochs) * 100
        fields = [
            {"name": "Progress", "value": f"{progress:.1f}% ({epoch}/{total_epochs})", "inline": True},
            {"name": "Loss", "value": f"{loss:.4f}", "inline": True},
            {"name": "Learning Rate", "value": f"{lr:.2e}", "inline": True}
        ]
        
        if samples_per_sec:
            fields.append({"name": "Data Speed", "value": f"{samples_per_sec:.1f} samples/sec", "inline": True})
        
        sms_msg = f"Training progress: {progress:.1f}% ({epoch}/{total_epochs}), Loss: {loss:.4f}"
        if samples_per_sec:
            sms_msg += f", Speed: {samples_per_sec:.0f} samples/sec"
        
        self.send_notification(
            title="📊 Training Progress",
            description=f"Epoch {epoch} completed",
            color=0xffaa00,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_completed(self, duration_hours: float, final_metrics: dict):
        """Notify training completion."""
        fields = [
            {"name": "Duration", "value": f"{duration_hours:.2f} hours", "inline": True}
        ]
        
        sms_msg = f"Training completed! Duration: {duration_hours:.2f}h"
        if final_metrics and 'train_loss' in final_metrics:
            sms_msg += f", Final loss: {final_metrics['train_loss']:.4f}"
        
        if final_metrics:
            for metric, value in final_metrics.items():
                fields.append({
                    "name": metric.replace("_", " ").title(),
                    "value": f"{value:.4f}" if isinstance(value, float) else str(value),
                    "inline": True
                })
        
        self.send_notification(
            title="✅ Training Completed",
            description="Model training finished successfully with ALL datasets!",
            color=0x00ff00,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_error(self, error_msg: str, epoch: int = None):
        """Notify training error."""
        fields = []
        if epoch is not None:
            fields.append({"name": "Failed at Epoch", "value": str(epoch), "inline": True})
        
        fields.append({"name": "Error", "value": error_msg[:1000], "inline": False})
        
        sms_msg = f"Training error"
        if epoch is not None:
            sms_msg += f" at epoch {epoch}"
        sms_msg += f": {error_msg[:100]}"
        
        self.send_notification(
            title="❌ Training Error",
            description="Training encountered an error!",
            color=0xff0000,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_crashed(self, error_msg: str, epoch: int = None):
        """Notify training crash with full stack trace."""
        # Truncate error message for Discord (embed limit is 4096 chars)
        truncated_error = error_msg[:2000] + "..." if len(error_msg) > 2000 else error_msg
        
        fields = []
        if epoch is not None:
            fields.append({"name": "Failed at Epoch", "value": str(epoch), "inline": True})
        
        fields.append({"name": "Stack Trace", "value": f"```\n{truncated_error}\n```", "inline": False})
        
        sms_msg = f"Training CRASHED"
        if epoch is not None:
            sms_msg += f" at epoch {epoch}"
        
        # Extract just the error type/message for SMS
        try:
            lines = error_msg.strip().split('\n')
            last_line = lines[-1] if lines else error_msg[:100]
            sms_msg += f": {last_line[:100]}"
        except:
            sms_msg += f": {error_msg[:80]}"
        
        self.send_notification(
            title="💥 TRAINING CRASHED",
            description="Training process has crashed unexpectedly!",
            color=0x8B0000,  # Dark red for crashes
            fields=fields,
            sms_message=sms_msg
        )
    
    def data_pipeline_bottleneck(self, bottleneck_type: str, details: dict):
        """Notify data pipeline bottleneck detection."""
        fields = [
            {"name": "Bottleneck Type", "value": bottleneck_type, "inline": True},
            {"name": "Current Speed", "value": f"{details.get('samples_per_sec', 0):.1f} samples/sec", "inline": True}
        ]
        
        if 'expected_speed' in details:
            fields.append({"name": "Expected Speed", "value": f"{details['expected_speed']:.1f} samples/sec", "inline": True})
        
        if 'recommendation' in details:
            fields.append({"name": "Recommendation", "value": details['recommendation'][:500], "inline": False})
        
        color = 0xff9900  # Orange for warning
        if details.get('severity') == 'critical':
            color = 0xff0000  # Red for critical
        
        sms_msg = f"Data pipeline bottleneck: {bottleneck_type}. Speed: {details.get('samples_per_sec', 0):.1f} samples/sec"
        
        self.send_notification(
            title="⚠️ Data Pipeline Bottleneck Detected",
            description=f"Performance issue detected in data loading pipeline: {bottleneck_type}",
            color=color,
            fields=fields,
            sms_message=sms_msg
        )
    
    def data_pipeline_optimized(self, old_speed: float, new_speed: float, optimization_applied: str):
        """Notify successful data pipeline optimization."""
        improvement = ((new_speed - old_speed) / old_speed) * 100
        
        fields = [
            {"name": "Old Speed", "value": f"{old_speed:.1f} samples/sec", "inline": True},
            {"name": "New Speed", "value": f"{new_speed:.1f} samples/sec", "inline": True},
            {"name": "Improvement", "value": f"{improvement:.1f}%", "inline": True},
            {"name": "Optimization", "value": optimization_applied, "inline": False}
        ]
        
        sms_msg = f"Data pipeline optimized! {old_speed:.1f} → {new_speed:.1f} samples/sec ({improvement:.1f}% improvement)"
        
        self.send_notification(
            title="🚀 Data Pipeline Optimized",
            description="Data loading performance successfully improved!",
            color=0x00ff00,
            fields=fields,
            sms_message=sms_msg
        )

    def hardware_warning(self, warning_msg: str):
        """Notify hardware warnings."""
        sms_msg = f"Hardware warning: {warning_msg[:120]}"
        
        self.send_notification(
            title="⚠️ Hardware Warning",
            description=warning_msg,
            color=0xff9900,
            sms_message=sms_msg
        )
    
    def data_pipeline_update(self, samples_per_sec: float, datasets_active: int, total_samples: int):
        """Notify about data pipeline performance."""
        fields = [
            {"name": "Data Speed", "value": f"{samples_per_sec:.1f} samples/sec", "inline": True},
            {"name": "Active Datasets", "value": str(datasets_active), "inline": True},
            {"name": "Total Samples", "value": f"{total_samples:,}", "inline": True}
        ]
        
        # Only send if speed is exceptional (to avoid spam)
        if samples_per_sec > 300:
            self.send_notification(
                title="🚀 High Performance Data Loading",
                description=f"Rust data loader achieving {samples_per_sec:.1f} samples/sec!",
                color=0x00ff88,
                fields=fields
            )

class TrainingLogger:
    """W&B logging with step management."""

    def __init__(self, use_wandb: bool = True, project_name: str = "chef-genius-recipe-training"):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.current_step = 0
        self._step_lock = threading.Lock()

        if self.use_wandb:
            try:
                import datetime
                run_name = f"optimized-training-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

                print(f"📊 Initializing W&B...")
                print(f"   Project: {project_name}")
                print(f"   Run name: {run_name}")

                # Get API key from environment
                wandb_api_key = os.environ.get('WANDB_API_KEY')

                # Login first if API key is available
                if wandb_api_key:
                    print(f"   Logging in with API key from environment...")
                    wandb.login(key=wandb_api_key, relogin=True)
                else:
                    print(f"⚠️  No WANDB_API_KEY found in environment, trying anonymous mode...")

                wandb.init(
                    project=project_name,
                    name=run_name,
                    mode="online",
                    settings=wandb.Settings(console='off'),
                    tags=["recipe-generation", "optimized", "rust-dataloader", "ryzen-5090"]
                )
                print(f"✅ W&B initialized successfully!")
                print(f"   Dashboard: {wandb.run.url}")
            except Exception as e:
                print(f"❌ W&B initialization failed: {e}")
                print(f"⚠️  Continuing training without W&B logging...")
                self.use_wandb = False
    
    def set_step(self, step: int):
        """Set the current training step."""
        with self._step_lock:
            self.current_step = step
    
    def log_metrics(self, metrics: Dict, step: int = None, commit: bool = True):
        """Log metrics with proper step management."""
        if not self.use_wandb or not metrics:
            return
            
        with self._step_lock:
            log_step = step if step is not None else self.current_step
            
            try:
                wandb.log(metrics, step=log_step, commit=commit)
            except Exception as e:
                print(f"⚠️  W&B logging failed: {e}")

class SystemMonitor:
    """System monitoring for hardware metrics."""
    
    def __init__(self, training_logger=None):
        self.training_logger = training_logger
        self.monitoring_available = SYSTEM_MONITORING_AVAILABLE
        self.last_metrics = {}
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        if not self.monitoring_available:
            return {}
        
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metrics.update({
                'system/cpu_percent': cpu_percent,
                'system/memory_percent': memory.percent,
                'system/memory_available_gb': memory.available / (1024**3),
            })
            
            # GPU metrics
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)

                # Auto-detect total GPU memory (works for 4090 24GB, 5090 32GB, etc.)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                metrics.update({
                    'system/gpu_memory_allocated_gb': gpu_memory_allocated,
                    'system/gpu_memory_reserved_gb': gpu_memory_reserved,
                    'system/gpu_memory_total_gb': gpu_memory_total,
                    'system/gpu_memory_percent': (gpu_memory_allocated / gpu_memory_total) * 100,
                })

                # GPU utilization
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # First GPU
                        metrics.update({
                            'system/gpu_utilization': gpu.load * 100,
                            'system/gpu_temperature': gpu.temperature,
                        })
                except:
                    pass
            
        except Exception as e:
            pass
        
        self.last_metrics = metrics
        return metrics

class CompleteOptimizedTrainer:
    """
    Complete training pipeline with:
    - Ryzen 3900X + RTX 4090 optimizations
    - ALL datasets unified  
    - W&B monitoring
    - Discord + SMS notifications
    - System monitoring
    - Data pipeline monitoring
    """
    
    def __init__(self, 
                 model, 
                 tokenizer,
                 output_dir: str = "./optimized_model",
                 batch_size: int = None,
                 discord_webhook: str = None,
                 alert_phone: str = None,
                 wandb_project: str = "chef-genius-optimized",
                 use_wandb: bool = True,
                 gradient_accumulation_steps: int = None,
                 enable_mixed_precision: bool = False,
                 disable_compilation: bool = False,
                 disable_cudagraphs: bool = True,
                 dataloader_num_workers: int = 8):
        
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        
        # Initialize configurations
        self.hw_config = Ryzen4090OptimizedConfig()
        
        # Override batch size if provided
        if batch_size is not None:
            self.hw_config.batch_size = batch_size
            
        # Override gradient accumulation steps if provided
        if gradient_accumulation_steps is not None:
            self.hw_config.gradient_accumulation_steps = gradient_accumulation_steps
        else:
            # Auto-configure based on model size if not specified
            if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
                model_name = str(model.config.name_or_path).lower()
                if 'xxl' in model_name:
                    # FLAN-T5-XXL (11B): Very aggressive accumulation for 32GB VRAM
                    self.hw_config.gradient_accumulation_steps = 16
                    print(f"🎯 Detected FLAN-T5-XXL (11B) - using gradient_accumulation_steps=16 for 32GB VRAM")
                elif 'xl' in model_name:
                    # FLAN-T5-XL (3B): Moderate accumulation
                    self.hw_config.gradient_accumulation_steps = max(8, 16 // (batch_size or self.hw_config.batch_size))
                elif 'large' in model_name and (batch_size or self.hw_config.batch_size) > 1:
                    self.hw_config.gradient_accumulation_steps = max(8, 16 // (batch_size or self.hw_config.batch_size))
                    
        # Store mixed precision setting
        self.enable_mixed_precision = enable_mixed_precision
        self.disable_compilation = disable_compilation
        self.disable_cudagraphs = disable_cudagraphs
        self.dataloader_num_workers = dataloader_num_workers
        
        # Initialize monitoring
        self.training_logger = TrainingLogger(use_wandb=use_wandb, project_name=wandb_project)
        self.system_monitor = SystemMonitor(self.training_logger)
        self.alerter = DiscordAlerter(webhook_url=discord_webhook, phone_number=alert_phone)
        
        # Training state
        self.start_time = None
        self.epoch_losses = []
        self.current_epoch = 0
        
        # Setup global crash handler
        self.crash_handler = CrashHandler(alerter=self.alerter)
        
        # Setup hardware optimizations
        self.setup_hardware_optimizations()
        
    def setup_hardware_optimizations(self):
        """Setup Ryzen 3900X + RTX 5090 optimizations with memory management."""

        # Suppress TF32 deprecation warnings globally
        import warnings
        warnings.filterwarnings('ignore', message='.*allow_tf32.*')

        # CUDA optimizations for RTX 5090
        if torch.cuda.is_available():
            # Use new PyTorch 2.9+ API for TF32 settings BEFORE any CUDA operations
            try:
                # New API (PyTorch 2.9+)
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
                torch.backends.cuda.matmul.fp32_precision = 'tf32'
                print("   ✅ TF32 enabled using new PyTorch 2.9+ API")
            except AttributeError:
                # Fallback to old API for older PyTorch versions
                # Suppress warnings when using old API
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                print("   ✅ TF32 enabled using legacy API")

            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

            # Memory optimization for RTX 5090 (32GB VRAM)
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:16,garbage_collection_threshold:0.7'

            # Check if model is already in bfloat16, if not convert it
            if self.model.dtype != torch.bfloat16:
                print("   Converting model to bfloat16 for memory efficiency...")
                old_model = self.model  # Keep reference to old model
                self.model = self.model.to(dtype=torch.bfloat16)
                # Delete old model and free memory
                del old_model
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                print("   ✅ Freed float32 model from memory")
            else:
                print("   ✅ Model already in bfloat16, skipping conversion")

        # CPU optimizations for Ryzen 3900X
        torch.set_num_threads(self.hw_config.cpu_threads)
        os.environ['OMP_NUM_THREADS'] = str(self.hw_config.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.hw_config.cpu_threads)

        # Enable gradient checkpointing for memory efficiency (saves ~50% memory)
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("   ✅ Gradient checkpointing enabled (saves ~50% memory)")

            # CRITICAL: Disable KV cache when using gradient checkpointing
            # This prevents OOM by not storing intermediate key-value states
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
                print("   ✅ KV cache disabled (required for gradient checkpointing)")
        else:
            print("   ⚠️  Gradient checkpointing not available for this model")

        # Performance optimizations
        torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes

        # Enable Flash Attention if available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            print("   ✅ Flash Attention SDP enabled")
        except (AttributeError, Exception):
            pass  # Older PyTorch version or not available
        
        # Disable CUDA Graphs globally to prevent tensor overwrite issues with T5 models
        if self.disable_cudagraphs:
            os.environ['TORCH_COMPILE_DISABLE_CUDAGRAPHS'] = '1'
            # Also disable other CUDA graph related optimizations
            os.environ['TORCH_CUDAGRAPHS'] = '0'
            # Disable Triton's CUDA graph usage
            os.environ['TRITON_DISABLE_CUDAGRAPH'] = '1'
            # Force default compilation mode to avoid CUDA graph issues
            os.environ['TORCH_COMPILE_MODE'] = 'default'
            print("🛡️  CUDA Graphs disabled to prevent tensor overwrite issues")
        
        # Compile model for faster execution (PyTorch 2.0+) - only when GPU is available
        if not self.disable_compilation:
            try:
                import platform
                # Only compile when CUDA is available and working, not in CPU-only Docker environments
                if platform.system() != 'Windows' and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    # Set environment variable for safer compilation
                    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'  # Single thread for debugging
                    os.environ['TORCH_LOGS'] = '+dynamo'  # Enable dynamo logging for debugging

                    if self.disable_cudagraphs:
                        print("🚀 Enabling torch.compile with 'reduce-overhead' mode (CUDA graphs disabled)...")
                        # Use 'reduce-overhead' mode - safer than 'default' and still fast
                        self.model = torch.compile(self.model, mode='reduce-overhead', disable=False)
                        print("⚡ reduce-overhead mode enabled: Optimized compilation for T5 models")
                    else:
                        print("🚀 Enabling torch.compile with 'reduce-overhead' for stability...")
                        # Use 'reduce-overhead' instead of 'max-autotune' to avoid crashes
                        self.model = torch.compile(self.model, mode='reduce-overhead', disable=False)
                        print("⚡ reduce-overhead enabled: Balanced performance and stability")
                else:
                    print("⚠️  Skipping torch.compile (CPU-only environment or Windows)")
            except Exception as e:
                print(f"⚠️  torch.compile failed, continuing without compilation: {e}")
                print(f"   This is not critical - training will continue without compilation")
                import traceback
                traceback.print_exc()
                # Don't re-raise, just continue without compilation
            
        # Pin memory for faster data transfer is handled by dataloader
    
    def create_unified_dataloader(self):
        """Create the unified data loader using validated JSONL data with memory optimization."""

        print("🚀 Creating memory-optimized Rust dataloader...")

        # Check if validated data exists, generate if not
        # Handle both Docker context (root) and local context (cli/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"🔍 Script directory: {script_dir}")

        if script_dir.endswith('/cli'):
            # Script is in cli directory, but check current working directory
            cwd = os.getcwd()
            print(f"📂 Current working directory: {cwd}")
            if cwd.endswith('/cli'):
                print("🏠 Detected: Running from cli directory")
                validated_data_dir = "./cli/validated_datasets"
                batch_script = "./cli/batch_validate_all_datasets.sh"
            else:
                print("🐳 Detected: Script in cli/ but running from root (Docker context)")
                validated_data_dir = "./cli/validated_datasets"
                batch_script = "./cli/batch_validate_all_datasets.sh"
        else:
            # Running from root (Docker context)
            print("🐳 Detected: Running from root directory (Docker context)")
            validated_data_dir = "cli/validated_datasets"
            batch_script = "./cli/batch_validate_all_datasets.sh"

        print(f"📁 Validated data dir: {validated_data_dir}")
        print(f"📜 Batch script path: {batch_script}")

        combined_file = f"{validated_data_dir}/combined_all_datasets_flan_t5.jsonl"

        if not os.path.exists(combined_file):
            print("⚠️  Validated JSONL data not found!")
            print(f"🔄 Running automatic data validation pipeline from: {batch_script}")

            # Ensure script is executable
            import subprocess
            subprocess.run(["chmod", "+x", batch_script], capture_output=True)

            # Run the batch validation script with proper working directory
            result = subprocess.run([batch_script],
                                  capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ Data validation failed!")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Data validation failed: {result.stderr}")

            print("✅ Data validation completed!")

        # Import memory-optimized dataloader
        try:
            from memory_optimized_training import create_memory_optimized_dataloader

            # Create memory-optimized Rust dataloader
            train_loader = create_memory_optimized_dataloader(
                tokenizer=self.tokenizer,
                validated_data_dir=validated_data_dir,
                batch_size=self.hw_config.batch_size,
                min_quality_score=0.6  # Use high-quality recipes only
            )

            print(f"✅ Memory-optimized Rust dataloader created!")
            print(f"   Batch size: {self.hw_config.batch_size}")
            print(f"   Gradient accumulation: {self.hw_config.gradient_accumulation_steps}")
            print(f"   Effective batch size: {self.hw_config.batch_size * self.hw_config.gradient_accumulation_steps}")
            print(f"🎯 Quality threshold: 0.6 (high-quality recipes only)")

        except ImportError as e:
            print(f"⚠️  Memory-optimized dataloader not available, using standard: {e}")
            # Fallback to standard dataloader
            from jsonl_dataloader import create_optimized_jsonl_dataloader

            train_loader = create_optimized_jsonl_dataloader(
                tokenizer=self.tokenizer,
                validated_data_dir=validated_data_dir,
                batch_size=self.hw_config.batch_size,
                min_quality_score=0.6
            )

        return train_loader
    
    def train_complete_optimized(self, epochs: int = 3, resume_checkpoint: str = None):
        """Run complete optimized training with full monitoring."""

        # Training started - notifications only

        self.start_time = time.time()
        
        try:
            # Create unified data loader
            train_loader = self.create_unified_dataloader()
        except Exception as e:
            error_msg = f"Failed to create data loader: {str(e)}\n{traceback.format_exc()}"
            self.alerter.training_crashed(error_msg)
            raise
        
        # Log initial configuration to W&B
        if self.training_logger.use_wandb:
            # Get dataset size safely
            try:
                if hasattr(train_loader, 'dataset'):
                    total_samples = len(train_loader.dataset)
                else:
                    total_samples = len(train_loader) * self.hw_config.batch_size
            except:
                total_samples = 2490151  # Fallback estimate

            config_metrics = {
                'config/batch_size': self.hw_config.batch_size,
                'config/gradient_accumulation_steps': self.hw_config.gradient_accumulation_steps,
                'config/effective_batch_size': self.hw_config.batch_size * self.hw_config.gradient_accumulation_steps,
                'config/cpu_threads': self.hw_config.cpu_threads,
                'config/total_samples': total_samples,
                'config/model_parameters': sum(p.numel() for p in self.model.parameters()),
            }
            self.training_logger.log_metrics(config_metrics, step=0)

        # Send training started notification
        try:
            if hasattr(train_loader, 'dataset'):
                dataset_info = f"JSONL datasets, {len(train_loader.dataset):,} samples"
            else:
                dataset_info = f"JSONL datasets, ~{len(train_loader) * self.hw_config.batch_size:,} samples"
        except:
            dataset_info = "JSONL datasets"
        self.alerter.training_started(
            model_type=type(self.model).__name__,
            epochs=epochs,
            batch_size=self.hw_config.batch_size,
            dataset_info=dataset_info
        )
        
        # Calculate total training steps for scheduler
        total_training_steps = len(train_loader) * epochs // self.hw_config.gradient_accumulation_steps
        warmup_steps = min(1000, total_training_steps // 10)  # 10% warmup, max 1000 steps
        
        # Setup optimizer with model-size-specific learning rate
        # Larger models need lower learning rates
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'name_or_path'):
            model_name = str(self.model.config.name_or_path).lower()
            if 'xxl' in model_name:
                learning_rate = 1e-4  # Conservative for 11B model
                print(f"🎯 Using learning rate {learning_rate} for FLAN-T5-XXL (11B)")
            elif 'xl' in model_name:
                learning_rate = 3e-4  # Moderate for 3B model
            else:
                learning_rate = 5e-4  # Higher for smaller models
        else:
            learning_rate = 5e-4

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )

        # Resume from checkpoint if provided
        starting_step = 0
        starting_epoch = 0
        if resume_checkpoint:
            print(f"📂 Resuming from checkpoint: {resume_checkpoint}")
            try:
                # Use CheckpointManager to load complete state including RNG
                checkpoint_info = CheckpointManager.load_checkpoint(
                    checkpoint_dir=resume_checkpoint,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )

                # Restore training state
                starting_step = checkpoint_info.get('global_step', 0)
                starting_epoch = checkpoint_info.get('epoch', 0)
                self.epoch_losses = checkpoint_info.get('epoch_losses', [])
                best_loss = checkpoint_info.get('best_loss', float('inf'))

                print(f"✅ Resumed from step {starting_step}, epoch {starting_epoch}")
                if best_loss != float('inf'):
                    print(f"   Previous best loss: {best_loss:.4f}")
                print(f"   RNG states restored: {checkpoint_info.get('has_rng_states', False)}")
                print(f"   Optimizer state restored: {checkpoint_info.get('has_optimizer_state', False)}")
                print(f"   Scheduler state restored: {checkpoint_info.get('has_scheduler_state', False)}")

            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                print(f"   Starting from scratch")
                traceback.print_exc()
                best_loss = float('inf')
        else:
            best_loss = float('inf')

        print(f"Training setup:")
        print(f"  Total steps: {total_training_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")
        print(f"  Peak learning rate: {learning_rate} (optimized for model size)")
        print(f"  Scheduler: Linear decay with warmup")
        print(f"  Batch size: {self.hw_config.batch_size}")
        print(f"  Gradient accumulation: {self.hw_config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.hw_config.batch_size * self.hw_config.gradient_accumulation_steps}")
        print(f"  Training format: Input→Target seq2seq for proper instruction following")
        if resume_checkpoint:
            print(f"  Resuming from: Step {starting_step}, Epoch {starting_epoch}")

        # Early stopping setup
        patience = 3  # Stop if no improvement for 3 checkpoints
        patience_counter = 0
        best_checkpoint_path = None
        
        # Setup gradient scaler for mixed precision (disabled for bfloat16)
        # GradScaler is not needed/compatible with bfloat16, only with float16
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        
        # Recipe quality validation prompts
        self.validation_prompts = [
            "Create a simple pasta dish with tomatoes and basil.",
            "Make a healthy breakfast with eggs and vegetables.",
            "Design a quick 15-minute dinner recipe."
        ]
        
        # Training loop
        self.model.train()
        total_steps = starting_step  # Resume from checkpoint step

        try:
            for epoch in range(starting_epoch, epochs):
                self.current_epoch = epoch + 1
                
                # Reset Rust dataloader for new epoch (required for multi-epoch training)
                if hasattr(train_loader, 'reset'):
                    train_loader.reset()
                    print(f"🔄 Reset dataloader for epoch {self.current_epoch}")
                
                # Epoch started
                print(f"🚀 Starting epoch {self.current_epoch}/{epochs}")
                epoch_loss = 0
                batch_count = 0
                epoch_start_time = time.time()
                optimizer.zero_grad()
                
                try:
                    for batch_idx, batch in enumerate(train_loader):
                        # AGGRESSIVE memory management - clear every 5 batches to prevent leaks
                        if batch_count % 5 == 0:
                            torch.cuda.empty_cache()
                            import gc
                            gc.collect()

                        # Move to GPU with non-blocking transfer for speed
                        batch = {k: v.to(self.model.device, non_blocking=True) if hasattr(v, 'to') else v
                                for k, v in batch.items()}
                        
                        # Forward pass with mixed precision and memory management
                        # Clone the tensor outside the compiled context to prevent overwriting
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.enable_mixed_precision):
                            # Filter batch to only include model inputs (exclude quality_scores)
                            model_inputs = {k: v for k, v in batch.items() if k != 'quality_scores'}
                            outputs = self.model(**model_inputs)
                            # Clone the loss tensor to prevent CUDA graph overwriting issues
                            loss = outputs.loss.clone() / self.hw_config.gradient_accumulation_steps
                            
                            # Check for NaN loss (indicates training instability)
                            if torch.isnan(loss):
                                print(f"⚠️  NaN loss detected at batch {batch_count}, skipping...")
                                continue
                        
                        # Store loss value before clearing tensors
                        loss_value = loss.item()
                        
                        # Backward pass (no scaling needed for bfloat16)
                        loss.backward()

                        # AGGRESSIVE cleanup - delete ALL intermediate tensors immediately
                        del outputs, loss, model_inputs
                        # Also clear the batch dict to free GPU memory
                        for k in list(batch.keys()):
                            if hasattr(batch[k], 'cpu'):
                                del batch[k]
                        del batch
                        
                        batch_count += 1
                        epoch_loss += loss_value
                        
                        # Gradient step (no scaling for bfloat16)
                        if batch_count % self.hw_config.gradient_accumulation_steps == 0:
                            # Clip gradients
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                            # Step optimizer and scheduler
                            optimizer.step()
                            scheduler.step()  # Update learning rate
                            optimizer.zero_grad(set_to_none=True)  # More memory efficient
                            total_steps += 1

                            # Update step for W&B
                            self.training_logger.set_step(total_steps)

                            # AGGRESSIVE: Clear cache after each optimizer step to prevent memory creep
                            torch.cuda.empty_cache()
                        
                        # Logging every 50 batches
                        if batch_count % 50 == 0:
                            self.log_progress(
                                epoch, epochs, batch_count, epoch_loss, total_steps, 
                                train_loader, optimizer.param_groups[0]['lr'], batch_idx
                            )
                        
                        # Save checkpoint every 1000 steps with best model tracking (skip step 0)
                        if total_steps > 0 and total_steps % 1000 == 0:
                            checkpoint_path = self.save_checkpoint(
                                total_steps,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=self.current_epoch
                            )

                            # Early stopping logic
                            current_loss = epoch_loss / max(batch_count, 1)
                            if current_loss < best_loss:
                                best_loss = current_loss
                                patience_counter = 0
                                best_checkpoint_path = checkpoint_path
                                print(f"✅ New best checkpoint at step {total_steps}: loss={current_loss:.4f}")
                            else:
                                patience_counter += 1
                                print(f"⚠️  No improvement for {patience_counter}/{patience} checkpoints")
                                
                            # Early stopping - only after minimum epochs completed
                            if patience_counter >= patience and total_steps > 5000 and epoch >= 2:  # Complete at least 3 epochs
                                print(f"🛑 Early stopping triggered after {epoch + 1} epochs! Best model: {best_checkpoint_path}")
                                self.alerter.training_completed(
                                    (time.time() - self.start_time) / 3600,  # duration_hours
                                    {'train_loss': best_loss, 'reason': 'Early stopping - no improvement'}  # final_metrics
                                )
                                return best_checkpoint_path
                
                except Exception as e:
                    error_msg = f"Training failed at epoch {self.current_epoch}, batch {batch_count}: {str(e)}\n{traceback.format_exc()}"
                    self.alerter.training_crashed(error_msg, epoch=self.current_epoch)
                    raise
            
                # End of epoch processing
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
                self.epoch_losses.append(avg_epoch_loss)
                
                print(f"Epoch {self.current_epoch}/{epochs}: Loss {avg_epoch_loss:.4f}, Time {epoch_time:.1f}s")
                
                # Generate sample recipe for quality monitoring every epoch (4090 optimized)
                if self.current_epoch % 1 == 0 and total_steps > 1000:  # After warmup only
                    self.generate_sample_recipe_4090(total_steps)
            
            # Log epoch metrics to W&B
            if self.training_logger.use_wandb:
                epoch_metrics = {
                    'train/epoch_loss': avg_epoch_loss,
                    'train/epoch_time': epoch_time,
                    'train/epoch': self.current_epoch,
                }
                
                # Add system metrics
                system_metrics = self.system_monitor.get_system_metrics()
                epoch_metrics.update(system_metrics)
                
                # Add data pipeline metrics
                epoch_metrics.update({
                    'data_pipeline/samples_processed': batch_count * self.hw_config.batch_size,
                    'data_pipeline/batches_processed': batch_count,
                })
                
                self.training_logger.log_metrics(epoch_metrics, step=total_steps)
            
            # Send progress notification
            samples_per_sec = (batch_count * self.hw_config.batch_size) / epoch_time if epoch_time > 0 else 0
            
            self.alerter.training_progress(
                epoch=self.current_epoch,
                total_epochs=epochs,
                loss=avg_epoch_loss,
                lr=optimizer.param_groups[0]['lr'],
                samples_per_sec=samples_per_sec
            )
        
        except Exception as e:
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.alerter.training_crashed(error_msg, epoch=self.current_epoch)
            raise
        
        # Training completed
        total_time = time.time() - self.start_time
        total_hours = total_time / 3600

        print(f"Training complete: {total_hours:.2f}h, saving to {self.output_dir}")

        try:
            # Save final model - handle torch.compile() wrapper
            model_to_save = self.model
            if hasattr(self.model, '_orig_mod'):
                print(f"  📦 Unwrapping torch.compile() model for final save...")
                model_to_save = self.model._orig_mod

            model_to_save.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"✅ Final model saved to {self.output_dir}")
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}\n{traceback.format_exc()}"
            self.alerter.training_crashed(error_msg)
            raise
        
        # Final metrics
        final_metrics = {
            'train_loss': self.epoch_losses[-1] if self.epoch_losses else 0,
            'total_epochs': epochs,
            'total_hours': total_hours,
            'total_steps': total_steps
        }
        
        # Send completion notification
        self.alerter.training_completed(
            duration_hours=total_hours,
            final_metrics=final_metrics
        )
        
        # Final W&B log
        if self.training_logger.use_wandb:
            final_wandb_metrics = {
                'final/total_time_hours': total_hours,
                'final/final_loss': final_metrics['train_loss'],
                'final/total_steps': total_steps,
            }
            self.training_logger.log_metrics(final_wandb_metrics, step=total_steps)
            
            # Finish W&B run
            try:
                wandb.finish()
                print("📊 W&B session completed")
            except:
                pass
    
    def log_progress(self, epoch, total_epochs, batch_count, epoch_loss, total_steps, train_loader, lr, batch_idx):
        """Log training progress."""
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        
        # Minimal logging - only critical info
        if batch_count % 500 == 0:  # Reduced frequency
            print(f"Step {total_steps:,} | Loss: {avg_loss:.4f}")
        
        # Calculate approximate samples per second
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        samples_processed = batch_count * self.hw_config.batch_size
        samples_per_sec = samples_processed / elapsed_time
        
        # Data pipeline bottleneck detection
        if samples_per_sec < 5 and batch_count > 10:  # After warmup
            bottleneck_details = {
                'samples_per_sec': samples_per_sec,
                'expected_speed': 50,
                'severity': 'critical' if samples_per_sec < 1 else 'warning',
                'recommendation': 'Consider optimizing data preprocessing or increasing batch size'
            }
            self.alerter.data_pipeline_bottleneck("Slow Data Loading", bottleneck_details)
        
        # Detect memory issues
        if torch.cuda.is_available():
            gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
            if gpu_memory_percent > 95:
                bottleneck_details = {
                    'samples_per_sec': samples_per_sec,
                    'gpu_memory_percent': gpu_memory_percent,
                    'severity': 'critical',
                    'recommendation': 'Reduce batch size or enable gradient checkpointing'
                }
                self.alerter.data_pipeline_bottleneck("GPU Memory Exhaustion", bottleneck_details)

        # W&B logging
        if self.training_logger.use_wandb:
            metrics = {
                'train/loss': avg_loss,
                'train/learning_rate': lr,
                'train/epoch': epoch + (batch_idx / len(train_loader)),  # True fractional epoch
            }
            
            # Add data pipeline metrics
            elapsed_time = time.time() - self.start_time if self.start_time else 1
            samples_processed = batch_count * self.hw_config.batch_size
            samples_per_sec = samples_processed / elapsed_time
            metrics.update({
                'data_pipeline/samples_processed': samples_processed,
                'data_pipeline/samples_per_second': samples_per_sec,
                'data_pipeline/batch_count': batch_count,
            })
            
            # Add system metrics
            system_metrics = self.system_monitor.get_system_metrics()
            metrics.update(system_metrics)
            
            self.training_logger.log_metrics(metrics, step=total_steps)
    
    def generate_sample_recipe_4090(self, step):
        """Generate a sample recipe for quality monitoring (4090 24GB optimized)."""
        try:
            # Clear cache before generation to free memory
            torch.cuda.empty_cache()
            
            self.model.eval()
            prompt = self.validation_prompts[step % len(self.validation_prompts)]
            
            # Very conservative tokenization for 4090
            inputs = self.tokenizer(
                f"Generate a detailed recipe: {prompt}",
                return_tensors="pt",
                truncation=True,
                max_length=32  # Very short for memory safety
            ).to(self.model.device)
            
            # Memory-optimized generation for 4090
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,  # Conservative for 4090
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False,  # Disable KV cache
                    num_beams=1  # No beam search to save memory
                )
            
            # Quick decode and cleanup
            recipe = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            recipe = recipe.replace(f"Generate a detailed recipe: {prompt}", "").strip()
            
            print(f"\n🍳 Sample Recipe (Step {step}):")
            print(f"Prompt: {prompt}")
            print(f"Recipe: {recipe[:120]}..." if len(recipe) > 120 else f"Recipe: {recipe}")
            print("-" * 50)
            
            # Cleanup tensors immediately
            del inputs, outputs
            
        except torch.cuda.OutOfMemoryError:
            print(f"⚠️  Skipping sample generation - GPU memory full")
        except Exception as e:
            print(f"❌ Sample generation failed: {e}")
        finally:
            torch.cuda.empty_cache()  # Always clear cache
            self.model.train()  # Return to training mode
    
    def save_checkpoint(self, step, optimizer=None, scheduler=None, epoch=None):
        """Save training checkpoint with full training state including RNG states."""
        checkpoint_dir = f"{self.output_dir}/checkpoint-{step}"

        try:
            # Use CheckpointManager to save complete state including RNG
            CheckpointManager.save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch if epoch is not None else self.current_epoch,
                global_step=step,
                best_loss=getattr(self, 'best_loss', float('inf')),
                epoch_losses=self.epoch_losses,
                tokenizer=self.tokenizer
            )

            print(f"💾 Complete checkpoint saved: {checkpoint_dir}")

        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            traceback.print_exc()
            print(f"⚠️  Training will continue, but resume may not work properly")

        return checkpoint_dir  # Return path for best model tracking

def main():
    """Main training function with argument parsing."""
    import argparse
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    parser = argparse.ArgumentParser(description='Complete Optimized Training Pipeline')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--enable-profiling', action='store_true', help='Enable profiling')
    parser.add_argument('--profile-schedule', type=str, help='Profiling schedule')
    parser.add_argument('--model-output', type=str, required=True, help='Output directory for model')
    parser.add_argument('--pretrained-model', type=str, required=False, help='Pretrained model name (not needed when resuming)')
    parser.add_argument('--alert-phone', type=str, help='Phone number for SMS alerts')
    parser.add_argument('--discord-webhook', type=str, help='Discord webhook URL')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint path')
    parser.add_argument('--resume-from-checkpoint', type=str, help='Resume from specific checkpoint path')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument('--enable-mixed-precision', action='store_true', help='Enable mixed precision training (FP16/BF16)')
    parser.add_argument('--disable-compilation', action='store_true', help='Disable torch.compile() optimization')
    parser.add_argument('--disable-cudagraphs', action='store_true', help='Disable CUDA Graphs to prevent tensor overwrite issues')
    parser.add_argument('--dataloader-num-workers', type=int, default=8, help='Number of data loading workers (use 12-16 for Ryzen 3900X)')

    args = parser.parse_args()

    # Training pipeline starting

    # Load model and tokenizer
    checkpoint_path = args.resume_from_checkpoint or args.resume

    # Validate that either checkpoint or pretrained model is provided
    if not checkpoint_path and not args.pretrained_model:
        parser.error("Either --pretrained-model or --resume-from-checkpoint must be provided")

    if checkpoint_path:
        print(f"📂 Loading from checkpoint: {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    else:
        print(f"📥 Loading pretrained model: {args.pretrained_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        
        # Enable Flash Attention 2 for 20-30% speedup
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.pretrained_model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            print("⚡ Flash Attention 2 enabled for 20-30% faster training!")
        except Exception as e:
            print(f"⚠️ Flash Attention 2 not available, falling back to standard attention: {e}")
            model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model is already on GPU from device_map="auto" or needs to be moved
    # Don't move again if already on GPU (device_map="auto" already placed it)
    if not next(model.parameters()).is_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        print(f"✅ Model moved to GPU")
    elif torch.cuda.is_available():
        print(f"✅ Model already on GPU (via device_map='auto')")
    
    # Create trainer
    trainer = CompleteOptimizedTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.model_output,
        batch_size=args.batch_size,
        discord_webhook=args.discord_webhook,
        alert_phone=args.alert_phone,
        wandb_project="chef-genius-optimized",
        use_wandb=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        enable_mixed_precision=args.enable_mixed_precision,
        disable_compilation=args.disable_compilation,
        disable_cudagraphs=args.disable_cudagraphs,
        dataloader_num_workers=args.dataloader_num_workers
    )
    
    # Start training with error handling
    try:
        trainer.train_complete_optimized(
            epochs=args.epochs,
            resume_checkpoint=checkpoint_path
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.alerter.training_crashed("Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        # Error already sent via global crash handler
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 Failed to start training: {e}")
        sys.exit(1)
