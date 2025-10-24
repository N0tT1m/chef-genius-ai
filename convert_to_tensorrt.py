#!/usr/bin/env python3
"""
Convert Transformers model to TensorRT for high-performance inference.
Supports: Transformers -> ONNX -> TensorRT conversion pipeline.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

import torch
import torch.onnx
import onnx
import onnxruntime as ort
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    GenerationConfig
)

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT not available. Install with: pip install nvidia-tensorrt")
    TRT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
    """High-performance model converter for Transformers -> TensorRT pipeline."""
    
    def __init__(self, model_path: str, output_dir: str, precision: str = "fp16"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.precision = precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for intermediate files
        self.onnx_path = self.output_dir / "model.onnx"
        self.tensorrt_path = self.output_dir / "model.trt"
        self.config_path = self.output_dir / "conversion_config.json"
        
        logger.info(f"Initialized converter:")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Precision: {self.precision}")
        logger.info(f"  Device: {self.device}")

    def load_model(self) -> Tuple[Any, Any, Any]:
        """Load Transformers model, tokenizer, and config."""
        logger.info("Loading Transformers model...")
        
        try:
            # Load config first to check model type
            config = AutoConfig.from_pretrained(self.model_path)
            logger.info(f"Model type: {config.model_type}")
            logger.info(f"Model architecture: {config.architectures}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.precision == "fp16" else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            model.eval()
            
            logger.info(f"Model loaded successfully:")
            logger.info(f"  Parameters: {model.num_parameters():,}")
            logger.info(f"  Vocab size: {config.vocab_size}")
            logger.info(f"  Max length: {getattr(config, 'max_position_embeddings', 'Unknown')}")
            
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def export_to_onnx(self, model: Any, tokenizer: Any, config: Any) -> bool:
        """Export Transformers model to ONNX format."""
        logger.info("Exporting to ONNX...")
        
        try:
            # Prepare sample inputs
            sample_text = "Generate a recipe with chicken and rice."
            inputs = tokenizer(
                sample_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            )
            
            # Move inputs to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            logger.info(f"Sample input shape: {input_ids.shape}")
            
            # Dynamic axes for variable length inputs
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    (input_ids, attention_mask),
                    str(self.onnx_path),
                    export_params=True,
                    opset_version=17,  # Latest stable opset
                    do_constant_folding=True,
                    input_names=["input_ids", "attention_mask"],
                    output_names=["logits"],
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            
            logger.info(f"ONNX export completed: {self.onnx_path}")
            
            # Verify ONNX model
            try:
                onnx_model = onnx.load(str(self.onnx_path))
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verification passed")
                
                # Get model info
                file_size = self.onnx_path.stat().st_size / (1024 * 1024)
                logger.info(f"ONNX model size: {file_size:.2f} MB")
                
            except Exception as e:
                logger.warning(f"ONNX verification failed: {e}")
                
            return True
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    def test_onnx_inference(self, tokenizer: Any) -> bool:
        """Test ONNX model inference."""
        logger.info("Testing ONNX inference...")
        
        try:
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            
            # Prepare test input
            test_text = "Create a recipe with beef and potatoes."
            inputs = tokenizer(
                test_text,
                return_tensors="np",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Run inference
            start_time = time.time()
            outputs = session.run(
                None,
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
            )
            inference_time = time.time() - start_time
            
            logger.info(f"ONNX inference successful:")
            logger.info(f"  Input shape: {inputs['input_ids'].shape}")
            logger.info(f"  Output shape: {outputs[0].shape}")
            logger.info(f"  Inference time: {inference_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX inference test failed: {e}")
            return False

    def convert_to_tensorrt(self) -> bool:
        """Convert ONNX model to TensorRT engine."""
        if not TRT_AVAILABLE:
            logger.error("TensorRT not available. Install nvidia-tensorrt package.")
            return False
            
        logger.info("Converting ONNX to TensorRT...")
        
        try:
            # TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)
            
            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(self.onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(f"Parser error: {parser.get_error(error)}")
                    return False
            
            # Builder configuration
            config = builder.create_builder_config()
            
            # Set precision
            if self.precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision")
            elif self.precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Enabled INT8 precision")
                
            # Memory pool
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
            
            # Optimization profiles for dynamic shapes
            profile = builder.create_optimization_profile()
            
            # Input shapes: [batch_size, sequence_length]
            min_shape = (1, 1)
            opt_shape = (1, 512)
            max_shape = (4, 1024)
            
            profile.set_shape("input_ids", min_shape, opt_shape, max_shape)
            profile.set_shape("attention_mask", min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            logger.info(f"Optimization profile:")
            logger.info(f"  Min shape: {min_shape}")
            logger.info(f"  Opt shape: {opt_shape}")
            logger.info(f"  Max shape: {max_shape}")
            
            # Build engine
            logger.info("Building TensorRT engine... (this may take several minutes)")
            start_time = time.time()
            
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
                
            build_time = time.time() - start_time
            logger.info(f"Engine build completed in {build_time:.2f}s")
            
            # Save engine
            with open(self.tensorrt_path, 'wb') as f:
                f.write(serialized_engine)
                
            # Get file size
            file_size = self.tensorrt_path.stat().st_size / (1024 * 1024)
            logger.info(f"TensorRT engine saved: {self.tensorrt_path}")
            logger.info(f"Engine size: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False

    def test_tensorrt_inference(self) -> bool:
        """Test TensorRT engine inference."""
        if not TRT_AVAILABLE:
            return False
            
        logger.info("Testing TensorRT inference...")
        
        try:
            # Load engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(self.tensorrt_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
                
            if engine is None:
                logger.error("Failed to load TensorRT engine")
                return False
                
            context = engine.create_execution_context()
            
            # Prepare test data
            batch_size = 1
            seq_length = 512
            
            input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.int32)
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int32)
            
            # Allocate GPU memory
            input_ids_gpu = cuda.mem_alloc(input_ids.nbytes)
            attention_mask_gpu = cuda.mem_alloc(attention_mask.nbytes)
            
            # Output allocation (estimate size)
            vocab_size = 50257  # Common GPT vocab size
            output_size = batch_size * seq_length * vocab_size * 4  # float32
            output_gpu = cuda.mem_alloc(output_size)
            
            # Copy inputs to GPU
            cuda.memcpy_htod(input_ids_gpu, input_ids.numpy().ascontiguousarray())
            cuda.memcpy_htod(attention_mask_gpu, attention_mask.numpy().ascontiguousarray())
            
            # Set binding shapes
            context.set_binding_shape(0, input_ids.shape)
            context.set_binding_shape(1, attention_mask.shape)
            
            # Execute inference
            start_time = time.time()
            context.execute_v2([int(input_ids_gpu), int(attention_mask_gpu), int(output_gpu)])
            cuda.Context.synchronize()
            inference_time = time.time() - start_time
            
            logger.info(f"TensorRT inference successful:")
            logger.info(f"  Input shape: {input_ids.shape}")
            logger.info(f"  Inference time: {inference_time:.3f}s")
            logger.info(f"  Speed improvement: {(inference_time < 0.1)} (< 100ms)")
            
            return True
            
        except Exception as e:
            logger.error(f"TensorRT inference test failed: {e}")
            return False

    def save_conversion_config(self, model_info: Dict[str, Any]) -> None:
        """Save conversion configuration for the Go API."""
        config = {
            "model_path": str(self.model_path),
            "onnx_path": str(self.onnx_path),
            "tensorrt_path": str(self.tensorrt_path),
            "precision": self.precision,
            "conversion_time": time.time(),
            "model_info": model_info,
            "api_config": {
                "max_batch_size": 4,
                "max_sequence_length": 1024,
                "vocab_size": model_info.get("vocab_size", 50257),
                "model_type": model_info.get("model_type", "causal_lm")
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        logger.info(f"Conversion config saved: {self.config_path}")

    def convert_full_pipeline(self) -> bool:
        """Run the complete conversion pipeline."""
        logger.info("Starting full conversion pipeline...")
        start_time = time.time()
        
        try:
            # Step 1: Load model
            model, tokenizer, config = self.load_model()
            
            model_info = {
                "model_type": config.model_type,
                "vocab_size": config.vocab_size,
                "num_parameters": model.num_parameters(),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', 2048),
                "architectures": config.architectures
            }
            
            # Step 2: Export to ONNX
            if not self.export_to_onnx(model, tokenizer, config):
                return False
                
            # Step 3: Test ONNX inference
            if not self.test_onnx_inference(tokenizer):
                logger.warning("ONNX inference test failed, but continuing...")
                
            # Step 4: Convert to TensorRT
            if not self.convert_to_tensorrt():
                return False
                
            # Step 5: Test TensorRT inference
            if not self.test_tensorrt_inference():
                logger.warning("TensorRT inference test failed, but conversion completed")
                
            # Step 6: Save configuration
            self.save_conversion_config(model_info)
            
            total_time = time.time() - start_time
            logger.info(f"Conversion pipeline completed successfully in {total_time:.2f}s")
            logger.info(f"Files created:")
            logger.info(f"  ONNX: {self.onnx_path}")
            logger.info(f"  TensorRT: {self.tensorrt_path}")
            logger.info(f"  Config: {self.config_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Conversion pipeline failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Convert Transformers model to TensorRT")
    parser.add_argument("--model-path", required=True, help="Path to Transformers model")
    parser.add_argument("--output-dir", required=True, help="Output directory for converted models")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16", 
                        help="Precision for TensorRT engine")
    parser.add_argument("--test-only", action="store_true", help="Only test existing models")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
        
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available. TensorRT conversion requires GPU.")
        sys.exit(1)
        
    logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create converter
    converter = ModelConverter(args.model_path, args.output_dir, args.precision)
    
    # Run conversion
    if args.test_only:
        # Test existing models
        success = True
        if converter.onnx_path.exists():
            success &= converter.test_onnx_inference(None)
        if converter.tensorrt_path.exists():
            success &= converter.test_tensorrt_inference()
    else:
        # Full conversion pipeline
        success = converter.convert_full_pipeline()
    
    if success:
        logger.info("✅ Conversion completed successfully!")
        logger.info(f"Use the TensorRT engine in your Go API: {converter.tensorrt_path}")
    else:
        logger.error("❌ Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()