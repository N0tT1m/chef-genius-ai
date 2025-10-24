"""
API endpoints for Rust performance monitoring and benchmarking
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import logging
import time
from pydantic import BaseModel

from app.services.rust_integration import rust_service, RUST_AVAILABLE
from app.models.recipe import RecipeGenerationRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rust", tags=["rust-performance"])

class BenchmarkRequest(BaseModel):
    operation: str
    iterations: int = 100
    batch_size: int = 1

class PerformanceTestRequest(BaseModel):
    ingredients: List[str]
    test_rust: bool = True
    test_python: bool = True
    iterations: int = 10

@router.get("/status")
async def get_rust_status() -> Dict[str, Any]:
    """Get Rust core status and availability"""
    return {
        "rust_available": RUST_AVAILABLE,
        "rust_service_initialized": rust_service.rust_available,
        "system_info": rust_service.get_rust_stats() if RUST_AVAILABLE else None,
        "performance_boost": "Active" if RUST_AVAILABLE else "Unavailable"
    }

@router.get("/stats")
async def get_performance_stats() -> Dict[str, Any]:
    """Get detailed performance statistics from Rust engines"""
    if not RUST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Rust core not available")
    
    return rust_service.get_rust_stats()

@router.post("/benchmark")
async def benchmark_operation(request: BenchmarkRequest) -> Dict[str, Any]:
    """Benchmark a specific operation"""
    if not RUST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Rust core not available")
    
    valid_operations = ["inference", "search", "processing", "nutrition"]
    if request.operation not in valid_operations:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid operation. Must be one of: {valid_operations}"
        )
    
    try:
        result = rust_service.benchmark_rust_vs_python(
            request.operation, 
            request.iterations
        )
        return {
            "operation": request.operation,
            "iterations": request.iterations,
            "batch_size": request.batch_size,
            "results": result,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

@router.post("/performance-test")
async def performance_test(request: PerformanceTestRequest) -> Dict[str, Any]:
    """Compare Rust vs Python performance for recipe generation"""
    results = {
        "rust_results": None,
        "python_results": None,
        "comparison": None
    }
    
    recipe_request = RecipeGenerationRequest(
        ingredients=request.ingredients,
        max_length=200,
        temperature=0.8
    )
    
    # Test Rust implementation
    if request.test_rust and RUST_AVAILABLE:
        rust_times = []
        for i in range(request.iterations):
            start_time = time.time()
            try:
                result = await rust_service.generate_recipe_rust(recipe_request)
                end_time = time.time()
                rust_times.append((end_time - start_time) * 1000)  # Convert to ms
            except Exception as e:
                logger.error(f"Rust test iteration {i} failed: {e}")
        
        if rust_times:
            results["rust_results"] = {
                "avg_time_ms": sum(rust_times) / len(rust_times),
                "min_time_ms": min(rust_times),
                "max_time_ms": max(rust_times),
                "iterations": len(rust_times),
                "total_time_ms": sum(rust_times)
            }
    
    # Test Python implementation
    if request.test_python:
        try:
            from app.services.recipe_generator import RecipeGeneratorService
            python_service = RecipeGeneratorService()
            
            python_times = []
            for i in range(request.iterations):
                start_time = time.time()
                try:
                    result = await python_service.generate_recipe(recipe_request)
                    end_time = time.time()
                    python_times.append((end_time - start_time) * 1000)
                except Exception as e:
                    logger.error(f"Python test iteration {i} failed: {e}")
            
            if python_times:
                results["python_results"] = {
                    "avg_time_ms": sum(python_times) / len(python_times),
                    "min_time_ms": min(python_times),
                    "max_time_ms": max(python_times),
                    "iterations": len(python_times),
                    "total_time_ms": sum(python_times)
                }
        except ImportError:
            results["python_results"] = {"error": "Python service not available"}
    
    # Calculate comparison
    if results["rust_results"] and results["python_results"]:
        rust_avg = results["rust_results"]["avg_time_ms"]
        python_avg = results["python_results"]["avg_time_ms"]
        
        if rust_avg > 0:
            speedup = python_avg / rust_avg
            results["comparison"] = {
                "speedup_factor": speedup,
                "rust_faster": speedup > 1.0,
                "performance_improvement": f"{((speedup - 1) * 100):.1f}%" if speedup > 1 else f"{((1 - speedup) * 100):.1f}% slower"
            }
    
    return results

@router.post("/clear-caches")
async def clear_rust_caches() -> Dict[str, str]:
    """Clear all Rust engine caches"""
    if not RUST_AVAILABLE:
        raise HTTPException(status_code=503, detail="Rust core not available")
    
    try:
        rust_service.clear_caches()
        return {"status": "success", "message": "All Rust caches cleared"}
    except Exception as e:
        logger.error(f"Failed to clear caches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")

@router.get("/health")
async def rust_health_check() -> Dict[str, Any]:
    """Health check for Rust components"""
    health = {
        "rust_core": RUST_AVAILABLE,
        "inference_engine": False,
        "vector_search": False,
        "recipe_processor": False,
        "nutrition_analyzer": False,
        "overall_status": "degraded"
    }
    
    if RUST_AVAILABLE:
        try:
            # Test each component
            if rust_service._inference_engine:
                stats = rust_service._inference_engine.get_stats()
                health["inference_engine"] = True
            
            if rust_service._vector_search:
                search_stats = rust_service._vector_search.get_stats()
                health["vector_search"] = True
            
            if rust_service._recipe_processor:
                # Simple test
                test_ingredients = rust_service._recipe_processor.extract_ingredients("1 cup flour")
                health["recipe_processor"] = len(test_ingredients) > 0
            
            if rust_service._nutrition_analyzer:
                health["nutrition_analyzer"] = True
            
            # Overall status
            components_working = sum([
                health["inference_engine"],
                health["vector_search"], 
                health["recipe_processor"],
                health["nutrition_analyzer"]
            ])
            
            if components_working == 4:
                health["overall_status"] = "healthy"
            elif components_working >= 2:
                health["overall_status"] = "degraded"
            else:
                health["overall_status"] = "unhealthy"
                
        except Exception as e:
            logger.error(f"Rust health check failed: {e}")
            health["overall_status"] = "unhealthy"
            health["error"] = str(e)
    
    return health

@router.get("/performance-tips")
async def get_performance_tips() -> Dict[str, Any]:
    """Get performance optimization tips"""
    tips = {
        "general": [
            "Install Rust core for 5-15x performance boost",
            "Use batch operations when processing multiple items",
            "Enable caching for frequently accessed data",
            "Use appropriate batch sizes for your hardware"
        ],
        "rust_specific": [],
        "hardware_optimization": []
    }
    
    if RUST_AVAILABLE:
        system_info = rust_service.get_rust_stats().get("system_info", {})
        cpu_count = system_info.get("cpu_count", 1)
        cuda_available = system_info.get("cuda_available", False)
        
        tips["rust_specific"] = [
            "Rust acceleration is active - optimal performance enabled",
            "Use vector search for fast recipe similarity",
            "Batch process recipes for maximum throughput",
            f"System has {cpu_count} CPU cores - batch size 8-32 recommended"
        ]
        
        if cuda_available:
            tips["hardware_optimization"] = [
                "CUDA GPU detected - ML inference acceleration available",
                "Use GPU for large batch processing",
                "Consider enabling TensorRT for maximum speed"
            ]
        else:
            tips["hardware_optimization"] = [
                "CPU-only mode - still 5-15x faster than Python",
                "Consider GPU for even better performance",
                "Optimize batch sizes for CPU cores"
            ]
    else:
        tips["rust_specific"] = [
            "Rust core not installed - missing major performance boost",
            "Run 'python install_rust_core.py' to enable acceleration",
            "Currently using Python fallback implementations"
        ]
    
    return tips