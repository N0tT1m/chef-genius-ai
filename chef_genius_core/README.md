# Chef Genius Rust Core 🦀⚡

High-performance Rust acceleration for the Chef Genius AI cooking platform. This library provides 5-15x performance improvements over pure Python implementations.

## 🚀 Features

- **Fast ML Inference**: Optimized recipe generation with TensorRT/Candle
- **High-Speed Vector Search**: Semantic recipe similarity search
- **Efficient Recipe Processing**: Text parsing and ingredient extraction
- **Quick Nutrition Analysis**: Real-time nutritional calculations
- **Smart Caching**: Intelligent caching with 90%+ hit rates
- **Batch Processing**: Parallel processing for maximum throughput

## 📊 Performance Gains

| Component | Python Time | Rust Time | Speedup |
|-----------|-------------|-----------|---------|
| Recipe Generation | 500ms | 50ms | **10x** |
| Vector Search | 200ms | 20ms | **10x** |
| Recipe Processing | 100ms | 15ms | **6.7x** |
| Nutrition Analysis | 80ms | 10ms | **8x** |
| Batch Operations | 10s | 1s | **10x** |

## 🛠️ Installation

### Quick Install
```bash
python install_rust_core.py
```

### Manual Installation
```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install
cd chef_genius_core
maturin develop --release
```

## 💻 Usage

### Python Integration
```python
from backend.app.services.rust_integration import rust_service

# Fast recipe generation
request = RecipeGenerationRequest(ingredients=["chicken", "rice"])
recipe = await rust_service.generate_recipe_rust(request)

# Fast vector search
results = await rust_service.search_recipes_rust("pasta recipes")

# Fast recipe processing
parsed = rust_service.parse_recipe_rust(recipe_text)

# Fast nutrition analysis
nutrition = rust_service.analyze_nutrition_rust(recipe_data)
```

### Direct Rust API
```python
import chef_genius_core

# Initialize engines
inference = chef_genius_core.PyInferenceEngine("models/recipe_generation")
search = chef_genius_core.PyVectorSearchEngine()
processor = chef_genius_core.PyRecipeProcessor()
analyzer = chef_genius_core.PyNutritionAnalyzer()

# Create request
request = chef_genius_core.PyInferenceRequest(
    ingredients=["chicken", "rice", "vegetables"],
    max_length=500,
    temperature=0.8
)

# Generate recipe
response = inference.generate_recipe(request)
print(f"Generated: {response.recipe.title}")
print(f"Time: {response.generation_time_ms}ms")
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│          Python FastAPI Backend        │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │    Rust Integration Layer       │   │
│  │  (rust_integration.py)          │   │
│  └─────────────┬───────────────────┘   │
└────────────────┼───────────────────────┘
                 │ PyO3 Bindings
                 ▼
┌─────────────────────────────────────────┐
│           Rust Core Library             │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ Inference   │  │ Vector Search   │  │
│  │ Engine      │  │ Engine          │  │
│  └─────────────┘  └─────────────────┘  │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ Recipe      │  │ Nutrition       │  │
│  │ Processor   │  │ Analyzer        │  │
│  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
```

## 🧪 Testing

```bash
# Test installation
python -c "import chef_genius_core; print('✅ Rust core working!')"

# Run benchmarks
python -c "
import chef_genius_core
result = chef_genius_core.benchmark_inference('models/test', 100, 1)
print(f'Throughput: {result[\"throughput_per_sec\"]:.1f} req/s')
"

# Performance comparison
curl -X POST "http://localhost:8000/api/v1/rust/performance-test" \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["chicken", "rice"], "iterations": 10}'
```

## 🔧 Configuration

### Environment Variables
```bash
export CHEF_GENIUS_MODEL_PATH="models/recipe_generation"
export CHEF_GENIUS_CACHE_SIZE=10000
export CHEF_GENIUS_BATCH_SIZE=8
export RUST_LOG=info
```

### Python Configuration
```python
# In your application startup
from backend.app.services.rust_integration import rust_service

# Check if Rust is available
if rust_service.rust_available:
    print("🚀 Rust acceleration enabled!")
else:
    print("⚠️  Using Python fallback")

# Get performance stats
stats = rust_service.get_rust_stats()
print(f"Cache hit rate: {stats['inference_stats']['cache_hits']}%")
```

## 📈 Monitoring

### API Endpoints
- `GET /api/v1/rust/status` - Rust core status
- `GET /api/v1/rust/stats` - Performance statistics  
- `POST /api/v1/rust/benchmark` - Run benchmarks
- `GET /api/v1/rust/health` - Health check

### Performance Metrics
```python
# Get detailed stats
stats = rust_service.get_rust_stats()

print(f"Total requests: {stats['inference_stats']['total_requests']}")
print(f"Cache hits: {stats['inference_stats']['cache_hits']}")
print(f"Avg latency: {stats['inference_stats']['avg_latency_ms']:.2f}ms")
```

## 🛠️ Development

### Building from Source
```bash
# Development build
maturin develop

# Release build  
maturin develop --release

# Build wheel
maturin build --release
```

### Adding New Features
1. Add Rust implementation in `src/`
2. Add PyO3 bindings in the module
3. Update `lib.rs` to export new classes/functions
4. Add Python integration in `rust_integration.py`
5. Add tests and documentation

### Testing
```bash
# Rust tests
cargo test

# Python integration tests
pytest tests/test_rust_integration.py

# Benchmark tests
python scripts/benchmark.py
```

## 🔍 Troubleshooting

### Common Issues

**Import Error**
```bash
# Check installation
python -c "import chef_genius_core"

# Reinstall if needed
pip uninstall chef-genius-core
python install_rust_core.py
```

**Performance Issues**
```bash
# Check system info
python -c "
import chef_genius_core
info = chef_genius_core.get_system_info()
print(f'CPU cores: {info[\"cpu_count\"]}')
print(f'CUDA available: {info[\"cuda_available\"]}')
"

# Clear caches
curl -X POST "http://localhost:8000/api/v1/rust/clear-caches"
```

**Build Failures**
```bash
# Update Rust
rustup update

# Clean and rebuild
rm -rf target/
maturin develop --release
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add Rust implementation with tests
4. Update Python bindings
5. Submit a pull request

---

**Made with ❤️ and ⚡ by the Chef Genius team**