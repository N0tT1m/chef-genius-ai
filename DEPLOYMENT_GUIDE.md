# Chef Genius MCP System - Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 16GB+ RAM (24GB recommended for RTX 4090)
- NVIDIA GPU with CUDA support (optional but recommended)
- Python 3.11+

### 1. Clone and Setup
```bash
cd /Users/timmy/workspace/ai-apps/chef-genius
chmod +x Makefile.mcp
```

### 2. Start the System
```bash
# Quick start (builds and starts everything)
make -f Makefile.mcp quick-start

# Or step by step
make -f Makefile.mcp build
make -f Makefile.mcp up
```

### 3. Verify Deployment
```bash
# Check all services
make -f Makefile.mcp health

# Test MCP servers
make -f Makefile.mcp test-mcp

# Check service endpoints
make -f Makefile.mcp check-services
```

## 🌐 Service URLs

Once deployed, access these services:

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main application interface |
| **Backend API** | http://localhost:8000 | REST API endpoints |
| **Recipe Server** | http://localhost:8001 | MCP recipe generation |
| **Knowledge Server** | http://localhost:8002 | MCP knowledge retrieval |
| **Tool Server** | http://localhost:8003 | MCP tool integration |
| **Weaviate** | http://localhost:8080 | Vector database |
| **Grafana** | http://localhost:3001 | Monitoring dashboard |
| **Prometheus** | http://localhost:9090 | Metrics collection |

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│                 Frontend (React/Next.js)                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND API                               │
│              (FastAPI + MCP Client)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ MCP Protocol
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  MCP SERVERS                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Recipe    │ │ Knowledge   │ │    Tool     │            │
│  │   Server    │ │   Server    │ │   Server    │            │
│  │ (T5-Large)  │ │   (RAG)     │ │ (Utilities) │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA LAYER                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │  Weaviate   │ │    Redis    │ │   Recipe    │            │
│  │ (Vectors)   │ │  (Cache)    │ │  Database   │            │
│  │             │ │             │ │ (4.1M recipes)           │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Environment Variables
Create `.env` file in the root directory:
```bash
# Database
DATABASE_URL=sqlite:///./chef_genius.db

# External APIs
NUTRITION_API_KEY=your_nutrition_api_key

# Model Paths
MODEL_PATH=/Users/timmy/workspace/ai-apps/chef-genius/models/recipe_generation

# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Monitoring
LOG_LEVEL=INFO
```

### Hardware Optimization
For RTX 4090 + Ryzen 9 3900X:
```yaml
# docker-compose.mcp-full.yml
recipe-server:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
      limits:
        memory: 16G
```

## 📈 Performance Monitoring

### Grafana Dashboard
1. Open http://localhost:3001
2. Login: admin/admin123
3. Navigate to Chef Genius MCP System dashboard

### Key Metrics to Monitor
- **Recipe Generation Rate**: Requests per second
- **Response Times**: 95th percentile latency
- **Cache Hit Rate**: Should be >90%
- **Vector Search Performance**: Query response time
- **System Resources**: CPU, memory, disk usage

### Health Checks
```bash
# All services status
curl http://localhost:8000/health

# Detailed health report
curl http://localhost:8000/health/detailed

# MCP servers status
curl http://localhost:8000/health/mcp

# RAG system status
curl http://localhost:8000/health/rag
```

## 🧪 Testing the System

### Basic Functionality Test
```bash
# Test recipe generation
curl -X POST http://localhost:8001/tools/generate_recipe \
  -H "Content-Type: application/json" \
  -d '{
    "ingredients": ["chicken", "rice", "broccoli"],
    "cuisine": "Asian",
    "dietary_restrictions": ["healthy"]
  }'

# Test knowledge search
curl -X POST http://localhost:8002/tools/search_recipes \
  -H "Content-Type: application/json" \
  -d '{
    "query": "healthy chicken recipes",
    "top_k": 5
  }'

# Test nutrition analysis
curl -X POST http://localhost:8003/tools/analyze_nutrition \
  -H "Content-Type: application/json" \
  -d '{
    "recipe": "Grilled chicken with rice and vegetables",
    "servings": 4
  }'
```

### Performance Benchmark
```bash
# Run benchmarks
make -f Makefile.mcp benchmark

# Load test (requires apache bench)
ab -n 100 -c 10 -T application/json -p test_recipe.json http://localhost:8001/tools/generate_recipe
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Weaviate Connection Failed
```bash
# Check Weaviate status
curl http://localhost:8080/v1/meta

# Restart Weaviate
docker-compose -f docker-compose.mcp-full.yml restart weaviate
```

#### 2. MCP Server Not Responding
```bash
# Check logs
make -f Makefile.mcp logs-recipe
make -f Makefile.mcp logs-knowledge
make -f Makefile.mcp logs-tools

# Restart specific server
docker-compose -f docker-compose.mcp-full.yml restart recipe-server
```

#### 3. Out of Memory Errors
```bash
# Check memory usage
docker stats

# Reduce model size or enable quantization
# Edit docker-compose.mcp-full.yml:
environment:
  - QUANTIZATION_BITS=4
  - ENABLE_GRADIENT_CHECKPOINTING=true
```

#### 4. Slow Recipe Generation
```bash
# Check GPU utilization
nvidia-smi

# Verify CUDA is available
docker-compose -f docker-compose.mcp-full.yml exec recipe-server python -c "import torch; print(torch.cuda.is_available())"

# Enable CPU fallback if needed
docker-compose -f docker-compose.mcp-full.yml up -d --scale recipe-server=2
```

### Log Analysis
```bash
# View all logs
make -f Makefile.mcp logs

# Search for errors
docker-compose -f docker-compose.mcp-full.yml logs | grep ERROR

# Monitor real-time
docker-compose -f docker-compose.mcp-full.yml logs -f --tail=50
```

## 📦 Data Management

### Backup System Data
```bash
# Backup all data
make -f Makefile.mcp backup

# Manual backup
docker run --rm -v chef-genius_weaviate_data:/data -v $(PWD)/backups:/backup alpine tar czf /backup/weaviate-backup.tar.gz -C /data .
```

### Seed Recipe Data
```bash
# Load 4.1M recipe dataset into Weaviate
make -f Makefile.mcp seed-data

# Verify data loaded
curl http://localhost:8080/v1/objects | jq '.objects | length'
```

### Database Migration
```bash
# Run database migrations
make -f Makefile.mcp migrate

# Reset database (caution!)
docker-compose -f docker-compose.mcp-full.yml down -v
make -f Makefile.mcp up
```

## 🔒 Security Considerations

### Production Deployment
1. **Change default passwords**:
   ```bash
   # Grafana admin password
   GF_SECURITY_ADMIN_PASSWORD=your_secure_password
   ```

2. **Enable authentication**:
   ```yaml
   # Add to docker-compose.mcp-full.yml
   environment:
     - ENABLE_AUTH=true
     - JWT_SECRET=your_jwt_secret
   ```

3. **Use SSL/TLS**:
   ```bash
   # Add SSL certificates to nginx/ssl/
   cp your_cert.pem nginx/ssl/
   cp your_key.pem nginx/ssl/
   ```

4. **Network security**:
   ```yaml
   # Restrict port exposure in production
   ports:
     - "127.0.0.1:8080:8080"  # Weaviate
     - "127.0.0.1:9090:9090"  # Prometheus
   ```

## 🎯 Performance Tuning

### For RTX 4090 (24GB VRAM)
```yaml
recipe-server:
  environment:
    - BATCH_SIZE=32
    - GRADIENT_ACCUMULATION_STEPS=1
    - USE_QUANTIZATION=false
    - ENABLE_MIXED_PRECISION=true
```

### For RTX 3080 (10GB VRAM)
```yaml
recipe-server:
  environment:
    - BATCH_SIZE=8
    - GRADIENT_ACCUMULATION_STEPS=4
    - USE_QUANTIZATION=true
    - QUANTIZATION_BITS=8
```

### CPU-Only Deployment
```yaml
recipe-server:
  environment:
    - DEVICE=cpu
    - WORKERS=8
    - USE_TORCH_COMPILE=false
  deploy:
    resources:
      limits:
        memory: 8G
        cpus: '4'
```

## 📋 Maintenance

### Regular Tasks
```bash
# Weekly cleanup
make -f Makefile.mcp clean

# Update system
docker-compose -f docker-compose.mcp-full.yml pull
make -f Makefile.mcp build
make -f Makefile.mcp up

# Monitor disk usage
df -h
docker system df
```

### Scaling
```bash
# Scale MCP servers
docker-compose -f docker-compose.mcp-full.yml up -d --scale recipe-server=3

# Load balancer configuration
# Edit nginx/nginx.conf for multiple backends
```

## 🎉 Success Metrics

Your deployment is successful when:

- ✅ All services show "healthy" status
- ✅ Recipe generation < 2 seconds response time
- ✅ Vector search returns results < 500ms
- ✅ Cache hit rate > 90%
- ✅ System memory usage < 85%
- ✅ No critical alerts in monitoring

## 📞 Support

For issues or questions:
1. Check logs: `make -f Makefile.mcp logs`
2. Review health status: `make -f Makefile.mcp health`
3. Run diagnostics: `make -f Makefile.mcp test-mcp`

The system is now ready for production use with advanced RAG + MCP capabilities!