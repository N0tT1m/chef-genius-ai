# Chef Genius API Specification

## Overview

This specification outlines the API design for Chef Genius, an AI-powered recipe generation platform built on a fine-tuned transformer model with RAG (Retrieval Augmented Generation) and MCP (Model Context Protocol) integrations.

## Architecture Summary

### Core Components
- **Recipe Generation Model**: Fine-tuned GPT-2/T5 transformer model trained on 2M+ recipes
- **RAG System**: Vector-based recipe knowledge retrieval using sentence transformers
- **MCP Integration**: Model Context Protocol for structured interactions
- **Performance Optimization**: 4-bit quantization, caching, and GPU optimization

## Authentication

All API endpoints require authentication via Bearer token:

```
Authorization: Bearer <your-api-token>
```

## Base URL

```
https://api.chef-genius.com/v1
```

## Core Endpoints

### Recipe Generation

#### POST /recipes/generate

Generate a new recipe based on specified criteria using the trained model with RAG enhancement.

**Request Body:**
```json
{
  "ingredients": ["chicken", "broccoli", "rice"],
  "cuisine": "Asian",
  "dietary_restrictions": ["gluten-free"],
  "cooking_time": "30 minutes",
  "difficulty": "medium",
  "servings": 4,
  "meal_type": "dinner",
  "preferences": {
    "spice_level": "mild",
    "cooking_method": "stir-fry"
  },
  "use_rag": true,
  "context_recipes": 3
}
```

**Response:**
```json
{
  "id": "uuid",
  "title": "Asian Chicken and Broccoli Stir-Fry",
  "description": "A delicious gluten-free Asian-inspired dish",
  "ingredients": [
    {
      "name": "chicken breast",
      "amount": 1.5,
      "unit": "lbs",
      "notes": "cut into bite-sized pieces"
    },
    {
      "name": "broccoli florets",
      "amount": 2,
      "unit": "cups",
      "notes": "fresh or frozen"
    }
  ],
  "instructions": [
    "Heat oil in large wok over high heat",
    "Add chicken and stir-fry for 5-6 minutes",
    "Add broccoli and cook for 3-4 minutes"
  ],
  "nutrition": {
    "calories": 320,
    "protein": 35,
    "carbs": 28,
    "fat": 8,
    "fiber": 4
  },
  "prep_time": 15,
  "cook_time": 15,
  "total_time": 30,
  "servings": 4,
  "difficulty": "medium",
  "cuisine": "Asian",
  "dietary_tags": ["gluten-free"],
  "generated_at": "2024-01-15T10:30:00Z",
  "model_version": "chef-genius-v2.1",
  "rag_enhanced": true,
  "context_recipes_used": 3,
  "confidence_score": 0.92
}
```

#### POST /recipes/generate/enhanced

Generate recipe with tool enhancements (nutrition analysis, substitutions, scaling).

**Request Body:**
```json
{
  "ingredients": ["beef", "potatoes", "carrots"],
  "cuisine": "American",
  "servings": 6,
  "dietary_restrictions": ["low-sodium"],
  "enable_tools": {
    "nutrition_analysis": true,
    "ingredient_substitutions": true,
    "recipe_scaling": true,
    "cooking_timers": true
  }
}
```

**Response:**
```json
{
  "recipe": {
    // Standard recipe object
  },
  "enhancements": {
    "nutrition": {
      "detailed_breakdown": {},
      "health_score": 8.5,
      "recommendations": []
    },
    "substitutions": [
      {
        "original": "beef",
        "alternatives": ["turkey", "plant-based protein"],
        "reason": "lower sodium content"
      }
    ],
    "scaled_ingredients": [
      {
        "name": "beef",
        "original_amount": 1,
        "scaled_amount": 1.5,
        "unit": "lbs"
      }
    ],
    "cooking_times": {
      "beef": {"estimated_time": 25, "method": "braising"},
      "potatoes": {"estimated_time": 20, "method": "roasting"}
    }
  },
  "tool_enhanced": true
}
```

### Recipe Management

#### GET /recipes

List recipes with filtering and pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)
- `cuisine`: Filter by cuisine type
- `dietary_tags`: Filter by dietary restrictions
- `difficulty`: Filter by difficulty level
- `search`: Search in title and ingredients
- `generated_only`: Show only AI-generated recipes

**Response:**
```json
{
  "recipes": [
    {
      // Recipe objects
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  },
  "filters_applied": {
    "cuisine": "Italian",
    "difficulty": "easy"
  }
}
```

#### GET /recipes/{id}

Get a specific recipe by ID.

#### PUT /recipes/{id}

Update an existing recipe.

#### DELETE /recipes/{id}

Delete a recipe.

## RAG System Endpoints

### POST /rag/search

Search the recipe knowledge base using vector similarity.

**Request Body:**
```json
{
  "query": "spicy vegetarian pasta with tomatoes",
  "top_k": 5,
  "min_similarity": 0.3,
  "filters": {
    "cuisine": ["Italian", "Mediterranean"],
    "dietary_tags": ["vegetarian"]
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "recipe_id": "uuid",
      "title": "Spicy Arrabbiata Pasta",
      "similarity_score": 0.87,
      "ingredients": ["pasta", "tomatoes", "chili"],
      "cuisine": "Italian",
      "snippet": "A fiery Italian pasta with tomatoes..."
    }
  ],
  "query_embedding": "base64-encoded-vector",
  "search_time_ms": 45
}
```

### POST /rag/enhance

Use RAG to enhance an existing recipe with related knowledge.

**Request Body:**
```json
{
  "recipe_id": "uuid",
  "enhancement_type": "variations",
  "context_recipes": 3
}
```

### GET /rag/stats

Get RAG system performance statistics.

**Response:**
```json
{
  "total_recipes": 2150000,
  "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
  "query_count": 15420,
  "cache_hit_rate": "78.5%",
  "average_search_time_ms": 23,
  "index_last_updated": "2024-01-15T08:00:00Z"
}
```

## MCP Integration Endpoints

### POST /mcp/chat

Interact with the recipe model using Model Context Protocol.

**Request Body:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Create a healthy breakfast recipe with oats and berries"
    }
  ],
  "context": {
    "dietary_preferences": ["healthy", "high-fiber"],
    "time_constraint": "15 minutes",
    "skill_level": "beginner"
  },
  "tools": ["nutrition_calculator", "ingredient_substitution"],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Response:**
```json
{
  "message": {
    "role": "assistant",
    "content": "Here's a nutritious overnight oats recipe...",
    "tool_calls": [
      {
        "id": "call_123",
        "type": "function",
        "function": {
          "name": "nutrition_calculator",
          "arguments": {
            "ingredients": ["oats", "berries", "yogurt"],
            "servings": 1
          }
        }
      }
    ]
  },
  "tool_results": [
    {
      "call_id": "call_123",
      "result": {
        "calories": 285,
        "protein": 12,
        "fiber": 8
      }
    }
  ],
  "usage": {
    "prompt_tokens": 256,
    "completion_tokens": 384,
    "total_tokens": 640
  }
}
```

### POST /mcp/function-call

Execute specific functions through MCP protocol.

**Request Body:**
```json
{
  "function": "ingredient_substitution",
  "arguments": {
    "ingredient": "butter",
    "dietary_restrictions": ["vegan"],
    "recipe_context": "baking cookies"
  }
}
```

**Response:**
```json
{
  "result": {
    "substitutions": [
      {
        "name": "vegan butter",
        "ratio": "1:1",
        "notes": "Best for texture"
      },
      {
        "name": "coconut oil",
        "ratio": "0.75:1",
        "notes": "Slight coconut flavor"
      }
    ],
    "confidence": 0.94
  },
  "execution_time_ms": 12
}
```

## Utility Endpoints

### POST /nutrition/analyze

Analyze nutritional content of ingredients or recipes.

**Request Body:**
```json
{
  "ingredients": [
    {
      "name": "chicken breast",
      "amount": 8,
      "unit": "oz"
    }
  ],
  "servings": 2
}
```

### POST /substitutions/suggest

Get ingredient substitution suggestions.

### POST /recipes/scale

Scale recipe ingredients for different serving sizes.

### POST /meal-plans/generate

Generate meal plans using recipe generation model.

## Performance & Monitoring

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "rag_status": "ready",
  "database_status": "connected",
  "uptime_seconds": 3600,
  "version": "2.1.0"
}
```

### GET /metrics

System performance metrics.

**Response:**
```json
{
  "requests_per_minute": 45,
  "average_response_time_ms": 234,
  "model_memory_usage_gb": 3.2,
  "gpu_utilization_percent": 67,
  "cache_hit_rate": 0.78,
  "error_rate": 0.02
}
```

## Rate Limiting

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1000 requests/hour  
- **Enterprise**: Custom limits

Rate limit headers included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1642680000
```

## Error Handling

Standard HTTP status codes with detailed error responses:

```json
{
  "error": {
    "code": "INVALID_INGREDIENTS",
    "message": "At least one ingredient is required",
    "details": {
      "field": "ingredients",
      "constraint": "min_length"
    },
    "request_id": "req_123abc"
  }
}
```

Common error codes:
- `INVALID_INGREDIENTS`: Invalid or missing ingredients
- `MODEL_UNAVAILABLE`: Recipe generation model is unavailable
- `RAG_SEARCH_FAILED`: Vector search failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INSUFFICIENT_CREDITS`: API usage limits exceeded

## WebSocket Support

### Real-time Recipe Generation

```javascript
ws://api.chef-genius.com/v1/ws/generate

// Send generation request
{
  "type": "generate_recipe",
  "data": {
    "ingredients": ["pasta", "tomatoes"],
    "cuisine": "Italian"
  }
}

// Receive progress updates
{
  "type": "generation_progress",
  "progress": 0.6,
  "stage": "ingredient_processing"
}

// Receive completed recipe
{
  "type": "recipe_complete",
  "data": {
    // Recipe object
  }
}
```

## SDK Examples

### Python SDK

```python
from chef_genius import ChefGeniusClient

client = ChefGeniusClient(api_key="your-api-key")

# Generate recipe
recipe = client.recipes.generate(
    ingredients=["chicken", "vegetables"],
    cuisine="Mediterranean",
    dietary_restrictions=["gluten-free"]
)

# Search recipes with RAG
results = client.rag.search(
    query="healthy breakfast recipes",
    top_k=5
)

# MCP chat interaction
response = client.mcp.chat(
    messages=[
        {"role": "user", "content": "Create a vegan dinner recipe"}
    ],
    tools=["nutrition_calculator"]
)
```

### JavaScript SDK

```javascript
import { ChefGeniusSDK } from '@chef-genius/sdk';

const client = new ChefGeniusSDK({
  apiKey: 'your-api-key'
});

// Generate recipe with async/await
const recipe = await client.recipes.generate({
  ingredients: ['salmon', 'asparagus'],
  cuisine: 'Nordic',
  cookingTime: '25 minutes'
});

// Real-time generation with WebSocket
client.ws.generateRecipe({
  ingredients: ['tofu', 'vegetables'],
  onProgress: (progress) => console.log(`Progress: ${progress}%`),
  onComplete: (recipe) => console.log('Recipe ready:', recipe)
});
```

## Training Model Integration

The API leverages models trained using the CLI training script (`cli/train_recipe_model.py`) which includes:

### Model Architecture
- **Base Models**: GPT-2, T5, or Phi-3.5-mini-instruct
- **Training Data**: 2M+ recipes from multiple datasets
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training
- **Evaluation**: ROUGE, BLEU, custom recipe quality metrics

### Training Configuration
```python
# Extracted from training script
SUPPORTED_MODELS = ["gpt2", "t5", "phi-3.5"]
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 3
```

### Recipe Format
The model is trained on structured recipe format:
```
<TITLE>Recipe Name</TITLE>
<CUISINE>Cuisine Type</CUISINE>
<INGREDIENTS>ingredient1 | ingredient2 | ingredient3</INGREDIENTS>
<INSTRUCTIONS>Step 1. Step 2. Step 3.</INSTRUCTIONS>
```

## Deployment Requirements

### Hardware
- **GPU**: RTX 4090 or equivalent (24GB VRAM recommended)
- **RAM**: 32GB minimum
- **Storage**: 500GB SSD for models and data

### Software Stack
- **Framework**: FastAPI + PyTorch + Transformers
- **Database**: PostgreSQL with vector extensions
- **Cache**: Redis for response caching
- **Monitoring**: Prometheus + Grafana

### Environment Variables
```bash
MODEL_PATH=/models/recipe_generation
RAG_DB_PATH=/data/recipe_database.json
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2
CUDA_VISIBLE_DEVICES=0
API_KEY=your-secret-key
DATABASE_URL=postgresql://user:pass@host:5432/chef_genius
```

This specification provides a comprehensive API design that leverages the trained model architecture with modern RAG and MCP capabilities for enterprise-grade recipe generation.