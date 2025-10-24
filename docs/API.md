# ChefGenius API Documentation

## Base URL
- Development: `http://localhost:8000`
- Production: `https://api.chefgenius.com`

## Authentication
The API uses JWT tokens for authentication. Include the token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

## Error Handling
All errors follow a consistent format:
```json
{
  "error": {
    "message": "Error description",
    "code": "ERROR_CODE",
    "details": {},
    "type": "ExceptionType"
  }
}
```

## Rate Limiting
- Default: 60 requests per minute per IP
- Headers returned: `X-RateLimit-Limit`, `X-RateLimit-Remaining`

## API Endpoints

### Recipes

#### List Recipes
```http
GET /api/v1/recipes
```

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 20, max: 100)
- `cuisine` (str): Filter by cuisine type
- `cooking_time` (int): Maximum cooking time in minutes
- `search` (str): Search in title and ingredients

**Response:**
```json
{
  "recipes": [
    {
      "id": "uuid",
      "title": "Recipe Title",
      "cuisine": "Italian",
      "cooking_time": 30,
      "servings": 4,
      "ingredients": ["ingredient1", "ingredient2"],
      "instructions": ["step1", "step2"],
      "nutrition": {
        "calories": 250,
        "protein": 15,
        "carbs": 30,
        "fat": 8
      },
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "pages": 5
  }
}
```

#### Get Recipe
```http
GET /api/v1/recipes/{recipe_id}
```

#### Create Recipe
```http
POST /api/v1/recipes
```

**Request Body:**
```json
{
  "title": "New Recipe",
  "cuisine": "Italian",
  "cooking_time": 30,
  "servings": 4,
  "ingredients": ["ingredient1", "ingredient2"],
  "instructions": ["step1", "step2"]
}
```

#### Update Recipe
```http
PUT /api/v1/recipes/{recipe_id}
```

#### Delete Recipe
```http
DELETE /api/v1/recipes/{recipe_id}
```

### Meal Planning

#### Get Meal Plans
```http
GET /api/v1/meal-plans
```

#### Generate Meal Plan
```http
POST /api/v1/meal-plans
```

**Request Body:**
```json
{
  "days": 7,
  "dietary_restrictions": ["vegetarian"],
  "calories_per_day": 2000,
  "preferences": {
    "cuisines": ["Italian", "Mediterranean"],
    "exclude_ingredients": ["nuts"]
  }
}
```

### Ingredient Substitutions

#### Get Substitutions
```http
POST /api/v1/substitutions
```

**Request Body:**
```json
{
  "ingredient": "butter",
  "amount": "1 cup",
  "recipe_context": "baking cookies"
}
```

**Response:**
```json
{
  "substitutions": [
    {
      "ingredient": "coconut oil",
      "amount": "3/4 cup",
      "confidence": 0.95,
      "notes": "Solid at room temperature, good for baking"
    }
  ]
}
```

### Vision Analysis

#### Analyze Food Image
```http
POST /api/v1/vision/analyze
```

**Request:** Multipart form data with image file

**Response:**
```json
{
  "ingredients": [
    {
      "name": "tomato",
      "confidence": 0.95,
      "bounding_box": [100, 100, 200, 200]
    }
  ],
  "suggested_recipes": ["recipe_id1", "recipe_id2"]
}
```

## WebSocket Endpoints

### Real-time Recipe Generation
```
ws://localhost:8000/ws/recipe-generation
```

Send JSON messages to generate recipes in real-time:
```json
{
  "type": "generate",
  "ingredients": ["chicken", "rice", "vegetables"],
  "cuisine": "Asian"
}
```

## Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

## SDKs and Examples

### Python SDK
```python
import chefgenius

client = chefgenius.Client(api_key="your-api-key")
recipes = client.recipes.list(cuisine="Italian")
```

### JavaScript/Node.js
```javascript
const ChefGenius = require('chefgenius-sdk');
const client = new ChefGenius({ apiKey: 'your-api-key' });
const recipes = await client.recipes.list({ cuisine: 'Italian' });
```

### cURL Examples
```bash
# Get recipes
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/recipes?cuisine=Italian"

# Create recipe
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title":"Pasta","cuisine":"Italian"}' \
  "http://localhost:8000/api/v1/recipes"
```