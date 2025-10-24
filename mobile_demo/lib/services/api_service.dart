import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../models/recipe.dart';

class ApiService {
  // Change this to your actual backend URL
  static const String baseUrl = 'http://localhost:8000';
  
  // For demo purposes, we'll also include a mock mode
  static const bool mockMode = true;
  
  late final http.Client _client;
  String? _sessionId;

  ApiService() {
    _client = http.Client();
  }

  Map<String, String> get _headers => {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  // Recipe Generation
  Future<Recipe> generateRecipe({
    required List<String> ingredients,
    String? cuisine,
    List<String>? dietaryRestrictions,
    String difficulty = 'medium',
    String? mealType,
    int servings = 4,
    int? cookingTime,
  }) async {
    if (mockMode) {
      return _getMockRecipe(ingredients, cuisine);
    }

    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/recipes/generate'),
        headers: _headers,
        body: jsonEncode({
          'ingredients': ingredients,
          'cuisine': cuisine,
          'dietary_restrictions': dietaryRestrictions ?? [],
          'difficulty': difficulty,
          'meal_type': mealType,
          'servings': servings,
          'cooking_time': cookingTime,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return Recipe.fromJson(data['recipe']);
      } else {
        throw Exception('Failed to generate recipe: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Vision Analysis
  Future<VisionAnalysisResult> analyzeImage(File imageFile, {String analysisType = 'comprehensive'}) async {
    if (mockMode) {
      return _getMockVisionAnalysis();
    }

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/vision/analyze-food-multimodal'),
      );
      
      request.fields['analysis_type'] = analysisType;
      request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

      final streamedResponse = await _client.send(request);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return VisionAnalysisResult.fromJson(data['results']);
      } else {
        throw Exception('Failed to analyze image: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Fridge Scan
  Future<Map<String, dynamic>> scanFridge(File imageFile, {bool generateRecipes = true}) async {
    if (mockMode) {
      return _getMockFridgeScan();
    }

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/vision/scan-fridge'),
      );
      
      request.fields['generate_recipes'] = generateRecipes.toString();
      request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

      final streamedResponse = await _client.send(request);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to scan fridge: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Chef Assistant
  Future<String> startChefSession({String? initialMessage}) async {
    if (mockMode) {
      _sessionId = 'mock_session_${DateTime.now().millisecondsSinceEpoch}';
      return _sessionId!;
    }

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/chef/start-session'),
      );
      
      if (initialMessage != null) {
        request.fields['initial_message'] = initialMessage;
      }

      final streamedResponse = await _client.send(request);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _sessionId = data['session_id'];
        return _sessionId!;
      } else {
        throw Exception('Failed to start chef session: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<ChefResponse> chatWithChef(String message, {File? image}) async {
    if (mockMode) {
      return _getMockChefResponse(message);
    }

    if (_sessionId == null) {
      await startChefSession();
    }

    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/chef/chat/$_sessionId'),
      );
      
      request.fields['message'] = message;
      if (image != null) {
        request.files.add(await http.MultipartFile.fromPath('image', image.path));
      }

      final streamedResponse = await _client.send(request);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return ChefResponse.fromJson(data);
      } else {
        throw Exception('Failed to chat with chef: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Mock data for demo purposes
  Recipe _getMockRecipe(List<String> ingredients, String? cuisine) {
    final mainIngredient = ingredients.isNotEmpty ? ingredients.first : 'vegetables';
    
    return Recipe(
      title: '${cuisine ?? 'Delicious'} $mainIngredient Stir Fry',
      description: 'A quick and healthy stir fry that brings out the best flavors of fresh ingredients.',
      ingredients: [
        Ingredient(name: mainIngredient, amount: 2, unit: 'cups'),
        Ingredient(name: 'olive oil', amount: 2, unit: 'tbsp'),
        Ingredient(name: 'garlic', amount: 3, unit: 'cloves'),
        Ingredient(name: 'soy sauce', amount: 3, unit: 'tbsp'),
        Ingredient(name: 'ginger', amount: 1, unit: 'tsp'),
      ],
      instructions: [
        'Heat olive oil in a large pan or wok over medium-high heat.',
        'Add minced garlic and ginger, stir for 30 seconds.',
        'Add $mainIngredient and stir-fry for 3-4 minutes.',
        'Add soy sauce and cook for another 2 minutes.',
        'Serve hot over rice or noodles.',
      ],
      prepTime: 10,
      cookTime: 15,
      servings: 4,
      difficulty: 'easy',
      cuisine: cuisine,
      dietaryTags: ['quick', 'healthy'],
    );
  }

  VisionAnalysisResult _getMockVisionAnalysis() {
    return VisionAnalysisResult(
      detectedIngredients: [
        DetectedIngredient(ingredient: 'tomatoes', confidence: 0.95, category: 'vegetables'),
        DetectedIngredient(ingredient: 'onions', confidence: 0.88, category: 'vegetables'),
        DetectedIngredient(ingredient: 'garlic', confidence: 0.82, category: 'aromatics'),
      ],
      cookingStage: {'stage': 'raw', 'confidence': 0.9},
      dishClassification: {'primary_dish_type': 'vegetables', 'confidence': 0.85},
      caption: 'Fresh vegetables ready for cooking',
      recipeSuggestions: ['Vegetable stir fry', 'Roasted vegetables', 'Vegetable soup'],
    );
  }

  Map<String, dynamic> _getMockFridgeScan() {
    return {
      'detected_ingredients': [
        {'name': 'eggs', 'confidence': 0.95, 'freshness': 'fresh'},
        {'name': 'milk', 'confidence': 0.90, 'freshness': 'fresh'},
        {'name': 'cheese', 'confidence': 0.88, 'freshness': 'good'},
        {'name': 'bell peppers', 'confidence': 0.85, 'freshness': 'fresh'},
        {'name': 'leftover pasta', 'confidence': 0.78, 'freshness': 'good'},
      ],
      'freshness_assessment': {
        'overall_score': 0.87,
        'items_to_use_soon': ['leftover pasta'],
      },
      'organization_tips': [
        'Store eggs in the main compartment, not the door',
        'Keep vegetables in the crisper drawer',
      ],
      'recipe_suggestions': [
        'Breakfast frittata with bell peppers',
        'Cheesy pasta bake',
        'Vegetable omelet',
      ],
    };
  }

  ChefResponse _getMockChefResponse(String message) {
    final responses = [
      'That sounds delicious! Let me help you with that recipe.',
      'Great choice! Here are some tips for making that dish perfectly.',
      'I love that combination of flavors. Here\'s what I recommend...',
      'Perfect! That\'s one of my favorite dishes to make. Let\'s get started!',
    ];
    
    return ChefResponse(
      response: responses[DateTime.now().millisecond % responses.length],
      actions: ['provide_recipe', 'suggest_techniques'],
      toolsUsed: ['recipe_generator', 'cooking_timer'],
      confidence: 0.92,
    );
  }

  void dispose() {
    _client.close();
  }
}