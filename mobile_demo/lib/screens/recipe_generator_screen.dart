import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../models/recipe.dart';
import '../widgets/recipe_card.dart';
import '../widgets/loading_animation.dart';

class RecipeGeneratorScreen extends StatefulWidget {
  const RecipeGeneratorScreen({super.key});

  @override
  State<RecipeGeneratorScreen> createState() => _RecipeGeneratorScreenState();
}

class _RecipeGeneratorScreenState extends State<RecipeGeneratorScreen> {
  final TextEditingController _ingredientsController = TextEditingController();
  final List<String> _ingredients = [];
  String? _selectedCuisine;
  String _selectedDifficulty = 'medium';
  final List<String> _dietaryRestrictions = [];
  bool _isLoading = false;
  Recipe? _generatedRecipe;

  final List<String> _cuisineTypes = [
    'Italian',
    'Mexican',
    'Asian',
    'Indian',
    'Mediterranean',
    'American',
    'French',
    'Thai',
    'Japanese',
    'Middle Eastern',
  ];

  final List<String> _difficultyLevels = ['easy', 'medium', 'hard'];
  
  final List<String> _dietaryOptions = [
    'vegetarian',
    'vegan',
    'gluten-free',
    'dairy-free',
    'low-carb',
    'keto',
    'paleo',
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Recipe Generator'),
        backgroundColor: theme.colorScheme.surface,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.orange.withOpacity(0.1),
                    Colors.deepOrange.withOpacity(0.1),
                  ],
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.auto_fix_high,
                    size: 32,
                    color: Colors.orange,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'AI Recipe Generator',
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Tell me what ingredients you have, and I\'ll create a perfect recipe for you!',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurface.withOpacity(0.7),
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Ingredients Section
            _buildSection(
              title: 'Ingredients',
              icon: Icons.shopping_basket,
              child: Column(
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _ingredientsController,
                          decoration: InputDecoration(
                            hintText: 'Add an ingredient...',
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 16,
                              vertical: 12,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      IconButton.filled(
                        onPressed: _addIngredient,
                        icon: const Icon(Icons.add),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  if (_ingredients.isNotEmpty)
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: _ingredients.map((ingredient) => Chip(
                        label: Text(ingredient),
                        deleteIcon: const Icon(Icons.close, size: 18),
                        onDeleted: () => _removeIngredient(ingredient),
                      )).toList(),
                    ),
                ],
              ),
            ),
            
            const SizedBox(height: 20),
            
            // Cuisine Selection
            _buildSection(
              title: 'Cuisine Type',
              icon: Icons.public,
              child: DropdownButtonFormField<String>(
                value: _selectedCuisine,
                decoration: InputDecoration(
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                ),
                hint: const Text('Select cuisine (optional)'),
                items: _cuisineTypes.map((cuisine) => DropdownMenuItem(
                  value: cuisine,
                  child: Text(cuisine),
                )).toList(),
                onChanged: (value) => setState(() => _selectedCuisine = value),
              ),
            ),
            
            const SizedBox(height: 20),
            
            // Difficulty Level
            _buildSection(
              title: 'Difficulty Level',
              icon: Icons.trending_up,
              child: SegmentedButton<String>(
                selected: {_selectedDifficulty},
                onSelectionChanged: (Set<String> selection) {
                  setState(() => _selectedDifficulty = selection.first);
                },
                segments: _difficultyLevels.map((level) => ButtonSegment(
                  value: level,
                  label: Text(level.toUpperCase()),
                )).toList(),
              ),
            ),
            
            const SizedBox(height: 20),
            
            // Dietary Restrictions
            _buildSection(
              title: 'Dietary Restrictions',
              icon: Icons.health_and_safety,
              child: Wrap(
                spacing: 8,
                runSpacing: 8,
                children: _dietaryOptions.map((option) => FilterChip(
                  label: Text(option),
                  selected: _dietaryRestrictions.contains(option),
                  onSelected: (selected) {
                    setState(() {
                      if (selected) {
                        _dietaryRestrictions.add(option);
                      } else {
                        _dietaryRestrictions.remove(option);
                      }
                    });
                  },
                )).toList(),
              ),
            ),
            
            const SizedBox(height: 32),
            
            // Generate Button
            SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: _ingredients.isNotEmpty && !_isLoading 
                    ? _generateRecipe 
                    : null,
                icon: _isLoading 
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.auto_awesome),
                label: Text(_isLoading ? 'Generating...' : 'Generate Recipe'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Generated Recipe
            if (_isLoading)
              const LoadingAnimation()
            else if (_generatedRecipe != null)
              RecipeCard(recipe: _generatedRecipe!),
          ],
        ),
      ),
    );
  }

  Widget _buildSection({
    required String title,
    required IconData icon,
    required Widget child,
  }) {
    final theme = Theme.of(context);
    
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, size: 20, color: theme.colorScheme.primary),
            const SizedBox(width: 8),
            Text(
              title,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.semibold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        child,
      ],
    );
  }

  void _addIngredient() {
    final ingredient = _ingredientsController.text.trim();
    if (ingredient.isNotEmpty && !_ingredients.contains(ingredient)) {
      setState(() {
        _ingredients.add(ingredient);
        _ingredientsController.clear();
      });
    }
  }

  void _removeIngredient(String ingredient) {
    setState(() {
      _ingredients.remove(ingredient);
    });
  }

  Future<void> _generateRecipe() async {
    if (_ingredients.isEmpty) return;

    setState(() {
      _isLoading = true;
      _generatedRecipe = null;
    });

    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      final recipe = await apiService.generateRecipe(
        ingredients: _ingredients,
        cuisine: _selectedCuisine,
        dietaryRestrictions: _dietaryRestrictions.isNotEmpty 
            ? _dietaryRestrictions 
            : null,
        difficulty: _selectedDifficulty,
        servings: 4,
      );

      setState(() {
        _generatedRecipe = recipe;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to generate recipe: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  void dispose() {
    _ingredientsController.dispose();
    super.dispose();
  }
}