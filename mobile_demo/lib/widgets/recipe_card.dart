import 'package:flutter/material.dart';
import '../models/recipe.dart';

class RecipeCard extends StatelessWidget {
  final Recipe recipe;

  const RecipeCard({
    super.key,
    required this.recipe,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        recipe.title,
                        style: theme.textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        recipe.description,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: theme.colorScheme.onSurface.withOpacity(0.7),
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: _getDifficultyColor(recipe.difficulty).withOpacity(0.1),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    recipe.difficulty.toUpperCase(),
                    style: TextStyle(
                      color: _getDifficultyColor(recipe.difficulty),
                      fontWeight: FontWeight.semibold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 20),
            
            // Info Row
            Row(
              children: [
                _InfoChip(
                  icon: Icons.timer,
                  label: recipe.totalTimeFormatted,
                  color: Colors.blue,
                ),
                const SizedBox(width: 12),
                _InfoChip(
                  icon: Icons.people,
                  label: '${recipe.servings} servings',
                  color: Colors.green,
                ),
                if (recipe.cuisine != null) ...[
                  const SizedBox(width: 12),
                  _InfoChip(
                    icon: Icons.public,
                    label: recipe.cuisine!,
                    color: Colors.orange,
                  ),
                ],
              ],
            ),
            
            const SizedBox(height: 20),
            
            // Dietary Tags
            if (recipe.dietaryTags != null && recipe.dietaryTags!.isNotEmpty) ...[
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: recipe.dietaryTags!.map((tag) => Chip(
                  label: Text(
                    tag,
                    style: const TextStyle(fontSize: 12),
                  ),
                  backgroundColor: theme.colorScheme.secondaryContainer,
                  side: BorderSide.none,
                )).toList(),
              ),
              const SizedBox(height: 20),
            ],
            
            // Ingredients Section
            Text(
              'Ingredients',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ...recipe.ingredients.map((ingredient) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                children: [
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      color: theme.colorScheme.primary,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      ingredient.displayText,
                      style: theme.textTheme.bodyMedium,
                    ),
                  ),
                ],
              ),
            )),
            
            const SizedBox(height: 20),
            
            // Instructions Section
            Text(
              'Instructions',
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ...recipe.instructions.asMap().entries.map((entry) {
              final index = entry.key;
              final instruction = entry.value;
              
              return Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 24,
                      height: 24,
                      decoration: BoxDecoration(
                        color: theme.colorScheme.primary,
                        shape: BoxShape.circle,
                      ),
                      child: Center(
                        child: Text(
                          '${index + 1}',
                          style: TextStyle(
                            color: theme.colorScheme.onPrimary,
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        instruction,
                        style: theme.textTheme.bodyMedium,
                      ),
                    ),
                  ],
                ),
              );
            }),
            
            const SizedBox(height: 12),
            
            // Action Buttons
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _shareRecipe(context),
                    icon: const Icon(Icons.share, size: 18),
                    label: const Text('Share'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _saveRecipe(context),
                    icon: const Icon(Icons.bookmark, size: 18),
                    label: const Text('Save'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Color _getDifficultyColor(String difficulty) {
    switch (difficulty.toLowerCase()) {
      case 'easy':
        return Colors.green;
      case 'medium':
        return Colors.orange;
      case 'hard':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  void _shareRecipe(BuildContext context) {
    // In a real app, this would use the share plugin
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Recipe sharing would be implemented here'),
      ),
    );
  }

  void _saveRecipe(BuildContext context) {
    // In a real app, this would save to local storage or user account
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Recipe saved! (Demo)'),
        backgroundColor: Colors.green,
      ),
    );
  }
}

class _InfoChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;

  const _InfoChip({
    required this.icon,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.medium,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
}