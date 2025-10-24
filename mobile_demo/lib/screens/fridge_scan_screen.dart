import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../widgets/loading_animation.dart';

class FridgeScanScreen extends StatefulWidget {
  const FridgeScanScreen({super.key});

  @override
  State<FridgeScanScreen> createState() => _FridgeScanScreenState();
}

class _FridgeScanScreenState extends State<FridgeScanScreen> {
  File? _imageFile;
  bool _isScanning = false;
  Map<String, dynamic>? _scanResult;
  final ImagePicker _picker = ImagePicker();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Fridge Scanner'),
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
                    Colors.green.withOpacity(0.1),
                    Colors.teal.withOpacity(0.1),
                  ],
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.kitchen,
                    size: 32,
                    color: Colors.green,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Smart Fridge Scanner',
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Scan your fridge to see what ingredients you have and get recipe suggestions.',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurface.withOpacity(0.7),
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Image Upload Section
            if (_imageFile == null) ...[
              Container(
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  border: Border.all(
                    color: theme.colorScheme.outline.withOpacity(0.3),
                    width: 2,
                    style: BorderStyle.solid,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.kitchen,
                      size: 48,
                      color: theme.colorScheme.onSurface.withOpacity(0.5),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Take a photo of your fridge',
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: theme.colorScheme.onSurface.withOpacity(0.7),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Open your fridge and take a clear photo',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: theme.colorScheme.onSurface.withOpacity(0.5),
                      ),
                    ),
                  ],
                ),
              ),
              
              const SizedBox(height: 20),
              
              // Camera/Gallery Buttons
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () => _pickImage(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => _pickImage(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Gallery'),
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ] else ...[
              // Selected Image
              Container(
                width: double.infinity,
                height: 300,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  image: DecorationImage(
                    image: FileImage(_imageFile!),
                    fit: BoxFit.cover,
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // Action Buttons
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _clearImage,
                      icon: const Icon(Icons.close),
                      label: const Text('Remove'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isScanning ? null : _scanFridge,
                      icon: _isScanning 
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.search),
                      label: Text(_isScanning ? 'Scanning...' : 'Scan Fridge'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ],
            
            const SizedBox(height: 24),
            
            // Scan Results
            if (_isScanning)
              const VisionLoadingAnimation()
            else if (_scanResult != null)
              _buildScanResults(),
          ],
        ),
      ),
    );
  }

  Widget _buildScanResults() {
    final theme = Theme.of(context);
    final result = _scanResult!;
    
    return Column(
      children: [
        // Detected Ingredients
        if (result['detected_ingredients'] != null) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.inventory,
                        color: theme.colorScheme.primary,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Detected Ingredients',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  ...((result['detected_ingredients'] as List).map((ingredient) => 
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Row(
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: _getFreshnessColor(ingredient['freshness']),
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              ingredient['name'],
                              style: theme.textTheme.bodyMedium,
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: _getFreshnessColor(ingredient['freshness'])
                                  .withOpacity(0.1),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              ingredient['freshness'],
                              style: TextStyle(
                                color: _getFreshnessColor(ingredient['freshness']),
                                fontWeight: FontWeight.medium,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  )),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],
        
        // Organization Tips
        if (result['organization_tips'] != null && 
            (result['organization_tips'] as List).isNotEmpty) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.tips_and_updates,
                        color: Colors.orange,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Organization Tips',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  ...((result['organization_tips'] as List).map((tip) => 
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Icon(
                            Icons.lightbulb,
                            size: 16,
                            color: Colors.orange,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              tip,
                              style: theme.textTheme.bodyMedium,
                            ),
                          ),
                        ],
                      ),
                    ),
                  )),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
        ],
        
        // Recipe Suggestions
        if (result['recipe_suggestions'] != null && 
            (result['recipe_suggestions'] as List).isNotEmpty) ...[
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.restaurant_menu,
                        color: Colors.green,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Recipe Suggestions',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  ...((result['recipe_suggestions'] as List).map((recipe) => 
                    Padding(
                      padding: const EdgeInsets.only(bottom: 12),
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.primaryContainer.withOpacity(0.3),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: theme.colorScheme.primary.withOpacity(0.2),
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.restaurant,
                              color: theme.colorScheme.primary,
                              size: 20,
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                recipe,
                                style: theme.textTheme.bodyMedium?.copyWith(
                                  fontWeight: FontWeight.medium,
                                ),
                              ),
                            ),
                            Icon(
                              Icons.arrow_forward_ios,
                              size: 16,
                              color: theme.colorScheme.primary,
                            ),
                          ],
                        ),
                      ),
                    ),
                  )),
                  
                  const SizedBox(height: 16),
                  
                  // Generate Recipe Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _generateRecipeFromIngredients,
                      icon: const Icon(Icons.auto_awesome),
                      label: const Text('Generate Recipe'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
        
        // Freshness Assessment
        if (result['freshness_assessment'] != null) ...[
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.assessment,
                        color: Colors.blue,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Freshness Assessment',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Text(
                        'Overall Score: ',
                        style: theme.textTheme.bodyMedium,
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.green.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          '${(result['freshness_assessment']['overall_score'] * 100).toInt()}%',
                          style: TextStyle(
                            color: Colors.green,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                  if (result['freshness_assessment']['items_to_use_soon'] != null &&
                      (result['freshness_assessment']['items_to_use_soon'] as List).isNotEmpty) ...[
                    const SizedBox(height: 12),
                    Text(
                      'Use Soon:',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.medium,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: (result['freshness_assessment']['items_to_use_soon'] as List)
                          .map((item) => Chip(
                            label: Text(item),
                            backgroundColor: Colors.orange.withOpacity(0.1),
                            side: BorderSide(color: Colors.orange.withOpacity(0.3)),
                          )).toList(),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }

  Color _getFreshnessColor(String freshness) {
    switch (freshness.toLowerCase()) {
      case 'fresh':
        return Colors.green;
      case 'good':
        return Colors.orange;
      case 'use_soon':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      
      if (image != null) {
        setState(() {
          _imageFile = File(image.path);
          _scanResult = null;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to pick image: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _clearImage() {
    setState(() {
      _imageFile = null;
      _scanResult = null;
    });
  }

  Future<void> _scanFridge() async {
    if (_imageFile == null) return;

    setState(() {
      _isScanning = true;
      _scanResult = null;
    });

    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      final result = await apiService.scanFridge(_imageFile!);
      
      setState(() {
        _scanResult = result;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Scan failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      setState(() {
        _isScanning = false;
      });
    }
  }

  void _generateRecipeFromIngredients() {
    if (_scanResult == null || _scanResult!['detected_ingredients'] == null) return;
    
    final ingredients = (_scanResult!['detected_ingredients'] as List)
        .map((item) => item['name'] as String)
        .toList();
    
    Navigator.pushNamed(
      context,
      '/recipe-generator',
      arguments: {'ingredients': ingredients},
    );
  }
}