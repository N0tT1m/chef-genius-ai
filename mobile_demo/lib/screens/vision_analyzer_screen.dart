import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../models/recipe.dart';
import '../widgets/loading_animation.dart';

class VisionAnalyzerScreen extends StatefulWidget {
  const VisionAnalyzerScreen({super.key});

  @override
  State<VisionAnalyzerScreen> createState() => _VisionAnalyzerScreenState();
}

class _VisionAnalyzerScreenState extends State<VisionAnalyzerScreen> {
  File? _imageFile;
  bool _isAnalyzing = false;
  VisionAnalysisResult? _analysisResult;
  final ImagePicker _picker = ImagePicker();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Vision Analyzer'),
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
                    Colors.blue.withOpacity(0.1),
                    Colors.lightBlue.withOpacity(0.1),
                  ],
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.camera_alt,
                    size: 32,
                    color: Colors.blue,
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'AI Vision Analysis',
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Take or upload a photo of food and let AI analyze it for you.',
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
                      Icons.add_photo_alternate,
                      size: 48,
                      color: theme.colorScheme.onSurface.withOpacity(0.5),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Add a photo to analyze',
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: theme.colorScheme.onSurface.withOpacity(0.7),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Take a photo or choose from gallery',
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
                      onPressed: _isAnalyzing ? null : _analyzeImage,
                      icon: _isAnalyzing 
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.psychology),
                      label: Text(_isAnalyzing ? 'Analyzing...' : 'Analyze'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ],
            
            const SizedBox(height: 24),
            
            // Analysis Results
            if (_isAnalyzing)
              const VisionLoadingAnimation()
            else if (_analysisResult != null)
              _buildAnalysisResults(),
          ],
        ),
      ),
    );
  }

  Widget _buildAnalysisResults() {
    final theme = Theme.of(context);
    final result = _analysisResult!;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Icon(
                  Icons.auto_awesome,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 8),
                Text(
                  'Analysis Results',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 20),
            
            // Caption
            if (result.caption != null) ...[
              _buildResultSection(
                title: 'AI Caption',
                icon: Icons.description,
                child: Text(
                  result.caption!,
                  style: theme.textTheme.bodyLarge,
                ),
              ),
              const SizedBox(height: 20),
            ],
            
            // Detected Ingredients
            if (result.detectedIngredients != null && 
                result.detectedIngredients!.isNotEmpty) ...[
              _buildResultSection(
                title: 'Detected Ingredients',
                icon: Icons.restaurant,
                child: Column(
                  children: result.detectedIngredients!.map((ingredient) => 
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              ingredient.ingredient,
                              style: theme.textTheme.bodyMedium,
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 4,
                            ),
                            decoration: BoxDecoration(
                              color: _getConfidenceColor(ingredient.confidence)
                                  .withOpacity(0.1),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              '${(ingredient.confidence * 100).toInt()}%',
                              style: TextStyle(
                                color: _getConfidenceColor(ingredient.confidence),
                                fontWeight: FontWeight.medium,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ).toList(),
                ),
              ),
              const SizedBox(height: 20),
            ],
            
            // Dish Classification
            if (result.dishClassification != null) ...[
              _buildResultSection(
                title: 'Dish Type',
                icon: Icons.category,
                child: Text(
                  result.dishClassification!['primary_dish_type'] ?? 'Unknown',
                  style: theme.textTheme.bodyMedium,
                ),
              ),
              const SizedBox(height: 20),
            ],
            
            // Cooking Stage
            if (result.cookingStage != null) ...[
              _buildResultSection(
                title: 'Cooking Stage',
                icon: Icons.whatshot,
                child: Text(
                  result.cookingStage!['stage'] ?? 'Unknown',
                  style: theme.textTheme.bodyMedium,
                ),
              ),
              const SizedBox(height: 20),
            ],
            
            // Recipe Suggestions
            if (result.recipeSuggestions != null && 
                result.recipeSuggestions!.isNotEmpty) ...[
              _buildResultSection(
                title: 'Recipe Suggestions',
                icon: Icons.lightbulb,
                child: Column(
                  children: result.recipeSuggestions!.map((suggestion) => 
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          Icon(
                            Icons.arrow_right,
                            size: 16,
                            color: theme.colorScheme.primary,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              suggestion,
                              style: theme.textTheme.bodyMedium,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ).toList(),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultSection({
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

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.orange;
    return Colors.red;
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
          _analysisResult = null;
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
      _analysisResult = null;
    });
  }

  Future<void> _analyzeImage() async {
    if (_imageFile == null) return;

    setState(() {
      _isAnalyzing = true;
      _analysisResult = null;
    });

    try {
      final apiService = Provider.of<ApiService>(context, listen: false);
      final result = await apiService.analyzeImage(_imageFile!);
      
      setState(() {
        _analysisResult = result;
      });
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Analysis failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      setState(() {
        _isAnalyzing = false;
      });
    }
  }
}