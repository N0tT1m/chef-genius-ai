import 'package:flutter/material.dart';
import 'package:shimmer/shimmer.dart';

class LoadingAnimation extends StatelessWidget {
  const LoadingAnimation({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header shimmer
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _shimmerContainer(
                        width: 200,
                        height: 24,
                        theme: theme,
                      ),
                      const SizedBox(height: 8),
                      _shimmerContainer(
                        width: double.infinity,
                        height: 16,
                        theme: theme,
                      ),
                    ],
                  ),
                ),
                _shimmerContainer(
                  width: 60,
                  height: 24,
                  theme: theme,
                ),
              ],
            ),
            
            const SizedBox(height: 24),
            
            // Info chips shimmer
            Row(
              children: [
                _shimmerContainer(width: 80, height: 32, theme: theme),
                const SizedBox(width: 12),
                _shimmerContainer(width: 100, height: 32, theme: theme),
                const SizedBox(width: 12),
                _shimmerContainer(width: 70, height: 32, theme: theme),
              ],
            ),
            
            const SizedBox(height: 24),
            
            // Ingredients title
            _shimmerContainer(width: 120, height: 20, theme: theme),
            const SizedBox(height: 16),
            
            // Ingredients list
            ...List.generate(5, (index) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: Row(
                children: [
                  _shimmerContainer(width: 6, height: 6, theme: theme),
                  const SizedBox(width: 12),
                  _shimmerContainer(
                    width: 150 + (index * 20).toDouble(),
                    height: 16,
                    theme: theme,
                  ),
                ],
              ),
            )),
            
            const SizedBox(height: 24),
            
            // Instructions title
            _shimmerContainer(width: 120, height: 20, theme: theme),
            const SizedBox(height: 16),
            
            // Instructions list
            ...List.generate(4, (index) => Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _shimmerContainer(width: 24, height: 24, theme: theme),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _shimmerContainer(
                          width: double.infinity,
                          height: 16,
                          theme: theme,
                        ),
                        if (index < 2) ...[
                          const SizedBox(height: 4),
                          _shimmerContainer(
                            width: 200,
                            height: 16,
                            theme: theme,
                          ),
                        ],
                      ],
                    ),
                  ),
                ],
              ),
            )),
            
            const SizedBox(height: 24),
            
            // Buttons shimmer
            Row(
              children: [
                Expanded(
                  child: _shimmerContainer(
                    width: double.infinity,
                    height: 40,
                    theme: theme,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _shimmerContainer(
                    width: double.infinity,
                    height: 40,
                    theme: theme,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _shimmerContainer({
    required double width,
    required double height,
    required ThemeData theme,
  }) {
    return Shimmer.fromColors(
      baseColor: theme.colorScheme.surfaceContainerHighest.withOpacity(0.3),
      highlightColor: theme.colorScheme.surface,
      child: Container(
        width: width,
        height: height,
        decoration: BoxDecoration(
          color: theme.colorScheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(8),
        ),
      ),
    );
  }
}

class VisionLoadingAnimation extends StatelessWidget {
  const VisionLoadingAnimation({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            // AI Analysis Icon
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: theme.colorScheme.primaryContainer,
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.psychology,
                size: 40,
                color: theme.colorScheme.primary,
              ),
            ),
            
            const SizedBox(height: 20),
            
            // Analyzing text with animation
            TweenAnimationBuilder<double>(
              duration: const Duration(seconds: 2),
              tween: Tween(begin: 0.0, end: 1.0),
              builder: (context, value, child) {
                return Column(
                  children: [
                    Text(
                      'AI is analyzing your image...',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.semibold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    LinearProgressIndicator(
                      value: value,
                      backgroundColor: theme.colorScheme.surfaceContainerHighest,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        theme.colorScheme.primary,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${(value * 100).toInt()}%',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurface.withOpacity(0.7),
                      ),
                    ),
                  ],
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}