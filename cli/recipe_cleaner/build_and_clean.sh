#!/bin/bash

set -e

echo "🦀 Building recipe cleaner (optimized release build)..."
cd "$(dirname "$0")"
cargo build --release

echo ""
echo "✅ Build complete!"
echo ""
echo "📖 Usage examples:"
echo ""
echo "# Dry run to preview what would be cleaned"
echo "./target/release/recipe_cleaner --input ../../data/datasets/ --recursive --dry-run"
echo ""
echo "# Clean all datasets recursively"
echo "./target/release/recipe_cleaner --input ../../data/datasets/ --output ../../data/cleaned_datasets/ --recursive"
echo ""
echo "# Clean a single file"
echo "./target/release/recipe_cleaner --input ../../data/training.json --output ../../data/training_cleaned.json"
echo ""
echo "# Verbose mode to see what's being rejected"
echo "./target/release/recipe_cleaner --input ../../data/datasets/ --recursive --verbose"
echo ""
echo "# Adjust quality thresholds"
echo "./target/release/recipe_cleaner --input ../../data/datasets/ --recursive --min-ingredients 3 --min-instructions-len 30"
echo ""
