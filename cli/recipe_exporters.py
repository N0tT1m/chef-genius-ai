#!/usr/bin/env python3
"""
Recipe Export Module
Exports recipes to multiple formats: PDF, Markdown, HTML, JSON, TXT
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime


class RecipeExporter:
    """Export recipes to various formats."""

    @staticmethod
    def export_to_markdown(results: Dict[str, Any], output_file: str):
        """Export recipes to Markdown format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# {results.get('recipe_type', 'Recipe').title()} Collection\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Model:** `{os.path.basename(results['checkpoint_path'])}`  \n")
            f.write(f"**Success Rate:** {results['success_rate']:.1%}  \n")
            f.write(f"**Average Quality:** {results['avg_quality_score']:.3f}  \n\n")

            f.write("---\n\n")

            # Table of Contents
            f.write("## Table of Contents\n\n")
            categories = {}
            for result in results['individual_results']:
                category = result['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)

            for category in categories:
                f.write(f"- [{category.replace('_', ' ').title()}](#{category.replace('_', '-')})\n")

            f.write("\n---\n\n")

            # Statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Total Recipes | {results['total_recipes']} |\n")
            f.write(f"| Successful | {results['successful_recipes']} |\n")
            f.write(f"| Success Rate | {results['success_rate']:.1%} |\n")
            f.write(f"| Avg Quality | {results['avg_quality_score']:.3f} |\n")
            f.write(f"| Avg Gen Time | {results['avg_generation_time']:.2f}s |\n")
            f.write(f"| Tokens/sec | {results['avg_tokens_per_second']:.1f} |\n")
            f.write("\n")

            # Category breakdown
            f.write("### Category Breakdown\n\n")
            f.write(f"| Category | Success Rate | Avg Quality |\n")
            f.write(f"|----------|--------------|-------------|\n")
            for category, stats in results['category_stats'].items():
                f.write(f"| {category.replace('_', ' ').title()} | {stats['success_rate']:.1%} | {stats['avg_quality']:.3f} |\n")

            f.write("\n---\n\n")

            # Recipes by category
            for category, recipes in categories.items():
                # Sort by quality
                recipes.sort(key=lambda x: x['overall_quality'], reverse=True)

                f.write(f"## {category.replace('_', ' ').title()}\n\n")

                for recipe in recipes:
                    if recipe['status'] != 'SUCCESS':
                        continue

                    f.write(f"### {recipe['name']}\n\n")

                    # Metadata
                    f.write(f"**Quality Score:** {recipe['overall_quality']:.3f} | ")
                    f.write(f"**Difficulty:** {recipe['difficulty'].title()} | ")
                    f.write(f"**Best Mode:** {recipe['best_mode'].title()}  \n")

                    if recipe.get('perplexity'):
                        f.write(f"**Perplexity:** {recipe['perplexity']:.2f}  \n")

                    f.write("\n")

                    # Recipe content
                    recipe_text = recipe['recipe']
                    f.write(recipe_text)
                    f.write("\n\n")

                    # Features
                    if recipe.get('features_found'):
                        f.write(f"**Features Found:** {', '.join(recipe['features_found'])}  \n")

                    # Validation info
                    if recipe.get('validation'):
                        val = recipe['validation']
                        if not val.get('is_valid'):
                            f.write(f"**⚠️ Validation Issues:** {', '.join(val.get('errors', []))}  \n")

                    f.write("\n---\n\n")

        print(f"✅ Exported to Markdown: {output_file}")

    @staticmethod
    def export_to_html(results: Dict[str, Any], output_file: str):
        """Export recipes to HTML format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            # HTML header with CSS
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recipe Collection</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .recipe-card {
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .recipe-card h2 {
            color: #667eea;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .recipe-meta {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin: 15px 0;
            font-size: 0.9em;
        }
        .meta-item {
            background: #f0f0f0;
            padding: 5px 10px;
            border-radius: 5px;
        }
        .quality-high { color: #10b981; font-weight: bold; }
        .quality-medium { color: #f59e0b; font-weight: bold; }
        .quality-low { color: #ef4444; font-weight: bold; }
        .recipe-content {
            line-height: 1.8;
            white-space: pre-wrap;
        }
        .category-section {
            margin: 40px 0;
        }
        .category-section h2 {
            background: #667eea;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .validation-warning {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .features {
            background: #e0e7ff;
            padding: 10px 15px;
            border-radius: 4px;
            margin: 10px 0;
        }
        @media print {
            body { background: white; }
            .recipe-card { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
""")

            # Header
            recipe_type = results.get('recipe_type', 'Recipe').title()
            f.write(f"""
    <div class="header">
        <h1>{recipe_type} Collection</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Model: {os.path.basename(results['checkpoint_path'])}</p>
    </div>
""")

            # Statistics
            f.write("""
    <div class="stats">
""")
            stats_items = [
                ("Total Recipes", results['total_recipes']),
                ("Success Rate", f"{results['success_rate']:.1%}"),
                ("Avg Quality", f"{results['avg_quality_score']:.3f}"),
                ("Avg Gen Time", f"{results['avg_generation_time']:.2f}s"),
                ("Tokens/sec", f"{results['avg_tokens_per_second']:.1f}"),
            ]

            for label, value in stats_items:
                f.write(f"""
        <div class="stat-card">
            <h3>{label}</h3>
            <div class="value">{value}</div>
        </div>
""")

            f.write("""
    </div>
""")

            # Category sections
            categories = {}
            for result in results['individual_results']:
                category = result['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)

            for category, recipes in categories.items():
                recipes.sort(key=lambda x: x['overall_quality'], reverse=True)

                f.write(f"""
    <div class="category-section">
        <h2>{category.replace('_', ' ').title()}</h2>
""")

                for recipe in recipes:
                    if recipe['status'] != 'SUCCESS':
                        continue

                    quality = recipe['overall_quality']
                    quality_class = 'quality-high' if quality > 0.7 else 'quality-medium' if quality > 0.5 else 'quality-low'

                    f.write(f"""
        <div class="recipe-card">
            <h2>{recipe['name']}</h2>
            <div class="recipe-meta">
                <span class="meta-item">Quality: <span class="{quality_class}">{quality:.3f}</span></span>
                <span class="meta-item">Difficulty: {recipe['difficulty'].title()}</span>
                <span class="meta-item">Mode: {recipe['best_mode'].title()}</span>
                <span class="meta-item">Words: {recipe['word_count']}</span>
""")
                    if recipe.get('perplexity'):
                        f.write(f"""
                <span class="meta-item">Perplexity: {recipe['perplexity']:.2f}</span>
""")

                    f.write("""
            </div>
""")

                    # Recipe content
                    recipe_text = recipe['recipe'].replace('<', '&lt;').replace('>', '&gt;')
                    f.write(f"""
            <div class="recipe-content">{recipe_text}</div>
""")

                    # Features
                    if recipe.get('features_found'):
                        features = ', '.join(recipe['features_found'])
                        f.write(f"""
            <div class="features">
                <strong>Features Found:</strong> {features}
            </div>
""")

                    # Validation warnings
                    if recipe.get('validation'):
                        val = recipe['validation']
                        if not val.get('is_valid') or val.get('warnings'):
                            issues = val.get('errors', []) + val.get('warnings', [])
                            if issues:
                                f.write(f"""
            <div class="validation-warning">
                <strong>⚠️ Validation Notes:</strong> {', '.join(issues[:3])}
            </div>
""")

                    f.write("""
        </div>
""")

                f.write("""
    </div>
""")

            # Close HTML
            f.write("""
</body>
</html>
""")

        print(f"✅ Exported to HTML: {output_file}")

    @staticmethod
    def export_to_pdf(results: Dict[str, Any], output_file: str):
        """Export recipes to PDF format (requires reportlab)."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        except ImportError:
            print("⚠️ reportlab not installed. Install with: pip install reportlab")
            print("   Falling back to text export...")
            RecipeExporter.export_to_text(results, output_file.replace('.pdf', '.txt'))
            return

        doc = SimpleDocTemplate(output_file, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)

        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12
        )

        # Title page
        recipe_type = results.get('recipe_type', 'Recipe').title()
        story.append(Paragraph(f"{recipe_type} Collection", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Model', os.path.basename(results['checkpoint_path'])],
            ['Total Recipes', str(results['total_recipes'])],
            ['Success Rate', f"{results['success_rate']:.1%}"],
            ['Avg Quality', f"{results['avg_quality_score']:.3f}"],
            ['Avg Gen Time', f"{results['avg_generation_time']:.2f}s"],
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(summary_table)
        story.append(PageBreak())

        # Recipes
        categories = {}
        for result in results['individual_results']:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        for category, recipes in categories.items():
            recipes.sort(key=lambda x: x['overall_quality'], reverse=True)

            story.append(Paragraph(category.replace('_', ' ').title(), heading_style))
            story.append(Spacer(1, 0.2 * inch))

            for recipe in recipes:
                if recipe['status'] != 'SUCCESS':
                    continue

                # Recipe title
                story.append(Paragraph(recipe['name'], styles['Heading3']))

                # Metadata
                meta_text = f"Quality: {recipe['overall_quality']:.3f} | Difficulty: {recipe['difficulty'].title()} | Mode: {recipe['best_mode'].title()}"
                story.append(Paragraph(meta_text, styles['Normal']))
                story.append(Spacer(1, 0.1 * inch))

                # Recipe content
                recipe_text = recipe['recipe'].replace('\n', '<br/>')
                story.append(Paragraph(recipe_text, styles['Normal']))
                story.append(Spacer(1, 0.3 * inch))

        doc.build(story)
        print(f"✅ Exported to PDF: {output_file}")

    @staticmethod
    def export_to_text(results: Dict[str, Any], output_file: str):
        """Export recipes to plain text format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            recipe_type = results.get('recipe_type', 'Recipe').title()
            f.write(f"{recipe_type} Collection\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {os.path.basename(results['checkpoint_path'])}\n")
            f.write(f"Success Rate: {results['success_rate']:.1%}\n")
            f.write(f"Average Quality: {results['avg_quality_score']:.3f}\n\n")
            f.write("=" * 80 + "\n\n")

            # Category stats
            f.write("CATEGORY STATISTICS\n")
            f.write("-" * 80 + "\n")
            for category, stats in results['category_stats'].items():
                f.write(f"{category.replace('_', ' ').title()}:\n")
                f.write(f"  Success Rate: {stats['success_rate']:.1%}\n")
                f.write(f"  Avg Quality: {stats['avg_quality']:.3f}\n\n")

            f.write("=" * 80 + "\n\n")

            # Recipes
            categories = {}
            for result in results['individual_results']:
                category = result['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(result)

            for category, recipes in categories.items():
                recipes.sort(key=lambda x: x['overall_quality'], reverse=True)

                f.write(f"\n{category.replace('_', ' ').upper()}\n")
                f.write("=" * 80 + "\n\n")

                for recipe in recipes:
                    if recipe['status'] != 'SUCCESS':
                        continue

                    f.write(f"{recipe['name']}\n")
                    f.write("-" * len(recipe['name']) + "\n")
                    f.write(f"Quality: {recipe['overall_quality']:.3f} | ")
                    f.write(f"Difficulty: {recipe['difficulty'].title()} | ")
                    f.write(f"Mode: {recipe['best_mode'].title()}\n\n")

                    f.write(recipe['recipe'])
                    f.write("\n\n" + "=" * 80 + "\n\n")

        print(f"✅ Exported to text: {output_file}")

    @staticmethod
    def export_to_json(results: Dict[str, Any], output_file: str):
        """Export recipes to JSON format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Exported to JSON: {output_file}")

    @staticmethod
    def export_all_formats(results: Dict[str, Any], base_filename: str):
        """Export to all available formats."""
        base_path = Path(base_filename)
        stem = base_path.stem

        formats = {
            'markdown': f"{stem}.md",
            'html': f"{stem}.html",
            'pdf': f"{stem}.pdf",
            'text': f"{stem}.txt",
            'json': f"{stem}.json"
        }

        print(f"\n📦 Exporting to all formats...")

        RecipeExporter.export_to_markdown(results, formats['markdown'])
        RecipeExporter.export_to_html(results, formats['html'])
        RecipeExporter.export_to_pdf(results, formats['pdf'])
        RecipeExporter.export_to_text(results, formats['text'])
        RecipeExporter.export_to_json(results, formats['json'])

        print(f"\n✅ Exported to all formats in: {base_path.parent}")
        return formats
