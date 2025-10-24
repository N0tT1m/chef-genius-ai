#!/usr/bin/env python3
"""
Enhanced Discord Notifier
Provides real-time updates and rich notifications for recipe generation
"""

import requests
import time
from typing import Dict, List, Any, Optional
from datetime import datetime


class DiscordNotifier:
    """Enhanced Discord notification system with real-time updates."""

    def __init__(self, webhook_url: str, username: str = "Chef Genius Bot"):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username for messages
        """
        self.webhook_url = webhook_url
        self.username = username
        self.enabled = webhook_url is not None and len(webhook_url) > 0

    def send_message(self, content: str = None, embed: Dict[str, Any] = None,
                    embeds: List[Dict[str, Any]] = None) -> bool:
        """
        Send a Discord message.

        Args:
            content: Plain text message content
            embed: Single embed object
            embeds: List of embed objects

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            payload = {"username": self.username}

            if content:
                payload["content"] = content

            if embed:
                payload["embeds"] = [embed]
            elif embeds:
                payload["embeds"] = embeds

            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            return True

        except Exception as e:
            print(f"⚠️ Discord notification failed: {e}")
            return False

    def send_session_start(self, recipe_type: str, total_recipes: int,
                          checkpoint: str, config: Dict[str, Any]) -> bool:
        """Send notification when generation session starts."""
        embed = {
            "title": "🚀 Recipe Generation Session Started",
            "description": f"Starting generation of **{total_recipes}** {recipe_type} recipes",
            "color": 0x3498db,  # Blue
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "📁 Model",
                    "value": f"`{checkpoint}`",
                    "inline": False
                },
                {
                    "name": "🎯 Configuration",
                    "value": f"Modes: {', '.join(config.get('generation_modes', []))}\nRetry: {config.get('retry_on_failure', False)}\nBeam Search: {config.get('enable_beam_search', False)}",
                    "inline": False
                },
                {
                    "name": "📊 Target Metrics",
                    "value": f"Quality Threshold: {config.get('quality_threshold', 0.6):.2f}\nMax Retries: {config.get('max_retries', 2)}",
                    "inline": True
                }
            ],
            "footer": {"text": "Chef Genius Recipe Generator"}
        }

        return self.send_message(embed=embed)

    def send_progress_update(self, current: int, total: int, category: str,
                            recent_results: List[Dict[str, Any]]) -> bool:
        """
        Send progress update during generation.

        Args:
            current: Number of recipes completed
            total: Total recipes to generate
            category: Current category being processed
            recent_results: Recent recipe generation results
        """
        progress_pct = (current / total * 100) if total > 0 else 0
        progress_bar = self._create_progress_bar(current, total)

        # Calculate stats from recent results
        recent_success = sum(1 for r in recent_results if r.get('status') == 'SUCCESS')
        recent_quality = sum(r.get('overall_quality', 0) for r in recent_results) / len(recent_results) if recent_results else 0

        embed = {
            "title": "⚙️ Generation Progress",
            "description": f"Processing **{category.replace('_', ' ').title()}** recipes",
            "color": 0xf39c12,  # Orange
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "📈 Progress",
                    "value": f"{progress_bar}\n{current}/{total} ({progress_pct:.1f}%)",
                    "inline": False
                },
                {
                    "name": "✅ Recent Success",
                    "value": f"{recent_success}/{len(recent_results)}",
                    "inline": True
                },
                {
                    "name": "🎯 Recent Quality",
                    "value": f"{recent_quality:.3f}",
                    "inline": True
                }
            ],
            "footer": {"text": f"Category: {category.replace('_', ' ').title()}"}
        }

        # Add recent recipe names
        if recent_results:
            recent_names = "\n".join([
                f"{'✅' if r.get('status') == 'SUCCESS' else '⚠️'} {r.get('name', 'Unknown')} ({r.get('overall_quality', 0):.3f})"
                for r in recent_results[-3:]
            ])
            embed["fields"].append({
                "name": "📝 Recent Recipes",
                "value": recent_names,
                "inline": False
            })

        return self.send_message(embed=embed)

    def send_category_complete(self, category: str, results: List[Dict[str, Any]]) -> bool:
        """Send notification when a category is completed."""
        total = len(results)
        successful = sum(1 for r in results if r.get('status') == 'SUCCESS')
        avg_quality = sum(r.get('overall_quality', 0) for r in results) / total if total > 0 else 0

        # Determine color based on success rate
        success_rate = successful / total if total > 0 else 0
        if success_rate >= 0.8:
            color = 0x2ecc71  # Green
            emoji = "✅"
        elif success_rate >= 0.5:
            color = 0xf39c12  # Orange
            emoji = "⚠️"
        else:
            color = 0xe74c3c  # Red
            emoji = "❌"

        embed = {
            "title": f"{emoji} Category Completed: {category.replace('_', ' ').title()}",
            "description": f"Finished generating {total} recipes in this category",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "📊 Results",
                    "value": f"Success: {successful}/{total} ({success_rate:.1%})\nAvg Quality: {avg_quality:.3f}",
                    "inline": False
                }
            ]
        }

        # Add top recipes from category
        sorted_results = sorted(results, key=lambda x: x.get('overall_quality', 0), reverse=True)
        top_recipes = sorted_results[:3]

        if top_recipes:
            top_text = "\n".join([
                f"🏆 {r.get('name', 'Unknown')}: {r.get('overall_quality', 0):.3f}"
                for r in top_recipes
            ])
            embed["fields"].append({
                "name": "🏆 Top Recipes",
                "value": top_text,
                "inline": False
            })

        return self.send_message(embed=embed)

    def send_session_complete(self, results: Dict[str, Any]) -> bool:
        """Send comprehensive results when session completes."""
        success_rate = results.get('success_rate', 0)
        avg_quality = results.get('avg_quality_score', 0)

        # Determine overall status
        if success_rate >= 0.8 and avg_quality >= 0.7:
            color = 0x2ecc71  # Green
            status_emoji = "🎉"
            status_text = "EXCELLENT RESULTS"
        elif success_rate >= 0.6 and avg_quality >= 0.5:
            color = 0xf39c12  # Orange
            status_emoji = "🎯"
            status_text = "GOOD RESULTS"
        else:
            color = 0xe74c3c  # Red
            status_emoji = "⚠️"
            status_text = "NEEDS IMPROVEMENT"

        recipe_type = results.get('recipe_type', 'Recipe').title()

        embed = {
            "title": f"{status_emoji} {recipe_type} Generation Complete!",
            "description": f"**{status_text}**\nGenerated {results['total_recipes']} recipes with {results['successful_recipes']} successes",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "📊 Overall Statistics",
                    "value": f"✅ Success Rate: {success_rate:.1%}\n🎯 Avg Quality: {avg_quality:.3f}\n📈 Total Recipes: {results['total_recipes']}",
                    "inline": True
                },
                {
                    "name": "⚡ Performance",
                    "value": f"🏃 Avg Gen Time: {results.get('avg_generation_time', 0):.2f}s\n🚀 Tokens/sec: {results.get('avg_tokens_per_second', 0):.1f}\n⏱️ Total Time: {results.get('total_generation_time', 0):.1f}s",
                    "inline": True
                }
            ],
            "footer": {"text": f"Model: {results.get('checkpoint_path', 'Unknown')}"}
        }

        # Category breakdown
        if 'category_stats' in results:
            category_text = ""
            for category, stats in results['category_stats'].items():
                emoji = "✅" if stats['success_rate'] >= 0.7 else "⚠️" if stats['success_rate'] >= 0.5 else "❌"
                category_text += f"{emoji} {category.replace('_', ' ').title()}: {stats['success_rate']:.1%} ({stats['successful']}/{stats['total']})\n"

            embed["fields"].append({
                "name": "📋 Category Breakdown",
                "value": category_text.strip(),
                "inline": False
            })

        # Top recipes
        if 'individual_results' in results:
            top_recipes = sorted(results['individual_results'],
                               key=lambda x: x.get('overall_quality', 0),
                               reverse=True)[:5]

            top_text = "\n".join([
                f"{'🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '🏆'} {r.get('name', 'Unknown')}: {r.get('overall_quality', 0):.3f} ({r.get('best_mode', 'unknown')})"
                for i, r in enumerate(top_recipes)
            ])

            embed["fields"].append({
                "name": "🏆 Top Performing Recipes",
                "value": top_text,
                "inline": False
            })

        return self.send_message(embed=embed)

    def send_recipe_preview(self, recipe_result: Dict[str, Any]) -> bool:
        """Send a preview of a generated recipe."""
        quality = recipe_result.get('overall_quality', 0)

        if quality >= 0.8:
            color = 0x2ecc71  # Green
            emoji = "🏆"
        elif quality >= 0.6:
            color = 0x3498db  # Blue
            emoji = "⭐"
        else:
            color = 0x95a5a6  # Gray
            emoji = "📝"

        # Extract recipe excerpt
        recipe_text = recipe_result.get('recipe', '')
        lines = recipe_text.split('\n')
        preview_lines = []

        for line in lines[:15]:  # First 15 lines
            if line.strip():
                preview_lines.append(line)

        preview = '\n'.join(preview_lines)
        if len(recipe_text.split('\n')) > 15:
            preview += "\n\n... (truncated)"

        embed = {
            "title": f"{emoji} {recipe_result.get('name', 'Recipe')}",
            "description": f"```\n{preview[:1800]}\n```",  # Discord embed limit
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [
                {
                    "name": "📊 Metrics",
                    "value": f"Quality: {quality:.3f}\nMode: {recipe_result.get('best_mode', 'unknown').title()}\nWords: {recipe_result.get('word_count', 0)}",
                    "inline": True
                },
                {
                    "name": "🎯 Details",
                    "value": f"Category: {recipe_result.get('category', 'unknown').replace('_', ' ').title()}\nDifficulty: {recipe_result.get('difficulty', 'normal').title()}",
                    "inline": True
                }
            ]
        }

        # Add features found
        if recipe_result.get('features_found'):
            features = ', '.join(recipe_result['features_found'][:5])
            embed["fields"].append({
                "name": "✨ Features",
                "value": features,
                "inline": False
            })

        return self.send_message(embed=embed)

    def send_error(self, error_message: str, context: str = "") -> bool:
        """Send error notification."""
        embed = {
            "title": "❌ Error Occurred",
            "description": f"```\n{error_message}\n```",
            "color": 0xe74c3c,  # Red
            "timestamp": datetime.utcnow().isoformat(),
            "fields": []
        }

        if context:
            embed["fields"].append({
                "name": "Context",
                "value": context,
                "inline": False
            })

        return self.send_message(embed=embed)

    def _create_progress_bar(self, current: int, total: int, length: int = 20) -> str:
        """Create a visual progress bar."""
        if total == 0:
            return "░" * length

        filled = int((current / total) * length)
        bar = "█" * filled + "░" * (length - filled)
        return f"[{bar}]"

    def send_batch_update(self, updates: List[str]) -> bool:
        """Send multiple updates as a single message."""
        if not updates:
            return False

        content = "\n".join(f"• {update}" for update in updates)

        embed = {
            "title": "📢 Batch Update",
            "description": content[:2000],  # Discord limit
            "color": 0x3498db,
            "timestamp": datetime.utcnow().isoformat()
        }

        return self.send_message(embed=embed)


class ProgressTracker:
    """Track progress and send periodic Discord updates."""

    def __init__(self, notifier: DiscordNotifier, total_items: int,
                 update_interval: int = 5):
        """
        Initialize progress tracker.

        Args:
            notifier: Discord notifier instance
            total_items: Total number of items to process
            update_interval: Send update every N items
        """
        self.notifier = notifier
        self.total_items = total_items
        self.update_interval = update_interval
        self.completed = 0
        self.results_buffer = []
        self.current_category = ""

    def update(self, result: Dict[str, Any], category: str = ""):
        """Update progress with a new result."""
        self.completed += 1
        self.results_buffer.append(result)
        self.current_category = category or self.current_category

        # Send periodic updates
        if self.completed % self.update_interval == 0 or self.completed == self.total_items:
            self.notifier.send_progress_update(
                self.completed,
                self.total_items,
                self.current_category,
                self.results_buffer[-self.update_interval:]
            )

    def send_preview(self, result: Dict[str, Any]):
        """Send a recipe preview for high-quality results."""
        quality = result.get('overall_quality', 0)

        # Only send previews for high-quality recipes
        if quality >= 0.8:
            self.notifier.send_recipe_preview(result)

    def category_complete(self, category: str, results: List[Dict[str, Any]]):
        """Mark category as complete and send notification."""
        self.notifier.send_category_complete(category, results)

    def reset_buffer(self):
        """Reset the results buffer."""
        self.results_buffer = []
