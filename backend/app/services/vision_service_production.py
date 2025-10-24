#!/usr/bin/env python3
"""
Production Vision Service
Uses trained YOLOv8 food detection model instead of generic model
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from PIL import Image
import io
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class ProductionVisionService:
    """Production vision service using trained food detection model."""

    def __init__(self, model_path='models/food_detector/train/weights/best.pt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load trained YOLOv8 food detection model."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model not found: {self.model_path}")
                logger.info("Train model first: python cli/train_food_detector.py")
                return

            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            logger.info(f"✅ Loaded trained food detection model from {self.model_path}")
            logger.info(f"📊 Model classes: {len(self.model.names)}")
            logger.info(f"🖥️  Device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    async def scan_fridge(
        self,
        image_data: bytes,
        confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Scan fridge contents using trained model.

        Args:
            image_data: Image bytes
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dict with ingredients, freshness, recipes
        """
        if self.model is None:
            return self._fallback_response()

        try:
            # Convert bytes to image
            image = self._bytes_to_image(image_data)

            # Run detection
            results = self.model.predict(
                image,
                conf=confidence_threshold,
                device=self.device,
                verbose=False
            )

            # Parse results
            ingredients = self._parse_detections(results[0])

            # Assess freshness (basic)
            freshness = self._assess_freshness(image, ingredients)

            # Generate recipe suggestions
            recipe_suggestions = self._generate_recipe_ideas(ingredients)

            # Organization tips
            organization_tips = self._generate_organization_tips(ingredients)

            return {
                "success": True,
                "ingredients": ingredients,
                "ingredient_count": len(ingredients),
                "freshness_assessment": freshness,
                "recipe_suggestions": recipe_suggestions,
                "organization_tips": organization_tips,
                "confidence_stats": self._get_confidence_stats(ingredients)
            }

        except Exception as e:
            logger.error(f"Fridge scanning failed: {e}")
            return self._fallback_response()

    async def identify_ingredients(
        self,
        image_data: bytes,
        confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Identify ingredients from an image.

        Args:
            image_data: Image bytes
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dict with detected ingredients and confidence scores
        """
        if self.model is None:
            return self._fallback_response()

        try:
            image = self._bytes_to_image(image_data)

            # Run detection
            results = self.model.predict(
                image,
                conf=confidence_threshold,
                device=self.device,
                verbose=False
            )

            # Parse results
            ingredients = self._parse_detections(results[0])

            return {
                "success": True,
                "ingredients": ingredients,
                "confidence_scores": {
                    ing["name"]: ing["confidence"]
                    for ing in ingredients
                },
                "suggestions": self._generate_ingredient_suggestions(ingredients)
            }

        except Exception as e:
            logger.error(f"Ingredient identification failed: {e}")
            return self._fallback_response()

    def _parse_detections(self, result) -> List[Dict[str, Any]]:
        """Parse YOLO detection results into ingredient list."""
        ingredients = []

        if result.boxes is None or len(result.boxes) == 0:
            return ingredients

        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            # Get class name from model
            class_name = self.model.names[class_id]

            ingredient = {
                "name": class_name,
                "confidence": round(confidence, 3),
                "category": self._categorize_ingredient(class_name),
                "bbox": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3])
                },
                "estimated_amount": self._estimate_amount(bbox),
                "freshness": "unknown"  # Could be enhanced with freshness detection
            }

            ingredients.append(ingredient)

        # Sort by confidence
        ingredients.sort(key=lambda x: x["confidence"], reverse=True)

        return ingredients

    def _bytes_to_image(self, image_data: bytes):
        """Convert bytes to OpenCV image."""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def _categorize_ingredient(self, ingredient_name: str) -> str:
        """Categorize ingredients for better organization."""
        ingredient_lower = ingredient_name.lower()

        # Define categories
        categories = {
            "produce": [
                "apple", "banana", "orange", "tomato", "carrot", "onion",
                "lettuce", "spinach", "broccoli", "cucumber", "potato",
                "pepper", "cabbage", "celery", "corn", "garlic"
            ],
            "protein": [
                "chicken", "beef", "pork", "fish", "salmon", "tuna",
                "egg", "turkey", "lamb", "shrimp"
            ],
            "dairy": [
                "milk", "cheese", "butter", "yogurt", "cream", "sour_cream"
            ],
            "grains": [
                "rice", "pasta", "bread", "cereal", "oats", "quinoa"
            ],
            "beverages": [
                "juice", "soda", "water", "tea", "coffee", "wine", "beer"
            ],
            "condiments": [
                "ketchup", "mustard", "mayo", "sauce", "dressing", "jam"
            ]
        }

        for category, items in categories.items():
            if any(item in ingredient_lower for item in items):
                return category

        return "other"

    def _estimate_amount(self, bbox: List[float]) -> str:
        """Estimate amount based on bounding box size."""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        # Simple estimation based on area
        if area > 50000:
            return "large amount"
        elif area > 20000:
            return "medium amount"
        else:
            return "small amount"

    def _assess_freshness(self, image, ingredients: List[Dict]) -> Dict[str, str]:
        """Basic freshness assessment."""
        freshness = {}

        for ingredient in ingredients:
            name = ingredient["name"]
            # Simple rule-based assessment
            # In production, you could train a separate freshness classifier
            category = ingredient["category"]

            if category == "produce":
                freshness[name] = "appears_fresh"
            elif category == "dairy":
                freshness[name] = "check_expiration_date"
            elif category == "protein":
                freshness[name] = "check_storage_time"
            else:
                freshness[name] = "good_condition"

        return freshness

    def _generate_recipe_ideas(self, ingredients: List[Dict]) -> List[str]:
        """Generate recipe ideas based on detected ingredients."""
        if len(ingredients) < 2:
            return ["Not enough ingredients detected for recipe suggestions"]

        ingredient_names = [ing["name"] for ing in ingredients]
        categories = set(ing["category"] for ing in ingredients)

        suggestions = []

        # Analyze ingredient combinations
        if "protein" in categories and "produce" in categories:
            protein = next(ing["name"] for ing in ingredients if ing["category"] == "protein")
            suggestions.append(f"Try making a {protein} stir-fry with vegetables")

        if "dairy" in categories and "grains" in categories:
            suggestions.append("Consider a pasta dish with cheese sauce")

        if len([ing for ing in ingredients if ing["category"] == "produce"]) >= 3:
            suggestions.append("You have great ingredients for a fresh salad")

        # Add generic suggestion
        if len(ingredient_names) >= 3:
            top_ingredients = ", ".join(ingredient_names[:3])
            suggestions.append(f"Recipe using: {top_ingredients}")

        return suggestions[:3]  # Return top 3 suggestions

    def _generate_organization_tips(self, ingredients: List[Dict]) -> List[str]:
        """Generate fridge organization tips."""
        tips = []

        categories = [ing["category"] for ing in ingredients]

        if "protein" in categories:
            tips.append("Store raw protein on bottom shelf to prevent drips")

        if "produce" in categories:
            tips.append("Keep vegetables in crisper drawer for optimal freshness")

        if "dairy" in categories:
            tips.append("Store dairy in main body of fridge, not the door")

        # Always add general tip
        tips.append("Group similar items together for easy access")

        return tips

    def _generate_ingredient_suggestions(self, ingredients: List[Dict]) -> List[str]:
        """Generate helpful suggestions based on detected ingredients."""
        suggestions = []

        if len(ingredients) == 0:
            return ["No ingredients detected. Try a clearer image."]

        if len(ingredients) >= 5:
            suggestions.append("Great variety! You have ingredients for multiple meals.")

        categories = set(ing["category"] for ing in ingredients)

        if "protein" not in categories:
            suggestions.append("Consider adding a protein source like chicken or fish")

        if "produce" in categories and len([i for i in ingredients if i["category"] == "produce"]) >= 3:
            suggestions.append("Excellent selection of fresh vegetables!")

        return suggestions

    def _get_confidence_stats(self, ingredients: List[Dict]) -> Dict[str, float]:
        """Calculate confidence statistics."""
        if not ingredients:
            return {"average": 0.0, "min": 0.0, "max": 0.0}

        confidences = [ing["confidence"] for ing in ingredients]

        return {
            "average": round(np.mean(confidences), 3),
            "min": round(min(confidences), 3),
            "max": round(max(confidences), 3),
            "median": round(np.median(confidences), 3)
        }

    def _fallback_response(self) -> Dict[str, Any]:
        """Fallback response when model is not available."""
        return {
            "success": False,
            "error": "Food detection model not available",
            "ingredients": [],
            "message": "Please train the model first: python cli/train_food_detector.py"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if self.model is None:
            return {"loaded": False, "error": "Model not loaded"}

        return {
            "loaded": True,
            "model_path": str(self.model_path),
            "num_classes": len(self.model.names),
            "classes": list(self.model.names.values()),
            "device": self.device,
            "sample_classes": list(self.model.names.values())[:10]
        }
