import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from PIL import Image
import io
import torch
from ultralytics import YOLO
from app.services.recipe_generator import RecipeGeneratorService
from app.models.recipe import RecipeGenerationRequest

logger = logging.getLogger(__name__)

class VisionService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ingredient_model = None
        self.food_model = None
        self.recipe_generator = RecipeGeneratorService()
        self._load_models()
        
        # Ingredient database for recognition
        self.ingredient_classes = [
            "apple", "banana", "orange", "tomato", "potato", "carrot", "onion",
            "garlic", "bell pepper", "broccoli", "spinach", "lettuce", "cucumber",
            "chicken", "beef", "fish", "eggs", "milk", "cheese", "bread",
            "rice", "pasta", "beans", "avocado", "lemon", "lime", "mushroom"
        ]
    
    def _load_models(self):
        """Load computer vision models."""
        try:
            # In production, use fine-tuned models for food/ingredient recognition
            # For now, using general object detection models
            self.food_model = YOLO('yolov8n.pt')  # Lightweight model
            logger.info("Vision models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load full vision models: {e}")
            # Fallback to basic image processing
            self.food_model = None
    
    async def identify_ingredients(self, image_data: bytes, context: Optional[str] = None) -> Dict[str, Any]:
        """Identify ingredients from an image."""
        try:
            # Convert bytes to image
            image = self._bytes_to_image(image_data)
            
            # Run object detection
            if self.food_model:
                results = self.food_model(image)
                detected_objects = self._parse_detection_results(results)
            else:
                # Fallback: basic image analysis
                detected_objects = self._basic_ingredient_detection(image)
            
            # Map detections to ingredients
            ingredients = self._map_to_ingredients(detected_objects)
            
            # Calculate confidence scores
            confidence_scores = {
                ingredient["name"]: ingredient.get("confidence", 0.5)
                for ingredient in ingredients
            }
            
            # Generate suggestions
            suggestions = self._generate_ingredient_suggestions(ingredients, context)
            
            return {
                "ingredients": ingredients,
                "confidence_scores": confidence_scores,
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Ingredient identification failed: {e}")
            return self._fallback_ingredient_response()
    
    async def scan_fridge(self, image_data: bytes) -> Dict[str, Any]:
        """Scan fridge contents and assess organization/freshness."""
        try:
            image = self._bytes_to_image(image_data)
            
            # Detect ingredients
            if self.food_model:
                results = self.food_model(image)
                detected_objects = self._parse_detection_results(results)
            else:
                detected_objects = self._basic_fridge_scan(image)
            
            ingredients = self._map_to_ingredients(detected_objects)
            
            # Assess freshness (simplified)
            freshness_assessment = self._assess_freshness(image, ingredients)
            
            # Generate organization tips
            organization_tips = self._generate_organization_tips(ingredients)
            
            return {
                "ingredients": ingredients,
                "freshness": freshness_assessment,
                "tips": organization_tips
            }
            
        except Exception as e:
            logger.error(f"Fridge scanning failed: {e}")
            return {"ingredients": [], "freshness": {}, "tips": []}
    
    async def suggest_recipes_from_ingredients(self, ingredients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recipe suggestions based on detected ingredients."""
        try:
            ingredient_names = [ing["name"] for ing in ingredients]
            
            # Generate multiple recipe suggestions
            suggestions = []
            cuisines = ["italian", "asian", "american", "mediterranean"]
            
            for cuisine in cuisines[:2]:  # Generate 2 suggestions
                request = RecipeGenerationRequest(
                    ingredients=ingredient_names[:5],  # Use up to 5 ingredients
                    cuisine=cuisine,
                    cooking_time="under 45 minutes",
                    difficulty="medium"
                )
                
                recipe = await self.recipe_generator.generate_recipe(request)
                suggestions.append({
                    "title": recipe.title,
                    "description": recipe.description,
                    "cuisine": cuisine,
                    "ingredients_used": ingredient_names,
                    "prep_time": recipe.prep_time,
                    "cook_time": recipe.cook_time
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Recipe suggestion failed: {e}")
            return []
    
    async def analyze_cooking_progress(self, image_data: bytes, recipe_step: Optional[str], expected_result: Optional[str]) -> Dict[str, Any]:
        """Analyze cooking progress and provide feedback."""
        try:
            image = self._bytes_to_image(image_data)
            
            # Analyze visual characteristics
            analysis = self._analyze_visual_characteristics(image)
            
            # Compare with expected result
            feedback = self._generate_cooking_feedback(analysis, recipe_step, expected_result)
            
            return {
                "visual_analysis": analysis,
                "feedback": feedback["feedback"],
                "next_steps": feedback["next_steps"],
                "warnings": feedback.get("warnings", []),
                "progress_score": feedback.get("score", 5)
            }
            
        except Exception as e:
            logger.error(f"Cooking progress analysis failed: {e}")
            return {"feedback": "Unable to analyze image", "progress_score": 5}
    
    async def analyze_plating(self, image_data: bytes, dish_type: Optional[str]) -> Dict[str, Any]:
        """Analyze food plating and provide presentation suggestions."""
        try:
            image = self._bytes_to_image(image_data)
            
            # Analyze composition and presentation
            plating_analysis = self._analyze_plating_composition(image)
            
            # Generate suggestions
            suggestions = self._generate_plating_suggestions(plating_analysis, dish_type)
            
            return {
                "composition_analysis": plating_analysis,
                "presentation_score": plating_analysis.get("score", 7),
                "suggestions": suggestions,
                "color_balance": plating_analysis.get("color_balance", "good"),
                "portion_assessment": plating_analysis.get("portion", "appropriate")
            }
            
        except Exception as e:
            logger.error(f"Plating analysis failed: {e}")
            return {"presentation_score": 7, "suggestions": ["Focus on color contrast and portion balance"]}
    
    async def reconstruct_recipe(self, image_data: bytes, additional_info: Optional[str]) -> Dict[str, Any]:
        """Attempt to reconstruct a recipe from a finished dish image."""
        try:
            image = self._bytes_to_image(image_data)
            
            # Identify visible ingredients and cooking methods
            visible_ingredients = await self.identify_ingredients(image_data)
            cooking_techniques = await self.identify_cooking_technique(image_data)
            
            # Generate recipe reconstruction
            estimated_recipe = await self._reconstruct_from_analysis(
                visible_ingredients, cooking_techniques, additional_info
            )
            
            # Calculate confidence based on visible elements
            confidence = self._calculate_reconstruction_confidence(
                visible_ingredients, cooking_techniques
            )
            
            # Identify missing information
            missing_info = self._identify_missing_information(estimated_recipe)
            
            return {
                "estimated_recipe": estimated_recipe,
                "confidence_score": confidence,
                "missing_information": missing_info
            }
            
        except Exception as e:
            logger.error(f"Recipe reconstruction failed: {e}")
            return {
                "estimated_recipe": {"title": "Unable to reconstruct recipe"},
                "confidence_score": 0.1,
                "missing_information": ["Unable to analyze image"]
            }
    
    async def identify_cooking_technique(self, image_data: bytes, context: Optional[str]) -> Dict[str, Any]:
        """Identify cooking techniques being used."""
        try:
            image = self._bytes_to_image(image_data)
            
            # Analyze visual cues for cooking techniques
            techniques = self._detect_cooking_techniques(image)
            
            # Generate recommendations
            recommendations = self._generate_technique_recommendations(techniques, context)
            
            return {
                "techniques": techniques["detected"],
                "confidence": techniques["confidence"],
                "recommendations": recommendations,
                "next_steps": self._suggest_next_cooking_steps(techniques)
            }
            
        except Exception as e:
            logger.error(f"Technique identification failed: {e}")
            return {
                "techniques": ["general cooking"],
                "confidence": {"general cooking": 0.5},
                "recommendations": ["Continue cooking as planned"]
            }
    
    def _bytes_to_image(self, image_data: bytes):
        """Convert bytes to OpenCV image."""
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    def _parse_detection_results(self, results):
        """Parse YOLO detection results."""
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)
                    
                    # Map class index to ingredient (simplified)
                    if cls < len(self.ingredient_classes):
                        detected_objects.append({
                            "class": self.ingredient_classes[cls],
                            "confidence": conf,
                            "bbox": box.xyxy.tolist()
                        })
        
        return detected_objects
    
    def _basic_ingredient_detection(self, image):
        """Basic fallback ingredient detection using color analysis."""
        # Simplified color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common ingredients
        color_ranges = {
            "tomato": [(0, 50, 50), (10, 255, 255)],
            "carrot": [(15, 50, 50), (25, 255, 255)],
            "lettuce": [(40, 50, 50), (80, 255, 255)],
            "banana": [(20, 100, 100), (30, 255, 255)]
        }
        
        detected = []
        for ingredient, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if cv2.countNonZero(mask) > 1000:  # Threshold for detection
                detected.append({
                    "class": ingredient,
                    "confidence": 0.6,
                    "method": "color_analysis"
                })
        
        return detected
    
    def _map_to_ingredients(self, detected_objects):
        """Map detected objects to ingredient format."""
        ingredients = []
        
        for obj in detected_objects:
            ingredient = {
                "name": obj["class"],
                "confidence": obj["confidence"],
                "category": self._categorize_ingredient(obj["class"]),
                "estimated_amount": "1 piece",  # Simplified
                "freshness": "good"  # Default assumption
            }
            ingredients.append(ingredient)
        
        return ingredients
    
    def _categorize_ingredient(self, ingredient_name: str) -> str:
        """Categorize ingredients for better organization."""
        categories = {
            "produce": ["apple", "banana", "tomato", "carrot", "onion", "lettuce", "spinach"],
            "protein": ["chicken", "beef", "fish", "eggs"],
            "dairy": ["milk", "cheese"],
            "grains": ["rice", "pasta", "bread"]
        }
        
        for category, items in categories.items():
            if ingredient_name.lower() in items:
                return category
        
        return "other"
    
    def _generate_ingredient_suggestions(self, ingredients: List[Dict], context: Optional[str]) -> List[str]:
        """Generate helpful suggestions based on detected ingredients."""
        suggestions = []
        
        if len(ingredients) >= 3:
            suggestions.append("You have enough ingredients for a complete meal!")
        
        categories = set(ing["category"] for ing in ingredients)
        if "protein" not in categories:
            suggestions.append("Consider adding a protein source like chicken, fish, or beans")
        
        if "produce" in categories and len([ing for ing in ingredients if ing["category"] == "produce"]) >= 2:
            suggestions.append("Great variety of vegetables for a nutritious meal")
        
        return suggestions
    
    def _basic_fridge_scan(self, image):
        """Basic fridge content analysis."""
        # Simplified analysis - in production would use specialized models
        height, width = image.shape[:2]
        
        # Simulate detection of common fridge items
        common_items = ["milk", "eggs", "cheese", "leftovers", "vegetables"]
        detected = []
        
        for item in common_items[:3]:  # Simulate finding 3 items
            detected.append({
                "class": item,
                "confidence": 0.7,
                "location": "middle_shelf"
            })
        
        return detected
    
    def _assess_freshness(self, image, ingredients: List[Dict]) -> Dict[str, Any]:
        """Assess freshness of detected ingredients."""
        # Simplified freshness assessment
        freshness = {}
        
        for ingredient in ingredients:
            name = ingredient["name"]
            # Simple rule-based assessment
            if name in ["milk", "eggs", "cheese"]:
                freshness[name] = "check_expiration_date"
            elif name in ["lettuce", "spinach", "herbs"]:
                freshness[name] = "appears_fresh"
            else:
                freshness[name] = "good_condition"
        
        return freshness
    
    def _generate_organization_tips(self, ingredients: List[Dict]) -> List[str]:
        """Generate fridge organization tips."""
        tips = [
            "Keep dairy products in the main body of the fridge, not the door",
            "Store vegetables in the crisper drawer for optimal freshness",
            "Keep raw meat on the bottom shelf to prevent drips"
        ]
        
        # Add specific tips based on detected items
        categories = [ing["category"] for ing in ingredients]
        if "produce" in categories:
            tips.append("Separate fruits and vegetables to prevent premature ripening")
        
        return tips[:3]  # Return top 3 tips
    
    def _analyze_visual_characteristics(self, image):
        """Analyze visual characteristics of cooking in progress."""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze brightness (indication of browning/cooking level)
        brightness = np.mean(gray)
        
        # Analyze color distribution
        dominant_colors = self._get_dominant_colors(image)
        
        # Detect steam/bubbling (simplified)
        steam_detected = self._detect_steam_bubbles(gray)
        
        return {
            "brightness": brightness,
            "dominant_colors": dominant_colors,
            "steam_detected": steam_detected,
            "cooking_stage": self._determine_cooking_stage(brightness, steam_detected)
        }
    
    def _get_dominant_colors(self, image):
        """Get dominant colors in the image."""
        # Simplified color analysis
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = 3
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers.tolist()
    
    def _detect_steam_bubbles(self, gray_image):
        """Detect steam or bubbling activity."""
        # Simple edge detection to identify steam/bubbles
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count small circular contours (potential bubbles)
        bubble_count = sum(1 for c in contours if cv2.contourArea(c) < 100 and len(c) > 5)
        
        return bubble_count > 10  # Threshold for steam detection
    
    def _determine_cooking_stage(self, brightness: float, steam_detected: bool) -> str:
        """Determine cooking stage based on visual cues."""
        if steam_detected:
            return "active_cooking"
        elif brightness < 100:
            return "browning"
        elif brightness > 150:
            return "early_stage"
        else:
            return "medium_stage"
    
    def _generate_cooking_feedback(self, analysis: Dict, recipe_step: Optional[str], expected_result: Optional[str]) -> Dict[str, Any]:
        """Generate cooking feedback based on analysis."""
        feedback = []
        next_steps = []
        warnings = []
        score = 7  # Default score
        
        stage = analysis.get("cooking_stage", "medium_stage")
        
        if stage == "browning":
            feedback.append("Good browning detected - flavors are developing nicely")
            next_steps.append("Monitor closely to prevent burning")
            score = 8
        elif stage == "active_cooking":
            feedback.append("Active cooking detected with steam/bubbling")
            next_steps.append("Maintain current heat level")
            score = 8
        elif stage == "early_stage":
            feedback.append("Cooking is in early stage")
            next_steps.append("Increase heat slightly if needed")
            score = 6
        
        if analysis.get("brightness", 128) < 50:
            warnings.append("Very dark coloring - check for burning")
            score -= 2
        
        return {
            "feedback": feedback,
            "next_steps": next_steps,
            "warnings": warnings,
            "score": max(1, score)
        }
    
    def _analyze_plating_composition(self, image):
        """Analyze plating composition and visual appeal."""
        # Color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        colors = self._analyze_color_distribution(hsv)
        
        # Composition analysis (rule of thirds, balance)
        composition_score = self._analyze_composition(image)
        
        return {
            "score": composition_score,
            "color_balance": colors["balance"],
            "color_variety": colors["variety"],
            "portion": "appropriate",  # Simplified
            "visual_appeal": composition_score
        }
    
    def _analyze_color_distribution(self, hsv_image):
        """Analyze color distribution for visual appeal."""
        # Count different hue ranges
        hue_ranges = {
            "red": (0, 30),
            "yellow": (30, 60),
            "green": (60, 120),
            "blue": (120, 180)
        }
        
        color_counts = {}
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        
        for color, (min_hue, max_hue) in hue_ranges.items():
            mask = cv2.inRange(hsv_image, (min_hue, 50, 50), (max_hue, 255, 255))
            color_counts[color] = cv2.countNonZero(mask) / total_pixels
        
        variety = sum(1 for count in color_counts.values() if count > 0.05)
        balance = "good" if variety >= 2 else "limited"
        
        return {"balance": balance, "variety": variety, "distribution": color_counts}
    
    def _analyze_composition(self, image):
        """Analyze visual composition."""
        # Simple composition scoring based on center of mass
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find center of mass of the food
        moments = cv2.moments(gray)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            
            # Check if center is reasonably centered
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Score based on how centered the composition is
            composition_score = max(1, 10 - (distance_from_center / max_distance) * 5)
        else:
            composition_score = 5
        
        return min(10, composition_score)
    
    def _generate_plating_suggestions(self, analysis: Dict, dish_type: Optional[str]) -> List[str]:
        """Generate plating improvement suggestions."""
        suggestions = []
        
        if analysis.get("color_balance") == "limited":
            suggestions.append("Add colorful garnishes like fresh herbs or vegetables")
        
        if analysis.get("score", 7) < 6:
            suggestions.append("Consider repositioning main elements for better visual balance")
        
        suggestions.append("Use odd numbers of elements for more dynamic presentation")
        suggestions.append("Add height variation with garnishes or sauce drizzles")
        
        return suggestions
    
    async def _reconstruct_from_analysis(self, ingredients: Dict, techniques: Dict, additional_info: Optional[str]) -> Dict[str, Any]:
        """Reconstruct recipe from visual analysis."""
        # Extract ingredient names
        ingredient_names = [ing["name"] for ing in ingredients.get("ingredients", [])]
        
        # Determine cooking method from techniques
        cooking_methods = techniques.get("techniques", ["general cooking"])
        primary_method = cooking_methods[0] if cooking_methods else "cooking"
        
        # Generate estimated recipe
        request = RecipeGenerationRequest(
            ingredients=ingredient_names,
            cooking_time="under 60 minutes",
            difficulty="medium"
        )
        
        base_recipe = await self.recipe_generator.generate_recipe(request)
        
        # Enhance with visual analysis
        estimated_recipe = {
            "title": f"Reconstructed {primary_method.title()} Recipe",
            "description": base_recipe.description,
            "visible_ingredients": ingredient_names,
            "estimated_cooking_method": primary_method,
            "instructions": base_recipe.instructions,
            "estimated_prep_time": base_recipe.prep_time,
            "estimated_cook_time": base_recipe.cook_time,
            "confidence_notes": "Recipe reconstructed from visual analysis"
        }
        
        return estimated_recipe
    
    def _calculate_reconstruction_confidence(self, ingredients: Dict, techniques: Dict) -> float:
        """Calculate confidence score for recipe reconstruction."""
        base_confidence = 0.3
        
        # Boost confidence based on clearly visible ingredients
        ingredient_count = len(ingredients.get("ingredients", []))
        ingredient_bonus = min(0.4, ingredient_count * 0.1)
        
        # Boost confidence based on identifiable techniques
        technique_count = len(techniques.get("techniques", []))
        technique_bonus = min(0.3, technique_count * 0.15)
        
        total_confidence = base_confidence + ingredient_bonus + technique_bonus
        return round(min(1.0, total_confidence), 2)
    
    def _identify_missing_information(self, recipe: Dict) -> List[str]:
        """Identify what information is missing from reconstruction."""
        missing = []
        
        missing.append("Exact ingredient quantities")
        missing.append("Specific seasonings and spices")
        missing.append("Cooking temperatures and times")
        missing.append("Preparation techniques")
        
        return missing
    
    def _detect_cooking_techniques(self, image):
        """Detect cooking techniques from visual cues."""
        # Simplified technique detection
        techniques = {"detected": [], "confidence": {}}
        
        # Analyze for frying (golden/brown colors)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brown_mask = cv2.inRange(hsv, (10, 50, 50), (30, 255, 255))
        brown_ratio = cv2.countNonZero(brown_mask) / (image.shape[0] * image.shape[1])
        
        if brown_ratio > 0.2:
            techniques["detected"].append("frying")
            techniques["confidence"]["frying"] = min(0.9, brown_ratio * 3)
        
        # Analyze for steaming (light colors, high brightness)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 180:
            techniques["detected"].append("steaming")
            techniques["confidence"]["steaming"] = 0.7
        
        # Default if nothing specific detected
        if not techniques["detected"]:
            techniques["detected"] = ["general cooking"]
            techniques["confidence"]["general cooking"] = 0.5
        
        return techniques
    
    def _generate_technique_recommendations(self, techniques: Dict, context: Optional[str]) -> List[str]:
        """Generate recommendations based on detected techniques."""
        recommendations = []
        
        detected = techniques.get("detected", [])
        
        if "frying" in detected:
            recommendations.append("Monitor heat to prevent burning")
            recommendations.append("Flip/stir occasionally for even cooking")
        
        if "steaming" in detected:
            recommendations.append("Keep lid on to maintain steam")
            recommendations.append("Check for doneness with a fork")
        
        if not recommendations:
            recommendations.append("Continue cooking as planned")
            recommendations.append("Check for desired doneness")
        
        return recommendations
    
    def _suggest_next_cooking_steps(self, techniques: Dict) -> List[str]:
        """Suggest next steps based on current cooking technique."""
        detected = techniques.get("detected", [])
        
        if "frying" in detected:
            return ["Check for golden brown color", "Test for doneness"]
        elif "steaming" in detected:
            return ["Check texture with fork", "Season to taste"]
        else:
            return ["Continue cooking", "Monitor progress"]
    
    def _fallback_ingredient_response(self) -> Dict[str, Any]:
        """Provide fallback response when ingredient detection fails."""
        return {
            "ingredients": [
                {"name": "mixed vegetables", "confidence": 0.3, "category": "produce"}
            ],
            "confidence_scores": {"mixed vegetables": 0.3},
            "suggestions": ["Unable to clearly identify specific ingredients"]
        }