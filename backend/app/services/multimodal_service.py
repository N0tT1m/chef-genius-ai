"""
Multimodal Service for Chef Genius

This service integrates vision and language models to provide comprehensive
food understanding, recipe analysis, and multimodal recipe generation.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
import numpy as np
from PIL import Image
import io
import base64
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel
)
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class MultimodalFoodService:
    """Advanced multimodal service for food image understanding and recipe generation."""
    
    def __init__(self, 
                 vision_language_model: str = "Salesforce/blip2-opt-2.7b",
                 clip_model: str = "openai/clip-vit-large-patch14",
                 food_detection_model: str = "yolov8n.pt"):
        """
        Initialize the multimodal service.
        
        Args:
            vision_language_model: Model for image captioning and VQA
            clip_model: CLIP model for image-text similarity
            food_detection_model: YOLO model for food object detection
        """
        self.vision_language_model_name = vision_language_model
        self.clip_model_name = clip_model
        self.food_detection_model_path = food_detection_model
        
        # Model components
        self.vl_processor = None
        self.vl_model = None
        self.clip_processor = None
        self.clip_model = None
        self.food_detector = None
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Food categories for classification
        self.food_categories = {
            "ingredients": [
                "vegetables", "fruits", "meat", "seafood", "dairy", "grains", 
                "herbs", "spices", "oils", "nuts", "legumes"
            ],
            "cooking_stages": [
                "raw", "chopped", "cooking", "boiling", "frying", "baking", 
                "grilling", "steaming", "simmering", "ready"
            ],
            "dish_types": [
                "appetizer", "main course", "dessert", "soup", "salad", 
                "pasta", "rice dish", "sandwich", "pizza", "curry"
            ],
            "cuisine_styles": [
                "italian", "chinese", "mexican", "indian", "japanese", 
                "french", "thai", "mediterranean", "american", "korean"
            ]
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all multimodal models."""
        try:
            logger.info("Initializing multimodal models...")
            
            # Initialize Vision-Language model (BLIP-2)
            self._initialize_vision_language_model()
            
            # Initialize CLIP model
            self._initialize_clip_model()
            
            # Initialize food detection model
            self._initialize_food_detector()
            
            logger.info("All multimodal models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize multimodal models: {e}")
            raise
    
    def _initialize_vision_language_model(self):
        """Initialize the vision-language model for image understanding."""
        try:
            logger.info(f"Loading vision-language model: {self.vision_language_model_name}")
            
            self.vl_processor = BlipProcessor.from_pretrained(self.vision_language_model_name)
            self.vl_model = BlipForConditionalGeneration.from_pretrained(
                self.vision_language_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            logger.info("Vision-language model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load vision-language model: {e}")
            # Fallback to a smaller model
            try:
                fallback_model = "Salesforce/blip-image-captioning-base"
                self.vl_processor = BlipProcessor.from_pretrained(fallback_model)
                self.vl_model = BlipForConditionalGeneration.from_pretrained(fallback_model)
                logger.info(f"Loaded fallback model: {fallback_model}")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                self.vl_processor = None
                self.vl_model = None
    
    def _initialize_clip_model(self):
        """Initialize CLIP model for image-text similarity."""
        try:
            logger.info(f"Loading CLIP model: {self.clip_model_name}")
            
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(
                self.clip_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_processor = None
            self.clip_model = None
    
    def _initialize_food_detector(self):
        """Initialize YOLO model for food object detection."""
        try:
            logger.info(f"Loading food detection model: {self.food_detection_model_path}")
            
            self.food_detector = YOLO(self.food_detection_model_path)
            
            logger.info("Food detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load food detection model: {e}")
            self.food_detector = None
    
    async def analyze_food_image(self, 
                                image_data: Union[bytes, str, Image.Image],
                                analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Comprehensive food image analysis.
        
        Args:
            image_data: Image data (bytes, base64 string, or PIL Image)
            analysis_type: Type of analysis ("comprehensive", "ingredients", "cooking_stage", "dish_type")
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Process image
            image = self._process_image_input(image_data)
            if image is None:
                return {"error": "Invalid image data"}
            
            results = {
                "analysis_type": analysis_type,
                "image_size": image.size,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Perform different types of analysis
            if analysis_type in ["comprehensive", "caption"]:
                results["caption"] = await self._generate_image_caption(image)
            
            if analysis_type in ["comprehensive", "ingredients"]:
                results["detected_ingredients"] = await self._detect_ingredients(image)
            
            if analysis_type in ["comprehensive", "cooking_stage"]:
                results["cooking_stage"] = await self._classify_cooking_stage(image)
            
            if analysis_type in ["comprehensive", "dish_type"]:
                results["dish_classification"] = await self._classify_dish_type(image)
            
            if analysis_type in ["comprehensive", "objects"]:
                results["object_detection"] = await self._detect_food_objects(image)
            
            if analysis_type in ["comprehensive", "quality"]:
                results["food_quality"] = await self._assess_food_quality(image)
            
            if analysis_type in ["comprehensive", "nutrition"]:
                results["visual_nutrition"] = await self._estimate_visual_nutrition(image)
            
            # Generate recipe suggestions based on visual analysis
            if analysis_type == "comprehensive":
                results["recipe_suggestions"] = await self._generate_recipe_suggestions(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Food image analysis failed: {e}")
            return {"error": str(e)}
    
    def _process_image_input(self, image_data: Union[bytes, str, Image.Image]) -> Optional[Image.Image]:
        """Process various image input formats into PIL Image."""
        try:
            if isinstance(image_data, Image.Image):
                return image_data
            elif isinstance(image_data, bytes):
                return Image.open(io.BytesIO(image_data)).convert("RGB")
            elif isinstance(image_data, str):
                # Assume base64 encoded image
                if image_data.startswith("data:image"):
                    # Remove data:image/jpeg;base64, prefix
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else:
                logger.error(f"Unsupported image format: {type(image_data)}")
                return None
        except Exception as e:
            logger.error(f"Failed to process image input: {e}")
            return None
    
    async def _generate_image_caption(self, image: Image.Image) -> str:
        """Generate descriptive caption for the food image."""
        if not self.vl_model or not self.vl_processor:
            return "Vision-language model not available"
        
        try:
            # Process image for captioning
            inputs = self.vl_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output = self.vl_model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True
                )
            
            caption = self.vl_processor.decode(output[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return "Failed to generate caption"
    
    async def _detect_ingredients(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect and identify ingredients in the image."""
        detected_ingredients = []
        
        try:
            # Use CLIP to classify ingredients
            if self.clip_model and self.clip_processor:
                ingredient_candidates = [
                    "tomatoes", "onions", "garlic", "peppers", "carrots", "potatoes",
                    "chicken", "beef", "fish", "pasta", "rice", "bread",
                    "cheese", "eggs", "milk", "olive oil", "herbs", "spices"
                ]
                
                # Process image and texts
                inputs = self.clip_processor(
                    text=ingredient_candidates,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                
                # Get top ingredients with confidence scores
                top_indices = torch.topk(probs, k=5)[1].cpu().numpy()[0]
                top_probs = torch.topk(probs, k=5)[0].cpu().numpy()[0]
                
                for idx, prob in zip(top_indices, top_probs):
                    if prob > 0.1:  # Confidence threshold
                        detected_ingredients.append({
                            "ingredient": ingredient_candidates[idx],
                            "confidence": float(prob),
                            "category": self._categorize_ingredient(ingredient_candidates[idx])
                        })
            
            # Enhance with object detection if available
            if self.food_detector:
                detection_results = await self._detect_food_objects(image)
                # Merge detection results with CLIP results
                # This is a simplified merge - in production, use more sophisticated fusion
                
            return detected_ingredients
            
        except Exception as e:
            logger.error(f"Ingredient detection failed: {e}")
            return []
    
    async def _classify_cooking_stage(self, image: Image.Image) -> Dict[str, Any]:
        """Classify the cooking stage of food in the image."""
        if not self.clip_model or not self.clip_processor:
            return {"stage": "unknown", "confidence": 0.0}
        
        try:
            cooking_stages = [
                "raw ingredients", "chopped ingredients", "cooking in progress",
                "boiling", "frying", "baking", "grilling", "steaming",
                "nearly done", "ready to serve", "plated dish"
            ]
            
            inputs = self.clip_processor(
                text=cooking_stages,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            # Get the most likely cooking stage
            max_prob_idx = torch.argmax(probs).item()
            max_prob = float(probs[0][max_prob_idx])
            
            return {
                "stage": cooking_stages[max_prob_idx],
                "confidence": max_prob,
                "all_stages": {
                    stage: float(probs[0][i]) 
                    for i, stage in enumerate(cooking_stages)
                }
            }
            
        except Exception as e:
            logger.error(f"Cooking stage classification failed: {e}")
            return {"stage": "unknown", "confidence": 0.0}
    
    async def _classify_dish_type(self, image: Image.Image) -> Dict[str, Any]:
        """Classify the type of dish in the image."""
        if not self.clip_model or not self.clip_processor:
            return {"dish_type": "unknown", "confidence": 0.0}
        
        try:
            dish_types = [
                "appetizer", "soup", "salad", "main course", "pasta dish",
                "rice dish", "sandwich", "pizza", "burger", "stir fry",
                "curry", "stew", "dessert", "cake", "cookies"
            ]
            
            inputs = self.clip_processor(
                text=dish_types,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            # Get top 3 dish types
            top_k = 3
            top_probs, top_indices = torch.topk(probs, k=top_k)
            
            results = {
                "primary_dish_type": dish_types[top_indices[0][0].item()],
                "confidence": float(top_probs[0][0]),
                "top_predictions": [
                    {
                        "dish_type": dish_types[top_indices[0][i].item()],
                        "confidence": float(top_probs[0][i])
                    }
                    for i in range(top_k)
                ]
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Dish type classification failed: {e}")
            return {"dish_type": "unknown", "confidence": 0.0}
    
    async def _detect_food_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect food objects and ingredients using YOLO."""
        if not self.food_detector:
            return []
        
        try:
            # Convert PIL image to numpy array for YOLO
            image_array = np.array(image)
            
            # Run YOLO detection
            results = self.food_detector(image_array, verbose=False)
            
            detected_objects = []
            if results and len(results) > 0:
                for result in results[0].boxes.data:
                    x1, y1, x2, y2, confidence, class_id = result.tolist()
                    
                    # Get class name
                    class_name = self.food_detector.names[int(class_id)]
                    
                    detected_objects.append({
                        "object": class_name,
                        "confidence": confidence,
                        "bounding_box": {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2
                        },
                        "area": (x2 - x1) * (y2 - y1)
                    })
            
            return detected_objects
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    async def _assess_food_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess the visual quality and freshness of food."""
        try:
            # Convert to OpenCV format for analysis
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Analyze color distribution
            color_analysis = self._analyze_colors(image_cv)
            
            # Analyze texture and sharpness
            texture_analysis = self._analyze_texture(image_cv)
            
            # Simple quality scoring based on visual features
            quality_score = self._calculate_quality_score(color_analysis, texture_analysis)
            
            return {
                "overall_quality": quality_score,
                "color_analysis": color_analysis,
                "texture_analysis": texture_analysis,
                "freshness_indicators": self._assess_freshness(color_analysis, texture_analysis)
            }
            
        except Exception as e:
            logger.error(f"Food quality assessment failed: {e}")
            return {"overall_quality": 0.5, "error": str(e)}
    
    def _analyze_colors(self, image_cv: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution in the image."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        
        # Calculate color statistics
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        # Calculate color variance (indicator of color richness)
        color_variance = np.var(hsv[:, :, 0])
        
        return {
            "hue_mean": float(h_mean),
            "saturation_mean": float(s_mean),
            "brightness_mean": float(v_mean),
            "color_variance": float(color_variance)
        }
    
    def _analyze_texture(self, image_cv: np.ndarray) -> Dict[str, Any]:
        """Analyze texture and sharpness of the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (sharpness indicator)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate texture using local binary patterns (simplified)
        texture_score = np.std(gray)
        
        return {
            "sharpness": float(laplacian_var),
            "texture_score": float(texture_score)
        }
    
    def _calculate_quality_score(self, color_analysis: Dict, texture_analysis: Dict) -> float:
        """Calculate overall quality score from visual features."""
        # Normalize and combine different quality indicators
        saturation_score = min(color_analysis["saturation_mean"] / 255.0, 1.0)
        brightness_score = 1.0 - abs(color_analysis["brightness_mean"] / 255.0 - 0.5) * 2
        sharpness_score = min(texture_analysis["sharpness"] / 1000.0, 1.0)
        texture_score = min(texture_analysis["texture_score"] / 100.0, 1.0)
        
        # Weighted combination
        overall_score = (
            saturation_score * 0.3 +
            brightness_score * 0.2 +
            sharpness_score * 0.3 +
            texture_score * 0.2
        )
        
        return float(np.clip(overall_score, 0.0, 1.0))
    
    def _assess_freshness(self, color_analysis: Dict, texture_analysis: Dict) -> List[str]:
        """Assess freshness indicators from visual analysis."""
        indicators = []
        
        # High saturation often indicates freshness
        if color_analysis["saturation_mean"] > 150:
            indicators.append("vibrant colors")
        
        # Good texture detail suggests freshness
        if texture_analysis["texture_score"] > 50:
            indicators.append("good texture detail")
        
        # Sharp images suggest fresh, crisp food
        if texture_analysis["sharpness"] > 500:
            indicators.append("sharp appearance")
        
        if not indicators:
            indicators.append("standard quality")
        
        return indicators
    
    async def _estimate_visual_nutrition(self, image: Image.Image) -> Dict[str, Any]:
        """Estimate nutritional information from visual analysis."""
        try:
            # Get ingredient detection results
            ingredients = await self._detect_ingredients(image)
            
            # Simple nutritional estimation based on detected ingredients
            nutrition_estimate = {
                "calories_per_serving": 0,
                "protein": "unknown",
                "carbs": "unknown",
                "fats": "unknown",
                "fiber": "unknown",
                "confidence": "low"
            }
            
            # Basic estimation logic (in production, use more sophisticated models)
            total_calories = 0
            protein_indicators = 0
            carb_indicators = 0
            fat_indicators = 0
            
            for ingredient in ingredients:
                name = ingredient["ingredient"].lower()
                confidence = ingredient["confidence"]
                
                # Simple calorie estimation
                if any(word in name for word in ["meat", "chicken", "beef", "fish"]):
                    total_calories += 200 * confidence
                    protein_indicators += confidence
                elif any(word in name for word in ["rice", "pasta", "bread", "potato"]):
                    total_calories += 150 * confidence
                    carb_indicators += confidence
                elif any(word in name for word in ["oil", "cheese", "nuts"]):
                    total_calories += 100 * confidence
                    fat_indicators += confidence
                elif any(word in name for word in ["vegetables", "salad", "herbs"]):
                    total_calories += 25 * confidence
            
            nutrition_estimate["calories_per_serving"] = int(total_calories)
            nutrition_estimate["protein"] = "high" if protein_indicators > 0.5 else "medium" if protein_indicators > 0.2 else "low"
            nutrition_estimate["carbs"] = "high" if carb_indicators > 0.5 else "medium" if carb_indicators > 0.2 else "low"
            nutrition_estimate["fats"] = "high" if fat_indicators > 0.5 else "medium" if fat_indicators > 0.2 else "low"
            
            return nutrition_estimate
            
        except Exception as e:
            logger.error(f"Visual nutrition estimation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_recipe_suggestions(self, analysis_results: Dict) -> List[Dict[str, Any]]:
        """Generate recipe suggestions based on visual analysis."""
        suggestions = []
        
        try:
            # Extract key information
            ingredients = analysis_results.get("detected_ingredients", [])
            cooking_stage = analysis_results.get("cooking_stage", {})
            dish_type = analysis_results.get("dish_classification", {})
            
            # Generate suggestions based on analysis
            if ingredients:
                primary_ingredients = [ing["ingredient"] for ing in ingredients[:3]]
                
                # Suggest recipes based on detected ingredients
                suggestions.append({
                    "type": "ingredient_based",
                    "title": f"Recipe with {', '.join(primary_ingredients)}",
                    "description": f"Create a delicious dish using the detected ingredients: {', '.join(primary_ingredients)}",
                    "confidence": 0.8
                })
            
            if dish_type.get("primary_dish_type"):
                dish = dish_type["primary_dish_type"]
                suggestions.append({
                    "type": "dish_type_based",
                    "title": f"Improved {dish}",
                    "description": f"Enhance this {dish} with additional ingredients or cooking techniques",
                    "confidence": dish_type.get("confidence", 0.5)
                })
            
            stage = cooking_stage.get("stage", "")
            if "raw" in stage or "chopped" in stage:
                suggestions.append({
                    "type": "cooking_next_step",
                    "title": "Cooking Instructions",
                    "description": "Get step-by-step instructions for cooking these prepared ingredients",
                    "confidence": 0.9
                })
            
            return suggestions[:3]  # Return top 3 suggestions
            
        except Exception as e:
            logger.error(f"Recipe suggestion generation failed: {e}")
            return []
    
    def _categorize_ingredient(self, ingredient: str) -> str:
        """Categorize an ingredient into a food category."""
        ingredient_lower = ingredient.lower()
        
        for category, items in self.food_categories["ingredients"]:
            if any(item in ingredient_lower for item in items):
                return category
        
        return "other"
    
    async def compare_images(self, 
                           image1: Union[bytes, str, Image.Image],
                           image2: Union[bytes, str, Image.Image]) -> Dict[str, Any]:
        """Compare two food images for similarity."""
        if not self.clip_model or not self.clip_processor:
            return {"error": "CLIP model not available"}
        
        try:
            # Process both images
            img1 = self._process_image_input(image1)
            img2 = self._process_image_input(image2)
            
            if img1 is None or img2 is None:
                return {"error": "Invalid image data"}
            
            # Get image embeddings
            inputs = self.clip_processor(
                images=[img1, img2],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = torch.cosine_similarity(
                image_features[0:1], 
                image_features[1:2]
            ).item()
            
            return {
                "similarity_score": float(similarity),
                "interpretation": self._interpret_similarity(similarity)
            }
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return {"error": str(e)}
    
    def _interpret_similarity(self, score: float) -> str:
        """Interpret similarity score."""
        if score > 0.9:
            return "Very similar dishes"
        elif score > 0.7:
            return "Similar dishes or ingredients"
        elif score > 0.5:
            return "Somewhat related"
        elif score > 0.3:
            return "Different but may share some elements"
        else:
            return "Very different dishes"
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all multimodal models."""
        return {
            "vision_language_model": {
                "loaded": self.vl_model is not None,
                "model_name": self.vision_language_model_name
            },
            "clip_model": {
                "loaded": self.clip_model is not None,
                "model_name": self.clip_model_name
            },
            "food_detector": {
                "loaded": self.food_detector is not None,
                "model_path": self.food_detection_model_path
            },
            "device": self.device
        }