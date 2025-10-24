"""
Weights & Biases Weave Integration for Chef Assistant

This module integrates W&B Weave for advanced ML observability, conversation tracking,
and intelligent chef assistant capabilities.
"""

import weave
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from dataclasses import dataclass
from app.services.recipe_generator import RecipeGeneratorService
from app.services.multimodal_service import MultimodalFoodService
from app.services.tool_system import ToolSystem, ToolCall
from app.services.rag_system import RAGSystem

logger = logging.getLogger(__name__)

@dataclass
class ChefMessage:
    """Represents a message in the chef assistant conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class CookingSession:
    """Represents a cooking session with the chef assistant."""
    session_id: str
    user_id: str
    messages: List[ChefMessage]
    current_recipe: Optional[Dict[str, Any]] = None
    active_timers: List[Dict[str, Any]] = None
    cooking_context: Dict[str, Any] = None
    created_at: datetime = None

class WeaveChefAssistant(weave.Model):
    """
    W&B Weave-powered Chef Assistant with conversation tracking and ML observability.
    """
    
    model_name: str = "chef-genius-assistant"
    version: str = "1.0.0"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize services
        self.recipe_generator = RecipeGeneratorService()
        self.multimodal_service = MultimodalFoodService()
        self.tool_system = ToolSystem()
        self.rag_system = RAGSystem()
        
        # Conversation management
        self.active_sessions = {}
        
        # Chef assistant capabilities
        self.capabilities = {
            "recipe_generation": True,
            "image_analysis": True,
            "nutrition_calculation": True,
            "cooking_timers": True,
            "ingredient_substitution": True,
            "meal_planning": True,
            "cooking_guidance": True,
            "shopping_lists": True
        }
    
    @weave.op()
    async def start_cooking_session(self, user_id: str, initial_message: str = None) -> str:
        """Start a new cooking session with the chef assistant."""
        
        session_id = f"session_{user_id}_{datetime.now().timestamp()}"
        
        session = CookingSession(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            active_timers=[],
            cooking_context={
                "preferences": {},
                "dietary_restrictions": [],
                "skill_level": "intermediate",
                "available_ingredients": [],
                "kitchen_equipment": []
            },
            created_at=datetime.now()
        )
        
        # Add system message
        system_message = ChefMessage(
            role="system",
            content="Hello! I'm your AI chef assistant. I can help you with recipes, cooking guidance, nutrition analysis, and more. What would you like to cook today?",
            timestamp=datetime.now(),
            metadata={"session_start": True}
        )
        session.messages.append(system_message)
        
        # Add initial user message if provided
        if initial_message:
            user_message = ChefMessage(
                role="user",
                content=initial_message,
                timestamp=datetime.now()
            )
            session.messages.append(user_message)
        
        self.active_sessions[session_id] = session
        
        # Log to Weave
        weave.log({
            "event": "session_started",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return session_id
    
    @weave.op()
    async def chat(self, session_id: str, message: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Process a chat message in the cooking session.
        
        Args:
            session_id: Active cooking session ID
            message: User message
            image_data: Optional image data for multimodal interaction
        
        Returns:
            Assistant response with actions and metadata
        """
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Add user message to session
        user_message = ChefMessage(
            role="user",
            content=message,
            timestamp=datetime.now(),
            metadata={"has_image": image_data is not None}
        )
        session.messages.append(user_message)
        
        # Analyze intent and generate response
        response = await self._process_chef_request(session, message, image_data)
        
        # Add assistant message to session
        assistant_message = ChefMessage(
            role="assistant",
            content=response["message"],
            timestamp=datetime.now(),
            metadata={
                "actions": response.get("actions", []),
                "tools_used": response.get("tools_used", []),
                "confidence": response.get("confidence", 0.0)
            }
        )
        session.messages.append(assistant_message)
        
        # Log interaction to Weave
        weave.log({
            "event": "chat_interaction",
            "session_id": session_id,
            "user_message": message,
            "assistant_response": response["message"],
            "actions": response.get("actions", []),
            "tools_used": response.get("tools_used", []),
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    @weave.op()
    async def _process_chef_request(self, session: CookingSession, message: str, image_data: Optional[bytes]) -> Dict[str, Any]:
        """Process a chef assistant request with intent detection and action execution."""
        
        # Analyze intent
        intent = await self._detect_intent(message, image_data)
        
        response = {
            "message": "",
            "actions": [],
            "tools_used": [],
            "confidence": 0.8
        }
        
        try:
            if intent["type"] == "recipe_request":
                response = await self._handle_recipe_request(session, intent, image_data)
            
            elif intent["type"] == "image_analysis":
                response = await self._handle_image_analysis(session, image_data, message)
            
            elif intent["type"] == "cooking_guidance":
                response = await self._handle_cooking_guidance(session, intent)
            
            elif intent["type"] == "timer_management":
                response = await self._handle_timer_request(session, intent)
            
            elif intent["type"] == "nutrition_question":
                response = await self._handle_nutrition_question(session, intent)
            
            elif intent["type"] == "substitution_request":
                response = await self._handle_substitution_request(session, intent)
            
            elif intent["type"] == "meal_planning":
                response = await self._handle_meal_planning(session, intent)
            
            else:
                response = await self._handle_general_chat(session, message)
        
        except Exception as e:
            logger.error(f"Error processing chef request: {e}")
            response = {
                "message": "I'm sorry, I encountered an issue. Could you please try rephrasing your request?",
                "actions": [],
                "tools_used": [],
                "confidence": 0.0,
                "error": str(e)
            }
        
        return response
    
    @weave.op()
    async def _detect_intent(self, message: str, image_data: Optional[bytes]) -> Dict[str, Any]:
        """Detect user intent from message and optional image."""
        
        message_lower = message.lower()
        
        # Simple rule-based intent detection (can be replaced with ML model)
        if image_data:
            return {"type": "image_analysis", "confidence": 0.9}
        
        elif any(word in message_lower for word in ["recipe", "cook", "make", "prepare"]):
            return {"type": "recipe_request", "confidence": 0.85}
        
        elif any(word in message_lower for word in ["timer", "time", "minutes", "alarm"]):
            return {"type": "timer_management", "confidence": 0.9}
        
        elif any(word in message_lower for word in ["nutrition", "calories", "healthy", "diet"]):
            return {"type": "nutrition_question", "confidence": 0.8}
        
        elif any(word in message_lower for word in ["substitute", "replace", "instead of", "alternative"]):
            return {"type": "substitution_request", "confidence": 0.85}
        
        elif any(word in message_lower for word in ["how to", "how do i", "cooking", "technique"]):
            return {"type": "cooking_guidance", "confidence": 0.8}
        
        elif any(word in message_lower for word in ["meal plan", "weekly", "menu", "planning"]):
            return {"type": "meal_planning", "confidence": 0.8}
        
        else:
            return {"type": "general_chat", "confidence": 0.6}
    
    @weave.op()
    async def _handle_recipe_request(self, session: CookingSession, intent: Dict, image_data: Optional[bytes]) -> Dict[str, Any]:
        """Handle recipe generation requests."""
        
        # Extract recipe requirements from conversation context
        requirements = self._extract_recipe_requirements(session)
        
        try:
            # Generate recipe
            recipe = await self.recipe_generator.generate_recipe_with_tools(requirements)
            
            # Update session context
            session.current_recipe = recipe["recipe"].__dict__ if hasattr(recipe["recipe"], '__dict__') else recipe["recipe"]
            
            # Format response
            recipe_text = self._format_recipe_response(recipe["recipe"])
            
            actions = [
                {
                    "type": "recipe_generated",
                    "recipe_id": session.current_recipe.get("title", "").replace(" ", "_").lower(),
                    "recipe": session.current_recipe
                }
            ]
            
            # Add nutrition info if available
            if recipe.get("enhancements", {}).get("nutrition"):
                actions.append({
                    "type": "nutrition_calculated",
                    "nutrition": recipe["enhancements"]["nutrition"]
                })
            
            return {
                "message": f"Here's a great recipe for you:\n\n{recipe_text}",
                "actions": actions,
                "tools_used": ["recipe_generator"],
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"Recipe generation failed: {e}")
            return {
                "message": "I'm having trouble generating a recipe right now. Could you provide more specific details about what you'd like to cook?",
                "actions": [],
                "tools_used": [],
                "confidence": 0.3
            }
    
    @weave.op()
    async def _handle_image_analysis(self, session: CookingSession, image_data: bytes, message: str) -> Dict[str, Any]:
        """Handle image analysis requests."""
        
        try:
            # Analyze the image
            analysis = await self.multimodal_service.analyze_food_image(image_data, "comprehensive")
            
            if "error" in analysis:
                return {
                    "message": "I couldn't analyze the image properly. Could you try uploading a clearer photo?",
                    "actions": [],
                    "tools_used": ["multimodal_service"],
                    "confidence": 0.2
                }
            
            # Format analysis response
            response_text = self._format_image_analysis_response(analysis)
            
            actions = [
                {
                    "type": "image_analyzed",
                    "analysis": analysis
                }
            ]
            
            # Suggest recipes based on analysis
            if analysis.get("recipe_suggestions"):
                actions.append({
                    "type": "recipe_suggestions",
                    "suggestions": analysis["recipe_suggestions"]
                })
            
            return {
                "message": response_text,
                "actions": actions,
                "tools_used": ["multimodal_service"],
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                "message": "I had trouble analyzing your image. Please try uploading a different photo.",
                "actions": [],
                "tools_used": [],
                "confidence": 0.2
            }
    
    @weave.op()
    async def _handle_timer_request(self, session: CookingSession, intent: Dict) -> Dict[str, Any]:
        """Handle cooking timer requests."""
        
        # Extract timer details from the last message
        last_message = session.messages[-1].content.lower()
        
        # Simple extraction (can be improved with NLP)
        import re
        time_match = re.search(r'(\d+)\s*(minute|min|hour|hr)', last_message)
        
        if time_match:
            duration = int(time_match.group(1))
            unit = time_match.group(2)
            
            # Convert to minutes
            if unit in ['hour', 'hr']:
                duration *= 60
            
            # Set timer using tool system
            timer_call = ToolCall(
                name="cooking_timer",
                arguments={
                    "action": "set",
                    "duration_minutes": duration,
                    "timer_name": f"cooking_timer_{len(session.active_timers) + 1}"
                },
                call_id="timer_set"
            )
            
            result = await self.tool_system.execute_tool_call(timer_call)
            
            if result.success:
                # Add to session timers
                timer_info = result.result
                session.active_timers.append(timer_info)
                
                return {
                    "message": f"✅ Timer set for {duration} minutes! I'll let you know when it's done.",
                    "actions": [
                        {
                            "type": "timer_set",
                            "timer": timer_info
                        }
                    ],
                    "tools_used": ["cooking_timer"],
                    "confidence": 0.95
                }
            else:
                return {
                    "message": "I couldn't set the timer. Could you specify the time more clearly?",
                    "actions": [],
                    "tools_used": [],
                    "confidence": 0.4
                }
        else:
            return {
                "message": "How long would you like me to set the timer for? Please specify minutes or hours.",
                "actions": [],
                "tools_used": [],
                "confidence": 0.7
            }
    
    @weave.op()
    async def _handle_nutrition_question(self, session: CookingSession, intent: Dict) -> Dict[str, Any]:
        """Handle nutrition-related questions."""
        
        if session.current_recipe:
            # Calculate nutrition for current recipe
            nutrition_call = ToolCall(
                name="nutrition_calculator",
                arguments={
                    "ingredients": session.current_recipe.get("ingredients", []),
                    "servings": session.current_recipe.get("servings", 1)
                },
                call_id="nutrition_calc"
            )
            
            result = await self.tool_system.execute_tool_call(nutrition_call)
            
            if result.success:
                nutrition = result.result.get("nutrition", {})
                response_text = self._format_nutrition_response(nutrition)
                
                return {
                    "message": response_text,
                    "actions": [
                        {
                            "type": "nutrition_calculated",
                            "nutrition": nutrition
                        }
                    ],
                    "tools_used": ["nutrition_calculator"],
                    "confidence": 0.9
                }
        
        return {
            "message": "I'd be happy to help with nutrition information! Do you have a specific recipe or ingredients you'd like me to analyze?",
            "actions": [],
            "tools_used": [],
            "confidence": 0.6
        }
    
    @weave.op()
    async def _handle_substitution_request(self, session: CookingSession, intent: Dict) -> Dict[str, Any]:
        """Handle ingredient substitution requests."""
        
        # Extract ingredient from message
        last_message = session.messages[-1].content
        # Simple extraction - can be improved
        words = last_message.split()
        
        # Look for ingredient mentions
        potential_ingredient = None
        for i, word in enumerate(words):
            if word.lower() in ["substitute", "replace"] and i + 1 < len(words):
                potential_ingredient = words[i + 1]
                break
        
        if potential_ingredient:
            sub_call = ToolCall(
                name="ingredient_substitution",
                arguments={
                    "ingredient": potential_ingredient,
                    "dietary_restrictions": session.cooking_context.get("dietary_restrictions", []),
                    "recipe_context": "general cooking"
                },
                call_id="substitution_find"
            )
            
            result = await self.tool_system.execute_tool_call(sub_call)
            
            if result.success:
                substitutions = result.result.get("substitutions", {})
                response_text = self._format_substitution_response(potential_ingredient, substitutions)
                
                return {
                    "message": response_text,
                    "actions": [
                        {
                            "type": "substitutions_found",
                            "ingredient": potential_ingredient,
                            "substitutions": substitutions
                        }
                    ],
                    "tools_used": ["ingredient_substitution"],
                    "confidence": 0.85
                }
        
        return {
            "message": "What ingredient would you like me to find substitutes for?",
            "actions": [],
            "tools_used": [],
            "confidence": 0.6
        }
    
    @weave.op()
    async def _handle_cooking_guidance(self, session: CookingSession, intent: Dict) -> Dict[str, Any]:
        """Handle cooking technique and guidance requests."""
        
        # This could integrate with a cooking knowledge base
        guidance_responses = {
            "how to chop onions": "To chop onions: 1) Cut off the top, leaving the root end intact. 2) Peel the outer layer. 3) Make horizontal cuts from the top, stopping before the root. 4) Make vertical cuts. 5) Finally, cut across to dice.",
            "how to cook pasta": "Bring a large pot of salted water to boil. Add pasta and cook according to package directions, stirring occasionally. Taste test 1-2 minutes before the suggested time. Pasta should be al dente (firm to the bite).",
            "how to season": "Season in layers throughout cooking. Start with salt and pepper, then add herbs and spices. Taste frequently and adjust. Remember: you can always add more, but you can't take it away!"
        }
        
        last_message = session.messages[-1].content.lower()
        
        # Find matching guidance
        for key, guidance in guidance_responses.items():
            if any(word in last_message for word in key.split()):
                return {
                    "message": guidance,
                    "actions": [
                        {
                            "type": "cooking_guidance",
                            "technique": key,
                            "guidance": guidance
                        }
                    ],
                    "tools_used": [],
                    "confidence": 0.8
                }
        
        return {
            "message": "I'd be happy to help with cooking techniques! What specific cooking method or technique would you like guidance on?",
            "actions": [],
            "tools_used": [],
            "confidence": 0.6
        }
    
    @weave.op()
    async def _handle_meal_planning(self, session: CookingSession, intent: Dict) -> Dict[str, Any]:
        """Handle meal planning requests."""
        
        # This would integrate with meal planning logic
        return {
            "message": "I'd love to help you plan your meals! Let me know your preferences, dietary restrictions, and how many days you'd like to plan for.",
            "actions": [
                {
                    "type": "meal_planning_started",
                    "preferences": session.cooking_context.get("preferences", {})
                }
            ],
            "tools_used": [],
            "confidence": 0.7
        }
    
    @weave.op()
    async def _handle_general_chat(self, session: CookingSession, message: str) -> Dict[str, Any]:
        """Handle general chat messages."""
        
        return {
            "message": "I'm here to help with all things cooking! I can generate recipes, analyze food images, set timers, calculate nutrition, find ingredient substitutions, and provide cooking guidance. What would you like to explore?",
            "actions": [],
            "tools_used": [],
            "confidence": 0.5
        }
    
    def _extract_recipe_requirements(self, session: CookingSession):
        """Extract recipe requirements from conversation context."""
        # This is a simplified version - would use NLP in production
        from app.models.recipe import RecipeGenerationRequest
        
        return RecipeGenerationRequest(
            ingredients=session.cooking_context.get("available_ingredients", []),
            cuisine=None,
            dietary_restrictions=session.cooking_context.get("dietary_restrictions", []),
            difficulty="medium",
            meal_type="main course"
        )
    
    def _format_recipe_response(self, recipe) -> str:
        """Format recipe for chat response."""
        if hasattr(recipe, '__dict__'):
            recipe_dict = recipe.__dict__
        else:
            recipe_dict = recipe
            
        response = f"🍳 **{recipe_dict.get('title', 'Delicious Recipe')}**\n\n"
        response += f"*{recipe_dict.get('description', '')}*\n\n"
        response += f"⏱️ Prep: {recipe_dict.get('prep_time', 0)} min | Cook: {recipe_dict.get('cook_time', 0)} min | Serves: {recipe_dict.get('servings', 0)}\n\n"
        
        response += "**Ingredients:**\n"
        for ing in recipe_dict.get('ingredients', []):
            if isinstance(ing, dict):
                response += f"• {ing.get('amount', '')} {ing.get('unit', '')} {ing.get('name', '')}\n"
            else:
                response += f"• {ing}\n"
        
        response += "\n**Instructions:**\n"
        for i, instruction in enumerate(recipe_dict.get('instructions', []), 1):
            response += f"{i}. {instruction}\n"
        
        return response
    
    def _format_image_analysis_response(self, analysis: Dict) -> str:
        """Format image analysis for chat response."""
        response = "🔍 **Image Analysis Results:**\n\n"
        
        if analysis.get("caption"):
            response += f"**What I see:** {analysis['caption']}\n\n"
        
        if analysis.get("detected_ingredients"):
            response += "**Detected Ingredients:**\n"
            for ing in analysis["detected_ingredients"][:5]:
                confidence = ing.get("confidence", 0) * 100
                response += f"• {ing.get('ingredient', '')} ({confidence:.0f}% confidence)\n"
            response += "\n"
        
        if analysis.get("cooking_stage"):
            stage = analysis["cooking_stage"].get("stage", "unknown")
            confidence = analysis["cooking_stage"].get("confidence", 0) * 100
            response += f"**Cooking Stage:** {stage} ({confidence:.0f}% confidence)\n\n"
        
        if analysis.get("recipe_suggestions"):
            response += "**Recipe Suggestions:**\n"
            for suggestion in analysis["recipe_suggestions"][:3]:
                response += f"• {suggestion.get('title', '')}\n"
        
        return response
    
    def _format_nutrition_response(self, nutrition: Dict) -> str:
        """Format nutrition information for chat response."""
        response = "🥗 **Nutritional Information:**\n\n"
        
        if nutrition.get("calories"):
            response += f"**Calories:** {nutrition['calories']} per serving\n"
        
        if nutrition.get("macros"):
            macros = nutrition["macros"]
            response += f"**Protein:** {macros.get('protein', 0)}g\n"
            response += f"**Carbs:** {macros.get('carbs', 0)}g\n"
            response += f"**Fat:** {macros.get('fat', 0)}g\n"
        
        return response
    
    def _format_substitution_response(self, ingredient: str, substitutions: Dict) -> str:
        """Format substitution suggestions for chat response."""
        response = f"🔄 **Substitutes for {ingredient}:**\n\n"
        
        if substitutions.get("substitutes"):
            for sub in substitutions["substitutes"][:3]:
                ratio = sub.get("ratio", "1:1")
                notes = sub.get("notes", "")
                response += f"• **{sub.get('substitute', '')}** ({ratio}) - {notes}\n"
        
        if substitutions.get("notes"):
            response += f"\n**Tips:** {'; '.join(substitutions['notes'])}"
        
        return response
    
    @weave.op()
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the cooking session."""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        
        summary = {
            "session_id": session_id,
            "user_id": session.user_id,
            "duration": (datetime.now() - session.created_at).total_seconds(),
            "message_count": len(session.messages),
            "recipes_generated": 1 if session.current_recipe else 0,
            "active_timers": len(session.active_timers),
            "tools_used": set(),
            "topics_discussed": []
        }
        
        # Analyze conversation
        for message in session.messages:
            if message.role == "assistant" and message.metadata:
                tools = message.metadata.get("tools_used", [])
                summary["tools_used"].update(tools)
        
        summary["tools_used"] = list(summary["tools_used"])
        
        return summary
    
    @weave.op()
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a cooking session and log final metrics."""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        summary = await self.get_session_summary(session_id)
        
        # Log session end to Weave
        weave.log({
            "event": "session_ended",
            "session_summary": summary,
            "timestamp": datetime.now().isoformat()
        })
        
        # Clean up
        del self.active_sessions[session_id]
        
        return {
            "message": "Thanks for cooking with me! Have a delicious meal! 👨‍🍳",
            "summary": summary
        }