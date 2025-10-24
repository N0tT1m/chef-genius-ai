"""
Chef Assistant API endpoints with W&B Weave integration

Provides conversational AI chef assistant with advanced ML observability
and tracking through Weights & Biases Weave.
"""

import weave
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import Dict, Any, Optional
import logging
from app.services.weave_chef_assistant import WeaveChefAssistant
from app.core.dependencies import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize Weave (in production, this would be configured with your W&B API key)
# weave.init("chef-genius-production")

# Initialize chef assistant
chef_assistant = WeaveChefAssistant()

@router.post("/start-session")
async def start_cooking_session(
    initial_message: Optional[str] = Form(default=None),
    current_user = Depends(get_current_user)
):
    """Start a new cooking session with the AI chef assistant."""
    try:
        session_id = await chef_assistant.start_cooking_session(
            user_id=str(current_user.id),
            initial_message=initial_message
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Cooking session started! How can I help you cook today?"
        }
        
    except Exception as e:
        logger.error(f"Failed to start cooking session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{session_id}")
async def chat_with_chef(
    session_id: str,
    message: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
    current_user = Depends(get_current_user)
):
    """Send a message to the chef assistant in an active session."""
    try:
        # Process image if provided
        image_data = None
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            image_data = await image.read()
        
        # Chat with assistant
        response = await chef_assistant.chat(
            session_id=session_id,
            message=message,
            image_data=image_data
        )
        
        return {
            "status": "success",
            "response": response["message"],
            "actions": response.get("actions", []),
            "tools_used": response.get("tools_used", []),
            "confidence": response.get("confidence", 0.0)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    current_user = Depends(get_current_user)
):
    """Get a summary of the cooking session."""
    try:
        summary = await chef_assistant.get_session_summary(session_id)
        
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        
        return {
            "status": "success",
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/end")
async def end_cooking_session(
    session_id: str,
    current_user = Depends(get_current_user)
):
    """End a cooking session."""
    try:
        result = await chef_assistant.end_session(session_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return {
            "status": "success",
            "message": result["message"],
            "summary": result["summary"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_chef_capabilities():
    """Get the list of chef assistant capabilities."""
    try:
        return {
            "status": "success",
            "capabilities": chef_assistant.capabilities,
            "version": chef_assistant.version,
            "model_name": chef_assistant.model_name
        }
        
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quick-chat")
async def quick_chef_chat(
    message: str = Form(...),
    image: Optional[UploadFile] = File(default=None),
    current_user = Depends(get_current_user)
):
    """Quick chat without session management (for simple queries)."""
    try:
        # Start temporary session
        session_id = await chef_assistant.start_cooking_session(
            user_id=f"temp_{current_user.id}",
            initial_message=message
        )
        
        # Process image if provided
        image_data = None
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            image_data = await image.read()
        
        # Get response
        response = await chef_assistant.chat(
            session_id=session_id,
            message=message,
            image_data=image_data
        )
        
        # End session immediately
        await chef_assistant.end_session(session_id)
        
        return {
            "status": "success",
            "response": response["message"],
            "actions": response.get("actions", []),
            "tools_used": response.get("tools_used", [])
        }
        
    except Exception as e:
        logger.error(f"Quick chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time chat (optional enhancement)
@router.websocket("/ws/{session_id}")
async def websocket_chef_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat with chef assistant."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not message:
                continue
            
            # Process with chef assistant
            response = await chef_assistant.chat(
                session_id=session_id,
                message=message
            )
            
            # Send response
            await websocket.send_json({
                "type": "chef_response",
                "message": response["message"],
                "actions": response.get("actions", []),
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()