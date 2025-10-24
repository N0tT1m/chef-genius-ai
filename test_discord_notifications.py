#!/usr/bin/env python3
"""
Test script for Discord notifications without requiring full training setup
"""

import requests
import time

class DiscordAlerter:
    """Discord webhook notifications for training events."""
    
    def __init__(self, webhook_url: str = None, phone_number: str = None):
        self.webhook_url = webhook_url
        self.phone_number = phone_number
        self.enabled = webhook_url is not None
        
    def send_notification(self, title: str, description: str, color: int = 0x00ff00, fields: list = None):
        """Send a Discord notification with rich embed."""
        if not self.enabled:
            print("Discord notifications disabled (no webhook URL)")
            return
            
        try:
            embed = {
                "title": title,
                "description": description,
                "color": color,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "footer": {
                    "text": "Chef Genius Training Bot"
                }
            }
            
            if fields:
                embed["fields"] = fields
                
            payload = {
                "embeds": [embed],
                "username": "Chef Genius Training"
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            print(f"✅ Discord notification sent successfully! Status: {response.status_code}")
            
        except Exception as e:
            print(f"❌ Failed to send Discord notification: {e}")
    
    def training_started(self, model_type: str, epochs: int, batch_size: int, dataset_info: str):
        """Notify training start."""
        fields = [
            {"name": "Model", "value": model_type, "inline": True},
            {"name": "Epochs", "value": str(epochs), "inline": True},
            {"name": "Batch Size", "value": str(batch_size), "inline": True},
            {"name": "Dataset", "value": dataset_info, "inline": False}
        ]
        
        self.send_notification(
            title="🚀 Training Started",
            description="Model training has begun!",
            color=0x0099ff,
            fields=fields
        )
    
    def training_completed(self, duration_hours: float, final_metrics: dict):
        """Notify training completion."""
        fields = [
            {"name": "Duration", "value": f"{duration_hours:.2f} hours", "inline": True}
        ]
        
        if final_metrics:
            for metric, value in final_metrics.items():
                fields.append({
                    "name": metric.replace("_", " ").title(),
                    "value": f"{value:.4f}" if isinstance(value, float) else str(value),
                    "inline": True
                })
        
        self.send_notification(
            title="✅ Training Completed",
            description="Model training finished successfully!",
            color=0x00ff00,
            fields=fields
        )

def test_discord_notifications():
    """Test Discord notifications"""
    # Use the webhook URL from the error message
    webhook_url = "https://discord.com/api/webhooks/1386109570283343953/uGkhj9dpuCg09SbKzZ0Tx2evugJrchQv-nrq3w0r_xi3w8si-XBpQJuxq_p_bcQlhB9W"
    
    print("Testing Discord notifications...")
    alerter = DiscordAlerter(webhook_url=webhook_url)
    
    # Test training started notification
    print("\n1. Testing training started notification...")
    alerter.training_started(
        model_type="T5-Large",
        epochs=5,
        batch_size=32,
        dataset_info="Train: 4,100,000 recipes, Val: 500,000 recipes"
    )
    
    time.sleep(2)  # Wait between notifications
    
    # Test training completed notification
    print("\n2. Testing training completed notification...")
    final_metrics = {
        "final_loss": 0.1234,
        "train_loss": 0.1150,
        "eval_loss": 0.1456,
        "learning_rate": 1.5e-6
    }
    alerter.training_completed(4.25, final_metrics)
    
    print("\n✅ Test completed! Check your Discord channel for notifications.")

if __name__ == "__main__":
    test_discord_notifications()