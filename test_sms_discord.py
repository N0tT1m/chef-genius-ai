#!/usr/bin/env python3
"""
Test script for Discord and SMS notifications
"""

import requests
import time
import urllib.parse
import urllib.request

class DiscordAlerter:
    """Discord webhook and SMS notifications for training events."""
    
    def __init__(self, webhook_url: str = None, phone_number: str = None):
        self.webhook_url = webhook_url
        self.phone_number = phone_number
        self.discord_enabled = webhook_url is not None
        self.sms_enabled = phone_number is not None
        self.enabled = self.discord_enabled or self.sms_enabled
        
    def send_sms(self, message: str):
        """Send SMS notification using multiple services as fallbacks."""
        if not self.sms_enabled:
            print("SMS disabled (no phone number)")
            return
            
        # Clean phone number (remove non-digits except +)
        phone = ''.join(c for c in self.phone_number if c.isdigit() or c == '+')
        print(f"Sending SMS to: {phone}")
        
        # Try TextBelt first (free service)
        try:
            data = {
                'phone': phone,
                'message': f"Chef Genius: {message}",
                'key': 'textbelt'  # Free tier
            }
            
            response = requests.post(
                'https://textbelt.com/text',
                data=data,
                timeout=10
            )
            
            result = response.json()
            if result.get('success', False):
                print(f"✅ SMS sent successfully via TextBelt!")
                return
            else:
                error = result.get('error', 'Unknown error')
                print(f"❌ TextBelt failed: {error}")
                
        except Exception as e:
            print(f"❌ TextBelt SMS failed: {e}")

    def send_notification(self, title: str, description: str, color: int = 0x00ff00, fields: list = None, sms_message: str = None):
        """Send both Discord and SMS notifications."""
        if not self.enabled:
            print("All notifications disabled")
            return
        
        # Send Discord notification
        if self.discord_enabled:
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
        
        # Send SMS notification
        if self.sms_enabled:
            if sms_message is None:
                # Create a simple text version of the notification
                sms_text = f"{title}: {description}"
                if fields and len(fields) > 0:
                    # Add key info from fields
                    key_info = ", ".join([f"{f['name']}: {f['value']}" for f in fields[:2]])
                    sms_text += f" ({key_info})"
            else:
                sms_text = sms_message
                
            # Truncate to SMS length limit
            if len(sms_text) > 160:
                sms_text = sms_text[:157] + "..."
                
            self.send_sms(sms_text)
    
    def training_started(self, model_type: str, epochs: int, batch_size: int, dataset_info: str):
        """Notify training start."""
        fields = [
            {"name": "Model", "value": model_type, "inline": True},
            {"name": "Epochs", "value": str(epochs), "inline": True},
            {"name": "Batch Size", "value": str(batch_size), "inline": True},
            {"name": "Dataset", "value": dataset_info, "inline": False}
        ]
        
        sms_msg = f"Training started: {model_type}, {epochs} epochs, batch size {batch_size}"
        
        self.send_notification(
            title="🚀 Training Started",
            description="Model training has begun!",
            color=0x0099ff,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_completed(self, duration_hours: float, final_metrics: dict):
        """Notify training completion."""
        fields = [
            {"name": "Duration", "value": f"{duration_hours:.2f} hours", "inline": True}
        ]
        
        # Build SMS message
        sms_msg = f"Training completed! Duration: {duration_hours:.2f}h"
        if final_metrics and 'train_loss' in final_metrics:
            sms_msg += f", Final loss: {final_metrics['train_loss']:.4f}"
        
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
            fields=fields,
            sms_message=sms_msg
        )

def test_notifications():
    """Test both Discord and SMS notifications"""
    # Configuration
    webhook_url = "https://discord.com/api/webhooks/1386109570283343953/uGkhj9dpuCg09SbKzZ0Tx2evugJrchQv-nrq3w0r_xi3w8si-XBpQJuxq_p_bcQlhB9W"
    phone_number = "+18125841533"  # Your phone number
    
    print("🧪 Testing Discord and SMS notifications...")
    print(f"Discord enabled: {webhook_url is not None}")
    print(f"SMS enabled: {phone_number is not None}")
    print()
    
    alerter = DiscordAlerter(webhook_url=webhook_url, phone_number=phone_number)
    
    # Test 1: Training started notification
    print("1. Testing training started notification...")
    alerter.training_started(
        model_type="T5-Large",
        epochs=5,
        batch_size=32,
        dataset_info="Train: 4,100,000 recipes, Val: 500,000 recipes"
    )
    
    time.sleep(3)  # Wait between notifications
    
    # Test 2: Training completed notification
    print("\n2. Testing training completed notification...")
    final_metrics = {
        "train_loss": 0.1234,
        "eval_loss": 0.1456,
        "learning_rate": 1.5e-6
    }
    alerter.training_completed(4.25, final_metrics)
    
    print("\n✅ Test completed!")
    print("\nCheck your:")
    print("📱 Phone for SMS messages")
    print("💬 Discord channel for rich notifications")
    print("\nNote: TextBelt free tier allows 1 SMS per day per IP address.")

if __name__ == "__main__":
    test_notifications()