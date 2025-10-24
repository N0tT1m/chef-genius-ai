#!/usr/bin/env python3
"""Test Discord webhook to verify it's working"""

import requests
import json

webhook_url = "https://discord.com/api/webhooks/1386109570283343953/uGkhj9dpuCg09SbKzZ0Tx2evugJrchQv-nrq3w0r_xi3w8si-XBpQJuxq_p_bcQlhB9W"

# Test simple message
try:
    payload = {
        "embeds": [{
            "title": "🧪 Discord Webhook Test",
            "description": "Testing webhook connection from Chef Genius training system",
            "color": 0x00ff00,
            "fields": [
                {"name": "Status", "value": "✅ Webhook is working!", "inline": True},
                {"name": "Test Time", "value": "Now", "inline": True}
            ],
            "footer": {"text": "Chef Genius Training Bot"}
        }],
        "username": "Chef Genius Training"
    }

    response = requests.post(webhook_url, json=payload, timeout=10)

    if response.status_code == 204:
        print("✅ Discord webhook test successful!")
        print("Check your Discord channel for the test message")
    else:
        print(f"❌ Discord webhook failed with status code: {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"❌ Error testing Discord webhook: {e}")
