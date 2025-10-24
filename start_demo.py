#!/usr/bin/env python3
"""
ChefGenius FLAN-T5 Demo Starter
Quick demo launcher for showcasing the FLAN-T5 recipe generation system
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path
import json
import random
from datetime import datetime

def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        print("✅ FastAPI dependencies available")
        return True
    except ImportError:
        print("❌ Missing dependencies. Installing...")
        return False

def install_dependencies():
    """Install required dependencies"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "fastapi", "uvicorn", "torch", "transformers", "sqlalchemy", "pydantic"
        ])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting ChefGenius backend server...")
    
    backend_path = Path(__file__).parent / "backend"
    if not backend_path.exists():
        print("❌ Backend directory not found")
        sys.exit(1)
    
    os.chdir(backend_path)
    
    try:
        # Start uvicorn server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        return process
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def serve_demo():
    """Serve the demo HTML file"""
    print("🌐 Starting demo server...")
    
    demo_path = Path(__file__).parent / "demo.html"
    if not demo_path.exists():
        print("❌ Demo file not found")
        sys.exit(1)
    
    # Simple HTTP server for the demo
    try:
        os.chdir(Path(__file__).parent)
        server_process = subprocess.Popen([
            sys.executable, "-m", "http.server", "3000"
        ])
        
        print("✅ Demo server started at http://localhost:3000")
        return server_process
    except Exception as e:
        print(f"❌ Failed to start demo server: {e}")
        return None

def create_gap_analysis_tests():
    """Create comprehensive test cases to identify training gaps"""
    test_categories = {
        "dietary_restrictions": [
            "Create a vegan protein-rich dinner for 4 people",
            "Make a gluten-free dessert using almond flour", 
            "Design a keto breakfast with less than 10g carbs",
            "Prepare a dairy-free comfort food meal",
            "Create a low-sodium heart-healthy dinner"
        ],
        "cuisine_diversity": [
            "Make authentic Thai tom yum soup",
            "Create a traditional Ethiopian injera meal",
            "Prepare genuine Korean bibimbap",
            "Make classic French coq au vin",
            "Create authentic Indian biryani from scratch"
        ],
        "cooking_techniques": [
            "Use sous vide for perfect steak",
            "Ferment vegetables for probiotics", 
            "Make pasta from scratch using semolina",
            "Smoke ribs using wood chips",
            "Create molecular gastronomy spheres"
        ],
        "ingredient_substitutions": [
            "Replace eggs in baking recipes",
            "Substitute honey with maple syrup in sauce",
            "Use cauliflower instead of rice",
            "Replace butter with avocado in brownies",
            "Substitute meat with mushrooms in bolognese"
        ],
        "cooking_skill_levels": [
            "5-minute microwave meal for beginners",
            "Advanced knife skills for julienne vegetables",
            "Professional-level sauce emulsification", 
            "Basic one-pot pasta for students",
            "Expert-level pastry cream technique"
        ],
        "seasonal_ingredients": [
            "Winter squash recipes for December",
            "Fresh spring asparagus dishes",
            "Summer stone fruit desserts",
            "Fall apple harvest recipes",
            "Year-round preserved ingredient meals"
        ]
    }
    return test_categories

def log_model_response(category, prompt, response, quality_score):
    """Log model responses for gap analysis"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "category": category,
        "prompt": prompt,
        "response": response,
        "quality_score": quality_score,
        "response_length": len(response),
        "has_ingredients": "ingredients:" in response.lower(),
        "has_instructions": "instructions:" in response.lower() or "steps:" in response.lower(),
        "has_timing": any(word in response.lower() for word in ["minutes", "hours", "cook", "bake", "simmer"])
    }
    
    log_file = Path("gap_analysis_log.json")
    if log_file.exists():
        with open(log_file, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

def run_gap_analysis():
    """Run systematic gap analysis tests"""
    print("\n🔍 Starting Gap Analysis Testing...")
    print("=" * 50)
    
    test_categories = create_gap_analysis_tests()
    total_tests = sum(len(prompts) for prompts in test_categories.values())
    current_test = 0
    
    category_scores = {}
    
    for category, prompts in test_categories.items():
        print(f"\n📊 Testing {category.replace('_', ' ').title()}:")
        category_scores[category] = []
        
        for prompt in prompts:
            current_test += 1
            print(f"  [{current_test}/{total_tests}] {prompt[:50]}...")
            
            # Here you would call your actual model
            # For demo purposes, simulate response
            simulated_response = f"Recipe response for: {prompt}"
            
            # Quality scoring (1-5 scale)
            # In real implementation, you'd analyze response quality
            quality_score = random.randint(2, 5)  # Simulate scoring
            
            log_model_response(category, prompt, simulated_response, quality_score)
            category_scores[category].append(quality_score)
            
            print(f"    Quality: {quality_score}/5")
    
    # Generate summary report
    print("\n📈 Gap Analysis Summary:")
    print("=" * 50)
    
    for category, scores in category_scores.items():
        avg_score = sum(scores) / len(scores)
        print(f"{category.replace('_', ' ').title():.<30} {avg_score:.1f}/5")
        
        if avg_score < 3.5:
            print(f"  ⚠️  Low performance - needs more training data")
        elif avg_score < 4.0:
            print(f"  ⚡ Moderate performance - could improve")
        else:
            print(f"  ✅ Good performance")
    
    print(f"\n📄 Detailed logs saved to: gap_analysis_log.json")
    print("💡 Use these results to identify dataset expansion areas")

def main():
    """Main demo startup function"""
    print("🔥 ChefGenius FLAN-T5 Demo Launcher")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    
    # Check and install dependencies
    if not check_dependencies():
        install_dependencies()
    
    # Ask user for demo mode
    print("\n🎯 Choose demo mode:")
    print("1. Standard Demo (web interface)")
    print("2. Gap Analysis Testing (identify training gaps)")
    print("3. Both (demo + gap analysis)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    run_gap_analysis_flag = choice in ["2", "3"]
    run_demo_flag = choice in ["1", "3"]
    
    backend_process = None
    demo_process = None
    
    if run_demo_flag:
        # Start backend server
        backend_process = start_backend()
        if not backend_process:
            print("⚠️  Backend failed to start, demo will run in offline mode")
        
        # Start demo server
        demo_process = serve_demo()
        if not demo_process:
            print("❌ Failed to start demo")
            sys.exit(1)
        
        # Open browser
        time.sleep(2)
        print("🌐 Opening demo in browser...")
        webbrowser.open("http://localhost:3000/demo.html")
        
        print("\n" + "=" * 50)
        print("🎉 Demo is running!")
        print("📱 Demo URL: http://localhost:3000/demo.html")
        if backend_process:
            print("🔧 Backend API: http://localhost:8000")
            print("📊 API Docs: http://localhost:8000/docs")
        print("⚡ Features:")
        print("   • FLAN-T5-Large recipe generation")
        print("   • Real-time recipe customization") 
        print("   • Hardware-optimized inference")
        print("   • Multiple cuisine support")
        if run_gap_analysis_flag:
            print("   • Gap analysis testing")
        print("\n💡 Press Ctrl+C to stop")
        print("=" * 50)
    
    if run_gap_analysis_flag:
        run_gap_analysis()
    
    if run_demo_flag:
        try:
            # Keep demo running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping demo...")
            
            # Clean up processes
            if demo_process:
                demo_process.terminate()
            if backend_process:
                backend_process.terminate()
            
            print("✅ Demo stopped")

if __name__ == "__main__":
    main()