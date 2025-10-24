#!/usr/bin/env python3
"""
Simple test to check what's happening with Rust imports
"""

print("🧪 Testing Rust imports step by step...")
print()

# Test 1: Direct module import
print("1. Testing chef_genius_dataloader direct import:")
try:
    import chef_genius_dataloader
    print("   ✅ chef_genius_dataloader module imported successfully")
    print(f"   📋 Available: {dir(chef_genius_dataloader)}")
    
    # Test specific functions
    try:
        from chef_genius_dataloader import FastDataLoader, create_fast_dataloader, benchmark_loading
        print("   ✅ All required functions available")
    except ImportError as e:
        print(f"   ❌ Function import failed: {e}")
        
except ImportError as e:
    print(f"   ❌ Module import failed: {e}")
    print("   🔍 Checking Python path...")
    import sys
    for path in sys.path:
        print(f"      {path}")

print()

# Test 2: fast_dataloader import 
print("2. Testing fast_dataloader import:")
try:
    import sys
    sys.path.insert(0, '/workspace/cli')
    from fast_dataloader import RUST_AVAILABLE
    print(f"   ✅ fast_dataloader imported, RUST_AVAILABLE: {RUST_AVAILABLE}")
    
    if not RUST_AVAILABLE:
        print("   🔍 Checking why RUST_AVAILABLE is False...")
        try:
            import chef_genius_dataloader
            print("   🤔 chef_genius_dataloader imports fine here...")
        except ImportError as e2:
            print(f"   💡 chef_genius_dataloader fails in fast_dataloader context: {e2}")
            
except Exception as e:
    print(f"   ❌ fast_dataloader import failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Test maturin/pip installation 
print("3. Testing installed packages:")
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
if 'chef-genius-dataloader' in result.stdout or 'chef_genius_dataloader' in result.stdout:
    print("   ✅ Found chef_genius_dataloader in pip list")
else:
    print("   ❌ chef_genius_dataloader not found in pip list")
    print("   📋 Installed packages containing 'chef' or 'rust':")
    for line in result.stdout.split('\n'):
        if 'chef' in line.lower() or 'rust' in line.lower() or 'maturin' in line.lower():
            print(f"      {line}")

print()
print("🧪 Test complete!")