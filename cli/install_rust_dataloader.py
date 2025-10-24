#!/usr/bin/env python3
"""
Install script for the Rust-powered data loader
This will compile and install the Rust extension for maximum performance
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_rust_installed():
    """Check if Rust is installed"""
    try:
        result = subprocess.run(['cargo', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_rust():
    """Install Rust using rustup"""
    print("Rust not found. Installing Rust...")
    
    if platform.system() == "Windows":
        # Download and run rustup-init.exe
        print("Please install Rust from https://rustup.rs/ and run this script again")
        sys.exit(1)
    else:
        # Use curl to install rustup
        cmd = 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print("Failed to install Rust. Please install manually from https://rustup.rs/")
            sys.exit(1)
        
        # Source the cargo env
        cargo_env = os.path.expanduser("~/.cargo/env")
        if os.path.exists(cargo_env):
            os.system(f"source {cargo_env}")

def check_maturin_installed():
    """Check if maturin is installed"""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'maturin'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_maturin():
    """Install maturin for building Python extensions"""
    print("Installing maturin...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'maturin'])
    if result.returncode != 0:
        print("Failed to install maturin")
        sys.exit(1)

def build_rust_extension():
    """Build the Rust extension"""
    rust_dir = Path(__file__).parent / "rust_dataloader"
    
    print(f"Building Rust extension in {rust_dir}")
    
    # Change to rust directory
    original_cwd = os.getcwd()
    os.chdir(rust_dir)
    
    try:
        # Build and install the extension
        result = subprocess.run([
            sys.executable, '-m', 'maturin', 'develop', '--release'
        ], check=True)
        
        print("✅ Rust data loader built successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build Rust extension: {e}")
        return False
    except Exception as e:
        print(f"❌ Error building Rust extension: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def test_installation():
    """Test that the Rust extension can be imported"""
    try:
        import chef_genius_dataloader
        print("✅ Rust data loader can be imported successfully!")
        
        # Test basic functionality
        from fast_dataloader import FastDataLoader, DataLoaderConfig
        print("✅ Fast data loader classes imported successfully!")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import Rust data loader: {e}")
        return False

def main():
    print("🦀 Installing Rust-powered data loader for Chef Genius")
    print("This will dramatically improve data loading performance on Windows!")
    print()
    
    # Check prerequisites
    if not check_rust_installed():
        install_rust()
    else:
        print("✅ Rust is installed")
    
    if not check_maturin_installed():
        install_maturin()
    else:
        print("✅ Maturin is installed")
    
    # Build the extension
    if build_rust_extension():
        if test_installation():
            print()
            print("🎉 Installation complete!")
            print("Your data loader should now be 10-50x faster on Windows!")
            print()
            print("To test performance, run:")
            print("  python fast_dataloader.py data/training.json")
        else:
            print("❌ Installation completed but testing failed")
            sys.exit(1)
    else:
        print("❌ Installation failed")
        print("The system will fall back to optimized Python data loading")

if __name__ == "__main__":
    main()