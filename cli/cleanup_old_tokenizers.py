#!/usr/bin/env python3
"""
🧹 TOKENIZER CLEANUP SCRIPT
Remove old tokenizer files and clean up checkpoints with outdated tokenizers
"""

import os
import shutil
from pathlib import Path
import glob

def cleanup_old_tokenizers():
    """Remove old tokenizer files and clean up workspace."""
    
    print("🧹 CLEANING UP OLD TOKENIZER FILES")
    print("=" * 50)
    
    # Files to remove (old tokenizer scripts)
    old_tokenizer_files = [
        "enhanced_tokenizer_setup.py",
        "tokenizer_patch.py", 
        "upgrade_tokenizer_now.py",
        "fix_existing_checkpoints.py"
    ]
    
    current_dir = Path(".")
    removed_count = 0
    
    print("🗑️  Removing old tokenizer scripts...")
    for filename in old_tokenizer_files:
        file_path = current_dir / filename
        if file_path.exists():
            os.remove(file_path)
            print(f"   ✅ Removed: {filename}")
            removed_count += 1
        else:
            print(f"   ⏭️  Not found: {filename}")
    
    # Clean up old checkpoints with problematic tokenizers
    print(f"\n🔍 Checking for checkpoints with old tokenizer issues...")
    
    # Look for checkpoint directories
    checkpoint_patterns = [
        "../docker/models/*/checkpoint-*",
        "./models/*/checkpoint-*",
        "../models/*/checkpoint-*"
    ]
    
    problematic_checkpoints = []
    
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(pattern)
        for checkpoint in checkpoints:
            checkpoint_path = Path(checkpoint)
            
            # Check if checkpoint has incomplete tokenizer files
            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json", 
                "special_tokens_map.json"
            ]
            
            missing_tokenizer_files = []
            for tokenizer_file in tokenizer_files:
                if not (checkpoint_path / tokenizer_file).exists():
                    missing_tokenizer_files.append(tokenizer_file)
            
            if missing_tokenizer_files:
                problematic_checkpoints.append({
                    "path": checkpoint_path,
                    "missing_files": missing_tokenizer_files
                })
    
    if problematic_checkpoints:
        print(f"   Found {len(problematic_checkpoints)} checkpoints with tokenizer issues")
        
        print(f"\n⚠️  Problematic checkpoints:")
        for checkpoint in problematic_checkpoints:
            print(f"   📁 {checkpoint['path']}")
            print(f"      Missing: {', '.join(checkpoint['missing_files'])}")
        
        # Ask user if they want to clean these up
        response = input(f"\n🤔 Remove problematic checkpoints? (y/N): ").strip().lower()
        
        if response == 'y':
            for checkpoint in problematic_checkpoints:
                try:
                    shutil.rmtree(checkpoint['path'])
                    print(f"   🗑️  Removed: {checkpoint['path']}")
                    removed_count += 1
                except Exception as e:
                    print(f"   ❌ Failed to remove {checkpoint['path']}: {e}")
        else:
            print("   ⏭️  Keeping problematic checkpoints")
    else:
        print("   ✅ No problematic checkpoints found")
    
    # Clean up any backup files
    print(f"\n🧹 Cleaning up backup files...")
    backup_patterns = ["*.py.backup", "*.py.bak", "*~"]
    
    for pattern in backup_patterns:
        backup_files = list(current_dir.glob(pattern))
        for backup_file in backup_files:
            os.remove(backup_file)
            print(f"   🗑️  Removed backup: {backup_file.name}")
            removed_count += 1
    
    # Clean up temporary files
    temp_patterns = ["*.tmp", "*.temp", "__pycache__"]
    
    for pattern in temp_patterns:
        if pattern == "__pycache__":
            pycache_dirs = list(current_dir.glob("**/__pycache__"))
            for pycache_dir in pycache_dirs:
                shutil.rmtree(pycache_dir)
                print(f"   🗑️  Removed cache: {pycache_dir}")
                removed_count += 1
        else:
            temp_files = list(current_dir.glob(pattern))
            for temp_file in temp_files:
                os.remove(temp_file)
                print(f"   🗑️  Removed temp: {temp_file.name}")
                removed_count += 1
    
    # Summary
    print(f"\n📊 CLEANUP SUMMARY")
    print("-" * 30)
    print(f"Files removed: {removed_count}")
    print(f"✅ Workspace cleaned!")
    
    # Show remaining enterprise tokenizer files
    print(f"\n📁 REMAINING ENTERPRISE TOKENIZER FILES:")
    enterprise_files = [
        "enterprise_recipe_tokenizer.py",
        "integrate_enterprise_tokenizer.py",
        "b2b_recipe_testing.py"
    ]
    
    for filename in enterprise_files:
        file_path = current_dir / filename
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"   ✅ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"   ❌ {filename} (missing)")
    
    return removed_count

def verify_enterprise_tokenizer():
    """Verify the enterprise tokenizer is working correctly."""
    
    print(f"\n🔍 VERIFYING ENTERPRISE TOKENIZER")
    print("-" * 40)
    
    try:
        from enterprise_recipe_tokenizer import EnterpriseRecipeTokenizer
        
        # Create tokenizer
        enterprise_tokenizer = EnterpriseRecipeTokenizer()
        tokenizer, num_added = enterprise_tokenizer.create_tokenizer()
        
        # Test token limit compliance
        test_prompt = enterprise_tokenizer.create_b2b_prompt(
            "Create chicken with 2 sides for restaurant",
            business_type="Restaurant",
            service_style="Fast Casual"
        )
        
        tokens = tokenizer.encode(test_prompt)
        
        print(f"✅ Enterprise tokenizer loaded successfully")
        print(f"📊 Vocabulary size: {len(tokenizer):,}")
        print(f"🎯 Special tokens added: {num_added:,}")
        print(f"📝 Test prompt tokens: {len(tokens)} (limit: 512)")
        
        if len(tokens) <= 512:
            print(f"✅ Token limit compliance: PASSED")
        else:
            print(f"⚠️  Token limit compliance: NEEDS TRUNCATION")
        
        return True
        
    except Exception as e:
        print(f"❌ Enterprise tokenizer verification failed: {e}")
        return False

def main():
    """Main cleanup workflow."""
    
    print("🧹 TOKENIZER CLEANUP & VERIFICATION")
    print("=" * 50)
    
    # Cleanup old files
    removed_count = cleanup_old_tokenizers()
    
    # Verify enterprise tokenizer
    if verify_enterprise_tokenizer():
        print(f"\n🎉 CLEANUP COMPLETE!")
        print(f"✅ Removed {removed_count} old files")
        print(f"✅ Enterprise tokenizer verified")
        print(f"\n💡 Next steps:")
        print(f"   1. Test: python3 enterprise_recipe_tokenizer.py")
        print(f"   2. Integrate: python3 integrate_enterprise_tokenizer.py")
        print(f"   3. B2B Test: python3 b2b_recipe_testing.py")
    else:
        print(f"\n❌ CLEANUP INCOMPLETE")
        print(f"⚠️  Enterprise tokenizer needs attention")

if __name__ == "__main__":
    main()