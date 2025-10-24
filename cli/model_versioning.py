#!/usr/bin/env python3
"""
Model Versioning and Checkpoint Management System
Handles model versions, rollbacks, and performance tracking
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelVersionManager:
    """Manages model versions, checkpoints, and metadata."""
    
    def __init__(self, base_model_dir: str):
        self.base_dir = Path(base_model_dir)
        self.versions_dir = self.base_dir / "versions"
        self.backups_dir = self.base_dir / "backups"
        self.metadata_file = self.base_dir / "version_metadata.json"
        
        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load version metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                
        return {
            "current_version": None,
            "versions": {},
            "latest_version_number": 0
        }
        
    def _save_metadata(self):
        """Save version metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _calculate_model_hash(self, model_path: Path) -> str:
        """Calculate hash of model files for integrity checking."""
        hasher = hashlib.sha256()
        
        # Hash all model files
        for file_path in sorted(model_path.glob("*.bin")):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
                        
        # Also hash config.json if it exists
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                hasher.update(f.read())
                
        return hasher.hexdigest()
        
    def create_version(self, 
                      model_path: str, 
                      version_name: Optional[str] = None,
                      description: str = "",
                      metrics: Optional[Dict] = None,
                      training_config: Optional[Dict] = None) -> str:
        """Create a new model version."""
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
            
        # Generate version info
        if version_name is None:
            self.metadata["latest_version_number"] += 1
            version_name = f"v{self.metadata['latest_version_number']}"
            
        timestamp = datetime.now().isoformat()
        model_hash = self._calculate_model_hash(model_path)
        
        # Create version directory
        version_dir = self.versions_dir / version_name
        if version_dir.exists():
            raise ValueError(f"Version {version_name} already exists")
            
        version_dir.mkdir(parents=True)
        
        # Copy model files
        for file_path in model_path.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, version_dir / file_path.name)
                
        # Create version metadata
        version_metadata = {
            "version_name": version_name,
            "created_at": timestamp,
            "description": description,
            "model_hash": model_hash,
            "metrics": metrics or {},
            "training_config": training_config or {},
            "source_path": str(model_path),
            "size_mb": sum(f.stat().st_size for f in version_dir.glob("*") if f.is_file()) / (1024 * 1024)
        }
        
        # Save version-specific metadata
        with open(version_dir / "version_info.json", 'w') as f:
            json.dump(version_metadata, f, indent=2)
            
        # Update global metadata
        self.metadata["versions"][version_name] = version_metadata
        self.metadata["current_version"] = version_name
        self._save_metadata()
        
        logger.info(f"Created model version {version_name} at {version_dir}")
        return version_name
        
    def list_versions(self) -> List[Dict]:
        """List all available versions."""
        versions = []
        for version_name, metadata in self.metadata["versions"].items():
            version_info = metadata.copy()
            version_path = self.versions_dir / version_name
            version_info["exists"] = version_path.exists()
            version_info["path"] = str(version_path)
            versions.append(version_info)
            
        # Sort by creation date
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions
        
    def get_version_info(self, version_name: str) -> Optional[Dict]:
        """Get detailed information about a specific version."""
        if version_name not in self.metadata["versions"]:
            return None
            
        version_info = self.metadata["versions"][version_name].copy()
        version_path = self.versions_dir / version_name
        
        # Add current status
        version_info["exists"] = version_path.exists()
        version_info["path"] = str(version_path)
        version_info["is_current"] = self.metadata["current_version"] == version_name
        
        # Verify integrity
        if version_path.exists():
            current_hash = self._calculate_model_hash(version_path)
            version_info["integrity_ok"] = current_hash == version_info["model_hash"]
        else:
            version_info["integrity_ok"] = False
            
        return version_info
        
    def set_current_version(self, version_name: str) -> bool:
        """Set a version as the current active version."""
        if version_name not in self.metadata["versions"]:
            logger.error(f"Version {version_name} does not exist")
            return False
            
        version_path = self.versions_dir / version_name
        if not version_path.exists():
            logger.error(f"Version {version_name} files do not exist")
            return False
            
        # Create backup of current model if it exists
        current_model_path = self.base_dir / "current"
        if current_model_path.exists():
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backups_dir / backup_name
            shutil.copytree(current_model_path, backup_path)
            logger.info(f"Backed up current model to {backup_path}")
            
        # Copy version to current location
        if current_model_path.exists():
            shutil.rmtree(current_model_path)
        shutil.copytree(version_path, current_model_path)
        
        # Update metadata
        self.metadata["current_version"] = version_name
        self._save_metadata()
        
        logger.info(f"Set version {version_name} as current")
        return True
        
    def rollback_to_version(self, version_name: str) -> bool:
        """Rollback to a previous version."""
        logger.info(f"Rolling back to version {version_name}")
        return self.set_current_version(version_name)
        
    def delete_version(self, version_name: str, force: bool = False) -> bool:
        """Delete a model version."""
        if version_name not in self.metadata["versions"]:
            logger.error(f"Version {version_name} does not exist")
            return False
            
        if self.metadata["current_version"] == version_name and not force:
            logger.error("Cannot delete current version without force=True")
            return False
            
        version_path = self.versions_dir / version_name
        if version_path.exists():
            shutil.rmtree(version_path)
            
        # Remove from metadata
        del self.metadata["versions"][version_name]
        
        # Update current version if we deleted it
        if self.metadata["current_version"] == version_name:
            remaining_versions = list(self.metadata["versions"].keys())
            self.metadata["current_version"] = remaining_versions[0] if remaining_versions else None
            
        self._save_metadata()
        
        logger.info(f"Deleted version {version_name}")
        return True
        
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two model versions."""
        info1 = self.get_version_info(version1)
        info2 = self.get_version_info(version2)
        
        if not info1 or not info2:
            raise ValueError("One or both versions do not exist")
            
        comparison = {
            "version1": version1,
            "version2": version2,
            "created_at_diff": info2["created_at"] > info1["created_at"],
            "size_diff_mb": info2["size_mb"] - info1["size_mb"],
            "metrics_comparison": {}
        }
        
        # Compare metrics if available
        metrics1 = info1.get("metrics", {})
        metrics2 = info2.get("metrics", {})
        
        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            if metric in metrics1 and metric in metrics2:
                comparison["metrics_comparison"][metric] = {
                    "version1": metrics1[metric],
                    "version2": metrics2[metric],
                    "improvement": metrics2[metric] - metrics1[metric]
                }
                
        return comparison
        
    def cleanup_old_versions(self, keep_count: int = 5) -> int:
        """Remove old versions, keeping only the most recent ones."""
        versions = self.list_versions()
        
        if len(versions) <= keep_count:
            logger.info(f"Only {len(versions)} versions exist, no cleanup needed")
            return 0
            
        # Sort by creation date and keep the most recent
        versions_to_delete = versions[keep_count:]
        deleted_count = 0
        
        for version in versions_to_delete:
            version_name = version["version_name"]
            if version_name != self.metadata["current_version"]:
                if self.delete_version(version_name):
                    deleted_count += 1
                    
        logger.info(f"Cleaned up {deleted_count} old versions")
        return deleted_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Model version management")
    parser.add_argument("--model-dir", required=True, help="Base model directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create version
    create_parser = subparsers.add_parser("create", help="Create new version")
    create_parser.add_argument("--model-path", required=True, help="Path to model to version")
    create_parser.add_argument("--name", help="Version name (auto-generated if not provided)")
    create_parser.add_argument("--description", default="", help="Version description")
    
    # List versions
    subparsers.add_parser("list", help="List all versions")
    
    # Set current
    current_parser = subparsers.add_parser("set-current", help="Set current version")
    current_parser.add_argument("version", help="Version name to set as current")
    
    # Rollback
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to version")
    rollback_parser.add_argument("version", help="Version name to rollback to")
    
    # Delete version
    delete_parser = subparsers.add_parser("delete", help="Delete version")
    delete_parser.add_argument("version", help="Version name to delete")
    delete_parser.add_argument("--force", action="store_true", help="Force delete current version")
    
    # Compare versions
    compare_parser = subparsers.add_parser("compare", help="Compare two versions")
    compare_parser.add_argument("version1", help="First version")
    compare_parser.add_argument("version2", help="Second version")
    
    # Cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old versions")
    cleanup_parser.add_argument("--keep", type=int, default=5, help="Number of versions to keep")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    manager = ModelVersionManager(args.model_dir)
    
    if args.command == "create":
        version_name = manager.create_version(
            args.model_path, 
            args.name, 
            args.description
        )
        print(f"Created version: {version_name}")
        
    elif args.command == "list":
        versions = manager.list_versions()
        if not versions:
            print("No versions found")
        else:
            print(f"{'Version':<15} {'Created':<20} {'Size (MB)':<10} {'Current':<8} {'Description'}")
            print("-" * 80)
            for v in versions:
                current_marker = "✓" if v.get("is_current", False) else ""
                print(f"{v['version_name']:<15} {v['created_at'][:19]:<20} {v['size_mb']:<10.1f} {current_marker:<8} {v['description']}")
                
    elif args.command == "set-current":
        if manager.set_current_version(args.version):
            print(f"Set {args.version} as current version")
        else:
            print(f"Failed to set {args.version} as current")
            
    elif args.command == "rollback":
        if manager.rollback_to_version(args.version):
            print(f"Rolled back to version {args.version}")
        else:
            print(f"Failed to rollback to {args.version}")
            
    elif args.command == "delete":
        if manager.delete_version(args.version, args.force):
            print(f"Deleted version {args.version}")
        else:
            print(f"Failed to delete version {args.version}")
            
    elif args.command == "compare":
        comparison = manager.compare_versions(args.version1, args.version2)
        print(f"Comparison: {args.version1} vs {args.version2}")
        print(f"Size difference: {comparison['size_diff_mb']:.1f} MB")
        if comparison["metrics_comparison"]:
            print("Metrics comparison:")
            for metric, data in comparison["metrics_comparison"].items():
                print(f"  {metric}: {data['version1']} → {data['version2']} (Δ{data['improvement']:.4f})")
                
    elif args.command == "cleanup":
        deleted = manager.cleanup_old_versions(args.keep)
        print(f"Cleaned up {deleted} old versions")


if __name__ == "__main__":
    main()