#!/usr/bin/env python3
"""
DVC Workflow Helper for AeroVision Project

This script provides convenient commands for managing DVC in the AeroVision project.
Usage:
    python dvc_workflow.py [command] [options]

Commands:
    status      - Show DVC status
    push        - Push DVC tracked files to remote
    pull        - Pull DVC tracked files from remote
    add         - Add files to DVC tracking
    commit      - Commit DVC changes
    init-model  - Initialize model directory with DVC
    init-data   - Initialize data directory with DVC
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path


class DVCWorkflow:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "dvc-data"
        
    def run_command(self, cmd, cwd=None):
        """Run a shell command and return the result"""
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=cwd or self.project_root,
                capture_output=True, 
                text=True,
                check=True
            )
            print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {cmd}")
            print(f"Error output: {e.stderr}")
            return None
    
    def status(self):
        """Show DVC status"""
        print("=== DVC Status ===")
        self.run_command("dvc status")
        
        print("\n=== Git Status ===")
        self.run_command("git status --porcelain")
        
        print("\n=== DVC Cache Info ===")
        self.run_command("dvc cache dir")
        
        print("\n=== DVC Remote Info ===")
        self.run_command("dvc remote list")
    
    def push(self):
        """Push DVC tracked files to remote"""
        print("Pushing DVC tracked files to remote...")
        result = self.run_command("dvc push")
        if result:
            print("✅ DVC push completed successfully")
        else:
            print("❌ DVC push failed")
    
    def pull(self):
        """Pull DVC tracked files from remote"""
        print("Pulling DVC tracked files from remote...")
        result = self.run_command("dvc pull")
        if result:
            print("✅ DVC pull completed successfully")
        else:
            print("❌ DVC pull failed")
    
    def add_model(self, model_path):
        """Add a model file to DVC tracking"""
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ Model file {model_path} does not exist")
            return
        
        print(f"Adding {model_path} to DVC tracking...")
        result = self.run_command(f"dvc add {model_path}")
        if result:
            print(f"✅ {model_path} added to DVC tracking")
            print(f"Remember to run: git add {model_path}.dvc")
        else:
            print(f"❌ Failed to add {model_path} to DVC tracking")
    
    def add_data(self, data_path):
        """Add a data file to DVC tracking"""
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"❌ Data file {data_path} does not exist")
            return
        
        print(f"Adding {data_path} to DVC tracking...")
        result = self.run_command(f"dvc add {data_path}")
        if result:
            print(f"✅ {data_path} added to DVC tracking")
            print(f"Remember to run: git add {data_path}.dvc")
        else:
            print(f"❌ Failed to add {data_path} to DVC tracking")
    
    def commit(self, message):
        """Commit DVC changes"""
        print("Committing DVC changes...")
        
        # First add all DVC files
        self.run_command("git add *.dvc")
        self.run_command("git add .dvc")
        
        # Then commit
        result = self.run_command(f'git commit -m "{message}"')
        if result:
            print("✅ DVC changes committed successfully")
        else:
            print("❌ Failed to commit DVC changes")
    
    def init_model_dir(self):
        """Initialize model directory with DVC"""
        print("Initializing model directory with DVC...")
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(exist_ok=True)
        
        # Create .gitkeeper file
        gitkeeper = self.models_dir / ".gitkeeper"
        gitkeeper.write_text("# This file keeps the models directory in git\n# Actual model files should be managed by DVC\n")
        
        print(f"✅ Model directory initialized at {self.models_dir}")
        print("Usage: python dvc_workflow.py add-model models/your_model.pt")
    
    def init_data_dir(self):
        """Initialize data directory with DVC"""
        print("Initializing data directory with DVC...")
        
        # Create data directories
        (self.data_dir / "training_data").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "outputs").mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Data directory initialized at {self.data_dir}")
        print("Usage: python dvc_workflow.py add-data dvc-data/your_data.file")
    
    def setup_remote(self, remote_path=None):
        """Setup DVC remote"""
        if remote_path:
            print(f"Setting up DVC remote at {remote_path}...")
            result = self.run_command(f"dvc remote add -d local {remote_path}")
        else:
            # Use default local remote
            remote_path = str(self.project_root / "dvc-remote")
            print(f"Setting up DVC remote at {remote_path}...")
            result = self.run_command(f"dvc remote add -d local {remote_path}")
        
        if result:
            print(f"✅ DVC remote setup completed at {remote_path}")
        else:
            print("❌ DVC remote setup failed")


def main():
    parser = argparse.ArgumentParser(description="DVC Workflow Helper for AeroVision")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Show DVC status")
    
    # Push command
    subparsers.add_parser("push", help="Push DVC tracked files to remote")
    
    # Pull command
    subparsers.add_parser("pull", help="Pull DVC tracked files from remote")
    
    # Add model command
    add_model_parser = subparsers.add_parser("add-model", help="Add model file to DVC tracking")
    add_model_parser.add_argument("model_path", help="Path to the model file")
    
    # Add data command
    add_data_parser = subparsers.add_parser("add-data", help="Add data file to DVC tracking")
    add_data_parser.add_argument("data_path", help="Path to the data file")
    
    # Commit command
    commit_parser = subparsers.add_parser("commit", help="Commit DVC changes")
    commit_parser.add_argument("message", help="Commit message")
    
    # Init commands
    subparsers.add_parser("init-model", help="Initialize model directory with DVC")
    subparsers.add_parser("init-data", help="Initialize data directory with DVC")
    
    # Setup remote command
    setup_remote_parser = subparsers.add_parser("setup-remote", help="Setup DVC remote")
    setup_remote_parser.add_argument("--path", help="Remote path (optional)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    workflow = DVCWorkflow()
    
    if args.command == "status":
        workflow.status()
    elif args.command == "push":
        workflow.push()
    elif args.command == "pull":
        workflow.pull()
    elif args.command == "add-model":
        workflow.add_model(args.model_path)
    elif args.command == "add-data":
        workflow.add_data(args.data_path)
    elif args.command == "commit":
        workflow.commit(args.message)
    elif args.command == "init-model":
        workflow.init_model_dir()
    elif args.command == "init-data":
        workflow.init_data_dir()
    elif args.command == "setup-remote":
        workflow.setup_remote(args.path)


if __name__ == "__main__":
    main()