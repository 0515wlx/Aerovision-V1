#!/usr/bin/env python3
"""
Example training script with DVC integration for AeroVision project.

This script demonstrates how to:
1. Load data managed by DVC
2. Train a simple model
3. Save the model with DVC tracking
4. Log metrics for DVC
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DVCTrainingPipeline:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "dvc-data"
        self.metrics_file = self.project_root / "metrics.json"
        
    def load_data(self, data_path):
        """Load training data (placeholder implementation)"""
        logger.info(f"Loading data from {data_path}")
        # In real implementation, this would load actual data
        # For now, we'll simulate loading data
        return {"samples": 1000, "features": 128}
    
    def train_model(self, data, model_config):
        """Train model (placeholder implementation)"""
        logger.info("Training model...")
        # Simulate training process
        import time
        time.sleep(2)  # Simulate training time
        
        # Simulate model metrics
        metrics = {
            "accuracy": 0.95,
            "loss": 0.05,
            "training_time": 2.0,
            "timestamp": datetime.now().isoformat()
        }
        
        return {"model_id": "model_001"}, metrics
    
    def save_model(self, model, model_path):
        """Save trained model"""
        logger.info(f"Saving model to {model_path}")
        # In real implementation, this would save the actual model
        # For now, we'll create a placeholder file
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'w') as f:
            f.write(f"# Trained model: {model['model_id']}\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write("# This file is managed by DVC\n")
        
        return model_path
    
    def save_metrics(self, metrics):
        """Save training metrics"""
        logger.info(f"Saving metrics to {self.metrics_file}")
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return self.metrics_file
    
    def run_training(self, data_file, model_name, model_config=None):
        """Run complete training pipeline"""
        logger.info(f"Starting training pipeline for {model_name}")
        
        # Default model config
        if model_config is None:
            model_config = {
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32
            }
        
        # Load data
        data_path = self.data_dir / "training_data" / data_file
        data = self.load_data(data_path)
        
        # Train model
        model, metrics = self.train_model(data, model_config)
        
        # Add model config to metrics
        metrics["model_config"] = model_config
        metrics["data_file"] = str(data_path)
        
        # Save model
        model_path = self.models_dir / f"{model_name}.pt"
        self.save_model(model, model_path)
        
        # Save metrics
        self.save_metrics(metrics)
        
        logger.info("Training pipeline completed!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Metrics saved to: {self.metrics_file}")
        logger.info(f"Final accuracy: {metrics['accuracy']:.3f}")
        
        return model_path, metrics
    
    def add_to_dvc(self, file_path):
        """Add file to DVC tracking"""
        import subprocess
        
        logger.info(f"Adding {file_path} to DVC tracking...")
        try:
            # Add to DVC
            result = subprocess.run(
                ["dvc", "add", str(file_path)],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("File added to DVC tracking")
            
            # Stage DVC file for git
            dvc_file = f"{file_path}.dvc"
            subprocess.run(["git", "add", dvc_file], check=True)
            logger.info(f"DVC file staged for git: {dvc_file}")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add file to DVC: {e}")
            return False
    
    def push_to_remote(self):
        """Push DVC tracked files to remote"""
        import subprocess
        
        logger.info("Pushing DVC tracked files to remote...")
        try:
            result = subprocess.run(
                ["dvc", "push"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("DVC push completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC push failed: {e}")
            return False


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model with DVC integration")
    parser.add_argument("--data", default="processed_data.pkl", help="Training data file")
    parser.add_argument("--model-name", default="aircraft_classifier", help="Model name")
    parser.add_argument("--config", help="Model config JSON file")
    parser.add_argument("--no-dvc", action="store_true", help="Skip DVC operations")
    parser.add_argument("--push", action="store_true", help="Push to DVC remote after training")
    
    args = parser.parse_args()
    
    # Load model config if provided
    model_config = None
    if args.config:
        with open(args.config, 'r') as f:
            model_config = json.load(f)
    
    # Initialize pipeline
    pipeline = DVCTrainingPipeline()
    
    # Run training
    model_path, metrics = pipeline.run_training(
        data_file=args.data,
        model_name=args.model_name,
        model_config=model_config
    )
    
    # Add model to DVC if not skipped
    if not args.no_dvc:
        success = pipeline.add_to_dvc(model_path)
        if success and args.push:
            pipeline.push_to_remote()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Loss: {metrics['loss']:.3f}")
    print(f"Training Time: {metrics['training_time']:.1f}s")
    print(f"Model Path: {model_path}")
    print(f"Metrics File: {pipeline.metrics_file}")
    print("="*50)


if __name__ == "__main__":
    main()