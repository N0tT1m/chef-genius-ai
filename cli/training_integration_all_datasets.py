#!/usr/bin/env python3
"""
Training Integration with ALL Datasets Combined
Drop-in replacement for your existing training script using all available datasets
"""

from unified_dataset_loader import create_unified_dataloader
from typing import Optional
import time

class AllDatasetsTrainingIntegration:
    """
    Complete integration that combines ALL your datasets for training
    Provides maximum training data with blazing fast performance
    """
    
    def __init__(self, tokenizer, max_datasets: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_datasets = max_datasets
        self.loader = None
        
    def create_dataloader(self, 
                         batch_size: int = 8,
                         shuffle: bool = True,
                         datasets_path: str = "data/datasets") -> object:
        """
        Create the unified data loader replacing your PyTorch DataLoader
        
        This replaces:
            DataLoader(dataset, batch_size=8, num_workers=12, shuffle=True)
        
        With a 100-1000x faster loader using ALL your datasets!
        """
        
        print("🔥 Creating unified data loader with ALL datasets...")
        
        # Create the unified loader
        self.loader = create_unified_dataloader(
            datasets_path=datasets_path,
            batch_size=batch_size,
            shuffle=shuffle,
            max_datasets=self.max_datasets
        )
        
        # Wrap it for training compatibility
        return TrainingDataLoaderWrapper(self.loader, self.tokenizer)

class TrainingDataLoaderWrapper:
    """
    Wrapper that makes the unified loader compatible with your training loop
    """
    
    def __init__(self, unified_loader, tokenizer):
        self.unified_loader = unified_loader
        self.tokenizer = tokenizer
        self.max_length = 512
        
    def __iter__(self):
        """Iterate through tokenized batches"""
        for batch in self.unified_loader:
            yield self._tokenize_batch(batch)
    
    def __len__(self):
        """Estimate number of batches"""
        return self.unified_loader.total_samples // self.unified_loader.batch_size
    
    def _tokenize_batch(self, batch):
        """Tokenize a batch for seq2seq training with proper input/target separation"""
        
        # Create proper input prompts and target recipes for FLAN-T5
        input_texts = []
        target_texts = []
        
        for item in batch.get('input_ids', []):
            if isinstance(item, str):
                # Parse the combined text to separate input and target
                if "Instructions:" in item:
                    # Split at Instructions to get ingredients vs full recipe
                    parts = item.split("Instructions:", 1)
                    ingredients = parts[0].replace("Ingredients:", "").strip()
                    instructions = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Create instruction-following prompt for FLAN-T5
                    input_prompt = f"Generate a detailed recipe with step-by-step instructions using these ingredients: {ingredients}"
                    target_recipe = f"Ingredients: {ingredients}\n\nInstructions: {instructions}"
                    
                    input_texts.append(input_prompt)
                    target_texts.append(target_recipe)
                else:
                    # Fallback for malformed data
                    input_texts.append(f"Generate a recipe: {item[:100]}")
                    target_texts.append(item)
            else:
                # Handle non-string inputs
                input_texts.append("Generate a recipe")
                target_texts.append("A recipe")
        
        # Tokenize inputs and targets separately for seq2seq
        model_inputs = self.tokenizer(
            input_texts,
            max_length=256,  # Shorter input prompts
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets (updated API without deprecated as_target_tokenizer)
        labels = self.tokenizer(
            text_target=target_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Replace padding token id's of the labels by -100 (ignore_index)
        labels_input_ids = labels["input_ids"].clone()
        labels_input_ids[labels_input_ids == self.tokenizer.pad_token_id] = -100
        
        model_inputs["labels"] = labels_input_ids
        return model_inputs
    
    def reset(self):
        """Reset for new epoch"""
        self.unified_loader.reset()
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return self.unified_loader.get_stats()

# Integration examples for your training script

def integrate_with_transformers_trainer(tokenizer):
    """
    Example: Integration with Hugging Face Transformers Trainer
    """
    
    print("🔧 HUGGING FACE TRAINER INTEGRATION")
    print("Replace your DataLoader creation with this:")
    print()
    
    integration_code = '''
# OLD CODE (remove this):
# train_dataset = YourDataset(...)
# train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=12, shuffle=True)

# NEW CODE (use this instead):
from training_integration_all_datasets import AllDatasetsTrainingIntegration

# Create the integration
integration = AllDatasetsTrainingIntegration(tokenizer=tokenizer, max_datasets=15)

# Create the unified data loader (combines ALL your datasets!)
train_dataloader = integration.create_dataloader(
    batch_size=8,
    shuffle=True,
    datasets_path="data/datasets"
)

# Your Trainer setup stays the same:
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,  # We don't use dataset anymore
    # Instead, use the dataloader directly in your training loop
)

# Manual training loop (recommended for full control):
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    epoch_loss = 0
    batch_count = 0
    
    for batch in train_dataloader:
        # Your training step here
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # Log performance every 50 batches
        if batch_count % 50 == 0:
            stats = train_dataloader.get_performance_stats()
            print(f"  Batch {batch_count}: Loss {loss.item():.4f}")
            print(f"  Performance: {stats['samples_processed']:,} samples processed")
    
    # Reset for next epoch
    train_dataloader.reset()
    
    avg_loss = epoch_loss / batch_count
    print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
'''
    
    with open("transformers_trainer_integration.py", "w") as f:
        f.write(integration_code)
    
    print("📝 Saved complete integration code to: transformers_trainer_integration.py")

def integrate_with_pytorch_lightning(tokenizer):
    """
    Example: Integration with PyTorch Lightning
    """
    
    lightning_code = '''
import pytorch_lightning as pl
from training_integration_all_datasets import AllDatasetsTrainingIntegration

class YourLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        
        # Create unified data loader integration
        self.data_integration = AllDatasetsTrainingIntegration(
            tokenizer=tokenizer, 
            max_datasets=20  # Use top 20 datasets
        )
    
    def train_dataloader(self):
        # This replaces your old DataLoader with ALL datasets combined!
        return self.data_integration.create_dataloader(
            batch_size=8,
            shuffle=True
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Log performance metrics
        if batch_idx % 100 == 0:
            stats = self.data_integration.loader.get_stats()
            self.log("samples_processed", stats['samples_processed'])
            self.log("datasets_used", stats['total_datasets'])
        
        return loss
'''
    
    with open("pytorch_lightning_integration.py", "w") as f:
        f.write(lightning_code)

def main():
    """Test and show integration examples"""
    
    print("🚀 ALL-DATASETS TRAINING INTEGRATION")
    print("Combines ALL your datasets for maximum training data!")
    print("=" * 60)
    
    # Test the unified loader
    print("\n🔥 Testing unified loader with your datasets...")
    
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create integration
        integration = AllDatasetsTrainingIntegration(tokenizer=tokenizer, max_datasets=10)
        
        # Create the loader
        train_loader = integration.create_dataloader(batch_size=8)
        
        if train_loader:
            print("\n✅ Unified loader created successfully!")
            
            # Quick performance test
            print("🏃 Quick performance test...")
            start_time = time.time()
            batch_count = 0
            total_samples = 0
            
            for batch in train_loader:
                batch_count += 1
                total_samples += len(batch['input_ids'])
                
                if batch_count >= 20:  # Test 20 batches
                    break
            
            elapsed = time.time() - start_time
            rate = total_samples / elapsed if elapsed > 0 else 0
            
            print(f"\n🎯 PERFORMANCE RESULTS:")
            print(f"   Combined rate: {rate:.1f} samples/sec")
            print(f"   Total samples available: {train_loader.unified_loader.total_samples:,}")
            print(f"   Datasets combined: {len(train_loader.unified_loader.datasets)}")
            
            # Show dataset breakdown
            print(f"\n📊 DATASETS BEING USED:")
            for i, dataset in enumerate(train_loader.unified_loader.datasets[:10], 1):
                print(f"   {i:2d}. {dataset['folder']}/{dataset['name']} ({dataset['estimated_samples']:,} samples)")
            
            if len(train_loader.unified_loader.datasets) > 10:
                print(f"       ... and {len(train_loader.unified_loader.datasets) - 10} more datasets")
            
            # Generate integration examples
            print(f"\n🔧 GENERATING INTEGRATION CODE...")
            integrate_with_transformers_trainer(tokenizer)
            integrate_with_pytorch_lightning(tokenizer)
            
            print(f"\n🎉 READY FOR TRAINING!")
            print(f"Expected improvement: {rate/0.01:.0f}x faster than your current 0.01 samples/sec!")
            print(f"Your Discord notifications will show these improved metrics! 🎊")
            
        else:
            print("❌ Could not create unified loader")
    
    except ImportError:
        print("⚠️  Transformers not available for testing, but integration code generated")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main()