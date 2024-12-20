import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.model import EfficientNetModel, CNNModel
from data.dataset import get_data_loaders
from data.data_processor import prepare_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Monitor validation accuracy
            factor=0.1,
            patience=3,
            verbose=True
        )
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize best metrics
        self.best_val_acc = 0.0

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, targets in loader:
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()
            
        accuracy = correct / total
        avg_loss = total_loss / len(loader)
        
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, val_acc, model_name, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / f'{model_name}_best_model.pth'
            logger.info(f"Saving best {model_name} model with validation accuracy: {val_acc:.2%}")
        else:
            path = self.checkpoint_dir / f'{model_name}_checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)

    def train(self, model_name):
        """Main training loop."""
        logger.info(f"Starting training {model_name} for {self.config['epochs']} epochs...")
        
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        for epoch in range(self.config['epochs']):
            logger.info(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.evaluate(self.val_loader)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            # Update scheduler
            self.scheduler.step(val_acc)
            
            # Print metrics
            logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2%}"
            )
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch + 1, val_acc, model_name, is_best)
        
        # Final evaluation
        test_loss, test_acc = self.evaluate(self.test_loader, "test")
        logger.info(
            f"\nFinal {model_name} Test Results:"
            f"\nTest Loss: {test_loss:.4f}"
            f"\nTest Accuracy: {test_acc:.2%}"
        )
        
        return {
            'metrics': metrics,
            'test_loss': test_loss,
            'test_acc': test_acc
        }


def train_model(model_name, model, train_loader, val_loader, test_loader, config):
    """Train a specific model."""
    logger.info(f"Training {model_name}...")
    
    # Update save directory for this model
    model_config = config.copy()
    model_config['save_dir'] = os.path.join(config['save_dir'], model_name)
    
    # trainer sınıfı
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=model_config
    )
    
    # Train and return results
    return trainer.train(model_name)


def main():
    # First, prepare the dataset
    logger.info("Preparing dataset...")
    prepare_dataset()
    
    # Training configuration
    base_config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'epochs': 10,
        'num_classes': 4,
        'save_dir': 'checkpoints'
    }
    
    # Get data loaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=base_config['batch_size']
    )
    
    # Train EfficientNet
    logger.info("Creating EfficientNet model...")
    efficientnet = EfficientNetModel(
        num_classes=base_config['num_classes'],
        model_name='efficientnet_b0',
        pretrained=True
    )
    efficientnet_results = train_model(
        'efficientnet',
        efficientnet,
        train_loader,
        val_loader,
        test_loader,
        base_config
    )
    
    # Train CNN
    logger.info("Creating CNN model...")
    cnn = CNNModel(num_classes=base_config['num_classes'])
    cnn_results = train_model(
        'cnn',
        cnn,
        train_loader,
        val_loader,
        test_loader,
        base_config
    )
    
    # Print final comparison
    logger.info("\nFinal Model Comparison:")
    logger.info(f"EfficientNet Test Accuracy: {efficientnet_results['test_acc']:.2%}")
    logger.info(f"CNN Test Accuracy: {cnn_results['test_acc']:.2%}")


if __name__ == "__main__":
    main()