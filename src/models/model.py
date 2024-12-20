from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BaseModel(nn.Module):
    """Base model class for animal classification."""
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        with torch.no_grad():
            logits = self(x)
            return F.softmax(logits, dim=1)

    @classmethod
    def load_from_checkpoint(
        cls,
        path: str,
        map_location: Any = None
    ) -> 'BaseModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(num_classes=checkpoint['config']['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_checkpoint(
        self,
        path: str,
        extra_data: Dict[str, Any] = None
    ) -> None:
        """Save model checkpoint."""
        data = {
            'model_state_dict': self.state_dict(),
            'config': {
                'num_classes': self.get_num_classes(),
                'model_type': self.__class__.__name__
            }
        }
        
        if extra_data:
            if 'config' in extra_data:
                data['config'].update(extra_data['config'])
                del extra_data['config']
            data.update(extra_data)
            
        torch.save(data, path)

    def get_num_classes(self) -> int:
        """Get number of output classes."""
        raise NotImplementedError


class CNNModel(BaseModel):    
    def __init__(self, num_classes: int, input_size: int = 224):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First block: 32 filters
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second block: 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third block: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return self.classifier(x)

    def get_num_classes(self) -> int:
        return self.classifier[-1].out_features

class EfficientNetModel(BaseModel):
    """EfficientNet-based model for animal classification."""
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True
    ):
        super(EfficientNetModel, self).__init__()
        
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.base_model(dummy_input)
            feature_dim = features.shape[1]

        # Simpler classifier structure matching the saved model
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)

    def get_num_classes(self) -> int:
        return self.classifier[-1].out_features

def get_model(model_type: str, num_classes: int, **kwargs) -> BaseModel:
    """Factory function to get model by type."""
    models = {
        'cnn': CNNModel,
        'efficientnet': EfficientNetModel
    }
    
    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported. Available models: {list(models.keys())}")
        
    return models[model_type](num_classes=num_classes, **kwargs)