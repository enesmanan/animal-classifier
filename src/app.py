import os
import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.model import EfficientNetModel, CNNModel

class AnimalClassifierApp:
    def __init__(self):
        """Initialize the application."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["bird", "cat", "dog", "horse"]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load models
        self.models = self.load_models()
        if not self.models:
            print("Warning: No models found in checkpoints directory!")

    def load_models(self):
        """Load both trained models."""
        models = {}
        
        # Load EfficientNet
        try:
            efficientnet = EfficientNetModel(num_classes=len(self.labels))
            efficientnet_path = os.path.join("checkpoints", "efficientnet", "efficientnet_best_model.pth")
            if os.path.exists(efficientnet_path):
                checkpoint = torch.load(efficientnet_path, map_location=self.device, weights_only=True)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                efficientnet.load_state_dict(state_dict, strict=False)
                efficientnet.eval()
                models['EfficientNet'] = efficientnet
                print("Successfully loaded EfficientNet model")
        except Exception as e:
            print(f"Error loading EfficientNet model: {str(e)}")
        
        # Load CNN
        try:
            cnn = CNNModel(num_classes=len(self.labels))
            cnn_path = os.path.join("checkpoints", "cnn", "cnn_best_model.pth")
            if os.path.exists(cnn_path):
                checkpoint = torch.load(cnn_path, map_location=self.device, weights_only=True)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                cnn.load_state_dict(state_dict, strict=False)
                cnn.eval()
                models['CNN'] = cnn
                print("Successfully loaded CNN model")
        except Exception as e:
            print(f"Error loading CNN model: {str(e)}")
        
        return models

    def predict(self, image: Image.Image):
        """Make predictions with both models and create comparison visualizations."""
        if not self.models:
            return "No trained models found. Please train the models first."
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions from both models
        results = {}
        probabilities = {}
        for model_name, model in self.models.items():
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
                probabilities[model_name] = probs
                
                # Get top prediction
                pred_idx = np.argmax(probs)
                pred_label = self.labels[pred_idx]
                pred_prob = probs[pred_idx]
                results[model_name] = (pred_label, pred_prob)
        
        # Create comparison plot
        fig = plt.figure(figsize=(12, 5))
        
        # Plot for EfficientNet
        if 'EfficientNet' in probabilities:
            plt.subplot(1, 2, 1)
            plt.bar(self.labels, probabilities['EfficientNet'], color='skyblue')
            plt.title('EfficientNet Predictions')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.ylabel('Probability')
        
        # Plot for CNN
        if 'CNN' in probabilities:
            plt.subplot(1, 2, 2)
            plt.bar(self.labels, probabilities['CNN'], color='lightcoral')
            plt.title('CNN Predictions')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.ylabel('Probability')
        
        plt.tight_layout()
        
        # Create results text
        text_results = "Model Predictions:\n\n"
        for model_name, (label, prob) in results.items():
            text_results += f"{model_name}:\n"
            text_results += f"Top prediction: {label} ({prob:.2%})\n"
            text_results += "All probabilities:\n"
            for label, prob in zip(self.labels, probabilities[model_name]):
                text_results += f"  {label}: {prob:.2%}\n"
            text_results += "\n"
        
        return [
            fig,           # Probability plots
            text_results   # Detailed text results
        ]

    def create_interface(self):
        """Create Gradio interface."""
        return gr.Interface(
            fn=self.predict,
            inputs=gr.Image(type="pil"),
            outputs=[
                gr.Plot(label="Prediction Probabilities"),
                gr.Textbox(label="Detailed Results", lines=10)
            ],
            title="Animal Classifier - Model Comparison",
            description=(
                "Upload an image of one of these animals: Bird, Cat, Dog, or Horse.\n"
                "The app will compare predictions from both EfficientNet and CNN models.\n\n"
                "Note: For best results, ensure the animal is clearly visible in the image."
            )
        )

def main():
    """Run the web application."""
    app = AnimalClassifierApp()
    interface = app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()