# Animal Classifier

A deep learning project that compares EfficientNet and CNN models for animal image classification. The project includes data preprocessing, model training, and a web interface for real-time predictions.

## Live Demo
[https://huggingface.co/spaces/enesmanan/dl-animal-classifier](https://huggingface.co/spaces/enesmanan/dl-animal-classifier)

## Project Overview

### Dataset
The project uses [Train Your First Classifier Dataset](https://huggingface.co/datasets/mertcobanov/train-your-first-classifier/blob/main/raw_images.zip) from Hugging Face Hub:

- 4 classes: Bird, Cat, Dog, Horse
- ~500 training images
  - Bird: 135 images
  - Cat: 116 images
  - Dog: 117 images
  - Horse: 151 images
- ~130 test images
  - Bird: 34 images
  - Cat: 29 images
  - Dog: 30 images
  - Horse: 38 images

#### Data Processing Pipeline
1. Download raw images from Hugging Face Hub
2. Apply smart cropping to focus on the animal
3. Resize to 224x224 pixels
4. Apply data normalization
5. Split into train/validation/test sets

The dataset is automatically downloaded and processed during training using the Hugging Face datasets library. The small dataset size particularly affects the CNN model's performance, while the pretrained EfficientNet model performs well due to transfer learning.


### Model Architecture and Performance

#### EfficientNet Model (98.95% Test Accuracy)
- Pretrained EfficientNet-B0 backbone
- Transfer learning approach
- Features:
  - Pretrained on ImageNet (1.2M images)
  - Compound scaling method
  - Efficient architecture design
  - Simple classifier head with dropout

#### Custom CNN Model (51.22% Test Accuracy)
- Lightweight CNN architecture
- Trained from scratch
- Features:
  - 3 convolutional blocks
  - Batch normalization
  - Global average pooling
  - Simple classifier head

#### Performance Analysis
The significant performance gap between models (47.73%) can be attributed to several factors:

1. **Limited Dataset Size**
   - Training dataset: ~500 images
   - Small dataset makes it challenging for CNN to learn robust features
   - EfficientNet benefits from pretrained weights

2. **Model Complexity**
   - EfficientNet: Complex, pretrained feature extractor
   - CNN: Simple architecture learning from scratch

3. **Transfer Learning Advantage**
   - EfficientNet utilizes knowledge from ImageNet
   - CNN needs to learn all features from limited data


### Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/enesmanan/animal-classifier.git
cd animal-classifier
```

2. Create virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Training
```bash
python src/train.py
```
This will:
- Download and process the dataset
- Train both models
- Save checkpoints and metrics

#### Web Interface
```bash
python src/app.py
```
This launches the Gradio interface for model comparison.

### Project Structure
```
animal-classifier/
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset handling
│   │   └── data_processor.py # Data preprocessing
│   ├── models/
│   │   └── model.py          # Model architectures
│   ├── utils/
│   │   └── image_utils.py    # Image processing utilities
│   ├── train.py              # Training script
│   └── app.py                # Gradio web interface
│
├── requirements.txt
└── README.md
```

### License
[MIT License](https://github.com/enesmanan/animal-classifier/blob/main/LICENSE) 

### Contact
- GitHub: [@enesmanan](https://github.com/enesmanan)
- Linkedin: [@enesfehmimanan](https://www.linkedin.com/in/enesfehmimanan/)
- Twitter: [@enesfehmimanan](https://x.com/enesfehmimanan)