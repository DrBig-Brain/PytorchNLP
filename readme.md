# Email Spam Detection with PyTorch LSTM

A deep learning model for email spam detection using PyTorch and LSTM (Long Short-Term Memory) networks.

## Project Overview

This project implements a text classification system to detect spam emails using natural language processing and deep learning. The model achieves approximately 97.29% accuracy on the test set.

## Features

- Text preprocessing and tokenization
- Custom vocabulary encoding
- LSTM-based neural network architecture
- PyTorch implementation
- Dataset handling with custom DataLoader

## Requirements

Install required packages using:

```bash
pip install -r requirements.txt
```

Dependencies:
```txt
torch>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
jupyter>=1.0.0
```

## Project Structure

```
PytorchNLP/
├── emails.csv          # Raw email dataset
├── processed.csv       # Preprocessed text data
├── encoder.pkl        # Saved vocabulary encoder
├── model.pth         # Trained model weights
├── preprocessing.ipynb # Data preprocessing notebook
├── model.ipynb       # Model training notebook
├── test.ipynb        # Model testing notebook
└── requirements.txt   # Project dependencies
```

## Model Architecture

- Embedding Layer
- LSTM Layer
- Fully Connected Layer
- Output Layer (2 classes: spam/not spam)

Key Parameters:
- Maximum sequence length: 50
- Batch size: 32
- Learning rate: 0.001
- Training epochs: 20

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```python
# Run preprocessing notebook to prepare the dataset
jupyter notebook preprocessing.ipynb
```

2. Training:
```python
# Train the model using model notebook
jupyter notebook model.ipynb
```

3. Testing:
```python
# Test the model on new data
jupyter notebook test.ipynb
```

## Model Performance

- Test Accuracy: 97.29%
- The model shows strong performance in distinguishing between spam and legitimate emails

## Saved Models

The trained model and encoder are saved as:
- `model.pth`: Contains the trained LSTM model weights
- `encoder.pkl`: Contains the text encoder for preprocessing new data

## License

This project is available for open use. Please provide appropriate attribution if you use or modify this code.

## Acknowledgments

- PyTorch documentation and tutorials
- Email spam classification dataset