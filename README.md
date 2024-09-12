
# Level1 Sketch Image Classification Project

This project implements an image classification model using PyTorch and timm.

## Project Structure

- `data/`: Contains train and test data
- `src/`: Source code for the project
  - `dataset.py`: CustomDataset class
  - `transforms.py`: Data transformations
  - `models.py`: Model architecture
  - `loss.py`: Loss function
  - `trainer.py`: Training loop
- `utils/`: Utility functions
- `configs/`: Configuration files
- `scripts/`: 
  - `train.py`: Script to train the model
  - `inference.py`: Script to run inference

## Usage

1. Prepare your data in the `data/` directory.
2. Adjust the configuration in `configs/config.json` if needed.
3. Run training:
   ```
   python scripts/train.py
   ```
4. Run inference:
   ```
   python scripts/inference.py
   ```

## Requirements

- pandas==2.1.4
- matplotlib==3.8.4
- seaborn==0.13.2
- Pillow==10.3.0
- numpy==1.26.3
- timm==0.9.16
- albumentations==1.4.4
- tqdm==4.66.1
- scikit-learn==1.4.2
- opencv-python==4.9.0.80

