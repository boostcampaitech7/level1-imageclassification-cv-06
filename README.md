 # Boostcamp AI Tech 7 CV 06
 
## Level1 Sketch Image Classification Competition
![image](https://github.com/user-attachments/assets/42997a1b-a4c3-4e5c-b67d-92b217dd277a)
### Description
The Sketch Image Classification competition focuses on developing a model using the provided dataset to classify the objects depicted in sketch images.

In computer vision, various forms of image data are utilized, and the accurate recognition and classification of unstructured data remain key challenges. While progress has been made based on general image data like photographs, sketch images, unlike everyday photos, are abstract and simplified, reflecting human imagination and conceptual understanding. Sketch data lacks color, texture, and fine details but instead focuses on basic shapes and structures. This highlights how sketches emphasize the essential features of objects.

The competition aims to develop models that can learn and recognize the fundamental shapes and structures of objects through sketch images, enhancing the ability to understand the differences from general image data and improving model development from a new perspective. Through this, participants can enhance creative approaches and processing abilities for complex and diverse real-world image data. Additionally, AI models using sketch data can be applied in various fields, such as digital art, game development, and educational content creation.

The original ImageNet Sketch dataset consists of 50,889 images, with approximately 50 images for each of the 1,000 object categories. It contains hand-drawn images of common objects, showcasing a variety of styles and features representing real-world objects.

In this competition, the provided dataset has been reviewed and refined from the original 1,000 classes, selecting the top 500 objects with the most images, resulting in a total of 25,035 images. This dataset is divided into 15,021 training images and 10,014 Private & Public evaluation images.

Developing a sketch image classification model will enhance the versatility of vision systems and build the capability to apply them in various real-world applications.

## Project Structure
This project implements an image classification model using PyTorch and timm.
- `data/`: Contains train and test data
- `eda/` : Code related to EDA
  - `augmentation_viewer.py`: Streamlit code for EDA  
(Augmentation visualization,  
GradCAM,  
Misclassification analysis)
  - `class_mapping.pkl`: PKL file for mapping class label onto text
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

