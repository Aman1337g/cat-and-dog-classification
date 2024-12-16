# Cat and Dog Classification

This project focuses on building a Convolutional Neural Network (CNN) model to classify images of cats and dogs using TensorFlow and Keras. The dataset for this project is obtained from Kaggle.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Overview
The goal of this project is to create a model capable of distinguishing between images of cats and dogs. The project employs:
- TensorFlow/Keras for deep learning
- Kaggle's dataset for training and validation
- Google Colab with GPU acceleration for efficient computation

---

## Features
- Efficient dataset download and processing with Kaggle CLI
- Data preprocessing using Keras utilities
- Implementation of a CNN with multiple convolutional and fully connected layers
- Use of image generators for memory-efficient data handling
- GPU-accelerated training using Google Colab

---

## Setup

### Prerequisites
1. Install Python 3.7+
2. Install the required libraries:
   ```bash
   pip install tensorflow keras
   ```
3. Access to Google Colab (recommended for GPU acceleration)

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Aman1337g/cat-and-dog-classification.git
   cd "cat-and-dog-classification"
   ```
2. Open the Jupyter Notebook (`cat_v_dog_classification.ipynb`) in Google Colab.
3. Follow the steps outlined in the notebook to download the dataset and train the model.

---

## Dataset

### Downloading the Dataset
1. Log in to your Kaggle account.
2. Go to your account settings and generate an API token (`kaggle.json`).
3. Upload the `kaggle.json` file to Colab:
   ```bash
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   ```
4. Download the dataset:
   ```bash
   !kaggle datasets download salader/dogs-vs-cats
   ```
5. Extract the dataset:
   ```python
   import zipfile
   zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip', 'r')
   zip_ref.extractall('/content')
   zip_ref.close()
   ```

### Dataset Structure
- **Training Data**: `/content/train`
- **Test Data**: `/content/test`

---

## Model

### CNN Architecture
1. **Convolutional Layers**:
   - Layer 1: 32 filters
   - Layer 2: 64 filters
   - Layer 3: 128 filters
2. **Fully Connected Layers**:
   - 3 layers with varying neurons

### Normalization
The images are normalized to ensure pixel values are scaled to the range [0, 1]:
```python
# Normalize
def process(image, label):
  image = tf.cast(image/255, tf.float32)
  return image, label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
```

### Image Generators
The data is divided into batches for efficient training using Keras image generators:
```python
# generators
train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels = 'inferred',
    label_mode = 'int',  # cats will be assigned 0 and dogs 1
    batch_size = 32,
    image_size = (256, 256)  # as model expect the images to be of same size
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels = 'inferred',
    label_mode = 'int',  # cats will be assigned 0 and dogs 1
    batch_size = 32,
    image_size = (256, 256)  # as model expect the images to be of same size
)
```

---

## Results
After training the CNN model, you can evaluate its performance using metrics like accuracy on the test dataset. Further improvements can be achieved through hyperparameter tuning and data augmentation.

---

## Acknowledgments
- Kaggle for providing the dataset
- TensorFlow and Keras for the deep learning framework
- Google Colab for free GPU resources

---

Feel free to explore the project and contribute! If you encounter any issues, please raise an issue in the repository.

---

### License
This project is licensed under the MIT License.
