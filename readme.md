# Song Genre Classification with CNN and KNN
This repository contains two approaches for classifying music genres: a Convolutional Neural Network (CNN) and a K-Nearest Neighbors (KNN) algorithm. Both models utilize audio features from music files to predict the genre of each track. The CNN approach leverages deep learning, while the KNN approach provides a non-parametric method for classification based on feature similarity.

## Directory Structure
- Data
  - ``` genres_original/ ```: Folder containing audio files organized by genre.
  - ``` features_30_sec.csv ```: Metadata file with filenames and corresponding genre labels.

- Notebooks
  - ``` cnn.ipynb ```: CNN-based genre classification.
  - ``` knn.ipynb ```: KNN-based genre classification.

## Requirements
Install the following dependencies before running the notebooks:

``` bash
pip install pandas numpy librosa matplotlib tqdm scikit-learn tensorflow scipy python_speech_features
```
## Methodology
1. Feature Extraction:

    - Both approaches utilize **MFCC** (Mel-frequency cepstral coefficients), a feature commonly used in audio processing, extracted using the librosa library.
    - The CNN model uses 40 MFCCs averaged over the audio sample, while the KNN model relies on MFCC features calculated with python_speech_features.

2. Model Architectures:
    - **CNN**:
      - The CNN model is designed to capture complex patterns in the MFCC features for high-accuracy genre classification.
    - **KNN**:
      - The KNN model classifies genres by measuring the distance between feature vectors, specifically using Mahalanobis distance.

3. Training and Evaluation:
    - **CNN**: Trains a deep learning model using extracted MFCC features and evaluates it on a test dataset.
    - **KNN**: Finds the nearest neighbors for a given audio sample and classifies based on the majority genre among neighbors.
## Usage
1. Prepare the Data:
    - Place audio files in Data/genres_original and ensure features_30_sec.csv is in Data/.
2. Run Feature Extraction:

    - CNN Notebook (``` cnn.ipynb ```): Extracts MFCCs, generates feature vectors, and prepares the dataset for CNN model training.
    - KNN Notebook (``` knn.ipynb ```): Extracts MFCC features and calculates distance between feature vectors for KNN classification.
3. Model Training and Testing:

    - **CNN**: Run the training cells to fit the CNN model and test its accuracy.
    - **KNN**: Execute distance calculations, find nearest neighbors, and classify genres.

## Notes
- **Performance**: The CNN model typically achieves higher accuracy due to its capability to capture intricate patterns in the data.
- **Manual Tuning**: You may adjust hyperparameters (e.g., k in KNN, number of epochs in CNN) to improve performance.

## Results and Evaluation
Evaluate and compare the results to determine which method works best for the dataset. The CNN model is generally expected to perform better with more data, while KNN offers simplicity and interpretability.

