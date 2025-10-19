# Source Code for "Investigating Audio Deepfakes and Countermeasures"

This directory contains the source code implementation for the Honours thesis titled "Investigating Audio Deepfakes and Countermeasures" by Makomborero Murwira, submitted to Rhodes University in October 2025.

The code implements and evaluates Light Convolutional Neural Network (LCNN) based audio deepfake countermeasures, using the ASVspoof 2019 Logical Access (LA) dataset.

## File Descriptions 

This repository includes four Jupyter Notebooks (`.ipynb`), each corresponding to a specific model configuration evaluated in the thesis:

1.  **`lfcc-lcnn-asoftmax1 (1).ipynb`**: Implements **Model-1**. This is the baseline LCNN model trained using the **Angular Margin Softmax (A-Softmax)** loss function. It replicates the LCNN architecture from Lavrentyeva et al. (2019).
2.  **`lcnn-asoftmax-freqaug1 (2).ipynb`**: Implements **Model-2**. This file uses the same architecture in Model-1 and adds **FreqAugment** data augmentation during training.
3.  **`lfcc-lcnn-cosface.ipynb`**: Implements **Model-3**. This model implements the same LCNN architecture but uses the **Large Margin Cosine Loss (Cosface)** function in place of A-Softmax loss.
4.  **`lfcc-lcnn-cosface-freqaug.ipynb`**: Implements **Model-4**. This is the final, optimised model, which combines the LCNN architecture with both **Cosface** loss and **FreqAugment** data augmentation.

Each notebook is structrued in the following format:
* Data loading and preprocessing (LFCC feature extraction).
* Dataset and DataLoader definition.
* LCNN model architecture definition (including Max-Feature-Map activation).
* Margin-based loss function definition (A-Softmax or Cosface).
* Training loop implementation (including FreqAugment where applicable).
* Evaluation loop implemenattion (calculating EER, F1-Score).
* Saving the trained model checkpoint.
* Generating confusion matrices and ROC curves.

## Dataset 

All experiments use the **ASVspoof 2019 Logical Access (LA) dataset**. Due to licensing restrictions, the dataset itself is not included in this repository. 
You can obtain access via the official ASVspoof 2019 challenge website: https://doi.org/10.7488/ds/2555

The notebooks expect the dataset to be structured as described in the ASVspoof 2019 documentation, with separate directories for `train`, `dev`, and `eval` sets containing `.flac` audio files and corresponding protocol files (`.txt`).
PS: The version of my dataset was converted to `.wav`; the original implementation i was replicating made use of a deprecated torchaudio.functional.decode_flac() function, requiring an alternate solution.

## Requirements and Setup 

The code was developed and tested in a **Kaggle Notebook environment** with GPU acceleration (NVIDIA Tesla P100).
You can replicate this in you own preferred IDE and appropriate GPU. While you can technically do it w/o GPU acceleration, it's a pain.

Key software dependencies include:
* Python 3.x (tested with 3.10+)
* PyTorch (tested with 2.0+)
* Torchaudio (tested with 2.0+)
* Librosa (for audio processing utilities)
* NumPy
* Matplotlib (for plotting)
* tqdm, helped for parallel compute
* scikit-learn (for metrics calculation)


############IMPORTANT##################
My original .ipynb files used tensorflow/keras, but i encountered problems trying to implement Cosface loss and A-Softmax layers, so i ported everything to use PyTorch and torchaudio.
I noticed a big mistake while i was already deep into testing and evaluating my models---i forgot to use a fixed seed in the final cleaned up notebooks added here.
Everything works as intedned, but due to the random weight init associated with not using a fixed seed, the EER may/may not be the same in subsequent runs:)

If you find this work useful, please pour yourself a glass of water, take frequent breaks from your screen, and don't forget to smile;)


