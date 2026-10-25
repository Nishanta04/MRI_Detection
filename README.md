# Brain Tumor MRI Classification

This project uses transfer learning with a VGG16 image classification model to classify brain MRI images into four dataset classes. The main workflow is in [`VGG16.ipynb`](VGG16.ipynb). [`first_notebook.ipynb`](first_notebook.ipynb) is a small notebook environment check and is not part of the model pipeline.

## Dataset

The main notebook downloads the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) with `kagglehub`. It expects the downloaded dataset to contain `Training/` and `Testing/` directories, each with one subdirectory per class.

The notebook reads the class folder names from `Training/`. Its expected four classes are **glioma**, **meningioma**, **notumor**, and **pituitary**. The dataset and pretrained VGG16 weights are downloaded when needed, so the first run requires an internet connection.

## Setup

Use Python in a Jupyter environment (for example, JupyterLab or the VS Code notebook extension). From the project directory, install the project and notebook dependencies:

```bash
python -m pip install -r requirements.txt kagglehub seaborn
```

`requirements.txt` includes TensorFlow, Jupyter, scikit-learn, NumPy, pandas, Matplotlib, and Pillow. TensorFlow availability depends on your Python version and platform; if installation fails, use a Python version supported by the TensorFlow release for your system.

## Run the model

1. Open `VGG16.ipynb` and run its cells from top to bottom.
2. The notebook downloads the dataset, loads and shuffles the training and test images, and displays sample training images.
3. Images are resized to **128 × 128**, brightness and contrast are randomly adjusted, and pixel values are scaled to `[0, 1]`.
4. A VGG16 model initialized with ImageNet weights is used as the feature extractor. The last three VGG16 layers are made trainable, followed by a dense classification head. Training uses a batch size of **20**, **5 epochs**, Adam, and sparse categorical cross-entropy.
5. The trained model and training history are saved under `VGG16_model/`:
   - `model.keras` — trained Keras model
   - `metrics.csv` — per-epoch training loss and accuracy
6. The evaluation cells load the saved model and report test-set classification metrics, show a confusion matrix, and plot one-vs-rest ROC curves with AUC values for all four classes.

The model output columns follow the class order returned by `os.listdir(train_dir)`. Keep the dataset class folders unchanged when running the notebook so its label encoding stays aligned with the model outputs.

## Evaluation notes

The notebook currently applies random brightness and contrast adjustments in `open_images()`, including when it loads the test set. This means test predictions and evaluation metrics can vary between runs. For a fixed evaluation, use deterministic preprocessing for test images (resize and normalize without random augmentation).

The ROC plot uses each class as positive against the other three classes (one-vs-rest). Each legend entry includes that class's AUC.

## Repository files

```text
.
├── VGG16.ipynb        # Dataset loading, training, and evaluation workflow
├── first_notebook.ipynb # Minimal notebook environment check
├── requirements.txt   # Python and Jupyter dependencies
└── README.md
```

## Intended use

This is an educational machine learning project. Its predictions are not a medical diagnosis and should not be used to guide clinical decisions.
