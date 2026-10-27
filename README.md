# Brain Tumor MRI Classification

This project uses transfer learning with a VGG16 image classification model to classify brain MRI images into four dataset classes. The main workflow is in [`VGG16.ipynb`](VGG16.ipynb). [`first_notebook.ipynb`](first_notebook.ipynb) is a small notebook environment check and is not part of the model pipeline.

## Dataset

The main notebook downloads the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) with `kagglehub`. It expects the downloaded dataset to contain `Training/` and `Testing/` directories, each with one subdirectory per class.

The notebook reads the class folder names from `Training/`. Its expected four classes are **glioma**, **meningioma**, **notumor**, and **pituitary**. The dataset and pretrained VGG16 weights are downloaded when needed, so the first run requires an internet connection.

The saved model's output columns follow the training-time `os.listdir(train_dir)` order, which was recorded during evaluation as **meningioma, pituitary, notumor, glioma**. The web app uses that order. If you retrain with a different folder order, update `CLASS_NAMES` in `app.py` (or set the `CLASS_NAMES` environment variable) to match.

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

## Run the web app

The Flask app serves a drag-and-drop front end and loads `VGG16_model/model.keras` at startup.

```bash
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>, upload a brain MRI image, and click **Analyze scan**. The page shows the predicted class, confidence, and the softmax probability for every class.

Configuration:

- `MODEL_PATH` — path to a different `.keras` model.
- `CLASS_NAMES` — comma-separated class names in the model's output order (default `meningioma,pituitary,notumor,glioma`).
- `HOST` / `PORT` — server bind address and port (defaults `127.0.0.1:5000`).
- `FLASK_DEBUG=1` — enable the Flask debug reloader.

Endpoints:

- `GET /` — upload interface.
- `POST /predict` — multipart form field `image`; returns JSON with the prediction and class probabilities.
- `GET /health` — liveness check.

Uploads are processed in memory and are not written to disk. The app accepts JPG, PNG, WEBP, BMP, and TIFF files up to 8 MB, and preprocesses them the same way as the notebook: EXIF orientation correction, RGB conversion, 128 × 128 nearest-neighbour resize, and scaling to `[0, 1]`.

## Deploy

The app is ready for a free Docker deployment on Hugging Face Spaces. See
[`DEPLOYMENT.md`](DEPLOYMENT.md) for the one-command deployment script, manual
Git steps, and free-tier notes.

```bash
python -m pip install -U huggingface_hub
export HF_TOKEN=hf_your_write_token
python deploy_to_hf.py your-username/brain-mri-classifier
```

## Evaluation notes

Training images are augmented with random brightness and contrast adjustments inside `open_images()`. The test set is **not** augmented: the evaluation cells use `load_test_images()`, which only resizes and scales pixels to `[0, 1]`. This keeps the reported metrics reproducible across runs.

The ROC plot uses each class as positive against the other three classes (one-vs-rest). Each legend entry includes that class's AUC, and the macro-average AUC is printed below the plot.

Current results for the saved 5-epoch model on the 1,600-image test set:

| Metric | Value |
| --- | --- |
| Test accuracy | 0.9100 |
| Macro-average AUC | 0.9771 |
| Best classes | notumor (recall 1.00), pituitary (0.97) |
| Weakest class | glioma (recall 0.72, mostly predicted as meningioma) |

## Repository files

```text
.
├── app.py             # Flask backend and model inference
├── templates/
│   └── index.html     # Upload and results interface
├── static/
│   ├── css/style.css  # Front-end styling
│   └── js/app.js      # Upload, preview, and prediction logic
├── Dockerfile         # Deployment image (Hugging Face Spaces)
├── requirements-deploy.txt # Runtime-only deployment dependencies
├── space_README.md    # Hugging Face Space metadata and description
├── deploy_to_hf.py    # One-command Space deployment script
├── DEPLOYMENT.md      # Deployment guide and troubleshooting
├── VGG16_model/
│   ├── model.keras    # Trained model (gitignored)
│   └── metrics.csv    # Per-epoch training metrics
├── VGG16.ipynb        # Dataset loading, training, and evaluation workflow
├── first_notebook.ipynb # Minimal notebook environment check
├── requirements.txt   # Python and Jupyter dependencies
└── README.md
```

## Intended use

This is an educational machine learning project. Its predictions are not a medical diagnosis and should not be used to guide clinical decisions.
