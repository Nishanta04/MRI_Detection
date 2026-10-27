from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "VGG16_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = hf_hub_download(
    repo_id="Nishanta04/brain-tumor-model",
    filename="model.keras",
)

destination = MODEL_DIR / "model.keras"
shutil.copy2(model_path, destination)

print(f"Model downloaded successfully: {destination}")
