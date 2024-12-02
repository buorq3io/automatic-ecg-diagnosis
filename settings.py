from pathlib import Path

PROJECT_PATH = Path("/opt/project")

# Models and Predictions
LOGS_PATH = PROJECT_PATH / "logs"
MODELS_PATH = PROJECT_PATH / "models"
BACKUPS_PATH = MODELS_PATH / "backup"
FIGURES_PATH = PROJECT_PATH / "figures"
PREDICTIONS_PATH = PROJECT_PATH / "predictions"

LOGS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)
BACKUPS_PATH.mkdir(exist_ok=True)
FIGURES_PATH.mkdir(exist_ok=True)
PREDICTIONS_PATH.mkdir(exist_ok=True)

# Data
TEST_DATA_PATH = PROJECT_PATH / "data" / "test"
TEST_ANNOTATIONS_PATH = TEST_DATA_PATH / "annotations"
INPUT_DATA_PATH = PROJECT_PATH / "data" / "train" / "inputs"
LABEL_DATA_PATH = PROJECT_PATH / "data" / "train" / "labels"

TEST_DATA_PATH.mkdir(parents=True, exist_ok=True)
TEST_ANNOTATIONS_PATH.mkdir(parents=True, exist_ok=True)
INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
LABEL_DATA_PATH.mkdir(parents=True, exist_ok=True)
