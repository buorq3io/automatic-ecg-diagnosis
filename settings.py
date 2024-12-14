from pathlib import Path

PROJECT_PATH = Path("/opt/project")
MODELS_PATH = PROJECT_PATH / "models"
MODELS_PATH.mkdir(exist_ok=True)

# Data
TEST_DATA_PATH = PROJECT_PATH / "data" / "test"
TEST_ANNOTATIONS_PATH = TEST_DATA_PATH / "annotations"
INPUT_DATA_PATH = PROJECT_PATH / "data" / "train" / "inputs"
LABEL_DATA_PATH = PROJECT_PATH / "data" / "train" / "labels"

TEST_DATA_PATH.mkdir(parents=True, exist_ok=True)
TEST_ANNOTATIONS_PATH.mkdir(parents=True, exist_ok=True)
INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
LABEL_DATA_PATH.mkdir(parents=True, exist_ok=True)
