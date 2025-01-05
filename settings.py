from pathlib import Path

PROJECT_PATH = Path("/opt/project")
MODELS_PATH = (PROJECT_PATH / "models").resolve()
MODELS_PATH.mkdir(exist_ok=True)

# Data
TEST_DATA_PATH = (PROJECT_PATH / "data" / "test").resolve()
TEST_ANNOTATIONS_PATH = (TEST_DATA_PATH / "annotations").resolve()
INPUT_DATA_PATH = (PROJECT_PATH / "data" / "train" / "inputs").resolve()
LABEL_DATA_PATH = (PROJECT_PATH / "data" / "train" / "labels").resolve()

TEST_DATA_PATH.mkdir(parents=True, exist_ok=True)
TEST_ANNOTATIONS_PATH.mkdir(parents=True, exist_ok=True)
INPUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
LABEL_DATA_PATH.mkdir(parents=True, exist_ok=True)

classes = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
