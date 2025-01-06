import enum
import pathlib


class ResourcePath:
    PROJECT = pathlib.Path("/opt/project")
    MODELS = (PROJECT / "models").resolve()
    MODELS.mkdir(exist_ok=True)

    TEST_INPUTS = (PROJECT / "data" / "test").resolve()
    TEST_ANNOTATIONS = (TEST_INPUTS / "annotations").resolve()
    TRAIN_INPUTS = (PROJECT / "data" / "train" / "inputs").resolve()
    TRAIN_LABELS = (PROJECT / "data" / "train" / "labels").resolve()

    TRAIN_INPUTS.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
    TEST_INPUTS.mkdir(parents=True, exist_ok=True)
    TEST_ANNOTATIONS.mkdir(parents=True, exist_ok=True)


class ArrhythmiaType(enum.StrEnum):
    AVB1D, RBBB, LBBB = "1dAVb", "RBBB", "LBBB"
    SB, AF, ST = "SB", "AF", "ST"


class CardiogramLead(enum.IntEnum):
    DI, DII, DIII = 0, 1, 2
    AVR, AVL, AVF = 3, 4, 5
    V1, V2, V3, V4, V5, V6 = 6, 7, 8, 9, 10, 11
