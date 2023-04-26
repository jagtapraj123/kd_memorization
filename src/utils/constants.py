from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from functools import partial

DATASET_ROOT = "dataset/"
DATASET_TRAIN_DATA_FILE = "dataset/train_image_data.csv"
DATASET_TEST_DATA_FILE = "dataset/test_image_data.csv"

GEN_DATASET_ROOT = "gen_data/"

SCORE_FUNCTIONS_CLASSIFICATION = [
    {"name": "f1_score_micro", "func": partial(f1_score, average="micro")},
    {"name": "f1_score_macro", "func": partial(f1_score, average="macro")},
    {"name": "precision_micro", "func": partial(precision_score, average="micro")},
    {"name": "precision_macro", "func": partial(precision_score, average="macro")},
    {"name": "recall_micro", "func": partial(recall_score, average="micro")},
    {"name": "recall_macro", "func": partial(recall_score, average="macro")},
    {"name": "accuracy", "func": accuracy_score},
]

SCORE_FUNCTIONS_REGRESSION = [
    {"name": "mae_s1", "func": mean_absolute_error},
    {"name": "mse_s1", "func": mean_squared_error},
    {"name": "r2_score_s1", "func": r2_score},
]
