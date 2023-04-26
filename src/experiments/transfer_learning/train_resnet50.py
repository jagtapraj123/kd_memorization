import sys
sys.path.append("../../")

import pandas as pd
import torch
import math
import pickle
import os

import models.resnet as ResNet

from utils.data_mappers import LabeledDatasetMapper
from utils.preprocessing import Preprocessor
from utils.model_utils import load_state_dict
from utils.constants import (
    DATASET_ROOT,
    DATASET_TRAIN_DATA_FILE,
    DATASET_TEST_DATA_FILE,
)
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION

from pipelines.classification_pipeline import Pipeline


if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    model_name = "transfer_resnet50_ft"

    data = pd.read_csv(os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE))

    preprocessor = Preprocessor()

    model = ResNet.resnet50(num_classes=len(data["label"].unique()), include_top=True)

    load_state_dict(model, "../../saved_models/resnet50_ft_weight.pkl")

    # load_state_dict(model, "../../saved_models/resnet50_scratch_weight.pkl")
    
    for params in model.parameters():
        params.requires_grad = False

    model.fc.reset_parameters()

    for params in model.fc.parameters():
        params.requires_grad = True

    # Creating dataset mapper instances
    train_set = LabeledDatasetMapper(
        os.path.join(PATH_PREFIX, DATASET_ROOT),
        os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE),
        preprocessor,
        augment=True,
    )
    test_set = LabeledDatasetMapper(
        os.path.join(PATH_PREFIX, DATASET_ROOT),
        os.path.join(PATH_PREFIX, DATASET_TEST_DATA_FILE),
        preprocessor,
        augment=False,
    )

    trainer = Pipeline(
        name=model_name,
        model=model,
        batch_size=16,
        workers=8,
        train_set=train_set,
        test_set=test_set,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
    )

    num_epochs = 100
    lr = 0.001
    step_size_func = lambda e: 1 / math.sqrt(1 + e)

    loss_func_with_grad = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.functional.cross_entropy

    training_log, validation_log = trainer.train(
        num_epochs=num_epochs,
        lr=lr,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        save_checkpoints_epoch=5,
        save_checkpoints_path="../../saved_models/",
    )

    os.makedirs(os.path.join("logs/logfiles", trainer.name))
    with open(
        os.path.join("logs/logfiles", trainer.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log, f)
