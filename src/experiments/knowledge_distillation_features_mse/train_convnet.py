import sys
sys.path.append("../../")

import pandas as pd
import torch
import math
import pickle
import os

import models.resnet as ResNet
import models.conv_net as ConvNet

from utils.data_mappers import LabeledDatasetMapper, GenDatasetMapper
from utils.preprocessing import Preprocessor
from utils.model_utils import load_state_dict
from utils.constants import (
    DATASET_ROOT,
    DATASET_TRAIN_DATA_FILE,
    DATASET_TEST_DATA_FILE,
    GEN_DATASET_ROOT,
)

from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION, SCORE_FUNCTIONS_REGRESSION

from pipelines.regression_pipeline import Pipeline as PipelineS1
from pipelines.classification_pipeline import Pipeline as PipelineS2


if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    model_name = "kd_convnet_resnet50_no_conn_mse"

    data = pd.read_csv(os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE))

    preprocessor = Preprocessor()

    teacher_model = ResNet.resnet50(
        num_classes=len(data["label"].unique()), include_top=False
    )

    load_state_dict(teacher_model, "../../saved_models/resnet50_ft_weight.pkl")

    # load_state_dict(teacher_model, "../../saved_models/resnet50_scratch_weight.pkl")
    
    for params in teacher_model.parameters():
        params.requires_grad = False

    teacher_model.eval()

    model = ConvNet.ConvNet(
        num_classes=len(data["label"].unique()), include_top=False, connector=2048
    )

    # Creating dataset mapper instances
    train_set_s1 = GenDatasetMapper(
        os.path.join(PATH_PREFIX, GEN_DATASET_ROOT),
        preprocessor,
        pretrain_size=8192,
        augment=True,
    )
    test_set = LabeledDatasetMapper(
        os.path.join(PATH_PREFIX, DATASET_ROOT),
        os.path.join(PATH_PREFIX, DATASET_TEST_DATA_FILE),
        preprocessor,
        augment=False,
    )

    trainer_s1 = PipelineS1(
        name=model_name + "_s1",
        model=model,
        batch_size=16,
        workers=8,
        train_set=train_set_s1,
        test_set=test_set,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        teacher_model=teacher_model,
    )

    num_epochs = 100
    lr_s1 = 0.001
    step_size_func_s1 = lambda e: 1 / math.sqrt(1 + e)

    loss_func_with_grad_s1 = torch.nn.MSELoss()
    loss_func_s1 = torch.nn.functional.mse_loss

    training_log_s1, validation_log_s1 = trainer_s1.train(
        num_epochs=num_epochs,
        teacher_weightage=1,
        lr=lr_s1,
        step_size_func=step_size_func_s1,
        loss_func_with_grad=loss_func_with_grad_s1,
        loss_func=loss_func_s1,
        score_functions=SCORE_FUNCTIONS_REGRESSION,
        validation_score_epoch=5,
        save_checkpoints_epoch=5,
        save_checkpoints_path="../../saved_models/",
    )

    os.makedirs(os.path.join("logs/logfiles", trainer_s1.name))
    with open(
        os.path.join("logs/logfiles", trainer_s1.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log_s1, f)

    with open(
        os.path.join("logs/logfiles", trainer_s1.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log_s1, f)

    model.include_top = True

    for params in model.parameters():
        params.requires_grad = False

    model.fc.reset_parameters()

    for params in model.fc.parameters():
        params.requires_grad = True

    train_set_s2 = LabeledDatasetMapper(
        os.path.join(PATH_PREFIX, DATASET_ROOT),
        os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE),
        preprocessor,
        augment=True,
    )

    trainer_s2 = PipelineS2(
        name=model_name + "_s2",
        model=model,
        batch_size=16,
        workers=8,
        train_set=train_set_s2,
        test_set=test_set,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
    )

    num_epochs = 100
    lr_s2 = 0.001
    step_size_func_s2 = lambda e: 1 / math.sqrt(1 + e)

    loss_func_with_grad_s2 = torch.nn.CrossEntropyLoss()
    loss_func_s2 = torch.nn.functional.cross_entropy

    training_log_s2, validation_log_s2 = trainer_s2.train(
        num_epochs=num_epochs,
        lr=lr_s2,
        step_size_func=step_size_func_s2,
        loss_func_with_grad=loss_func_with_grad_s2,
        loss_func=loss_func_s2,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        save_checkpoints_epoch=5,
        save_checkpoints_path="../../saved_models/",
    )

    os.makedirs(os.path.join("logs/logfiles", trainer_s2.name))
    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log_s2, f)

    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log_s2, f)
