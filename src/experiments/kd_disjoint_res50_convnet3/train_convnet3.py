import sys
sys.path.append("../../")

import pandas as pd
import torch
import math
import pickle
import os

import models.resnet as ResNet
import models.conv_simple as ConvNet

from utils.data_mappers import LabeledPickleDatasetMapper, GenDatasetMapper
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

import argparse
from functools import partial
from torchsummary import summary

if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    parser = argparse.ArgumentParser(prog="train_resnet18", description="Training ResNet18 from scratch")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    parser.add_argument("subset_csv_path", type=str, help='Path to subset csv')
    parser.add_argument("save_models", type=str, help='Path to directory to save models')
    parser.add_argument("teacher_path", type=str, help='Path to saved teacher model')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")


    args = parser.parse_args()

    model_name = args.model_name
    subset_file = args.subset_csv_path
    save_dir = args.save_models
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    teacher_path = args.teacher_path

    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Teacher Model: {}".format(teacher_path))
    print("Cuda Num: {}".format(cuda_num))


    preprocessor = Preprocessor()

    teacher_model = ResNet.resnet50(
        num_classes=100,
        include_top=True,
        # connector=512,
        # connector_sigmoid=True,
        final_activation=True
    )

    # load_state_dict(teacher_model, "../../saved_models/resnet50_ft_weight.pkl")
    # teacher_model.load_state_dict(torch.load("../../saved_models/teacher_resnet50_0-1_5_conn_sig_20230216-002615/checkpoints/125/model.pth")["model_state_dict"])
    
    # load_state_dict(teacher_model, "../../saved_models/resnet50_scratch_weight.pkl")
    teacher_model.load_state_dict(torch.load(teacher_path)["model_state_dict"])
    
    for params in teacher_model.parameters():
        params.requires_grad = False

    teacher_model.eval()

    # model = ResNet.resnet18(
    #     num_classes=100, 
    #     include_top=True, 
    #     # connector=512,
    #     final_activation=True
    # )
    model = ConvNet.convnet3(
        num_classes=100,
        include_top=True, 
        # connector=512,
        final_activation=True
    )
    print(model)
    summary(model, (3, 32, 32), device='cpu')

    # # Creating dataset mapper instances
    # train_set_s1 = GenDatasetMapper(
    #     os.path.join(PATH_PREFIX, GEN_DATASET_ROOT),
    #     preprocessor,
    #     pretrain_size=8192,
    #     augment=True,
    # )
    # test_set = LabeledDatasetMapper(
    #     os.path.join(PATH_PREFIX, DATASET_ROOT),
    #     os.path.join(PATH_PREFIX, DATASET_TEST_DATA_FILE),
    #     preprocessor,
    #     augment=False,
    # )

    with open("../../../dataset/cifar-100-python/train", 'rb') as fo:
        cifar_train = pickle.load(fo, encoding='bytes')

    with open("../../../dataset/cifar-100-python/test", 'rb') as fo:
        cifar_test = pickle.load(fo, encoding='bytes')

    train_set = LabeledPickleDatasetMapper(
        cifar_train[b'data'].copy(),
        cifar_train[b'fine_labels'].copy(),
        subset_file,
        preprocessor,
        augment=True
    )

    full_train_set = LabeledPickleDatasetMapper(
        cifar_train[b'data'].copy(),
        cifar_train[b'fine_labels'].copy(),
        None,
        preprocessor,
        augment=False
    )

    test_set = LabeledPickleDatasetMapper(
        cifar_test[b'data'].copy(),
        cifar_test[b'fine_labels'].copy(),
        None,
        preprocessor,
        augment=False
    )

    # num_epochs = 100
    # lr_s1 = 0.01
    # step_size_func_s1 = lambda e: 1 / math.sqrt(1 + e)

    # loss_func_with_grad_s1 = torch.nn.BCEWithLogitsLoss()
    # loss_func_s1 = torch.nn.functional.binary_cross_entropy_with_logits

    # training_log_s1, validation_log_s1 = trainer_s1.train(
    #     num_epochs=num_epochs,
    #     teacher_weightage=1,
    #     lr=lr_s1,
    #     step_size_func=step_size_func_s1,
    #     loss_func_with_grad=loss_func_with_grad_s1,
    #     loss_func=loss_func_s1,
    #     postprocess_out=torch.sigmoid,
    #     score_functions=SCORE_FUNCTIONS_REGRESSION,
    #     validation_score_epoch=5,
    #     save_checkpoints_epoch=25,
    #     save_checkpoints_path="../../saved_models/",
    # )

    # os.makedirs(os.path.join("logs/logfiles", trainer_s1.name))
    # with open(
    #     os.path.join("logs/logfiles", trainer_s1.name, "training_log.pkl"), "wb"
    # ) as f:
    #     pickle.dump(training_log_s1, f)

    # with open(
    #     os.path.join("logs/logfiles", trainer_s1.name, "validation_log.pkl"), "wb"
    # ) as f:
    #     pickle.dump(validation_log_s1, f)

    # model.include_top = True

    # for params in model.parameters():
    #     params.requires_grad = False

    # model.fc.reset_parameters()

    # for params in model.fc.parameters():
    #     params.requires_grad = True

    # train_set_s2 = LabeledDatasetMapper(
    #     os.path.join(PATH_PREFIX, DATASET_ROOT),
    #     os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE),
    #     preprocessor,
    #     augment=True,
    # )
    
    # train_set_s2 = train_set

    trainer_s2 = PipelineS2(
        name=model_name,
        model=model,
        batch_size=256,
        workers=num_workers,
        train_set=train_set,
        test_sets=[{"name": "full_train_set", "dataset": full_train_set}, {"name": "test_set", "dataset": test_set}],
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        teacher_model=teacher_model,
        cuda_num=cuda_num
    )

    num_epochs = 200
    lr_s2 = 0.01
    step_size_func_s2 = lambda e: 1 / math.sqrt(1 + e)

    loss_func_with_grad_s2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    loss_func_s2 = partial(torch.nn.functional.kl_div, reduction='batchmean', log_target=True)

    training_log_s2, validation_log_s2 = trainer_s2.train(
        num_epochs=num_epochs,
        teacher_weightage=1,
        lr=lr_s2,
        step_size_func=step_size_func_s2,
        loss_func_with_grad=loss_func_with_grad_s2,
        loss_func=loss_func_s2,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        # save_checkpoints_epoch=25,
        # save_checkpoints_path="../../saved_models/",
    )

    trainer_s2.save(save_dir, num_epochs)

    os.makedirs(os.path.join("logs/logfiles", trainer_s2.name))
    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log_s2, f)

    with open(
        os.path.join("logs/logfiles", trainer_s2.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log_s2, f)
