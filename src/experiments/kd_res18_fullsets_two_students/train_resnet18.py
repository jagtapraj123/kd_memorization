import sys
sys.path.append("../../")

import pandas as pd
import torch
import math
import pickle
import os

import models.resnet as ResNet

from utils.data_mappers import LabeledPickleDatasetMapper
from utils.preprocessing import Preprocessor
from utils.constants import (
    DATASET_ROOT,
    DATASET_TRAIN_DATA_FILE,
    DATASET_TEST_DATA_FILE,
)
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION

from pipelines.two_students_classification_pipeline import Pipeline

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
    parser.add_argument("--soft_temp", dest="soft_temp", default=1, type=float, help="Smoothening parameter for Softmax")
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")
    parser.add_argument("--reduce", dest="dim_scale_factor", type=int,help="Scale down student model dimenstions byt factor")
    parser.add_argument("--drop", dest="drop_factor", type=float, help="Drop training samples with high gradients")
    parser.add_argument("--drop_epoch", dest="drop_epoch", type=int, help="Start dropping at given epoch")


    args = parser.parse_args()

    model_name = args.model_name
    subset_file = args.subset_csv_path
    save_dir = args.save_models
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    dim_scale_factor = args.dim_scale_factor
    drop_factor = args.drop_factor
    drop_epoch = args.drop_epoch
    teacher_path = args.teacher_path
    soft_temp = args.soft_temp

    # model_name = "teacher_resnet50_0-0.1_sub_set3_6_conn_sig"
    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Teacher Model: {}".format(teacher_path))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Dimension Reducing: {}".format(dim_scale_factor), flush=True)
    print("Softmax Temperature: {}".format(soft_temp), flush=True)
    print("Drop Factor: {}".format(drop_factor))
    print("Drop Epoch: {}".format(drop_epoch))


    # data = pd.read_csv(os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE))
    # subset_file = "../../../dataset/scratch_res50_fullsets/subset_0-0.1.csv"
    # subset_file = pd.read_csv()

    preprocessor = Preprocessor((32,32))

    teacher_model = ResNet.resnet50(
        num_classes=100,
        include_top=True,
        # connector=512,
        # connector_sigmoid=True,
        final_activation=True,
        inplanes=64,
        temperature=soft_temp,
    )

    print(teacher_model)
    summary(teacher_model, (3, 32, 32), device='cpu')

    teacher_model.load_state_dict(torch.load(teacher_path)["model_state_dict"])
    
    for params in teacher_model.parameters():
        params.requires_grad = False

    teacher_model.eval()

    model_s1 = ResNet.resnet18(
        num_classes=100, 
        include_top=True, 
        # connector=512,
        final_activation=True,
        inplanes=64//dim_scale_factor
    )
    print(model_s1)
    summary(model_s1, (3, 32, 32), device='cpu')

    model_s2 = ResNet.resnet18(
        num_classes=100, 
        include_top=True, 
        # connector=512,
        final_activation=True,
        inplanes=64//dim_scale_factor
    )
    print(model_s2)
    summary(model_s2, (3, 32, 32), device='cpu')

    # # Creating dataset mapper instances
    # train_set = LabeledDatasetMapper(
    #     os.path.join(PATH_PREFIX, DATASET_ROOT),
    #     os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE),
    #     preprocessor,
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
        augment=True,
        return_idx=True
    )

    full_train_set = LabeledPickleDatasetMapper(
        cifar_train[b'data'].copy(),
        cifar_train[b'fine_labels'].copy(),
        None,
        preprocessor,
        augment=False
    )

    low_mem_test_names = [
        "train_set_0-{}".format(max_mem) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]
    print(["../../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]])
    low_mem_part_train_sets = [
        LabeledPickleDatasetMapper(
            cifar_train[b'data'].copy(),
            cifar_train[b'fine_labels'].copy(),
            "../../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem),
            preprocessor,
            augment=False
        ) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]
    
    score_sets_low_mem = [
        {"name": low_mem_test_names[i], "dataset": low_mem_part_train_sets[i]} for i in range(len(low_mem_test_names))
    ]

    high_mem_test_names = [
        "train_set_{}-1".format(min_mem) for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]
    print(["../../../dataset/high_mem/subset_{}-1.csv".format(min_mem) for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]])
    high_mem_part_train_sets = [
        LabeledPickleDatasetMapper(
            cifar_train[b'data'].copy(),
            cifar_train[b'fine_labels'].copy(),
            "../../../dataset/high_mem/subset_{}-1.csv".format(min_mem),
            preprocessor,
            augment=False
        ) for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]

    score_sets_high_mem = [
        {"name": high_mem_test_names[i], "dataset": high_mem_part_train_sets[i]} for i in range(len(high_mem_test_names))
    ]

    test_set = LabeledPickleDatasetMapper(
        cifar_test[b'data'].copy(),
        cifar_test[b'fine_labels'].copy(),
        None,
        preprocessor,
        augment=False
    )

    score_sets = [
        {"name": "full_train_set", "dataset": full_train_set}, 
        *score_sets_low_mem,
        *score_sets_high_mem,
        {"name": "test_set", "dataset": test_set},
    ]

    print(score_sets)

    trainer = Pipeline(
        name=model_name,
        model_s1=model_s1,
        model_s2=model_s2,
        batch_size=256,
        workers=num_workers,
        train_set=train_set,
        test_sets=score_sets,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        teacher_model=teacher_model,
        cuda_num=cuda_num
    )

    num_epochs = 200
    lr = 0.4
    # step_size_func = lambda e: 1 / math.sqrt(1 + e)
    step_size_func = lambda e: ((e - num_epochs*0.15)/(num_epochs*0.15) + 1) if e <= num_epochs*0.15 else (num_epochs - e)/(num_epochs*0.85)

    loss_func_with_grad_s1 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    loss_func_s1 = partial(torch.nn.functional.kl_div, reduction='batchmean', log_target=True)

    loss_func_with_grad_s2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    loss_func_s2 = partial(torch.nn.functional.kl_div, reduction='batchmean', log_target=True)


    training_log, validation_log, grad_log1, grad_log2 = trainer.train(
        num_epochs=num_epochs,
        teacher_weightage=1,
        score_on_gnd_truth=True,
        lr=lr,
        step_size_func_s1=step_size_func,
        step_size_func_s2=step_size_func,
        loss_func_with_grad_s1=loss_func_with_grad_s1,
        loss_func_s1=loss_func_s1,
        loss_func_with_grad_s2=loss_func_with_grad_s2,
        loss_func_s2=loss_func_s2,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        # save_checkpoints_epoch=25,
        # save_checkpoints_path="../../saved_models/",
        validation_score_epoch=5,
        drop_percentage=drop_factor,
        drop_epoch=drop_epoch,
    )

    save_path = (trainer.save(save_dir, num_epochs))

    with open("{}_save_path.txt".format(model_name), 'w') as f:
        f.write(save_path)

    os.makedirs(os.path.join("logs/logfiles", trainer.name))
    with open(
        os.path.join("logs/logfiles", trainer.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "grad_log1.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log1, f)

    with open(
        os.path.join("logs/logfiles", trainer.name, "grad_log2.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log2, f)
