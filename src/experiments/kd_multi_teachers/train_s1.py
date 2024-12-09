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

from pipelines.two_student_avg_classification_pipeline import PipelineS1, PipelineS2

import argparse
from functools import partial
from torchsummary import summary

if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    parser = argparse.ArgumentParser(
        prog="train_resnet18", description="Training ResNet18 from scratch"
    )
    parser.add_argument("model_name", type=str, help="Model Name (without spaces)")
    parser.add_argument("subset_csv_path", type=str, help="Path to subset csv")
    parser.add_argument(
        "save_models", type=str, help="Path to directory to save models"
    )
    parser.add_argument("teacher_path", type=str, help="Path to saved teacher model")
    parser.add_argument(
        "--soft_temp",
        dest="soft_temp",
        default=1,
        type=float,
        help="Smoothening parameter for Softmax",
    )
    parser.add_argument(
        "--cuda-num", dest="cuda_num", type=int, help="Device number for cuda"
    )
    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--reduce_t",
        dest="dim_scale_factor_t",
        type=int,
        help="Scale down teacher model dimenstions byt factor",
    )
    parser.add_argument(
        "--reduce_s",
        dest="dim_scale_factor_s",
        type=int,
        help="Scale down student model dimenstions byt factor",
    )
    args = parser.parse_args()

    model_name = args.model_name
    subset_file = args.subset_csv_path
    save_dir = args.save_models
    teacher_path = args.teacher_path
    soft_temp = args.soft_temp
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    dim_scale_factor_t = args.dim_scale_factor_t
    dim_scale_factor_s = args.dim_scale_factor_s
    

    # model_name = "teacher_resnet50_0-0.1_sub_set3_6_conn_sig"
    print("Name: {}".format(model_name))
    print("Subset File: {}".format(subset_file))
    print("Save Dir: {}".format(save_dir))
    print("Teacher Model: {}".format(teacher_path))
    print("Softmax Temperature: {}".format(soft_temp), flush=True)
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Dimension Reducing (Teacher): {}".format(dim_scale_factor_t), flush=True)
    print("Dimension Reducing (Student): {}".format(dim_scale_factor_s), flush=True)
    

    # data = pd.read_csv(os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE))
    # subset_file = "../../../dataset/scratch_res50_fullsets/subset_0-0.1.csv"
    # subset_file = pd.read_csv()

    preprocessor = Preprocessor((32, 32))

    teacher_models = []
    teacher_paths = os.listdir(teacher_path)
    for i in range(len(teacher_paths)):
        teacher_model = ResNet.resnet18(
            num_classes=100,
            include_top=True,
            final_activation=True,
            inplanes=64 // dim_scale_factor_t,
            temperature=soft_temp,
        )
        print(teacher_model)
        summary(teacher_model, (3, 32, 32), device="cpu")

        teacher_model.load_state_dict(torch.load(os.path.join(teacher_path, teacher_paths[i], "checkpoints", "200", "model.pth"))["model_state_dict"])

        for params in teacher_model.parameters():
            params.requires_grad = False

        teacher_model.eval()
        teacher_models.append(teacher_model)

    model_s1 = ResNet.resnet18(
        num_classes=100,
        include_top=True,
        final_activation=True,
        inplanes=64 // dim_scale_factor_s,
    )
    print(model_s1)
    summary(model_s1, (3, 32, 32), device="cpu")

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

    with open("../../../dataset/cifar-100-python/train", "rb") as fo:
        cifar_train = pickle.load(fo, encoding="bytes")

    with open("../../../dataset/cifar-100-python/test", "rb") as fo:
        cifar_test = pickle.load(fo, encoding="bytes")

    train_set = LabeledPickleDatasetMapper(
        cifar_train[b"data"].copy(),
        cifar_train[b"fine_labels"].copy(),
        subset_file,
        preprocessor,
        augment=True,
        return_idx=True,
    )

    full_train_set = LabeledPickleDatasetMapper(
        cifar_train[b"data"].copy(),
        cifar_train[b"fine_labels"].copy(),
        None,
        preprocessor,
        augment=False,
    )

    low_mem_test_names = [
        "train_set_0-{}".format(max_mem) for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]
    print(
        [
            "../../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem)
            for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
        ]
    )
    low_mem_part_train_sets = [
        LabeledPickleDatasetMapper(
            cifar_train[b"data"].copy(),
            cifar_train[b"fine_labels"].copy(),
            "../../../dataset/scratch_fullsets/subset_0-{}.csv".format(max_mem),
            preprocessor,
            augment=False,
        )
        for max_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]

    score_sets_low_mem = [
        {"name": low_mem_test_names[i], "dataset": low_mem_part_train_sets[i]}
        for i in range(len(low_mem_test_names))
    ]

    high_mem_test_names = [
        "train_set_{}-1".format(min_mem) for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]
    print(
        [
            "../../../dataset/high_mem/subset_{}-1.csv".format(min_mem)
            for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
        ]
    )
    high_mem_part_train_sets = [
        LabeledPickleDatasetMapper(
            cifar_train[b"data"].copy(),
            cifar_train[b"fine_labels"].copy(),
            "../../../dataset/high_mem/subset_{}-1.csv".format(min_mem),
            preprocessor,
            augment=False,
        )
        for min_mem in [0.1, 0.2, 0.4, 0.6, 0.8]
    ]

    score_sets_high_mem = [
        {"name": high_mem_test_names[i], "dataset": high_mem_part_train_sets[i]}
        for i in range(len(high_mem_test_names))
    ]

    test_set = LabeledPickleDatasetMapper(
        cifar_test[b"data"].copy(),
        cifar_test[b"fine_labels"].copy(),
        None,
        preprocessor,
        augment=False,
    )

    score_sets = [
        {"name": "full_train_set", "dataset": full_train_set},
        *score_sets_low_mem,
        *score_sets_high_mem,
        {"name": "test_set", "dataset": test_set},
    ]

    print(score_sets)

    trainer_s1 = PipelineS1(
        name=model_name,
        model=model_s1,
        batch_size=256,
        workers=num_workers,
        train_set=train_set,
        test_sets=score_sets,
        preprocessor=preprocessor,
        log_files_path="logs/fit/",
        teacher_models=teacher_models,
        cuda_num=cuda_num,
    )

    num_epochs = 200
    lr = 0.4
    # step_size_func = lambda e: 1 / math.sqrt(1 + e)
    step_size_func = (
        lambda e: ((e - num_epochs * 0.15) / (num_epochs * 0.15) + 1)
        if e <= num_epochs * 0.15
        else (num_epochs - e) / (num_epochs * 0.85)
    )

    loss_func_with_grad = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    loss_func = partial(
        torch.nn.functional.kl_div, reduction="batchmean", log_target=True
    )

    (
        training_log,
        validation_log,
        grad_log1,
        grad_log2,
        samples_bitvector,
    ) = trainer_s1.train(
        num_epochs=num_epochs,
        teacher_weightage=1,
        score_on_gnd_truth=True,
        lr=lr,
        score_functions=SCORE_FUNCTIONS_CLASSIFICATION,
        step_size_func=step_size_func,
        loss_func_with_grad=loss_func_with_grad,
        loss_func=loss_func,
        # save_checkpoints_epoch=25,
        # save_checkpoints_path="../../saved_models/",
        validation_score_epoch=5,
    )

    save_path = trainer_s1.save(save_dir, num_epochs)

    with open("{}_save_path.txt".format(model_name), "w") as f:
        f.write(save_path)

    os.makedirs(os.path.join("logs/logfiles", trainer_s1.name))
    with open(
        os.path.join("logs/logfiles", trainer_s1.name, "training_log.pkl"), "wb"
    ) as f:
        pickle.dump(training_log, f)

    with open(
        os.path.join("logs/logfiles", trainer_s1.name, "validation_log.pkl"), "wb"
    ) as f:
        pickle.dump(validation_log, f)

    with open(
        os.path.join("logs/logfiles", trainer_s1.name, "grad_log1.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log1, f)

    with open(
        os.path.join("logs/logfiles", trainer_s1.name, "grad_log2.pkl"), "wb"
    ) as f:
        pickle.dump(grad_log2, f)

    with open(
        os.path.join(
            "logs/logfiles", trainer_s1.name, "notdropped_samples_bitvector.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(samples_bitvector, f)
