import sys
sys.path.append("../../")

import pandas as pd
import torch
import math
import pickle
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
import models.resnet as ResNet

from utils.data_mappers import LabeledPickleDatasetMapper
from utils.preprocessing import Preprocessor
# from utils.constants import (
#     DATASET_ROOT,
#     DATASET_TRAIN_DATA_FILE,
#     DATASET_TEST_DATA_FILE,
# )
from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION

# from pipelines.classification_pipeline import Pipeline
import argparse

if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    parser = argparse.ArgumentParser(prog="testing", description="Testing ResNet18")
    parser.add_argument("model_name", type=str, help='Model Name (without spaces)')
    parser.add_argument("test_name", type=str, help='Test Name (without spaces)')
    parser.add_argument("saved_models", type=str, help='Path to directory to save models')
    parser.add_argument("--subset_csv_path", dest="subset_csv_path", type=str, default=None, help='Path to subset csv')
    parser.add_argument("--cuda-num", dest="cuda_num", type=int, help="Device number for cuda")
    parser.add_argument("--num-workers", dest="num_workers", type=int, help="Number of workers for dataloader")
    parser.add_argument("--reduce", dest="dim_scale_factor", type=int,help="Scale down student model dimenstions byt factor")
    
    args = parser.parse_args()

    model_name = args.model_name
    test_name = args.test_name
    subset_file = args.subset_csv_path
    saved_models = args.saved_models
    saved_models = saved_models.split()
    cuda_num = args.cuda_num
    num_workers = args.num_workers
    dim_scale_factor = args.dim_scale_factor

    print("Model Name: {}".format(model_name))
    print("Test Name: {}".format(test_name))
    print("Subset File: {}".format(subset_file))
    print("Saved Models: {}".format(saved_models))
    print("Cuda Num: {}".format(cuda_num), flush=True)
    print("Dimension Reducing: {}".format(dim_scale_factor), flush=True)
    

    preprocessor = Preprocessor((32,32))
    device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")

    model_s1 = ResNet.resnet18(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=False)
    # model = ResNet.resnet50(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=True)
    print("Epoch: {}".format(torch.load("{}".format(saved_models[0]), map_location=torch.device('cpu'))["epoch"]))
    model_s1.load_state_dict(torch.load("{}".format(saved_models[0]), map_location=torch.device('cpu'))["model_state_dict"])

    model_s1.to(device)
    model_s1.eval()

    model_s2 = ResNet.resnet18(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=False)
    # model = ResNet.resnet50(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=True)
    print("Epoch: {}".format(torch.load("{}".format(saved_models[1]), map_location=torch.device('cpu'))["epoch"]))
    model_s2.load_state_dict(torch.load("{}".format(saved_models[1]), map_location=torch.device('cpu'))["model_state_dict"])

    model_s2.to(device)
    model_s2.eval()

    # with open("../../../dataset/cifar-100-python/train", 'rb') as fo:
    #     cifar_train = pickle.load(fo, encoding='bytes')

    # test_set = LabeledPickleDatasetMapper(
    #     cifar_train[b'data'].copy(),
    #     cifar_train[b'fine_labels'].copy(),
    #     # subset_file,
    #     # None,
    #     "../../../dataset/scratch_fullsets/subset_0-0.1.csv",
    #     # "../../../dataset/high_mem/subset_0.8-1.csv",
    #     preprocessor,
    #     augment=False
    # )

    # test_loader = DataLoader(
    #         test_set, batch_size=512, num_workers=num_workers
    #     )

    with open("../../../dataset/cifar-100-python/train", "rb") as fo:
        cifar_train = pickle.load(fo, encoding="bytes")

    with open("../../../dataset/cifar-100-python/test", "rb") as fo:
        cifar_test = pickle.load(fo, encoding="bytes")

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

    loss_func = torch.nn.functional.cross_entropy

    for t in score_sets:
        test_loader = DataLoader(
            t["dataset"], batch_size=512, num_workers=num_workers
        )
    
        ys = []
        y_preds = []
        with torch.no_grad():
            for x_test, y_test in tqdm(test_loader):
                x = x_test.type(torch.FloatTensor).to(device)
                y_truth = y_test.type(torch.LongTensor).to(device)

                y_pred = torch.softmax(model_s1(x) + model_s2(x), dim=1)

                # loss = loss_func(y_pred, y_truth)

                ys += list(y_truth.cpu().detach().numpy())

                y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

        score_functions=SCORE_FUNCTIONS_CLASSIFICATION
        validation_scores = []
        if isinstance(score_functions, list) and len(score_functions) > 0:
            for score_func in score_functions:
                score = score_func["func"](ys, y_preds)
                validation_scores.append({score_func["name"]: score})

            print(
                "Testing {}, Set:{}, Validation Scores:{}".format(
                    test_name, t["name"], validation_scores
                ),
                flush=True,
            )
