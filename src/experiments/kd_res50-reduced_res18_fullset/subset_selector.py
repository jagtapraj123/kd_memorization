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

if __name__ == "__main__":
    PATH_PREFIX = "../../../"

    dim_scale_factor = 1
    min_mem = 0
    max_mem = 1


    # test_name = "train_kd_res50_0-{}_S1-reduced_by{}_res18_disjoint_sets_0-{}_S{}_2".format(mem1, dim_scale_factor, mem1, j)
    test_name = "teacher_acc"

    preprocessor = Preprocessor((32,32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet.resnet50(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=True, temperature=8)
    # model = ResNet.resnet50(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=True)
    # model_path = input()
    model_path = "../../saved_models/kd_full_sets_0-1/teacher/res50/train_scratch_res50_0-1_20230402-151007"
    model.load_state_dict(torch.load("{}/checkpoints/200/model.pth".format(model_path), map_location=torch.device('cpu'))["model_state_dict"])

    model.to(device)
    model.eval()

    with open("../../../dataset/cifar-100-python/train", 'rb') as fo:
        cifar_train = pickle.load(fo, encoding='bytes')

    test_set = LabeledPickleDatasetMapper(
        cifar_train[b'data'].copy(),
        cifar_train[b'fine_labels'].copy(),
        # None,
        "../../../dataset/scratch_fullsets/subset_{}-{}.csv".format(min_mem, max_mem),
        # "../../../dataset/high_mem/subset_{}-{}.csv".format(min_mem, max_mem),
        preprocessor,
        augment=False
    )

    test_loader = DataLoader(
            test_set, batch_size=512, num_workers=8
        )
    
    loss_func = torch.nn.functional.cross_entropy

    subset = set()
    ys = []
    y_preds = []
    with torch.no_grad():
        for x_test, y_test, img_idx in tqdm(test_loader):
            x = x_test.type(torch.FloatTensor).to(device)
            y_truth = y_test.type(torch.LongTensor).to(device)

            y_pred = model(x)

            loss = loss_func(y_pred, y_truth)

            # ys += list(y_truth.cpu().detach().numpy())
            # print(torch.max(y_pred, dim=1).values.cpu().detach().numpy())
            # y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

            ys = (y_truth.cpu().detach().numpy())
            maxs = (torch.max(y_pred, dim=1).values.cpu().detach().numpy())
            print(maxs)
            y_preds = (torch.argmax(y_pred, dim=1).cpu().detach().numpy())

            for i in range(len(ys)):
                if (ys[i] == y_preds[i]) and (maxs[i] >= 0.90):
                    subset.add(img_idx[i].cpu().detach().numpy().item())

    print(len(subset))
    print(subset)
    subset_df = pd.DataFrame(sorted(list(subset)), columns=["idx"])
    subset_df.to_csv("high_conf_subset_{}-{}.csv".format(min_mem, max_mem), index=False)


    # score_functions=SCORE_FUNCTIONS_CLASSIFICATION
    # validation_scores = []
    # if isinstance(score_functions, list) and len(score_functions) > 0:
    #     for score_func in score_functions:
    #         score = score_func["func"](ys, y_preds)
    #         validation_scores.append({score_func["name"]: score})

    #     print(
    #         "Testing {}, Validation Scores:{}".format(
    #             test_name, validation_scores
    #         ),
    #         flush=True,
    #     )
