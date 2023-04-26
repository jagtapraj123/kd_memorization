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

import matplotlib.pyplot as plt

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
    
    model_s1.load_state_dict(torch.load("{}".format(saved_models[0]), map_location=torch.device('cpu'))["model_state_dict"])

    model_s1.to(device)
    model_s1.eval()

    model_s2 = ResNet.resnet18(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=False)
    # model = ResNet.resnet50(num_classes=100, include_top=True, inplanes=64//dim_scale_factor, final_activation=True)
    
    model_s2.load_state_dict(torch.load("{}".format(saved_models[1]), map_location=torch.device('cpu'))["model_state_dict"])

    model_s2.to(device)
    model_s2.eval()

    with open("../../../dataset/cifar-100-python/train", 'rb') as fo:
        cifar_train = pickle.load(fo, encoding='bytes')

    test_set = LabeledPickleDatasetMapper(
        cifar_train[b'data'].copy(),
        cifar_train[b'fine_labels'].copy(),
        subset_file,
        # None,
        # "../../../dataset/scratch_fullsets/subset_0-0.1.csv",
        # "../../../dataset/high_mem/subset_0.8-1.csv",
        preprocessor,
        augment=False
    )

    test_loader = DataLoader(
            test_set, batch_size=512, num_workers=8
        )
    
    loss_func = torch.nn.functional.cross_entropy

    for scale_i in range(11):
        scale = scale_i * 0.1
        
        ys = []
        y_preds = []
        with torch.no_grad():
            for x_test, y_test in tqdm(test_loader):
                x = x_test.type(torch.FloatTensor).to(device)
                y_truth = y_test.type(torch.LongTensor).to(device)

                y_pred_s1 = model_s1(x)
                y_pred_s1 = torch.div(y_pred_s1.T, torch.max(y_pred_s1, dim=1).values).T

                y_pred_s2 = model_s2(x)
                y_pred_s2 = torch.div(y_pred_s2.T, torch.max(y_pred_s2, dim=1).values).T

                y_pred = torch.softmax((1-scale) * y_pred_s1 + scale * y_pred_s2, dim=1)
                # print(y_truth[0])
                # print(y_pred_s1[0])
                # print(torch.softmax(y_pred_s1, dim=1)[0], torch.argmax(y_pred_s1[0]))
                
                # print(y_pred_s2[0])
                # print(torch.softmax(y_pred_s2, dim=1)[0], torch.argmax(y_pred_s2[0]))
                
                # print(((1-scale) * y_pred_s1 + scale * y_pred_s2)[0], torch.argmax(((1-scale) * y_pred_s1 + scale * y_pred_s2)[0]))
                # print(y_pred[0])


                # plt.plot(y_pred_s1[0].cpu().detach().numpy(), label="s1")
                # plt.plot(y_pred_s2[0].cpu().detach().numpy(), label="s2")
                # plt.plot(((1-scale) * y_pred_s1 + scale * y_pred_s2)[0].cpu().detach().numpy(), label="final")
                
                # # plt.plot(torch.softmax(y_pred_s1, dim=1)[0].cpu().detach().numpy(), label="s1")
                # # plt.plot(torch.softmax(y_pred_s2, dim=1)[0].cpu().detach().numpy(), label="s2")
                # # plt.plot(torch.softmax(((1-scale) * y_pred_s1 + scale * y_pred_s2), dim=1)[0].cpu().detach().numpy(), label="final")
                # plt.legend()
                # plt.savefig("s1s2_preds.png")
                # exit()

                loss = loss_func(y_pred, y_truth)

                ys += list(y_truth.cpu().detach().numpy())

                y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

        score_functions=SCORE_FUNCTIONS_CLASSIFICATION
        validation_scores = []
        if isinstance(score_functions, list) and len(score_functions) > 0:
            for score_func in score_functions:
                score = score_func["func"](ys, y_preds)
                validation_scores.append({score_func["name"]: score})

            print(
                "Testing {} Scale:{}, Validation Scores:{}".format(
                    test_name, scale, validation_scores
                ),
                flush=True,
            )
