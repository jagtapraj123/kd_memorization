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

    with open("../../../dataset/cifar-100-python/train", 'rb') as fo:
        cifar_train = pickle.load(fo, encoding='bytes')

    test_set = LabeledPickleDatasetMapper(
        cifar_train[b'data'].copy(),
        cifar_train[b'fine_labels'].copy(),
        # subset_file,
        # None,
        "../../../dataset/scratch_fullsets/subset_0-0.1.csv",
        # "../../../dataset/high_mem/subset_0.8-1.csv",
        preprocessor,
        augment=False
    )

    test_loader = DataLoader(
            test_set, batch_size=512, num_workers=num_workers
        )
    
    loss_func = torch.nn.functional.cross_entropy

    for thresh_i in range(1, 21):
        thresh = thresh_i * 0.05
        ys = []
        y_preds = []
        with torch.no_grad():
            for x_test, y_test in tqdm(test_loader):
                x = x_test.type(torch.FloatTensor).to(device)
                y_truth = y_test.type(torch.LongTensor).to(device)

                y_pred = torch.softmax(model_s1(x), dim=1)

                high_conf = torch.max(y_pred, dim=1).values >= thresh
                re_predict = torch.max(y_pred, dim=1).values < thresh
                # print("repredicting: {}".format(torch.sum(re_predict)))

                if torch.sum(re_predict) > 0:
                    y_pred[re_predict] = torch.softmax(model_s2(x[re_predict]), dim=1)

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
                "Testing {}, Thresh:{}, Validation Scores:{}".format(
                    test_name, thresh, validation_scores
                ),
                flush=True,
            )
