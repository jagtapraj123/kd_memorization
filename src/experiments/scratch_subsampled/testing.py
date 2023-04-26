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

    test_name = "kd_resnet18_resnet50_0-0.01_5_conn_bce_0-0.01_test2"
    print(test_name)

    # data = pd.read_csv(os.path.join(PATH_PREFIX, DATASET_TRAIN_DATA_FILE))
    # subset_file = "../../../dataset/set3/subset_0-0.1.csv"
    # subset_file = pd.read_csv()

    preprocessor = Preprocessor((32,32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = ResNet.resnet50(num_classes=100, include_top=True, connector=512, connector_sigmoid=True)
    # model.load_state_dict(torch.load("../../saved_models/teacher_resnet50_0-0.1_sub_set3_6_conn_sig_20230217-010003/checkpoints/200/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    # model.load_state_dict(torch.load("../../saved_models/teacher_resnet50_0-1_sub_set3_6_conn_sig_20230216-211912/checkpoints/200/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    
    model = ResNet.resnet18(num_classes=100, include_top=True, connector=512, connector_sigmoid=True)
    # model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-1_5_conn_bce_0-1_s1_20230216-014227/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    # model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-1_5_conn_bce_0-1_s2_20230216-021551/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    
    # model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-1_5_conn_bce_0-0.01_s1_20230216-014800/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    # model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-1_5_conn_bce_0-0.01_s2_20230216-020156/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    
    # model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-0.01_5_conn_bce_0-1_s1_20230216-120341/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    # model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-0.01_5_conn_bce_0-1_s2_20230216-124936/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"])
    
    # print(model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-0.01_5_conn_bce_0-0.01_s1_20230216-120332/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"]))
    print(model.load_state_dict(torch.load("../../saved_models/kd_resnet18_resnet50_0-0.01_5_conn_bce_0-0.01_s2_20230216-121753/checkpoints/100/model.pth", map_location=torch.device('cpu'))["model_state_dict"]))
    
    model.to(device)
    model.eval()

    with open("../../../dataset/cifar-100-python/train", 'rb') as fo:
        cifar_train = pickle.load(fo, encoding='bytes')

    test_set = LabeledPickleDatasetMapper(
        cifar_train[b'data'].copy(),
        cifar_train[b'fine_labels'].copy(),
        None,
        preprocessor,
        augment=False
    )

    test_loader = DataLoader(
            test_set, batch_size=512, num_workers=8
        )
    
    loss_func = torch.nn.functional.cross_entropy

    ys = []
    y_preds = []
    with torch.no_grad():
        for x_test, y_test in tqdm(test_loader):
            x = x_test.type(torch.FloatTensor).to(device)
            y_truth = y_test.type(torch.LongTensor).to(device)

            y_pred = model(x)

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
            "Testing {}, Validation Scores:{}".format(
                test_name, validation_scores
            ),
            flush=True,
        )
