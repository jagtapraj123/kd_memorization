import torch
import numpy as np

from utils.preprocessing import Preprocessor

import os
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION
from typing import List, Dict

from functools import partial


class Pipeline:
    def __init__(
        self,
        name: str,
        model_s1: torch.nn.Module,
        model_s2: torch.nn.Module,
        batch_size: int,
        workers: int,
        train_set: Dataset,
        test_sets: List[Dict],
        preprocessor: Preprocessor,
        log_files_path: str,
        teacher_model: torch.nn.Module = None,
        cuda_num: int = 0,
        **kwargs
    ):
        self.name = name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model_s1 = model_s1
        self.model_s2 = model_s2
        self.teacher_model = teacher_model

        self.batch_size = batch_size

        self.preprocessor = preprocessor

        self.args = kwargs

        # Set training device (CUDA-GPU / CPU)
        self.device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")
        print("Training Device: {}".format(self.device))
        self.model_s1.to(self.device)
        self.model_s2.to(self.device)

        if self.teacher_model is not None:
            self.teacher_model.to(self.device)

        # Creating dataset loader to load data parallelly
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=workers,
            shuffle=True,
            persistent_workers=True,
            **kwargs
        )
        
        # new
        self.test_loaders = {}
        for t in test_sets:
            self.test_loaders[t['name']] = DataLoader(
                t['dataset'], batch_size=self.batch_size, num_workers=workers, **kwargs
            )

        # Create summary writers for tensorboard logs
        self.train_writer = SummaryWriter(
            os.path.join(log_files_path, self.name, "train")
        )

        self.train_writer_s2 = SummaryWriter(
            os.path.join(log_files_path, self.name, "train")
        )

        # new
        self.valid_writers = {}
        for k in self.test_loaders.keys():
            self.valid_writers[k] = SummaryWriter(
            os.path.join(log_files_path, self.name, "validation_{}".format(k))
        )

    def train(
        self,
        num_epochs: int = 100,
        teacher_weightage: float = 0,
        score_on_gnd_truth: bool = True,
        lr: float = 0.001,
        score_functions: list = SCORE_FUNCTIONS_CLASSIFICATION,
        optimizer_s1: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_s2: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler_s1: torch.optim.lr_scheduler._LRScheduler = None,
        lr_scheduler_s2: torch.optim.lr_scheduler._LRScheduler = None,
        step_size_func_s1=lambda e: 1,
        step_size_func_s2=lambda e: 1,
        loss_func_s1=torch.nn.functional.cross_entropy,
        loss_func_with_grad_s1=torch.nn.CrossEntropyLoss(),
        loss_func_s2=partial(torch.nn.functional.kl_div, reduction='batchmean', log_target=True),
        loss_func_with_grad_s2=torch.nn.KLDivLoss(reduction='batchmean', log_target=True),
        postprocess_out = None,
        validation_score_epoch: int = 1,
        save_checkpoints_epoch: int = -1,
        save_checkpoints_path: str = "",
        drop_percentage: float = 0,
        drop_epoch: int = -1,
        samples_bitvector: np.ndarray = None,
    ):

        self.epochs = num_epochs

        # Setting optimzer
        # optimizer = optimizer(self.model_s1.parameters(), lr=lr)
        optimizer_s1 = torch.optim.SGD(self.model_s1.parameters(), lr=lr, momentum=0.9)
        optimizer_s2 = torch.optim.SGD(self.model_s2.parameters(), lr=lr, momentum=0.9)

        # Learning rate scheduler for changing learning rate during training
        if lr_scheduler_s1 is None:
            lr_scheduler_s1 = torch.optim.lr_scheduler.LambdaLR(optimizer_s1, step_size_func_s1)

        if lr_scheduler_s2 is None:
            lr_scheduler_s2 = torch.optim.lr_scheduler.LambdaLR(optimizer_s2, step_size_func_s2)

        training_log = {"errors": [], "scores": []}
        # validation_log = {"errors": [], "scores": []}
        
        # new
        validation_log = {}
        for k in self.test_loaders.keys():
            validation_log[k] = {"errors": [], "scores": []}

        # Training
        # pbar = tqdm(range(self.epochs), desc="Training epoch")
        
        train_size = 50000
        if samples_bitvector is None:
            samples_bitvector = np.ones((train_size), dtype=np.int8)
        
        grad_log1 = np.zeros((50000, self.epochs))
        grad_log2 = np.zeros((50000, self.epochs))

        for epoch in range(1, self.epochs + 1):
            print("lr_s1: {}, lr_s2: {}".format(lr_scheduler_s1.get_last_lr(), lr_scheduler_s2.get_last_lr()))

            # Putting models in training mode to calculate back gradients
            self.model_s1.train()
            self.model_s2.train()

            ys = []
            y_preds = []

            # Batch-wise optimization
            pbar = tqdm(self.train_loader, desc="Training S1 epoch {}".format(epoch))
            for x_train, y_train, idx in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                y_truth = y_train.type(torch.LongTensor).to(self.device)

                if epoch > drop_epoch:
                    s1_selector = samples_bitvector[idx] == 1
                    x = x[s1_selector]
                    y_truth = y_truth[s1_selector]
                    idx = idx[s1_selector]

                    # self.model_s1.eval()
                    # with torch.no_grad():
                    #     y_pred_s12 = self.model_s1(x_s2)
                    # self.model_s1.train()

                    # loss_s2 = loss_func_with_grad_s2(y_pred_s2, torch.sigmoid(y_teacher_s2 - y_pred_s12))

                    
                    if epoch == drop_epoch+1:
                        print("dropping: {}".format(np.sum(samples_bitvector[idx] == 0)))
                        print("x: {}".format(x.shape))
                    if x.shape[0] == 0:
                        continue

                # Forward pass
                y_pred = self.model_s1(x)

                if teacher_weightage > 0:
                    if self.teacher_model is not None:
                        y_teacher = self.teacher_model(x)
                    else:
                        raise RuntimeError("Using un-specified teacher model")
                

                # Clearing previous epoch gradients
                optimizer_s1.zero_grad()

                # Calculating loss
                if teacher_weightage == 0:
                    loss = loss_func_with_grad_s1(y_pred, y_truth)
                elif teacher_weightage == 1:
                    loss = loss_func_with_grad_s1(y_pred, y_teacher)
                else:
                    loss = teacher_weightage * loss_func_with_grad_s1(
                        y_pred, y_teacher
                    ) + (1 - teacher_weightage) * loss_func_with_grad_s1(y_pred, y_truth)

                # Backward pass to calculate gradients
                loss.backward(retain_graph=True)

                for i in range(len(idx)):
                    if teacher_weightage == 0:
                        grad = torch.autograd.grad(loss_func_with_grad_s1(y_pred[i], y_truth[i]), self.model_s1.fc.parameters(), create_graph=True)
                        # print(sum(p.numel() for p in grad))
                        # print(len(grad), [grad[i].shape for i in range(len(grad))], [torch.mean(torch.abs(grad[i])) for i in range(len(grad))])
                        # print(grad)
                        grad_log1[idx[i]][epoch-1] = torch.mean(torch.abs(grad[0]))
                        grad_log2[idx[i]][epoch-1] = torch.mean(torch.abs(grad[1]))
                    else:
                        grad = torch.autograd.grad(loss_func_with_grad_s1(y_pred[i], y_teacher[i]), self.model.fc.parameters(), create_graph=True)
                        # print(sum(p.numel() for p in grad))
                        # print(len(grad), [grad[i].shape for i in range(len(grad))], [torch.mean(torch.abs(grad[i])) for i in range(len(grad))])
                        # print(grad)
                        grad_log1[idx[i]][epoch-1] = torch.mean(torch.abs(grad[0]))
                        grad_log2[idx[i]][epoch-1] = torch.mean(torch.abs(grad[1]))
                    
                # Update gradients
                optimizer_s1.step()

                # Save/show loss per step of training batches
                pbar.set_postfix({"training error": loss.item()})
                training_log["errors"].append({"epoch": epoch, "loss": loss.item()})

                self.train_writer.add_scalar("loss", loss.item(), epoch)
                self.train_writer.flush()

                # Save y_true and y_pred in lists for calculating epoch-wise scores
                if teacher_weightage == 1 and not score_on_gnd_truth:
                    ys += list(torch.argmax(y_teacher, dim=1).cpu().detach().numpy())
                else:
                    ys += list(y_truth.cpu().detach().numpy())
                
                if postprocess_out is not None:
                    y_pred = postprocess_out(y_pred)

                y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

            if epoch > drop_epoch:
                pbar = tqdm(self.train_loader, desc="Training S2 epoch {}".format(epoch))
                for x_train, y_train, idx in pbar:
                    x = x_train.type(torch.FloatTensor).to(self.device)
                    y_truth = y_train.type(torch.LongTensor).to(self.device)

                    s2_selector = samples_bitvector[idx] == 0
                    x_s2 = x[s2_selector]
                    y_truth_s2 = y_truth[s2_selector]
                    idx_s2 = idx[s2_selector]

                    if epoch == drop_epoch+1:
                        print("Using: {}".format(np.sum(samples_bitvector[idx] == 0)))
                        print("x: {}".format(x.shape))
                    if x.shape[0] == 0:
                        continue

                    # Forward pass
                    y_pred_s2 = self.model_s2(x_s2)

                    if teacher_weightage > 0:
                        if self.teacher_model is not None:
                            y_teacher_s2 = self.teacher_model(x_s2)
                        else:
                            raise RuntimeError("Using un-specified teacher model")
                
                    # Clearing previous epoch gradients
                    optimizer_s2.zero_grad()

                    # Calculating loss
                    if teacher_weightage == 0:
                        loss_s2 = loss_func_with_grad_s2(y_pred_s2, y_truth_s2)
                    elif teacher_weightage == 1:
                        loss_s2 = loss_func_with_grad_s2(y_pred_s2, y_teacher_s2)
                    else:
                        loss_s2 = teacher_weightage * loss_func_with_grad_s2(
                            y_pred_s2, y_teacher_s2
                        ) + (1 - teacher_weightage) * loss_func_with_grad_s2(y_pred_s2, y_truth_s2)

                    # Backward pass to calculate gradients
                    loss_s2.backward(retain_graph=True)

                    for i in range(len(idx_s2)):
                        if teacher_weightage == 0:
                            grad = torch.autograd.grad(loss_func_with_grad_s2(y_pred_s2[i], y_truth_s2[i]), self.model_s2.fc.parameters(), create_graph=True)
                            # print(sum(p.numel() for p in grad))
                            # print(len(grad), [grad[i].shape for i in range(len(grad))], [torch.mean(torch.abs(grad[i])) for i in range(len(grad))])
                            # print(grad)
                            grad_log1[idx_s2[i]][epoch-1] = torch.mean(torch.abs(grad[0]))
                            grad_log2[idx_s2[i]][epoch-1] = torch.mean(torch.abs(grad[1]))
                        else:
                            grad = torch.autograd.grad(loss_func_with_grad_s2(y_pred_s2[i], y_teacher_s2[i]), self.model_s2.fc.parameters(), create_graph=True)
                            # print(sum(p.numel() for p in grad))
                            # print(len(grad), [grad[i].shape for i in range(len(grad))], [torch.mean(torch.abs(grad[i])) for i in range(len(grad))])
                            # print(grad)
                            grad_log1[idx_s2[i]][epoch-1] = torch.mean(torch.abs(grad[0]))
                            grad_log2[idx_s2[i]][epoch-1] = torch.mean(torch.abs(grad[1]))
                    
                    # Update gradients
                    optimizer_s2.step()

                    # Save/show loss per step of training batches
                    pbar.set_postfix({"training error": loss_s2.item()})
                    training_log["errors"].append({"epoch": epoch, "loss": loss_s2.item()})

                    self.train_writer_s2.add_scalar("loss", loss_s2.item(), epoch)
                    self.train_writer_s2.flush()

                    # Save y_true and y_pred in lists for calculating epoch-wise scores
                    if teacher_weightage == 1 and not score_on_gnd_truth:
                        ys += list(torch.argmax(y_teacher_s2, dim=1).cpu().detach().numpy())
                    else:
                        ys += list(y_truth_s2.cpu().detach().numpy())
                    
                    if postprocess_out is not None:
                        y_pred = postprocess_out(y_pred_s2)

                    y_preds += list(torch.argmax(y_pred_s2, dim=1).cpu().detach().numpy())


            # Update learning rate as defined above
            lr_scheduler_s1.step()
            lr_scheduler_s2.step()

            if epoch == drop_epoch:
                drop_size = int((1-drop_percentage)*train_size)
                thresh = np.partition(grad_log2[:,epoch-1], drop_size)[drop_size]
                samples_bitvector[:] = (grad_log2[:,epoch-1] <= thresh)
                print("Dropping {} Samples".format(np.sum(samples_bitvector == 0)))

            # print(grad_log1[:,0])
            # print(grad_log2[:,0])

            # Save/show training scores per epoch
            training_scores = []
            if isinstance(score_functions, list) and len(score_functions) > 0:
                for score_func in score_functions:
                    score = score_func["func"](ys, y_preds)
                    training_scores.append({score_func["name"]: score})
                    self.train_writer.add_scalar(score_func["name"], score, epoch)

                self.train_writer.flush()
                print(
                    "epoch:{}, Training Scores:{}".format(epoch, training_scores),
                    flush=True,
                )
                training_log["scores"].append(
                    {"epoch": epoch, "scores": training_scores}
                )

            if epoch == 1 or epoch % validation_score_epoch == 0:

                for test_name, test_loader in self.test_loaders.items():

                    ys = []
                    y_preds = []

                    # Putting model in evaluation mode to stop calculating back gradients
                    self.model_s1.eval()
                    with torch.no_grad():
                        for x_test, y_test in tqdm(
                            test_loader, desc="Validation '{}' epoch {}".format(test_name, epoch)
                        ):
                            x = x_test.type(torch.FloatTensor).to(self.device)
                            y_truth = y_test.type(torch.LongTensor).to(self.device)

                            if teacher_weightage > 0:
                                if self.teacher_model is not None:
                                    y_teacher = self.teacher_model(x)
                                else:
                                    raise RuntimeError("Using un-specified teacher model")

                            # Predicting
                            y_pred = torch.softmax(self.model_s1(x), dim=1)

                            high_conf = torch.max(y_pred, dim=1).values >= 0.75
                            re_predict = torch.max(y_pred, dim=1).values < 0.75

                            if epoch > drop_epoch and torch.sum(re_predict) > 0:
                                y_pred[re_predict] = torch.softmax(self.model_s2(x[re_predict]), dim=1)

                                # Calculating loss
                                if teacher_weightage == 0:
                                    loss_s1 = loss_func_s1(y_pred[high_conf], y_truth[high_conf]) + loss_func_s2(y_pred[re_predict], y_truth[re_predict])
                                elif teacher_weightage == 1:
                                    loss_s1 = loss_func_s1(y_pred[high_conf], y_teacher[high_conf]) + loss_func_s2(y_pred[re_predict], y_teacher[re_predict]) 
                                else:
                                    loss_s1 = teacher_weightage * loss_func_s1(y_pred[high_conf], y_teacher[high_conf]) + (
                                        1 - teacher_weightage
                                    ) * loss_func_s1(y_pred[high_conf], y_truth[high_conf])
                                    + teacher_weightage * loss_func_s2(y_pred[re_predict], y_teacher[re_predict]) + (
                                        1 - teacher_weightage
                                    ) * loss_func_s2(y_pred[re_predict], y_truth[re_predict])
                            else:
                                # Calculating loss
                                if teacher_weightage == 0:
                                    loss_s1 = loss_func_s1(y_pred, y_truth)
                                elif teacher_weightage == 1:
                                    loss_s1 = loss_func_s1(y_pred, y_teacher)
                                else:
                                    loss_s1 = teacher_weightage * loss_func_s1(y_pred, y_teacher) + (
                                        1 - teacher_weightage
                                    ) * loss_func_s1(y_pred, y_truth)

                            # Save/show loss per batch of validation data
                            # pbar.set_postfix({"test error": loss})
                            validation_log[test_name]["errors"].append(
                                {"epoch": epoch, "loss": loss_s1.item()}
                            )
                            self.valid_writers[test_name].add_scalar("loss", loss_s1.item(), epoch)

                            # Save y_true and y_pred in lists for calculating epoch-wise scores
                            if teacher_weightage == 1 and not score_on_gnd_truth:
                                ys += list(
                                    torch.argmax(y_teacher, dim=1).cpu().detach().numpy()
                                )
                            else:
                                ys += list(y_truth.cpu().detach().numpy())
                            
                            if postprocess_out is not None:
                                y_pred = postprocess_out(y_pred)

                            y_preds += list(
                                torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                            )

                    # Save/show validation scores per epoch
                    validation_scores = []
                    if isinstance(score_functions, list) and len(score_functions) > 0:
                        for score_func in score_functions:
                            score = score_func["func"](ys, y_preds)
                            validation_scores.append({score_func["name"]: score})
                            self.valid_writers[test_name].add_scalar(score_func["name"], score, epoch)

                        self.valid_writers[test_name].flush()
                        print(
                            "epoch:{}, Validation '{}' Scores:{}".format(
                                epoch, test_name, validation_scores
                            ),
                            flush=True,
                        )
                        validation_log[test_name]["scores"].append(
                            {"epoch": epoch, "scores": validation_scores}
                        )

            # Saving models at specified checkpoints
            if save_checkpoints_epoch > 0:
                if epoch % save_checkpoints_epoch == 0:
                    chkp_path = os.path.join(
                        save_checkpoints_path,
                        self.name,
                        "checkpoints",
                        "{}".format(epoch),
                    )
                    os.makedirs(chkp_path)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model_s1.state_dict(),
                            "optimizer_state_dict": optimizer_s1.state_dict(),
                            "loss": loss_s1,
                        },
                        chkp_path + "/model_s1.pth",
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model_s2.state_dict(),
                            "optimizer_state_dict": optimizer_s2.state_dict(),
                            "loss": loss_s2,
                        },
                        chkp_path + "/model_s2.pth",
                    )

        
        return training_log, validation_log, grad_log1, grad_log2, samples_bitvector

    def save(self, save_dir_path, epoch):
        for p in self.model_s1.parameters():
            p.requires_grad = True

        for p in self.model_s2.parameters():
            p.requires_grad = True
        
        model_path = os.path.join(
            save_dir_path,
            self.name,
            "checkpoints",
            "{}".format(epoch),
        )

        os.makedirs(model_path)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model_s1.state_dict(),
            },
            model_path + "/model_s1.pth",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model_s2.state_dict(),
            },
            model_path + "/model_s2.pth",
        )

        return model_path + "/model_s1.pth", model_path + "/model_s2.pth"