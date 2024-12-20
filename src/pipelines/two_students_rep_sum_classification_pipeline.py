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
from sam.sam import SAM



class GenPipeline:
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
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

        self.model = model
        self.teacher_model = teacher_model

        self.batch_size = batch_size

        self.preprocessor = preprocessor

        self.args = kwargs

        # Set training device (CUDA-GPU / CPU)
        self.device = torch.device(
            "cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu"
        )
        print("Training Device: {}".format(self.device))
        self.model.to(self.device)

        if self.teacher_model is not None:
            self.teacher_model.to(self.device)

        # Creating dataset loader to load data parallelly
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            num_workers=workers,
            shuffle=True,
            **kwargs
        )

        # new
        self.test_loaders = {}
        for t in test_sets:
            self.test_loaders[t["name"]] = DataLoader(
                t["dataset"], batch_size=self.batch_size, num_workers=workers, **kwargs
            )

        # Create summary writers for tensorboard logs
        self.train_writer = SummaryWriter(
            os.path.join(log_files_path, self.name, "train")
        )

        # new
        self.valid_writers = {}
        for k in self.test_loaders.keys():
            self.valid_writers[k] = SummaryWriter(
                os.path.join(log_files_path, self.name, "validation_{}".format(k))
            )

    def train_rep(self):
        raise NotImplementedError()
    
    def train_cls(self):
        raise NotImplementedError()

    def save(self, save_dir_path, epoch):
        for p in self.model.parameters():
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
                "model_state_dict": self.model.state_dict(),
            },
            model_path + "/model.pth",
        )

        return model_path + "/model.pth"


class PipelineS1(GenPipeline):
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
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
        super().__init__(
            name + "_s1",
            model,
            batch_size,
            workers,
            train_set,
            test_sets,
            preprocessor,
            log_files_path,
            teacher_model,
            cuda_num,
            **kwargs
        )

    def train(
        self,
        num_epochs: int = 100,
        teacher_weightage: float = 0,
        score_on_gnd_truth: bool = True,
        lr: float = 0.001,
        score_functions: list = SCORE_FUNCTIONS_CLASSIFICATION,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        step_size_func=lambda e: 1,
        loss_func=torch.nn.functional.cross_entropy,
        loss_func_with_grad=torch.nn.CrossEntropyLoss(),
        validation_score_epoch: int = 1,
        save_checkpoints_epoch: int = -1,
        save_checkpoints_path: str = "",
        drop_percentage: float = 0,
        drop_epoch: int = -1,
        samples_bitvector: np.ndarray = None,
    ):
        self.epochs = num_epochs

        # Setting optimzer
        # optimizer = optimizer(self.model.parameters(), lr=lr)
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer = SAM(self.model.parameters(), base_optimizer)
        optimizer.param_groups[0]['lr'] = lr
        # print(optimizer.param_groups)

        # Learning rate scheduler for changing learning rate during training
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, step_size_func)

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
            print("lr: {}".format(lr_scheduler.get_last_lr()))

            # Putting model in training mode to calculate back gradients
            self.model.train()

            ys = []
            y_preds = []

            # Batch-wise optimization
            pbar = tqdm(self.train_loader, desc="Training epoch {}".format(epoch))
            for x_train, y_train, idx in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                y_truth = y_train.type(torch.LongTensor).to(self.device)

                if epoch > drop_epoch:
                    s1_selector = samples_bitvector[idx] == 1
                    x = x[s1_selector]
                    y_truth = y_truth[s1_selector]
                    idx = idx[s1_selector]
                    if epoch == drop_epoch + 1:
                        print(
                            "dropping: {}".format(np.sum(samples_bitvector[idx] == 0))
                        )
                        print("x: {}".format(x.shape))
                    if x.shape[0] == 0:
                        continue

                if teacher_weightage > 0:
                    if self.teacher_model is not None:
                        y_teacher = self.teacher_model(x)
                        # y_teacher = (y_teacher - torch.mean(y_teacher, dim=1, keepdim=True)) / torch.std(y_teacher, dim=1, keepdim=True)
                    else:
                        raise RuntimeError("Using un-specified teacher model")

                # Forward pass
                y_pred = self.model(x)
                # y_pred = (y_pred - torch.mean(y_pred, dim=1, keepdim=True)) / torch.std(y_pred, dim=1, keepdim=True)

                # Clearing previous epoch gradients
                optimizer.zero_grad()

                # Calculating loss
                if teacher_weightage == 0:
                    loss = loss_func_with_grad(y_pred, y_truth)
                elif teacher_weightage == 1:
                    loss = loss_func_with_grad(y_pred, y_teacher)
                else:
                    loss = teacher_weightage * loss_func_with_grad(
                        y_pred, y_teacher
                    ) + (1 - teacher_weightage) * loss_func_with_grad(y_pred, y_truth)

                # Backward pass to calculate gradients
                loss.backward(retain_graph=True)

                for i in range(len(idx)):
                    if teacher_weightage == 0:
                        grad = torch.autograd.grad(
                            loss_func_with_grad(y_pred[i], y_truth[i]),
                            self.model.fc.parameters(),
                            create_graph=True,
                        )
                        # print(sum(p.numel() for p in grad))
                        # print(len(grad), [grad[i].shape for i in range(len(grad))], [torch.mean(torch.abs(grad[i])) for i in range(len(grad))])
                        # print(grad)
                        grad_log1[idx[i]][epoch - 1] = torch.mean(torch.abs(grad[0]))
                        grad_log2[idx[i]][epoch - 1] = torch.mean(torch.abs(grad[1]))
                    else:
                        grad = torch.autograd.grad(
                            loss_func_with_grad(y_pred[i], y_teacher[i]),
                            self.model.fc.parameters(),
                            create_graph=True,
                        )
                        # print(sum(p.numel() for p in grad))
                        # print(len(grad), [grad[i].shape for i in range(len(grad))], [torch.mean(torch.abs(grad[i])) for i in range(len(grad))])
                        # print(grad)
                        grad_log1[idx[i]][epoch - 1] = torch.mean(torch.abs(grad[0]))
                        grad_log2[idx[i]][epoch - 1] = torch.mean(torch.abs(grad[1]))


                def sam_closure():
                    # Calculating loss
                    if teacher_weightage == 0:
                        loss = loss_func_with_grad(y_pred, y_truth)
                    elif teacher_weightage == 1:
                        loss = loss_func_with_grad(y_pred, y_teacher)
                    else:
                        loss = teacher_weightage * loss_func_with_grad(
                            y_pred, y_teacher
                        ) + (1 - teacher_weightage) * loss_func_with_grad(y_pred, y_truth)
                    loss.backward(retain_graph=True)
                    return loss

                # Update gradients
                # optimizer.step()
                optimizer.step(sam_closure)

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

                y_preds += list(torch.argmax(y_pred, dim=1).cpu().detach().numpy())

            # Update learning rate as defined above
            lr_scheduler.step()

            if epoch == drop_epoch:
                drop_size = int((1 - drop_percentage) * train_size)
                thresh = np.partition(grad_log2[:, epoch - 1], drop_size)[drop_size]
                samples_bitvector[:] = grad_log2[:, epoch - 1] <= thresh
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
                    self.model.eval()
                    with torch.no_grad():
                        for x_test, y_test in tqdm(
                            test_loader,
                            desc="Validation '{}' epoch {}".format(test_name, epoch),
                        ):
                            x = x_test.type(torch.FloatTensor).to(self.device)
                            y_truth = y_test.type(torch.LongTensor).to(self.device)

                            if teacher_weightage > 0:
                                if self.teacher_model is not None:
                                    y_teacher = self.teacher_model(x)
                                    # y_teacher = (y_teacher - torch.mean(y_teacher, dim=1, keepdim=True)) / torch.std(y_teacher, dim=1, keepdim=True)
                                else:
                                    raise RuntimeError(
                                        "Using un-specified teacher model"
                                    )

                            # Predicting
                            y_pred = self.model(x)
                            # y_pred = (y_pred - torch.mean(y_pred, dim=1, keepdim=True)) / torch.std(y_pred, dim=1, keepdim=True)

                            # Calculating loss
                            if teacher_weightage == 0:
                                loss = loss_func(y_pred, y_truth)
                            elif teacher_weightage == 1:
                                loss = loss_func(y_pred, y_teacher)
                            else:
                                loss = teacher_weightage * loss_func(
                                    y_pred, y_teacher
                                ) + (1 - teacher_weightage) * loss_func(y_pred, y_truth)

                            # Save/show loss per batch of validation data
                            # pbar.set_postfix({"test error": loss})
                            validation_log[test_name]["errors"].append(
                                {"epoch": epoch, "loss": loss.item()}
                            )
                            self.valid_writers[test_name].add_scalar(
                                "loss", loss.item(), epoch
                            )

                            # Save y_true and y_pred in lists for calculating epoch-wise scores
                            if teacher_weightage == 1 and not score_on_gnd_truth:
                                ys += list(
                                    torch.argmax(y_teacher, dim=1)
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                            else:
                                ys += list(y_truth.cpu().detach().numpy())

                            y_preds += list(
                                torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                            )

                    # Save/show validation scores per epoch
                    validation_scores = []
                    if isinstance(score_functions, list) and len(score_functions) > 0:
                        for score_func in score_functions:
                            score = score_func["func"](ys, y_preds)
                            validation_scores.append({score_func["name"]: score})
                            self.valid_writers[test_name].add_scalar(
                                score_func["name"], score, epoch
                            )

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

            # Saving model at specified checkpoints
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
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        chkp_path + "/model.pth",
                    )

        return training_log, validation_log, grad_log1, grad_log2, samples_bitvector


class PipelineS2(GenPipeline):
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
        super().__init__(
            name + "_s2",
            model_s2,
            batch_size,
            workers,
            train_set,
            test_sets,
            preprocessor,
            log_files_path,
            teacher_model,
            cuda_num,
            **kwargs
        )
        self.model_s1 = model_s1
        self.model_s1.to(self.device)

    def train(
        self,
        num_epochs: int = 100,
        teacher_weightage: float = 0,
        score_on_gnd_truth: bool = True,
        lr: float = 0.001,
        score_functions: list = SCORE_FUNCTIONS_CLASSIFICATION,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        step_size_func=lambda e: 1,
        loss_func=torch.nn.functional.cross_entropy,
        loss_func_with_grad=torch.nn.CrossEntropyLoss(),
        validation_score_epoch: int = 1,
        save_checkpoints_epoch: int = -1,
        save_checkpoints_path: str = "",
        samples_bitvector: np.ndarray = None,
    ):
        self.epochs = num_epochs

        # Setting optimzer
        # optimizer = optimizer(self.model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        optimizer = SAM(self.model.parameters(), base_optimizer)
        optimizer.param_groups[0]['lr'] = lr

        # Learning rate scheduler for changing learning rate during training
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, step_size_func)

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

        self.model_s1.include_top = False
        self.model.include_top = False
        # self.teacher_model.include_top = False

        for epoch in range(1, self.epochs + 1):
            print("lr: {}".format(lr_scheduler.get_last_lr()))

            # Putting model in training mode to calculate back gradients
            self.model.train()

            ys = []
            y_preds = []

            # Batch-wise optimization
            pbar = tqdm(self.train_loader, desc="Training epoch {}".format(epoch))
            for x_train, y_train, idx in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                y_truth = y_train.type(torch.LongTensor).to(self.device)

                s1_selector = samples_bitvector[idx] == 1
                x = x[s1_selector]
                y_truth = y_truth[s1_selector]
                idx = idx[s1_selector]
                if epoch == 1:
                    print("dropping: {}".format(np.sum(samples_bitvector[idx] == 0)))
                    print("x: {}".format(x.shape))
                if x.shape[0] == 0:
                    continue

                if teacher_weightage > 0:
                    if self.teacher_model is not None:
                        y_teacher = self.teacher_model(x)
                        # y_teacher = torch.div(y_teacher.T, torch.max(y_teacher, dim=1).values).T
                        # y_teacher = (y_teacher - torch.mean(y_teacher, dim=1, keepdim=True)) / torch.std(y_teacher, dim=1, keepdim=True)
                    else:
                        raise RuntimeError("Using un-specified teacher model")

                # Forward pass
                y_rep_s1 = self.model_s1(x)
                # y_pred_s1 = torch.div(y_pred_s1.T, torch.max(y_pred_s1, dim=1).values).T
                # y_pred_s1 = (y_pred_s1 - torch.mean(y_pred_s1, dim=1, keepdim=True)) / torch.std(y_pred_s1, dim=1, keepdim=True)

                y_rep = self.model(x)
                # y_pred = torch.div(y_pred.T, torch.max(y_pred, dim=1).values).T
                # y_pred = (y_pred - torch.mean(y_pred, dim=1, keepdim=True)) / torch.std(y_pred, dim=1, keepdim=True)
                
                # y_pred_smx = torch.log_softmax(y_pred, dim=1)
                # target = torch.log_softmax(y_teacher - y_pred_s1, dim=1)
                y_pred = self.model.rep_to_class(y_rep + y_rep_s1)


                # Clearing previous epoch gradients
                optimizer.zero_grad()

                # Calculating loss
                loss = loss_func_with_grad(y_pred, y_teacher)
                """
                if teacher_weightage == 0:
                    loss = loss_func_with_grad(y_pred, y_truth)
                elif teacher_weightage == 1:
                    loss = loss_func_with_grad(y_pred, y_teacher)
                else:
                    loss = teacher_weightage * loss_func_with_grad(
                        y_pred, y_teacher
                    ) + (1 - teacher_weightage) * loss_func_with_grad(y_pred, y_truth)
                """

                # Backward pass to calculate gradients
                loss.backward(retain_graph=True)

                for i in range(len(idx)):
                    grad = torch.autograd.grad(
                        loss_func_with_grad(y_pred[i], y_teacher[i]),
                        self.model.fc.parameters(),
                        create_graph=True,
                    )
                    # print(sum(p.numel() for p in grad))
                    # print(len(grad), [grad[i].shape for i in range(len(grad))], [torch.mean(torch.abs(grad[i])) for i in range(len(grad))])
                    # print(grad)
                    grad_log1[idx[i]][epoch - 1] = torch.mean(torch.abs(grad[0]))
                    grad_log2[idx[i]][epoch - 1] = torch.mean(torch.abs(grad[1]))

                def sam_closure():
                    # Calculating loss
                    loss = loss_func_with_grad(y_pred, y_teacher)
                    loss.backward(retain_graph=True)
                    return loss
                
                # Update gradients
                # optimizer.step()
                optimizer.step(sam_closure)

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

                y_preds += list(
                    torch.argmax(y_pred, dim=1).cpu().detach().numpy()
                )

            # Update learning rate as defined above
            lr_scheduler.step()

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
                    self.model.eval()
                    with torch.no_grad():
                        for x_test, y_test in tqdm(
                            test_loader,
                            desc="Validation '{}' epoch {}".format(test_name, epoch),
                        ):
                            x = x_test.type(torch.FloatTensor).to(self.device)
                            y_truth = y_test.type(torch.LongTensor).to(self.device)

                            if teacher_weightage > 0:
                                if self.teacher_model is not None:
                                    y_teacher = self.teacher_model(x)
                                    # y_teacher = torch.div(y_teacher.T, torch.max(y_teacher, dim=1).values).T
                                    # y_teacher = (y_teacher - torch.mean(y_teacher, dim=1, keepdim=True)) / torch.std(y_teacher, dim=1, keepdim=True)
                                else:
                                    raise RuntimeError(
                                        "Using un-specified teacher model"
                                    )

                            # S1 Forward Pass
                            y_rep_s1 = self.model_s1(x)
                            # y_pred_s1 = torch.div(y_pred_s1.T, torch.max(y_pred_s1, dim=1).values).T
                            # y_pred_s1 = (y_pred_s1 - torch.mean(y_pred_s1, dim=1, keepdim=True)) / torch.std(y_pred_s1, dim=1, keepdim=True)

                            # Predicting
                            y_rep = self.model(x)
                            # y_pred = torch.div(y_pred.T, torch.max(y_pred, dim=1).values).T
                            # std/nostd
                            # y_pred = (y_pred - torch.mean(y_pred, dim=1, keepdim=True)) / torch.std(y_pred, dim=1, keepdim=True)
                            
                            # y_pred_smx = torch.log_softmax(y_pred, dim=1)
                            # target = torch.log_softmax(y_teacher - y_pred_s1, dim=1)
                            y_pred = self.model.rep_to_class(y_rep + y_rep_s1)

                            # Calculating loss
                            loss = loss_func(y_pred, y_teacher)
                            """if teacher_weightage == 0:
                                loss = loss_func(y_pred, y_truth)
                            elif teacher_weightage == 1:
                                loss = loss_func(y_pred, y_teacher)
                            else:
                                loss = teacher_weightage * loss_func(
                                    y_pred, y_teacher
                                ) + (1 - teacher_weightage) * loss_func(y_pred, y_truth)
                            """

                            # Save/show loss per batch of validation data
                            # pbar.set_postfix({"test error": loss})
                            validation_log[test_name]["errors"].append(
                                {"epoch": epoch, "loss": loss.item()}
                            )
                            self.valid_writers[test_name].add_scalar(
                                "loss", loss.item(), epoch
                            )

                            # Save y_true and y_pred in lists for calculating epoch-wise scores
                            if teacher_weightage == 1 and not score_on_gnd_truth:
                                ys += list(
                                    torch.argmax(y_teacher, dim=1)
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                            else:
                                ys += list(y_truth.cpu().detach().numpy())

                            y_preds += list(
                                torch.argmax(y_pred, dim=1)
                                .cpu()
                                .detach()
                                .numpy()
                            )

                    # Save/show validation scores per epoch
                    validation_scores = []
                    if isinstance(score_functions, list) and len(score_functions) > 0:
                        for score_func in score_functions:
                            score = score_func["func"](ys, y_preds)
                            validation_scores.append({score_func["name"]: score})
                            self.valid_writers[test_name].add_scalar(
                                score_func["name"], score, epoch
                            )

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

            # Saving model at specified checkpoints
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
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                        },
                        chkp_path + "/model.pth",
                    )

        self.model_s1.include_top = True
        self.model.include_top = True

        return training_log, validation_log, grad_log1, grad_log2, samples_bitvector
