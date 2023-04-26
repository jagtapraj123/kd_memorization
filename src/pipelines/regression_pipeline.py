import torch

from utils.preprocessing import Preprocessor

import os
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from utils.constants import SCORE_FUNCTIONS_CLASSIFICATION


class Pipeline:
    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        batch_size: int,
        workers: int,
        train_set: Dataset,
        test_set: Dataset,
        preprocessor: Preprocessor,
        log_files_path: str,
        teacher_model: torch.nn.Module = None,
        **kwargs
    ):
        self.name = name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

        self.model = model
        self.teacher_model = teacher_model

        self.batch_size = batch_size

        self.preprocessor = preprocessor

        self.args = kwargs

        # Set training device (CUDA-GPU / CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.test_loader = DataLoader(
            test_set, batch_size=self.batch_size, num_workers=workers, **kwargs
        )

        # Create summary writers for tensorboard logs
        self.train_writer = SummaryWriter(
            os.path.join(log_files_path, self.name, "train")
        )
        self.valid_writer = SummaryWriter(
            os.path.join(log_files_path, self.name, "validation")
        )

    def train(
        self,
        num_epochs: int = 100,
        teacher_weightage: float = 0,
        lr: float = 0.001,
        score_functions: list = SCORE_FUNCTIONS_CLASSIFICATION,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        step_size_func=lambda e: 1,
        loss_func=torch.nn.functional.cross_entropy,
        loss_func_with_grad=torch.nn.CrossEntropyLoss,
        postprocess_out = None,
        validation_score_epoch: int = 1,
        save_checkpoints_epoch: int = -1,
        save_checkpoints_path: str = "",
    ):

        self.epochs = num_epochs

        # Setting optimzer
        optimizer = optimizer(self.model.parameters(), lr=lr)

        # Learning rate scheduler for changing learning rate during training
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, step_size_func)

        training_log = {"errors": [], "scores": []}
        validation_log = {"errors": [], "scores": []}

        # Training
        # pbar = tqdm(range(self.epochs), desc="Training epoch")
        for epoch in range(1, self.epochs + 1):
            print("lr: {}".format(lr_scheduler.get_last_lr()))

            # Putting model in training mode to calculate back gradients
            self.model.train()

            ys = []
            y_preds = []

            # Batch-wise optimization
            pbar = tqdm(self.train_loader, desc="Training epoch {}".format(epoch))
            for x_train, y_train in pbar:
                x = x_train.type(torch.FloatTensor).to(self.device)
                y_truth = y_train.type(torch.FloatTensor).to(self.device)

                if teacher_weightage > 0:
                    if self.teacher_model is not None:
                        y_teacher = self.teacher_model(x)
                    else:
                        raise RuntimeError("Using un-specified teacher model")

                # Forward pass
                y_pred = self.model(x)

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
                loss.backward()

                # Update gradients
                optimizer.step()

                # Save/show loss per step of training batches
                pbar.set_postfix({"training error": loss.item()})
                training_log["errors"].append({"epoch": epoch, "loss": loss.item()})

                self.train_writer.add_scalar("loss", loss.item(), epoch)
                self.train_writer.flush()

                # Save y_true and y_pred in lists for calculating epoch-wise scores
                if teacher_weightage == 1:
                    ys += list(torch.squeeze(y_teacher).cpu().detach().numpy())
                else:
                    ys += list(torch.squeeze(y_truth).cpu().detach().numpy())
                
                if postprocess_out is not None:
                    y_pred = postprocess_out(y_pred)
                    
                y_preds += list(torch.squeeze(y_pred).cpu().detach().numpy())

            # Update learning rate as defined above
            lr_scheduler.step()

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
                ys = []
                y_preds = []

                # Putting model in evaluation mode to stop calculating back gradients
                self.model.eval()
                with torch.no_grad():
                    for x_test, y_test in tqdm(
                        self.test_loader, desc="Validation epoch {}".format(epoch)
                    ):
                        x = x_test.type(torch.FloatTensor).to(self.device)
                        y_truth = y_test.type(torch.FloatTensor).to(self.device)

                        if teacher_weightage > 0:
                            if self.teacher_model is not None:
                                y_teacher = self.teacher_model(x)
                            else:
                                raise RuntimeError("Using un-specified teacher model")

                        # Predicting
                        y_pred = self.model(x)

                        # Calculating loss
                        if teacher_weightage == 0:
                            loss = loss_func(y_pred, y_truth)
                        elif teacher_weightage == 1:
                            loss = loss_func(y_pred, y_teacher)
                        else:
                            loss = teacher_weightage * loss_func(y_pred, y_teacher) + (
                                1 - teacher_weightage
                            ) * loss_func(y_pred, y_truth)

                        # Save/show loss per batch of validation data
                        # pbar.set_postfix({"test error": loss})
                        validation_log["errors"].append(
                            {"epoch": epoch, "loss": loss.item()}
                        )
                        self.valid_writer.add_scalar("loss", loss.item(), epoch)

                        # Save y_true and y_pred in lists for calculating epoch-wise scores
                        if teacher_weightage == 1:
                            ys += list(torch.squeeze(y_teacher).cpu().detach().numpy())
                        else:
                            ys += list(torch.squeeze(y_truth).cpu().detach().numpy())
                        
                        if postprocess_out is not None:
                            y_pred = postprocess_out(y_pred)

                        y_preds += list(torch.squeeze(y_pred).cpu().detach().numpy())

                # Save/show validation scores per epoch
                validation_scores = []
                if isinstance(score_functions, list) and len(score_functions) > 0:
                    for score_func in score_functions:
                        score = score_func["func"](ys, y_preds)
                        validation_scores.append({score_func["name"]: score})
                        self.valid_writer.add_scalar(score_func["name"], score, epoch)

                    self.valid_writer.flush()
                    print(
                        "epoch:{}, Validation Scores:{}".format(
                            epoch, validation_scores
                        ),
                        flush=True,
                    )
                    validation_log["scores"].append(
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

        return training_log, validation_log
