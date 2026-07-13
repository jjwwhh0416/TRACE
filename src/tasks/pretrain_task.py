import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from wandb import AlertLevel
from src.utils.tools import MetricsStore, dtype_map, make_dir_if_not_exists, count_parameters
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.utils.tools import gather_all_tensor_with_padding
from .base import Tasks
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from src.utils.tools import EarlyStopping
warnings.filterwarnings("ignore")


class Pretraining(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self._build_model()
        count_parameters(self.model)
    
    def validation(self, data_loader, return_preds: bool = False, split: str = "val"):
        all_preds, all_valid_labels = [], []
        loss = {"forecast_losses": [], "classification_losses": [], "total_losses": []}

        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device) #[B, C, L]
                input_mask = batch_x.input_mask.long().to(self.device) #[B, C, L]
                labels = torch.tensor(batch_x.labels, dtype=torch.long).reshape(-1).to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )
                
                if (self.args.encoder_type == "MOMENT"):
                    recon = outputs.reconstruction
                    B, C, L = timeseries.shape # [B, 7, 186]

                    if recon.shape != timeseries.shape:
                        if recon.dim() == 3 and recon.shape[1] == recon.shape[2]:
                            recon = torch.nn.functional.interpolate(recon, size=C, mode='linear', align_corners=False)
                            recon = recon.transpose(1, 2)
                        
                        elif recon.shape[1] == L and recon.shape[2] == C:
                            recon = recon.transpose(1, 2)
                        
                        elif recon.shape != timeseries.shape:
                            recon = recon.view(timeseries.shape)

                    recon_loss = self.forecast_criterion(recon, timeseries)
                else:
                    recon_loss = self.forecast_criterion(outputs.reconstruction, timeseries)  #[B, C, L]
                
                # compute loss on (input_mask = 1 & pre-train_mask = 0)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)  #[B, C, L]
                masked_loss = observed_mask * recon_loss
                forecast_loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)
                preds = outputs.classification.argmax(dim=1)

                # Support -100 padding
                valid_mask = labels != -100
                all_preds.append(preds[valid_mask])
                all_valid_labels.append(labels[valid_mask])
                
                labeled_mask =(labels != -100)
                if labeled_mask.any():
                    classification_loss = self.classification_criterion(outputs.classification, labels)  #[B, n_classes]
                    total_loss = forecast_loss + self.args.beta * classification_loss
                else:
                    classification_loss = torch.tensor(0.0, device=outputs.classification.device)
                    total_loss = forecast_loss
                    
                #### get metrics from all GPUs #####
                if self.args.world_size > 1:
                    tensor_forecast_loss = torch.tensor(forecast_loss, device=self.device)
                    dist.all_reduce(tensor_forecast_loss, op=dist.ReduceOp.SUM)
                    forecast_loss = (tensor_forecast_loss / self.args.world_size)
                    
                    tensor_classification_loss = torch.tensor(classification_loss, device=self.device)
                    dist.all_reduce(tensor_classification_loss, op=dist.ReduceOp.SUM)
                    classification_loss = (tensor_classification_loss / self.args.world_size)
                    
                    total_loss = forecast_loss + self.args.beta * classification_loss
                
              
                loss["forecast_losses"].append(forecast_loss.item())
                loss["classification_losses"].append(classification_loss.item())
                loss["total_losses"].append(total_loss.item())

        all_preds = gather_all_tensor_with_padding(torch.cat(all_preds, dim=0))
        all_valid_labels = gather_all_tensor_with_padding(torch.cat(all_valid_labels, dim=0))
        all_preds_np = all_preds.cpu().numpy()
        all_valid_labels_np = all_valid_labels.cpu().numpy()
        if self.args.rank == 0:
            accuracy = accuracy_score(all_valid_labels_np, all_preds_np)
            precision = precision_score(all_valid_labels_np, all_preds_np, average="macro", zero_division=0)
            recall = recall_score(all_valid_labels_np, all_preds_np, average="macro", zero_division=0)
            f1 = f1_score(all_valid_labels_np, all_preds_np, average="macro", zero_division=0)    
            self.logger.log({
                f"{split}_accuracy": accuracy,
                f"{split}_precision": precision,
                f"{split}_recall": recall,
                f"{split}_f1": f1
            })
        
        average_total_loss = np.average(np.array(loss["total_losses"]))
        average_forecast_loss = np.average(np.array(loss["forecast_losses"]))
        average_classification_loss = np.average(np.array(loss["classification_losses"]))
        average_losses = {f"{split}_total_loss": average_total_loss, f"{split}_recon_loss": average_forecast_loss, f"{split}_classification_loss": average_classification_loss}
        self.model.train()
        return average_losses

    def train(self):
        if self.args.rank == 0:
            self.run_name = self.logger.name
            path = os.path.join("./checkpoints/", self.run_name)
            make_dir_if_not_exists(path, verbose=True)

        self.optimizer = self._select_optimizer()
        self.forecast_criterion = self._select_criterion()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler()
        self.model.to(self.args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.rank], find_unused_parameters=True )
        self.early_stopping = EarlyStopping(patience=self.args.patience, delta=self.args.delta)
        # self.evaluate_model()
        
        
        opt_steps = 0
        cur_epoch = 0
        while cur_epoch < self.args.max_epoch:
            print(f"Epoch {cur_epoch} of {self.args.max_epoch}")
            self.model.train()
            if self.args.distributed and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(cur_epoch)
                
            accumulation_steps = 4  # 64(배치) x 4(누적) = 256 효과
            self.optimizer.zero_grad(set_to_none=True) # 루프 시작 전 초기화

            for i, batch_x in enumerate(tqdm(
                self.train_dataloader, total=len(self.train_dataloader)
            )):
                # self.optimizer.zero_grad(set_to_none=True) <- 기존의 이 줄은 삭제합니다!
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)  #[B, C, L]
                labels = torch.tensor(batch_x.labels, dtype=torch.long).reshape(-1).to(self.device)
                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(x_enc=timeseries, input_mask=input_mask)

                    if (self.args.encoder_type == "MOMENT"):
                        recon = outputs.reconstruction
                        B, C, L = timeseries.shape

                        if recon.shape != timeseries.shape:
                            if recon.dim() == 3 and recon.shape[1] == recon.shape[2]:
                                recon = torch.mean(recon, dim=1, keepdim=True).repeat(1, C, 1)
                            
                            elif recon.shape[1] == L and recon.shape[2] == C:
                                recon = recon.transpose(1, 2)
                            
                            if recon.shape != timeseries.shape:
                                recon = recon.view(B, C, L)

                        recon_loss = self.forecast_criterion(recon, timeseries)  #[B, C, L]
                    else:
                        recon_loss = self.forecast_criterion(outputs.reconstruction, timeseries)  #[B, C, L]
                    
                    observed_mask = input_mask * (1 - outputs.pretrain_mask)  #[B, C, L]
                    masked_loss = observed_mask * recon_loss  #[B, C, L]
                    recon_loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)  #[B, C, L]
                    labeled_mask = (labels != -100)
                    if labeled_mask.any():
                        classification_loss = self.classification_criterion(
                            outputs.classification, labels
                        )
                        print("classification_loss:", classification_loss.item())
                    else:
                        classification_loss = 0.0 * outputs.classification.sum()
                    
                    total_loss = recon_loss + self.args.beta * classification_loss
                
                if self.args.rank == 0:
                    self.logger.log(
                        {
                            "train_total_loss": total_loss.item(),
                            "train_recon_loss": recon_loss.item(),
                            "train_classification_loss": classification_loss.item(),
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )
                    print(f"Epoch: {cur_epoch} | Total: {total_loss.item():.4f} | Recon: {recon_loss.item():.4f} | Class: {classification_loss.item():.4f}")

                if self.args.debug and opt_steps >= 1:
                    self.debug_model_outputs(total_loss, outputs, batch_x)

                # 1. 4번 누적할 것이므로 Loss를 4로 나누어 평균을 유지합니다.
                loss = total_loss / accumulation_steps
                loss.backward()

                # 2. 4번째 스텝이거나, 데이터로더의 마지막 배치일 때만 업데이트를 진행합니다.
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_dataloader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True) # 업데이트 후 기울기 초기화
                    
                    opt_steps = opt_steps + 1
                    self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)

            if cur_epoch % self.args.log_interval == 0:
                if self.args.distributed and isinstance(self.val_dataloader.sampler, DistributedSampler):
                    self.val_dataloader.sampler.set_epoch(cur_epoch)
                    self.val_dataloader.sampler.set_epoch(cur_epoch)
                eval_metrics =self.evaluate_and_log()
                
                if self.args.rank == 0 and self.args.early_stop == True:
                    self.early_stopping(eval_metrics.val_loss["val_total_loss"], self.model, path=os.path.join(path, "best_checkpoint.pth"))

                stop_signal = torch.tensor(float(self.early_stopping.early_stop), device=self.device)
                dist.broadcast(stop_signal, src=0)

                if stop_signal.item() > 0:
                    break
            
            #### save checkpoints every epoch #####
            if cur_epoch % self.args.checkpoint_interval == 0:
                ## save checkpoints every epoch
                if self.args.rank == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                        )
                    self.save_model(model=self.model, path=path, opt_steps=None, optimizer=self.optimizer, scaler=self.scaler)
            ########################################
            
            cur_epoch = cur_epoch + 1

        return self.model

    def evaluate_and_log(self):
        eval_metrics = MetricsStore(val_loss=self.validation(self.val_dataloader, split="val"), test_loss=self.validation(self.test_dataloader, split="test"))
        if self.args.rank == 0:
            self.logger.log(eval_metrics.val_loss)
            self.logger.log(eval_metrics.test_loss)
        return eval_metrics

