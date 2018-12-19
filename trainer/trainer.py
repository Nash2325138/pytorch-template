import numpy as np
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from utils.util import get_lr


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, losses, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None,
                 show_all_loss=False):
        super(Trainer, self).__init__(model, losses, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.show_all_loss = show_all_loss

    def _eval_metrics(self, data, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data, output)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        np.random.seed()
        self.model.train()
        self.logger.info(f'Current lr:{get_lr(self.optimizer)}')

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            for key in data.keys():
                data[key] = data[key].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            losses = []
            for loss_name, (loss_instance, loss_weight) in self.losses.items():
                if loss_weight <= 0.0:
                    continue
                loss = loss_instance(data, output) * loss_weight
                losses.append(loss)
                self.writer.add_scalar(f'{loss_name}', loss.item())
            loss = sum(losses)
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('total_loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(data, output)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                self.writer.add_image('input', make_grid(data['frame'][0].cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                for key in data.keys():
                    data[key] = data[key].to(self.device)
                output = self.model(data)

                losses = []
                for loss_name, (loss_instance, loss_weight) in self.losses.items():
                    if loss_weight <= 0.0:
                        continue
                    loss = loss_instance(data, output) * loss_weight
                    losses.append(loss)
                    self.writer.add_scalar(f'{loss_name}', loss.item())
                loss = sum(losses)

                self.writer.add_scalar('total_loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(data, output)
                self.writer.add_image('input', make_grid(data['frame'][0].cpu(), nrow=8, normalize=True))

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
