import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from typing import Any

from .utils import init_overload_state, is_method_overloaded, prettify_dict
from .tracker import Tracker


class Boiler(nn.Module):

    def __init__(self, model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, patience=None, save_path=None, load_path=None, mixed_precision=False):
        super(Boiler, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.patience = patience
        
        self.save_path = save_path
        self.load_path = load_path

        self.mixed_precision = mixed_precision

        # Initialize tracker
        self.tracker = Tracker()
        self.patience_counter = 0

        # Initialize mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        
        # Load model state
        self.start_epoch = 0
        self.best_validation_loss = float('inf')
        if self.load_path is not None:
            print('\nBoiler | Loading from {}'.format(self.load_path))
            loaded_object = torch.load(self.load_path)
            self.model.load_state_dict(loaded_object['model_state_dict'])
            self.optimizer.load_state_dict(loaded_object['optimizer_state_dict'])
            self.scaler.load_state_dict(loaded_object['scaler_state_dict'])
            self.scheduler.load_state_dict(loaded_object['scheduler_state_dict'])
            self.tracker.load_state_dict(loaded_object['tracker'])
            self.start_epoch = loaded_object['epoch'] + 1
            self.best_validation_loss = loaded_object['best_validation_loss']

    @init_overload_state
    def pre_process(self, x):
        """Computes the pre processing steps. Must be overloaded for usage.

        Args:
            x (Any): The input to the pre processor

        Raises:
            NotImplementedError: If this function is not implemented and is called.
        """
        raise NotImplementedError()
    
    @init_overload_state
    def post_process(self, x):
        """Computes the post processing steps. Must be overloaded for usage.

        Args:
            x (Any): The input to the post processor (output from the model forward pass).

        Raises:
            NotImplementedError: If this function is not implemented and is called.
        """
        raise NotImplementedError()

    @init_overload_state
    def forward(self, x):
        """Computes the forward pass, with additional pre and post processing.

        Args:
            x (torch.Tensor|Any): The tensor input to the model forward pass, or input to the pre processor.

        Returns:
            torch.Tensor|Any: The tensor output of the model, or the output from the post processor.
        """
        if is_method_overloaded(self.pre_process):
            x = self.pre_process(x)
        with torch.autocast('cuda', dtype=torch.float16, enabled=(self.mixed_precision and self.training)):
            x = self.model(x)
        if is_method_overloaded(self.post_process):
            x = self.post_process(x)
        return x

    @init_overload_state
    def loss(self, output, data):
        """Computes the loss. Must be overloaded for usage.

        Args:
            output(torch.Tensor|Any): The output from the model forward pass, or output from the post procesor.
            data(torch.Tensor|Any): The data batch loaded from the dataloader.

        Raises:
            NotImplementedError: If this function is not implemented and is called.
        """
        raise NotImplementedError()

    @init_overload_state
    def performance(self, output, data):
        """Computes performance metric. Must be overloaded for usage.

        Args:
            output (torch.Tensor|Any): The output from the model forward pass, or output from the post processor.
            data (torch.Tensor|Any): The data batch loaded from the dataloader.

        Returns:
            torch.Tensor|Any: The performance metric computed on the current batch.
        """
        raise NotImplementedError()

    @init_overload_state
    def infer(self, x):
        """Computes the inference in eval mode.

        Args:
            x (torch.Tensor|Any): The tensor input to the model forward pass, or input to the pre processor.

        Returns:
            torch.Tensor|Any: The tensor output of the model, or the output from the post processor.
        """
        # If model is in training mode, convert to eval
        training_state_true = self.training
        if training_state_true:
            self.eval()

        with torch.no_grad():
            output = self.forward(x)
        
        # If model was in training mode, convert back to train from eval
        if training_state_true:
            self.train()

        return output

    @init_overload_state
    def decode_item_type(self, item, require_summary=True):
        """Decodes the type of the item. This function is used to handle multiple metrics

        Args:
            item (torch.Tensor|np.ndarray|dict): Either a tensor describing a loss/metric, or a dictionary of named tensors.
            require_summary (bool): If True then the item of type dictionary will be enforced to have a key 'summary' if more than one keys are present.

        Returns:
            dict: A dictionary with named tensors, where "summary" is a special key that specifies the loss to be backpropagated / metric to be tracked.
        """
        assert type(item) in [torch.Tensor, np.ndarray, dict], f"Either a tensor, numpy array, or a dictionary of (tag: value) must be passed. Given {type(item)}."
        if type(item) == torch.Tensor or type(item) == np.ndarray:
            return {'summary': item}
        elif type(item) == dict:
            keys = list(item.keys())
            if len(keys) == 1:
                item['summary'] = item[keys[0]]
            if require_summary:
                assert 'summary' in item.keys(), "If a dictionary is returned from loss / performance, it must contain the key called 'summary' indicating the loss to backpropagate / metric to track."
            return item

    @init_overload_state
    def train_epoch(self):
        """Trains the model for one epoch.

        Args:
            None
        
        Returns:
            Tracker: The tracker object.
        """
        self.train()  # Set the mode to training
        for data in tqdm(self.train_dataloader):
            # Set optimizer zero grad
            self.optimizer.zero_grad()

            # Compute model output
            model_output = self.forward(data)

            # Compute loss & Update tracker
            loss = self.loss(model_output, data)
            decoded_loss = self.decode_item_type(loss, require_summary=True)
            self.tracker.update(key='training_loss', 
                                value=decoded_loss['summary'].cpu().detach().numpy())
            for key in decoded_loss:
                if key != 'summary':
                    self.tracker.update(key='training_{}'.format(key), 
                                        value=decoded_loss[key].cpu().detach().numpy())

            # Compute performance and update tracker
            if is_method_overloaded(self.performance):
                perf = self.performance(model_output, data)
                decoded_perf = self.decode_item_type(perf, require_summary=True)
                self.tracker.update(key='training_perf', 
                                    value=decoded_perf['summary'].cpu().detach().numpy() \
                                        if type(decoded_perf['summary'])==torch.Tensor \
                                        else decoded_perf['summary'])
                for key in decoded_perf:
                    if key != 'summary':
                        self.tracker.update(key='training_{}'.format(key), 
                                            value=decoded_perf[key].cpu().detach().numpy() \
                                                  if type(decoded_perf[key])==torch.Tensor \
                                                  else decoded_perf[key])

            if self.mixed_precision:
                # Scale the loss
                scaled_loss = self.scaler.scale(decoded_loss['summary'])
                # Backpropagate loss
                scaled_loss.backward()
                # Update model parameters
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Backpropagate loss
                decoded_loss['summary'].backward()
                # Update model parameters
                self.optimizer.step()
        return self.tracker

    @init_overload_state
    def eval_epoch(self):
        """Evaluates the model for one epoch.

        Args:
            None
        
        Returns:
            Tracker: The tracker object.
        """
        self.eval()  # Set the mode to evaluation
        for data in tqdm(self.val_dataloader):
            # Compute model output
            model_output = self.infer(data)

            # Compute loss & Update tracker
            loss = self.loss(model_output, data)
            decoded_loss = self.decode_item_type(loss, require_summary=True)
            self.tracker.update('validation_loss', decoded_loss['summary'].cpu().detach().numpy())
            for key in decoded_loss:
                if key != 'summary':
                    self.tracker.update('validation_{}'.format(key), decoded_loss[key].cpu().detach().numpy())

            # Compute performance & Update tracker
            if is_method_overloaded(self.performance):
                perf = self.performance(model_output, data)
                decoded_perf = self.decode_item_type(perf, require_summary=True)
                self.tracker.update('validation_perf', decoded_perf['summary'].cpu().detach().numpy() if type(decoded_perf['summary'])==torch.Tensor else decoded_perf['summary'])
                for key in decoded_perf:
                    if key != 'summary':
                        self.tracker.update('validation_{}'.format(key), decoded_perf[key].cpu().detach().numpy() if type(decoded_perf[key])==torch.Tensor else decoded_perf[key])

        return self.tracker

    @init_overload_state
    def fit(self):
        """Fits the model for the given number of epochs.

        Returns:
            Boiler: Self.
        """
        best_validation_loss = self.best_validation_loss
        start_epoch = self.start_epoch

        for e in range(start_epoch, self.epochs):
            print('\nBoiler | Training epoch {}/{}...'.format(e+1, self.epochs))
            self.train_epoch()
            self.eval_epoch()
            self.scheduler.step()
            summary = self.tracker.summarize()
            print(prettify_dict(summary))
            self.tracker.stash()

            if summary['validation_loss'] < best_validation_loss:
                self.patience_counter = 0
                to_save = True
            else:
                self.patience_counter += 1
                to_save = False
        
            if self.save_path is not None and to_save:
                best_validation_loss = summary['validation_loss']
                print('Saving training state at {}'.format(self.save_path))
                os.makedirs(os.path.abspath(os.path.dirname(self.save_path)), exist_ok=True)
                saved_object = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'tracker': self.tracker.state_dict(),
                    'epoch': e,
                    'best_validation_loss': best_validation_loss
                }
                torch.save(saved_object, self.save_path)
            
            if (self.patience is not None) and (self.patience_counter > self.patience):
                print('Maximum patience reached. Stopping training.')
                break

        return self
