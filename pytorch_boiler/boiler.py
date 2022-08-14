import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Any

from .utils import init_overload_state, is_method_overloaded
from .tracker import Tracker


class Boiler(nn.Module):

    def __init__(self, model, optimizer, train_dataloader, val_dataloader, epochs, save_path=None, load_path=None):
        super(Boiler, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        
        self.save_path = save_path
        self.load_path = load_path

        self.tracker = Tracker()

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
    def train_epoch(self):
        """Trains the model for one epoch.

        Args:
            None
        
        Returns:
            Tracker: The tracker object.
        """
        for data in tqdm(self.train_dataloader):

            self.optimizer.zero_grad()
            
            model_output = self.forward(data)
            loss = self.loss(model_output, data)
            self.tracker.update('training_loss', loss.cpu().detach().numpy())

            if is_method_overloaded(self.performance):
                perf = self.performance(model_output, data)
                self.tracker.update('training_perf', perf.cpu().detach().numpy())

            loss.backward()
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
        for data in tqdm(self.val_dataloader):
            model_output = self.infer(data)
            loss = self.losS(model_output, data)
            self.tracker.update('validation_loss', loss.cpu().detach().numpy())

            if is_method_overloaded(self.performance):
                perf = self.performance(model_output, data)
                self.tracker.update('validation_perf', perf.cpu().detach().numpy())
        return self.tracker

    @init_overload_state
    def fit(self):
        """Fits the model for the given number of epochs.

        Returns:
            Boiler: Self.
        """
        if self.load_path is not None:
            print('\nBoiler | Loading from {}'.format(self.load_path))
            loaded_object = torch.load(self.load_path)
            self.load_state_dict(loaded_object['state_dict'])
            self.tracker.load_state_dict(loaded_object['tracker'])

        for e in range(self.epochs):
            print('\nBoiler | Training epoch {}/{}...'.format(e+1, self.epochs))
            self.train_epoch()
            self.eval_epoch()
            print(self.tracker.summarize())
            self.tracker.stash()
        
        if self.save_path is not None:
            os.makedirs(os.path.abspath(os.path.dirname(self.save_path)), exist_ok=True)
            saved_object = {
                'state_dict': self.state_dict(),
                'tracker': self.tracker.state_dict()
            }
            torch.save(saved_object, self.save_path)

        return self
