"""Example usage of Pytorch Boiler"""
import torch

from pytorch_boiler import Boiler, overload
from example_model import TinyResNet
from example_dataloader import cifar10_dataloader


class Trainer(Boiler):

    @overload
    def pre_process(self, data):
        images, labels = data
        return images

    @overload
    def loss(self, model_output, data):
        images, labels = data
    @overload
    def performance(self, model_output, data):
        images, labels = data
        preds = model_output.argmax(dim=-1)
        acc = (preds == labels.cuda()).float().mean()
        return acc


if __name__ == '__main__':
    model = TinyResNet(in_channels=3, hidden_channels=16, output_channels=10, num_layers=3, expansion_factor=2)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, weight_decay=1e-5)
    train_dataloader = cifar10_dataloader(root='./data/cifar10', batch_size=4, train=True, shuffle=True, drop_last=True)
    val_dataloader = cifar10_dataloader(root='./data/cifar10', batch_size=4, train=False, shuffle=False, drop_last=False)

    trainer = Trainer(model=model, optimizer=optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                      epochs=10, save_path='./state_dict.pt', load_path=None).fit()
