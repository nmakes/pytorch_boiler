"""Example usage of Pytorch Boiler"""
import torch

from pytorch_boiler import Boiler, overload
from .model import TinyResNet
from .dataloader import mnist_dataloader, cifar10_dataloader


class Trainer(Boiler):

    @overload
    def pre_process(self, data):
        images, labels = data
        return images.cuda()

    @overload
    def loss(self, model_output, data):  # Overload the loss function
        images, labels = data
        xe_loss = torch.nn.functional.cross_entropy(model_output, labels.cuda(), reduction='mean')
        return xe_loss  # Can return a tensor, or a dictinoary like {'xe_loss': xe_loss} with multiple losses. See README.

    @overload
    def performance(self, model_output, data):
        images, labels = data
        preds = model_output.argmax(dim=-1)
        acc = (preds == labels.cuda()).float().mean()
        return acc.cpu().detach().numpy()  # Can return a tensor, or a dictinoary like {'acc': acc} with multiple metrics. See README.


if __name__ == '__main__':
    # dataloader = mnist_dataloader; dataset = 'mnist'; in_channels = 1; batch_size = 256; exp_tag='classifier_mnist'
    dataloader = cifar10_dataloader; dataset = 'cifar10'; in_channels = 3; batch_size = 256; exp_tag='classifier_cifar10'
    model = TinyResNet(in_channels=in_channels, hidden_channels=4, output_channels=10, num_layers=3, expansion_factor=2).cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=1)
    train_dataloader = dataloader(root=f'./data/{dataset}', batch_size=batch_size, train=True, shuffle=True, drop_last=True)
    val_dataloader = dataloader(root=f'./data/{dataset}', batch_size=batch_size, train=False, shuffle=False, drop_last=False)

    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, 
                      train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                      epochs=10, save_path=f'./model/{exp_tag}/state_dict.pt', load_path=None, mixed_precision=True)
    trainer.fit()
