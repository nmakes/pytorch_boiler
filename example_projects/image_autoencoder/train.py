"""Example usage of Pytorch Boiler"""
import torch

from pytorch_boiler import Boiler, overload
from .model import TinyResNetAE
from .dataloader import mnist_dataloader, cifar10_dataloader


class Trainer(Boiler):

    @overload
    def pre_process(self, data):
        images, labels = data
        return images.cuda()

    @overload
    def loss(self, model_output, data):  # Overload the loss function
        images, labels = data
        latent_encoding, predicted_images = model_output
        l2_loss = ((images.cuda() - predicted_images) ** 2).mean()
        return l2_loss  # Can return a tensor, or a dictinoary like {'xe_loss': xe_loss} with multiple losses. See README.

    @overload
    def performance(self, model_output, data):
        images, labels = data
        latent_encoding, predicted_images = model_output
        diff = (images.cuda() - predicted_images) ** 2
        reconstruction_error = diff.mean()
        max_error = diff.mean(dim=(1, 2, 3)).amax(dim=0)
        min_error = diff.mean(dim=(1, 2, 3)).amin(dim=0)
        return_values = {
            'avg_reconstruction_error': reconstruction_error,
            'avg_max_error': max_error,
            'avg_min_error': min_error,
            'summary': reconstruction_error
        }
        return return_values  # Can return a tensor, or a dictinoary like {'acc': acc} with multiple metrics. See README.


if __name__ == '__main__':
    dataloader = mnist_dataloader; dataset = 'mnist'; in_channels = 1; batch_size = 256; exp_tag='autoencoder_mnist'
    # dataloader = cifar10_dataloader; dataset = 'cifar10'; in_channels = 3; batch_size = 256; exp_tag='autoencoder_cifar10'
    model = TinyResNetAE(in_channels=in_channels, hidden_channels=4, expansion_factor=2, latent_image_size=(4, 4)).cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_dataloader = dataloader(root=f'./data/{dataset}', batch_size=batch_size, train=True, shuffle=True, drop_last=True)
    val_dataloader = dataloader(root=f'./data/{dataset}', batch_size=batch_size, train=False, shuffle=False, drop_last=False)

    trainer = Trainer(model=model, optimizer=optimizer, train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                      epochs=10, save_path=f'./model/{exp_tag}/state_dict.pt', load_path=None, mixed_precision=False)
    trainer.fit()
