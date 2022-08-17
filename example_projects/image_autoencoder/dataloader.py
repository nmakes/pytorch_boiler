from torch.utils.data import DataLoader
import torchvision.datasets as tvd
import torchvision.transforms as tvt


def cifar10_dataloader(root='./data', train=True, batch_size=4, drop_last=True, shuffle=True, num_workers=4):
    if train:
        transforms = tvt.Compose([
            tvt.RandomHorizontalFlip(),
            tvt.RandomResizedCrop(size=32, scale=(0.08, 1)),
            tvt.ToTensor(),
            tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transforms = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    loader = DataLoader(
        dataset=tvd.CIFAR10(root=root, train=train, transform=transforms, download=True),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
    )

    return loader


def mnist_dataloader(root='./data', train=True, batch_size=4, drop_last=True, shuffle=True, num_workers=4):
    if train:
        transforms = tvt.Compose([
            tvt.Resize((32, 32)),
            tvt.ToTensor(),
            tvt.Normalize(0.5, 0.5)
        ])
    else:
        transforms = tvt.Compose([
            tvt.Resize((32, 32)),
            tvt.ToTensor(),
            tvt.Normalize(0.5, 0.5)
        ])

    loader = DataLoader(
        dataset=tvd.MNIST(root=root, train=train, transform=transforms, download=True),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
    )

    return loader
