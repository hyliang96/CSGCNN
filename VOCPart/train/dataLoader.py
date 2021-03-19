import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import sys
import os
here = os.path.dirname(os.path.realpath(__file__)) # get absoltae path to the dir this file is in
sys.path.append(os.path.join(here,'..',  'VOC'))
from VOCPart import VOCPart
from seed import set_work_init_fn


VOC_path = os.path.join(here,'..',  '__data__')

class DataLoader():
    def __init__(self,dataset, batch_size, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

    def load_data(self, img_size=32):
        data_dir = '/home/zengyuyuan/data/CIFAR10'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }
        if self.dataset == 'cifar-10':
            data_train = datasets.CIFAR10(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR10(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)
        if self.dataset == 'cifar-100':
            data_train = datasets.CIFAR100(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR100(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)
        if self.dataset == 'mnist':
            data_dir = '/home/zengyuyuan/data/MNIST'
            mnist_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307,], [0.3081])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307, ], [0.3081])
                ])
            }
            data_train = datasets.MNIST(root=data_dir,
                                        transform = mnist_transforms['train'],
                                        train=True,
                                        download=True)
            data_test = datasets.MNIST(root=data_dir,
                                       transform=mnist_transforms['val'],
                                       train=False,
                                       download=True)
        if self.dataset == 'VOCpart':
            data_train = VOCPart(VOC_path, train=True ,requires=['img'], size=img_size)
            data_test = VOCPart(VOC_path, train=False, requires=['img'], size=img_size)

        image_datasets = {'train': data_train, 'val': data_test}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                                           batch_size=self.batch_size, pin_memory=True,
                                                           shuffle=True, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)
        dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                         batch_size=self.batch_size, pin_memory=True,
                                                         shuffle=False, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        return dataloaders,dataset_sizes




