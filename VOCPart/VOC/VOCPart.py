import torch.utils.data as data
import numpy as np
import torch
import os
from torchvision.transforms import transforms
from PIL import Image
import glob

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class VOCPart(data.Dataset):
    def __init__(self, root_dir, train=True, transform=None, requires=['img'], size=64):
        assert size in [32, 64, 128]
        assert set(requires) <= set(['img','obj_mask','part_mask'])

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(size, padding=int(size/8)),
            transforms.RandomHorizontalFlip(),
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        self.val_transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

        self._ToTensor = transforms.ToTensor()


        self.root_dir = root_dir
        self.train = train
        self.transform = transform or ( self.train_transform if train else self.val_transform)
        self.requires  = requires
        self.classes = ['bird', 'cat',  'dog', 'cow','horse', 'sheep']


        self.metadata_file = os.path.join(self.root_dir, 'processed', 'metadata', 'train.txt' if train else 'val.txt' )
        with open(self.metadata_file,'r') as f:
            self.img_name_list = list(filter(lambda x:x!='', f.read().split('\n')))

        self.imgs, self.labels, self.obj_masks, self.part_masks = [], [], [], []
        self.processed_dir = os.path.join(self.root_dir, 'processed', '%dx%d' % (size, size) )
        self.obj_img_dir = os.path.join(self.processed_dir, 'obj_img')

        for file_name in os.listdir(self.obj_img_dir):
            class_name, obj_id, tmp = file_name.split('-')
            img_name, _ = tmp.split('.')
            if img_name not in self.img_name_list:
                continue
            file_name_body = file_name.replace('.jpg', '')

        # if 'img' in self.requires:
            label = self.classes.index(class_name) # id from 0 to 5
            img_path = os.path.join(self.obj_img_dir, file_name)
            self.imgs.append(img_path)
            self.labels.append(label)
        # if 'obj_mask' in self.requires:
            obj_mask_path = os.path.join(self.processed_dir, 'obj_mask', file_name_body+'.mask.bmp')
            self.obj_masks.append(obj_mask_path)
        # if 'part_mask' in self.requires:
            part_mask_dir = os.path.join(self.processed_dir, 'part_mask')
            part_mask_files = os.listdir(os.path.join(part_mask_dir,file_name_body))
            part_names = [file.split('.')[0] for file in part_mask_files]
            part_mask_paths = [os.path.join(part_mask_dir,file_name_body, part_mask_file) for part_mask_file in part_mask_files]
            part_mask_dict = {part_name:part_mask_path for part_name,part_mask_path in zip(part_names, part_mask_paths)}
            self.part_masks.append(part_mask_dict)

        print('\ntrain set' if train else 'val set')
        print('    image size %dx%d' % (size, size) )
        print('    image num = %d' % self.__len__())

    def __len__(self):
        return len(self.imgs)

    def img_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def mask_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L')

    def __getitem__(self, index):
        result = []
        if 'img' in self.requires:
            img_path, label = self.imgs[index], int(self.labels[index])

            img = self.img_loader(img_path)
            img = self.transform(img) # torch.Size([3, 128, 128]) torch.float32

            result.append(img)
            result.append(label)

        if 'obj_mask' in self.requires:
            obj_mask_path = self.obj_masks[index]
            obj_mask = self.mask_loader(obj_mask_path)
            obj_mask = self._ToTensor(obj_mask) # (max= tensor(1.), min = tensor(0.), torch.Size([1, 128, 128]), torch.float32)
            result.append(obj_mask)

        if 'part_mask' in self.requires:
            part_mask_dict = {}
            for part_name, part_mask_path in self.part_masks[index].items():
                part_mask = self.mask_loader(part_mask_path)
                part_mask = self._ToTensor(part_mask) # (max= tensor(1.), min=tensor(0.), torch.Size([1, 128, 128]), torch.float32)
                part_mask_dict[part_name] = part_mask
            result.append(part_mask_dict)

        return tuple(result)
