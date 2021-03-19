# %%
gauss_radius = 30
gradmap_threshold = 1
ap_thread = 0.2
per_filter = True
n_top_channel = 30

num_channel = 2048
if_show = False
showimg_id = 5

gpuid = 0

# %%
import sys, os, torch, torchvision
import numpy as np
import torch
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here,'..', 'train'))

from resnet_std import resent as resnet_std
from resnet_std import ResNet
if per_filter:
    ResNet.forward = ResNet.get_feature_map

sys.path.append(os.path.join(here,'..',  'VOC'))
from VOCPart import VOCPart
from torch.nn.functional import softmax
from tqdm import tqdm


import matplotlib.pyplot as plt
import math
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter
import kornia

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
device_str = "cuda:0"
device = torch.device(device_str)




# %%

checkpoint_path = \
{
    'CSG': os.path.join(here,'..',  'checkpoints', 'CSG.pt'),
    'STD': os.path.join(here,'..',  'checkpoints', 'STD.pt')
}

# checkpoint_path = \
# {
#     'CSG': '/raid/haoyu/result/SLGCNN_VOC_new/VOCPart_128x128_pretrained/res152_bs32_adam_lr1e-5_lrreg1e-3_lmd1e-3_warm10_mask-epoch_mod3geq2_learnable-final-STD5-CSG4-layers_cudnn-slow/seed12/CSG/run0/checkpoints/epoch_150.pt',
#     'STD': '/raid/haoyu/result/SLGCNN_VOC_new/VOCPart_128x128_pretrained/res152_bs32_adam_lr1e-5_lrreg1e-3_lmd1e-3_warm10_mask-epoch_mod3geq2_learnable-final-STD5-CSG4-layers_cudnn-slow/seed12/STD/run0/checkpoints/epoch_150.pt'
# }

# VOC dataset preprocesed
VOC_path = os.path.join(here,'..',  '__data__')

model = {}
for method in ['CSG', 'STD']:
    ifmask = method == 'CSG'
    model[method] = resnet_std(depth=152, num_classes=6, pretrained=True, ifmask=ifmask)
    model[method] = model[method].to(device)
    model[method] = torch.nn.DataParallel(model[method]) # device_ids=args.gpu_ids
    unfinished_model_path = os.path.join(checkpoint_path[method])
    checkpoint = torch.load(unfinished_model_path,map_location={'cuda:0': device_str})
    model[method].load_state_dict(checkpoint['model_state_dict'])
    model[method].eval()

# %%

dataset = VOCPart(VOC_path , train=False, requires=['img','obj_mask'], size=128)

dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=64, pin_memory=True,
                shuffle=False, num_workers=16)

dataloader_monoimage = torch.utils.data.DataLoader(dataset,
                batch_size=1, pin_memory=True,
                shuffle=False, num_workers=1)
# %%


def proc_gradmap(grad):
    # grad: [bs, C, H, W]
    x = grad
    x = x.norm(p=2, dim=1, keepdim=True)  # [bs,1, H, W]
    mean_x2 = x.norm(p=2, dim=[2,3], keepdim=True) / math.sqrt(x[0].numel()) # [bs,1, H, W]
    x = x / mean_x2  # [bs,1, H, W]
    kernel_size = 2*round(gauss_radius) + 1
    gauss = kornia.filters.GaussianBlur2d(
        (kernel_size, kernel_size),
        (gauss_radius, gauss_radius))
    x = gauss(x)  # [bs,1, H, W]
    x = (x > gradmap_threshold).type(grad.dtype)  # [bs,1, H, W]
    return x # [bs,1, H, W]

def to_image(imgs):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    # imgs  [bs, C, H, W]
    shape = imgs.shape
    imgs = imgs * std.view(1, -1, 1, 1).repeat(shape[0], 1, *shape[2:]) + \
                mean.view(1, -1, 1, 1).repeat(shape[0], 1, *shape[2:])
    return imgs

def IoU(mask1s, mask2s):
    # mask1s, mask2s: [bs, 1, H, W]
    intersect = torch.min(mask1s, mask2s)
    union = torch.max(mask1s, mask2s)
    IoUs = intersect.sum(dim=[1, 2, 3]) / union.sum(dim=[1, 2, 3])
    return IoUs


def imgshow(img, colorbar=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    img = img.detach()
    npimg = img.numpy()
    if npimg.shape[0] == 1:
        npimg = npimg[0]
    else:
        npimg = np.transpose(npimg, (1, 2, 0))

    print('range = ', npimg.max(), npimg.min())
    print('shape =', npimg.shape)
    im = ax.imshow(npimg)  #  cmap=plt.cm.hot_r
    if colorbar:
        plt.colorbar(im)
    plt.show()



class Report(object):
    def __init__(self):
        self.num_pbar = 8
        self.pbar = [tqdm(bar_format='{desc}',leave=True) for i in range(self.num_pbar)]

    def update(self, IoU_list, label_list):
        setnum = len(dataset.classes) + 1

        IoU_set = {'CSG':    [[] for i in range(setnum)],
                    'STD': [[] for i in range(setnum)]}

        for method in ['CSG', 'STD']:
            for iou, label in zip(IoU_list[method], label_list):
                IoU_set[method][label].append(iou)
                IoU_set[method][-1].append(iou)


        def mean(l):
            return 0 if len(l) == 0 else sum(l) / len(l)

        IoU_mean = {'CSG': [], 'STD': []}
        for method in ['CSG', 'STD']:
            for IoUs in IoU_set[method]:
                IoU_mean[method].append(mean(IoUs))

        self.pbar[0].set_description_str('IoU')
        self.pbar[1].set_description_str('%-7s' * (1 + setnum) % ('model', *dataset.classes, 'total'))
        for idx, method in enumerate(['CSG', 'STD']):
            self.pbar[2+idx].set_description_str('%-7s'%method + ' ' . join( [ '%.4f' ]*setnum) % (* IoU_mean[method], ) )


        ap_mean = {'CSG': [], 'STD': []}
        for method in ['CSG', 'STD']:
            for IoUs in IoU_set[method]:
                aps = [1 if iou>=ap_thread else 0 for iou in IoUs]
                ap_mean[method].append(mean(aps))

        self.pbar[4].set_description_str('AP@%d'%(100*ap_thread))
        self.pbar[5].set_description_str('%-7s'*(1+setnum) % ('model', *dataset.classes, 'total') )
        for idx, method in enumerate(['CSG', 'STD']):
            self.pbar[6+idx].set_description_str('%-7s'%method + ' ' . join( [ '%.4f' ]*setnum) % (* ap_mean[method], ) )

        for i in range(self.num_pbar):
            self.pbar[i].update()

    def close(self):
        for i in range(self.num_pbar):
            self.pbar[i].close()

# %%



if per_filter:

    num_class = len(dataset.classes)
    act_sum = {'CSG': torch.zeros(num_class, num_channel ),
               'STD': torch.zeros(num_class, num_channel ) } # [C, K]
    act_num = {'CSG': torch.zeros(num_class, num_channel ),
               'STD': torch.zeros(num_class, num_channel ) } # [C, K]

    pbar = tqdm(bar_format='{desc}',leave=True)
    for batch_id, (imgs, labels, obj_masks) in enumerate(dataloader):
        pbar.set_description_str(f'batch = {batch_id} / {len(dataloader)}')
        pbar.update()
        for method in ['CSG', 'STD']:
            activation_maps = model[method](imgs)  # [bs, K=2048, h=4, w=4]
            activations = activation_maps.sum(dim=[2,3]) # [bs, K]

            for imgid, label in enumerate(labels): # [1] ~ [1...C]
                act_sum[method][label] += activations.detach().cpu()[imgid]
                act_num[method][label] += 1
    pbar.close()

    act_mean = {}
    top_channels = {}
    for method in ['CSG', 'STD']:
        act_mean[method] = act_sum[method] / act_num[method]
        _, top_channels[method] = torch.topk(act_mean[method], n_top_channel, dim=1, sorted=True, largest=True)


# %%
if per_filter:
    IoU_list = {'CSG': [], 'STD': []}
    label_list = []


    report = Report()
    pbar = tqdm(bar_format='{desc}',leave=True)

    for img_id, (img, label, obj_mask) in enumerate(dataloader_monoimage):
        pbar.set_description_str(f'image = {img_id+1} / {len(dataloader_monoimage)}')
        pbar.update()
        label_list.append(label.item())

        if if_show and showimg_id == img_id:
            showable_img = to_image(img)
            imgshow(showable_img[0], colorbar=False)

        for method in ['CSG', 'STD']:
            imgs = img.repeat([n_top_channel,1,1,1])
            imgs.requires_grad = True
            activation_maps = model[method](imgs)  # [n_top_channel, K=2048, h=4, w=4]
            activations = activation_maps.mean(dim=[2, 3])  # [n_top_channel, K]
            target = torch.gather(activations , 1, top_channels[method][label.item()].view(-1,1).to(device) ) # [n_top_channel, 1]
            target.sum().backward()
            imgrad = proc_gradmap(imgs.grad)

            if if_show and showimg_id == img_id:
                imgshow(imgrad[0])

            obj_masks = obj_mask.repeat(n_top_channel,1,1,1)
            IoUs = IoU(imgrad, obj_masks)
            IoU_list[method].append(IoUs.mean())

        report.update(IoU_list, label_list)
        if if_show and showimg_id == img_id:
            break

    report.close()
    pbar.close()


# %%
if not per_filter:
    IoU_list = {'CSG': [], 'STD': []}
    label_list = []

    pbar = tqdm(bar_format='{desc}',leave=True)

    for batch_id, (imgs, labels, obj_masks) in enumerate(dataloader):
        pbar.set_description_str(f'batch id = {batch_id} / {len(dataloader)}')
        pbar.update()

        if if_show:
            showable_imgs = to_image(imgs)
            imgshow(showable_imgs[showimg_id], colorbar=False)
        label_list += labels.tolist()

        for method in ['CSG', 'STD']:
            imgs.requires_grad = True
            logits = model[method](imgs)   # [bs, C]  # C: class_num
            probs = softmax(logits, dim=1)  # [bs, C]

            target = torch.gather(probs, 1, labels.view(-1, 1).to(device))  # [bs, 1]
            target.sum().backward()

            imgrad = proc_gradmap(imgs.grad)
            if if_show:
                imgshow(imgrad[showimg_id])

            IoUs = IoU(imgrad, obj_masks)
            IoU_list[method] += IoUs.tolist()

        if if_show:
            break

    pbar.close()
