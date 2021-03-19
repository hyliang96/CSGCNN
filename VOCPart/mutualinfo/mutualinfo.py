# %%
gpuid = "0"


if_balance_sample = False
if_normalize = False
# %%
import sys, os, torch, torchvision
import numpy as np
from sklearn.feature_selection import mutual_info_classif
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here,'..', 'train'))

from resnet_std import resent as resnet_std
from resnet_std import ResNet
ResNet.forward = ResNet.get_feature_map

sys.path.append(os.path.join(here,'..',  'VOC'))
from VOCPart import VOCPart
from torch.nn.functional import softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import norm


os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
device_str = "cuda:0"
device = torch.device(device_str)


font = {'family' : 'Times New Roman',
        # 'weight' : 'bold',
        'size'   : 30}
matplotlib.rc('font', **font)

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

dataset = VOCPart(VOC_path , train=False, requires=['img'], size=128)

dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=64, pin_memory=True,
                shuffle=True, num_workers=16)

# %%

MutaulInfo = {}

for method in ['CSG', 'STD']:
    print(f'method = {method}')
    feature_list = []
    label_list = []

    for img, label in dataloader:
        feature = model[method](img)
        feature = feature.mean(dim=[2,3])
        feature_list.append(feature.detach().cpu())
        label_list.append(label.detach().cpu())

    features = torch.cat(feature_list, dim=0).numpy()
    labels = torch.cat(label_list, dim=0).numpy()

    print('features.shape =', features.shape)

    M_info = []
    num_class = len(dataset.classes)
    for c in range(num_class):
        print(f'class {c} / {num_class}')
        y = labels == c
        # print('y.shape', y.shape) # [N=num_img=1700, K=num_channel=2048]

        if if_balance_sample:
            positive_feature = features[np.argwhere(y == True).reshape(-1)]
            negative_feature = features[np.argwhere(y == False).reshape(-1)]
            n_sample = positive_feature.shape[0]
            negative_feature = np.random.permutation(negative_feature)[:n_sample]

            sampled_features = np.concatenate([positive_feature, negative_feature], axis=0)
            sampled_labels = np.concatenate([
                np.ones(n_sample, dtype=bool),
                np.zeros(n_sample, dtype=bool)
            ], axis=0)
            mu_info = mutual_info_classif(sampled_features, sampled_labels)
        else:
            mu_info = mutual_info_classif(features, y)

        # print('mu_info.shape =',mu_info.shape) # [K]
        mu_info = np.reshape(mu_info, (1, features.shape[1]))
        M_info.append(mu_info)

    MutaulInfo[method] = np.concatenate(M_info, axis=0)  # [C,K]
    # print(MutaulInfo[method])

for method in ['CSG', 'STD']:
    np.save(os.path.join(here, f'MI_matrix{method}.npy'), MutaulInfo[method] )

# %%
MutaulInfo = {}
for method in ['CSG', 'STD']:
    MutaulInfo[method] = np.load(os.path.join(here, f'MI_matrix{method}.npy'))

idxes = np.random.permutation(MutaulInfo[method].shape[1])[:64]

from scipy.special import softmax
from scipy.stats import entropy

for method in ['CSG', 'STD']:
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(111)
    X = MutaulInfo[method]  # [C,K]
    if if_normalize:
        X = X / norm(X,ord=1, axis=0, keepdims=True)  # CSG:STD = 0.33510 : 0.334148
            # [C]                   [C,1]

    im = ax.imshow(MutaulInfo[method][:, idxes], cmap=plt.cm.jet)
    plt.title(f'{method} CNN mutaul infomation matrix')
    plt.xlabel('class id'   )
    plt.ylabel('channel id')
    plt.colorbar(im, cax = fig.add_axes([0.93, 0.27, 0.01, 0.38] ))
    plt.show()

    metric = np.mean(np.max(X, axis=0))
    print(method, 'MIS = %.4f' % metric, metric)


# %%
