# %%

cam_top_rate = 0.3 # 0.3
cam_UoI_threshold = 0.3

threshold_probability = 0.3  # 0.1  #  0.005
UoI_threshold = 0.3 # 0.04

gpuid = 0
# %%
import sys, os, torch, torchvision
import numpy as np
here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here,'..', 'train'))

from resnet_std import resent as resnet_std
from resnet_std import ResNet
ResNet.forward = ResNet.get_feature_map

sys.path.append(os.path.join(here,'..',  'VOC'))
from VOCPart import VOCPart

from tqdm import tqdm
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
mask = model['CSG'].module.lmask.mask.detach().cpu().numpy()
from matplotlib import pyplot as plt
plt.imshow(mask[0:20,:])
plt.show()

num_channel, num_class = mask.shape




# %%

print('cam   : top_rate =',cam_top_rate,           ', UoI_threshold =', cam_UoI_threshold )
print('filter: top_rate =', threshold_probability, ', UoI_threshold =', UoI_threshold)


dataset = VOCPart(VOC_path, train=False, requires=['img'], size=128)
dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=32, pin_memory=True,
                shuffle=True, num_workers=16)
img_num = len(dataset)


num_large = int( threshold_probability * 4 *4 * len(dataset) )


threshold = {}
for method in ['CSG', 'STD']:
    activations = np.zeros((num_channel, len(dataset)*16))

    for batch_id, (imgs, labels) in enumerate(dataloader):

        fmap = model[method](imgs)
        bs, n_channel, _, _ =fmap.shape
        fmap = fmap.view(bs, n_channel, -1)
        activations_channels = torch.cat((*fmap, ), 1) # num_channel, batch_size * 16
        activations[:, (batch_id*bs*16):((batch_id+1)*bs*16)] = activations_channels.detach().cpu().numpy()

    activations_sort = np.sort(activations, axis=1)
    threshold[method] = activations_sort[:,-num_large]


# %%
dataset = VOCPart(VOC_path , train=False, requires=['img','obj_mask', 'part_mask'], size=128)
dataloader = torch.utils.data.DataLoader(dataset,
                batch_size=32, pin_memory=True,
                shuffle=True, num_workers=16)
img_num = len(dataset)

# %%
num_class = 6

def dev_zero_replace(a,b):
    return np.divide(a, b,
             out=np.zeros_like(a/1.0),
             where=(b!=0))


def get_IoU(fmap, obj_mask, _threshold=None, top_rate=None):
    # all torch.tensor(1,128,128)
    fmap = torch.nn.functional.interpolate(fmap.view(1,1,4,4), 128, mode='bilinear', align_corners=False) [0]
    fmap = fmap[0]
    obj_mask = obj_mask[0]

    if top_rate!=None and _threshold==None:
        top_num = int(torch.numel(fmap) * top_rate)
        value,index=fmap.view([-1]).sort()
        _threshold = value[-top_num]

    fmaask=(fmap >= _threshold).float()
    union = torch.max(fmaask,obj_mask)
    intersect = torch.min(fmaask,obj_mask)
    _UoI = intersect.sum()/union.sum()

    return _UoI.item()




# %%
print('\n----------------------------------------------------------------------------------------------')
print('CAM     '+'| '+(((num_class+1)*'%-6s ') % (*dataset.classes,'total')))

pbar1 = tqdm(bar_format='{desc}',leave=True)
pbar2 = tqdm(bar_format='{desc}',leave=True)
pbar3 = tqdm(bar_format='{desc}',leave=True)
pbar4 = tqdm(bar_format='{desc}',leave=True)
pbar5 = tqdm(bar_format='{desc}',leave=True)


sum_cam_IoU = {'CSG':  0.0 , 'STD' : 0.0 }
sum_cam_related =  {'CSG':  0 , 'STD' : 0 }
num_cam =  {'CSG':  0.0 , 'STD' : 0.0 }
mean_cam_IoU = {'CSG':  0.0 , 'STD' : 0.0 }
mean_cam_related = {'CSG':  0.0 , 'STD' : 0.0 }

sum_cam_UoI_class = {'CSG': np.zeros(num_class), 'STD': np.zeros(num_class)}
sum_cam_related_class = {'CSG': np.zeros(num_class), 'STD': np.zeros(num_class)}
num_cam_UoI_class = {'CSG': np.zeros(num_class), 'STD' :  np.zeros(num_class)}
mean_cam_UoI_class = {'CSG': np.zeros(num_class), 'STD' :  np.zeros(num_class)}
mean_cam_related_class = {'CSG': np.zeros(num_class), 'STD': np.zeros(num_class)}

for img_id,(img, label, obj_mask, part_mask) in enumerate(dataset):
    img = img.view(1,*img.shape).to(device)
    for method in ['CSG', 'STD']:
        feature_map = model[method](img)[0].detach().cpu()

        weight = model[method].module.fc.weight.detach().cpu() # [6, 2048]
        weight = weight[label]
        cam = ( feature_map * weight.view([-1,1,1]) ).sum(axis=0)
        cam_IoU = get_IoU(cam, obj_mask, top_rate=cam_top_rate)

        sum_cam_IoU[method] += cam_IoU
        sum_cam_related[method] += int(cam_IoU>cam_UoI_threshold)
        num_cam[method] += 1

        mean_cam_IoU[method] = sum_cam_IoU[method] / num_cam[method]
        mean_cam_related[method] = sum_cam_related[method] / num_cam[method]

        sum_cam_UoI_class[method][label] += cam_IoU
        sum_cam_related_class[method][label] += int(cam_IoU>cam_UoI_threshold)
        num_cam_UoI_class[method][label] += 1

        mean_cam_UoI_class[method] = dev_zero_replace(sum_cam_UoI_class[method], num_cam_UoI_class[method])
        mean_cam_related_class[method] = dev_zero_replace(sum_cam_related_class[method], num_cam_UoI_class[method])


    pbar1.set_description_str(('IoU CSG | ' + num_class * '%.4f ' + '%.4f') % (
         *mean_cam_UoI_class['CSG'], mean_cam_IoU['CSG']))
    pbar2.set_description_str(('IoU STD | ' + num_class * '%.4f ' + '%.4f') % (
         *mean_cam_UoI_class['STD'], mean_cam_IoU['STD']))

    pbar3.set_description_str(('AP  CSG | ' + num_class * '%.4f ' + '%.4f') % (
         *mean_cam_related_class['CSG'], mean_cam_related['CSG']))
    pbar4.set_description_str(('AP  STD | ' + num_class * '%.4f ' + '%.4f') % (
         *mean_cam_related_class['STD'], mean_cam_related['STD']))
    pbar5.set_description_str('img %d / %d' % (img_id + 1, img_num))

    pbar1.update()
    pbar2.update()
    pbar3.update()
    pbar4.update()
    pbar5.update()

pbar1.close()
pbar2.close()
pbar3.close()
pbar4.close()
pbar5.close()

# %%
print('\n----------------------------------------------------------------------------------------------')
print('filter activation map')
print('\ngo thru the dataset')

def take(x,coords):
    # coords = np.array(each column is a coordinate)
    return np.take(x, np.ravel_multi_index(coords, x.shape))


pbar = tqdm(bar_format='{desc}',leave=True)

sum_IoU = {'CSG': np.zeros(mask.shape), 'STD' :  np.zeros(mask.shape)}
sum_success = {'CSG': np.zeros(mask.shape), 'STD' :  np.zeros(mask.shape)}
num = {'CSG': np.zeros(mask.shape), 'STD' :  np.zeros(mask.shape)}
mean_IoU = {'CSG': 0.0, 'STD' : 0.0}
mean_score_filter = {'CSG': 0.0, 'STD' : 0.0}

for img_id, (img, label, obj_mask, part_mask) in enumerate(dataset):
    img = img.view(1,*img.shape).to(device)
    for method in ['CSG', 'STD']:
        feature_map = model[method](img)[0].detach().cpu()
        for channel, fmap in enumerate(feature_map):
            fmap = fmap
            IoU = get_IoU(fmap, obj_mask, threshold[method][channel])

            sum_IoU[method][channel, label] += IoU
            sum_success[method][channel, label] += int(IoU>UoI_threshold)
            num[method][channel, label] += 1

        mean_success = dev_zero_replace(sum_success[method], num[method])
        score_filter = mean_success.max(axis=1)
        mean_score_filter[method] = score_filter.mean(axis=0)

        idx = mean_success.argmax(axis=1)
        mean_IoU_matrix = dev_zero_replace(sum_IoU[method], num[method])
        coords = np.array([range(mean_IoU_matrix.shape[0]),idx])
        IoU_related_class = take(mean_IoU_matrix, coords)
        mean_IoU[method] = IoU_related_class.mean(axis=0)

    pbar.set_description_str( 'img %d / %d | CSG : STD | total UoI %.4f : %.4f | total AP %.4f : %.4f |'  % ( img_id + 1, img_num,
        mean_IoU['CSG'], mean_IoU['STD'] ,  mean_score_filter['CSG'], mean_score_filter['STD'] ) )
    pbar.update()

pbar.close()


# %%
# result

related_filter_num = {}
related_filter_mean_score = {}
related_filter_mean_IoU = {}


for method in ['CSG', 'STD']:

    mean_success = dev_zero_replace(sum_success[method], num[method])
    score_filter = mean_success.max(axis=1)
    related_filter_num[method]=np.zeros(num_class)
    related_filter_mean_score[method] = np.zeros([num_class])
    related_filter_mean_IoU[method] = np.zeros([num_class])

    filter_related_class = mean_success.argmax(axis=1)
    mean_IoU_matrix = dev_zero_replace(sum_IoU[method], num[method])
    coords = np.array([range(mean_IoU_matrix.shape[0]),filter_related_class])
    IoU_filter = take(mean_IoU_matrix, coords)

    for cid, class_name in enumerate(dataset.classes):
        score_related_to_class = score_filter[filter_related_class==cid]
        related_filter_num[method][cid] = len(score_related_to_class)
        related_filter_mean_score[method][cid] = score_related_to_class.mean()

        IoU_related_to_class = IoU_filter[filter_related_class == cid]
        if len(IoU_related_to_class) != related_filter_num[method][cid]:
            print('inequal', len(IoU_related_to_class), related_filter_num[method][cid])
        related_filter_mean_IoU[method][cid] = IoU_related_to_class.mean()


# %%
# print table

print('activMap' + '| ' + (((num_class + 1) * '%-6s ') % (*dataset.classes, 'total')))

for method in ['CSG', 'STD']:
    print(('RFN %s | ' % method) + ((num_class+1) * '%-6d ') % (*related_filter_num[method], related_filter_num[method].sum() ))

for method in ['CSG', 'STD']:
    print('IoU %s | ' % method + ((num_class + 1) * '%.4f ') % (*related_filter_mean_IoU[method],
    (related_filter_num[method]*related_filter_mean_IoU[method]).sum() / related_filter_num[method].sum() ) )

for method in ['CSG', 'STD']:
    print('AP  %s | ' % method + ((num_class + 1) * '%.4f ') % (*related_filter_mean_score[method],
    (related_filter_num[method]*related_filter_mean_score[method]).sum() / related_filter_num[method].sum() ) )

print('RFN: related filter number')
