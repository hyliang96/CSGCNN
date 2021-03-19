# -------------------------------------------------------------
# experiment settings:

gpu_ids = '0'
ifmask = True
train = True

# -------------------------------------------------------------
# hyper-parameters:
seed = 12
repeat_times = 1 # how much times repeat it
cudnn_behavoir = 'slow' # 'benchmark' 'normal' 'slow' 'none'

# model and dataset
datasets = ['VOCpart']
img_size = 128
depth = 152
batchsize = 32

# training
epoch = 150
optim = 'adam'
lr = '1e-5' # finetune resnet152: 1e-5
lr_reg = '1e-3'
lambda_reg = '1e-3' # reg. coef.
warmup_epochs = 10
layers_learnable = 5
layers_learnable_reg = 4
mask_period = 3
mask_epoch_min = 2
frozen = True

# -------------------------------------------------------------
import os, subprocess
here = os.path.dirname(os.path.realpath(__file__))  # get absoltae path to the dir this file is in

exp_dir_root = os.path.join(here, '..', '__result__', f'VOCPart_{img_size}x{img_size}_pretrained' )

if train:
    load_checkpoint = ''
else:
    if ifmask:
        load_checkpoint = os.path.join(here, '..', '__result__/VOCPart_128x128_pretrained/res152_bs32_adam_lr1e-5_lrreg1e-3_lmd1e-3_warm10_mask-epoch_mod3geq2_learnable-final-STD5-CSG4-layers_cudnn-slow/seed12/CSG/run0/checkpoints/epoch_150.pt')
        # load_checkpoint = os.path.join(here, '..', 'checkpoints', 'CSG.pt')
    else:
        load_checkpoint = os.path.join(here, '..', '__result__/VOCPart_128x128_pretrained/res152_bs32_adam_lr1e-5_lrreg1e-3_lmd1e-3_warm10_mask-epoch_mod3geq2_learnable-final-STD5-CSG4-layers_cudnn-slow/seed12/STD/run0/checkpoints/epoch_150.pt')
        # load_checkpoint = os.path.join(here, '..', 'checkpoints', 'STD.pt')

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

if not train:
    repeat_times = 1

for run_id in range(repeat_times):
    for data in datasets:
        train_strategy = 'CSG' if ifmask else 'STD'

        exp_dir = \
            f"{exp_dir_root}/res{depth}_bs{batchsize}_{optim}_lr{lr}_lrreg{lr_reg}_lmd{lambda_reg}" +\
            f"_warm{warmup_epochs}_mask-epoch_mod{mask_period}geq{mask_epoch_min}" + \
            (f"_learnable-final-STD{layers_learnable}-CSG{layers_learnable_reg}-layers" if frozen else '') +\
            f"_cudnn-{cudnn_behavoir}/seed{seed}/{train_strategy}/run{run_id}"

        print('model name: ', exp_dir.split('/')[-1])

        cmds = [
                "python3", f"{here}/train.py",
                "--dataset", f"{data}",
                "--depth", f"{depth}",
                "--ifmask", f"{ifmask}",
                "--gpu-ids", f"{gpu_ids}",
                "--optim", f"{optim}",
                "--batch_size", f"{batchsize}",
                "--epoch", f"{epoch}",
                "--exp_dir", f"{exp_dir}",
                "--lr", f"{lr}",
                "--img_size", f"{img_size}",
                "--lambda_reg", f"{lambda_reg}",
                "--warmup_epochs", f"{warmup_epochs}",
                "--mask_period", f"{mask_period}",
                "--mask_epoch_min", f"{mask_epoch_min}",
                "--layers_learnable", f"{layers_learnable}",
                "--layers_learnable_reg", f"{layers_learnable_reg}",
                "--frozen", f"{frozen}",
                "--lr_reg", f"{lr_reg}",
                "--train", f"{train}",
                "--seed", f"{seed}",
                "--cudnn_behavoir", f"{cudnn_behavoir}",
                "--load_checkpoint", f"'{load_checkpoint}'"
            ]

        cmd = ' '.join(cmds) + f" | tee -a {exp_dir}/log"

        if train or load_checkpoint == "":
            os.makedirs(exp_dir, exist_ok=True) # if no such path exists, iteratively created the dir

        print()
        print(cmd)
        subprocess.run(['bash', '-c', cmd])





