import argparse
import pickle

import itk
import monai
import numpy as np
import torch
from torch.utils import data
from utils import ShapedUnet3D, GaussianDiffusion3D

monai.utils.set_determinism(seed=2938649572)

itk.ProcessObject.SetGlobalWarningDisplay(False)

parser = argparse.ArgumentParser(description="3D Diffusion trainer")
parser.add_argument("--dataset_path", type=str, required=True, help="path to the training dataset")
parser.add_argument("--d_ckpt", type=str, required=True, help="path to the stored diffusion model checkpoint")
parser.add_argument("--output_path", type=str, required=True, help="path to the folder to save the segmentation models")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
parser.add_argument("--iters", type=int, default=50, help="total training iterations")
parser.add_argument("--img_size", type=int, default=50,
                    help="resolution of image for diffusion models (50 for synthetic, 128 for BraTS)")
parser.add_argument("--timesteps", type=int, default=250, help="number of diffusion timesteps")
parser.add_argument("--batch", type=int, default=32, help="batch size")
parser.add_argument("--n_seg_classes", type=int, default=4, help="number of desired segmentation classes")
parser.add_argument("--lr", type=float, default=0.0001, help="number of diffusion timesteps")

args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu_id))

toy_data_dir = args.dataset_path + '/im_{}.p'
all_inputs = []
for i in range(80):
    curr_path = toy_data_dir.format(str(i))
    toy = pickle.load(open(curr_path, 'rb'))
    all_inputs.append(toy.astype(np.float32))
train_dataset = data.TensorDataset(torch.Tensor(all_inputs))
train_dataloader = data.DataLoader(train_dataset, batch_size=4)

model = ShapedUnet3D(img_size=args.img_size, dim=64, dim_mults=(1, 2, 4, 8), channels=1).to(device)
diffusion = GaussianDiffusion3D(model, image_size=args.img_size, channels=1, timesteps=args.timesteps,
                                loss_type='l1').to(device)
dicts = torch.load(args.d_ckpt, map_location=device)
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in dicts['model'].items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

diffusion.load_state_dict(new_state_dict)

dice_loss2 = monai.losses.DiceLoss(
    include_background=True,
    to_onehot_y=False,
    softmax=False,
    reduction="mean"
)


def consistency_loss(masks, image):
    weighted_regions = masks.unsqueeze(2) * image.unsqueeze(1)
    mask_sum = masks.sum(4).sum(3).sum(2, keepdim=True)
    means = weighted_regions.sum(5).sum(4).sum(3) / (mask_sum + 1e-5)
    diff_sq = (image.unsqueeze(1) - means.unsqueeze(3).unsqueeze(4).unsqueeze(5)) ** 2
    loss = (diff_sq * masks.unsqueeze(2)).mean(5).mean(4).mean(3)
    return loss.sum(2).sum(1).mean()


num_segmentation_classes = args.n_seg_classes

seg_net = torch.nn.Sequential(torch.nn.Upsample((48, 48, 48)),  # resize to the closest power of 2 for easy use of UNet
                              monai.networks.nets.UNet(
                                  3,  # spatial dims
                                  1,  # input channels
                                  num_segmentation_classes,  # output channels
                                  (8, 16, 16, 32, 32, 64, 64),  # channel sequence
                                  (1, 2, 1, 2, 1, 2),  # convolutional strides
                                  dropout=0.2,
                                  norm='batch'
                              ),
                              torch.nn.Upsample((50, 50, 50))).to(device)

learning_rate = args.lr
optimizer = torch.optim.Adam(seg_net.parameters(), learning_rate)

max_epochs = args.iters
training_losses = []
timestamp = 25
lambda_rgb = 1
lambda_sc = 1
lambda_inv = 1

interp = torch.nn.Upsample(size=(50, 50, 50), mode='trilinear', align_corners=True)

best_val_dice = 0
for epoch_number in range(max_epochs):

    print(f"Epoch {epoch_number + 1}/{max_epochs}:")

    seg_net.train()
    losses = []
    for batch in train_dataloader:
        imgs = batch[0].unsqueeze(1).to(device)

        optimizer.zero_grad()
        predicted_segs = seg_net(imgs).softmax(dim=1)

        activation = {}


        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook


        if args.n_seg_classes == 2:
            diffusion.denoise_fn.unet.ups[0][2].register_forward_hook(getActivation('ups'))
        elif args.n_seg_classes == 4:
            diffusion.denoise_fn.unet.ups[1][2].register_forward_hook(getActivation('ups'))
        elif args.n_seg_classes == 8:
            diffusion.denoise_fn.unet.ups[2][2].register_forward_hook(getActivation('ups'))

        imgs = (imgs - imgs.amin(dim=(1, 2, 3, 4), keepdim=True)) / (
                imgs.amax(dim=(1, 2, 3, 4), keepdim=True) - imgs.amin(dim=(1, 2, 3, 4), keepdim=True))
        d = imgs * 2 - 1  # normalize to -1 and 1
        t = torch.tensor([timestamp] * d.shape[0], device=device).long()
        noise = torch.randn_like(d)
        x = diffusion.q_sample(x_start=d.to(device), t=t, noise=noise.to(device))

        out = diffusion.denoise_fn(x, t)
        feats = interp(activation['ups'])

        gamma = np.random.uniform(0.9, 1.1)

        predicted_segs_gamma = seg_net(imgs.amin(dim=(1, 2, 3, 4), keepdim=True) + imgs ** gamma * (
                imgs.amax(dim=(1, 2, 3, 4), keepdim=True) - imgs.amin(dim=(1, 2, 3, 4), keepdim=True))).softmax(
            dim=1)

        loss_rgb = lambda_rgb * consistency_loss(predicted_segs, imgs)
        loss_sc = lambda_sc * consistency_loss(predicted_segs, feats)
        loss_inv = lambda_inv * dice_loss2(predicted_segs, predicted_segs_gamma)
        loss = loss_rgb + loss_sc + loss_inv
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    training_loss = np.mean(losses)
    print(
        f"\ttraining loss: {training_loss:.4f}, \trgb loss: {loss_rgb:.4f}, \tloss_sc: {loss_sc:.4f}, \tloss_inv: {loss_inv:.4f}")
    training_losses.append([epoch_number, training_loss])
