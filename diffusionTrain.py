import argparse

import itk
import monai
import torch
from utils import ShapedUnet3D, GaussianDiffusion3D, Trainer3D

monai.utils.set_determinism(seed=2938649572)

itk.ProcessObject.SetGlobalWarningDisplay(False)

parser = argparse.ArgumentParser(description="3D Diffusion trainer")
parser.add_argument("--dataset_path", type=str, required=True, help="path to the training dataset")
parser.add_argument("--output_path", type=str, required=True, help="path to the folder to save the diffusion models")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
parser.add_argument("--iters", type=int, default=50000, help="total training iterations")
parser.add_argument("--img_size", type=int, default=50,
                    help="resolution of image for diffusion models (50 for synthetic, 128 for BraTS)")
parser.add_argument("--timesteps", type=int, default=250, help="number of diffusion timesteps")
parser.add_argument("--batch", type=int, default=4, help="batch size")
parser.add_argument("--lr", type=float, default=0.0001, help="number of diffusion timesteps")

args = parser.parse_args()

device = torch.device("cuda:" + str(args.gpu_id))

model = ShapedUnet3D(img_size=args.img_size, dim=64, dim_mults=(1, 2, 4, 8), channels=1).to(device)

diffusion = GaussianDiffusion3D(model, image_size=args.img_size, channels=1, timesteps=args.timesteps,
                                objective='pred_x0', loss_type='l1').to(device)

trainer = Trainer3D(
    diffusion,
    args.dataset_path,
    image_size=diffusion.image_size,
    train_batch_size=args.batch,
    train_lr=args.lr,
    train_num_steps=args.iters,
    gradient_accumulate_every=2,
    amp=True,
    results_folder=args.output_path
)

trainer.train()
