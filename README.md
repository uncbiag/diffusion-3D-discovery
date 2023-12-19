# Unsupervised Discovery of 3D Hierarchical Structure with Generative Diffusion Features 

This repo includes the code for [Unsupervised Discovery of 3D Hierarchical Structure with Generative Diffusion Features](https://arxiv.org/abs/2305.00067). 

We show our method on the Synthetic 3D Dataset from Hsu et al ( [regular]([https://nda.nih.gov/oai/](https://drive.google.com/file/d/1mdRuSkXmTof9vq62FSmoZXneUme_97dc/view)) , [irregular]([https://nda.nih.gov/oai/](https://drive.google.com/file/d/1XGx8GQlNGCStmxjYatWGBGAW25e2zxTn/view)) ) and also on [BraTS'19](https://www.med.upenn.edu/cbica/brats-2019/)


### Pipeline

1. Diffusion model training
   ```
   python diffusionTrain.py --dataset_path <folder with training images> -output_path <folder to save diffusion models> 
   ```  
    
2. Unsupervised segmentation/discovery model training
   ```
   python segTrain.py --dataset_path <folder with training images> --d_ckpt <folder with saved diffusion models> --output_path <folder to save segmentation models>
   ```  


### Citation
If you find this project useful, please cite:

```
@inproceedings{tursynbek2023unsupervised,
  title={Unsupervised Discovery of 3D Hierarchical Structure with Generative Diffusion Features},
  author={Tursynbek, Nurislam and Niethammer, Marc},
  booktitle={Medical Image Computing and Computer Assisted Intervention – MICCAI 2023},
  year={2023},
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="320--330",
  isbn="978-3-031-43907-0"
}
```
