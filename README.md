# NYCU-Intro-Machine-Learning-Project
 FGVC Classifier on CUB-200-2011

## Performance
- Accuracy 85.9% on both public and private leaderboards
- [**Leaderboard on Kaggle contest**](https://www.kaggle.com/competitions/nycu2023mlfinalproject/leaderboard)

## Training Detail
- Training environment
    - Python version: 3.10.13
    - framework: Pytorch
    - Hardware: NVIDIA 1080Ti with CUDA 12.3 & Intel i7-12700
- Pretrained model<br>
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
- Hyper-parameters
    - epoch: 5
    - batch size: 10
    - learning rate: 2e-5
    - weight decay: 1e-4