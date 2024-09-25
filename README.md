<div align="center">
<h1>UltraFastCrackSeg </h1>
<h3>A Lightweight Real-Time Crack Segmentation Model with Task-Oriented Pretraining</h3>

Weiqing Qi<sup>1</sup>, Guoyang Zhao<sup>1</sup>, Fulong Ma<sup>1</sup>, Ming Liu<sup>1</sup>\*

<sup>1</sup> Hong Kong University of Science and Technology (Guangzhou)




</div>

## 🔥🔥Highlights🔥🔥
### *1. UltraFastCrackSeg contains only 0.47M parameters and 0.76 GFLOPs, with an inference speed exceeding 1700 FPS on high-end GPUs and over 80 FPS on low-power CPUs. It achieves top performance across multiple crack segmentation datasets*</br>
### *2. Task-oriented pretraining using a masked image modeling strategy further improves accuracy without adding computational overhead.*</br>

## News🚀

(2024.09.26) ***Initial Code Release.***

### Abstract
Crack segmentation is pivotal for structural health monitoring, enabling the timely maintenance of critical in- frastructure such as bridges and roads. However, existing deep learning models are often too computationally intensive for deployment on resource-constrained devices. To address this limitation, we introduce UltraFastCrackSeg, a lightweight model designed for real-time crack segmentation that effectively balances high accuracy with low computational demands. Featuring an efficient encoder-decoder architecture, our model significantly reduces parameter count and floating point operations (FLOPs) compared to current methods, as illustrated in Figure 1. We further enhance performance through a self- supervised pretraining approach that employs a novel, task- oriented masking strategy, thereby improving feature extrac- tion. Experiments across multiple datasets demonstrate that UltraFastCrackSeg achieves state-of-the-art Intersection over Union (IoU) and F1 scores while maintaining a compact model size and high inference speed. Evaluations on a low-power CPU device confirm its capability to achieve up to 80 frames per second (FPS) with ONNX runtime optimization, making it highly suitable for real-time, on-site applications. These findings establish UltraFastCrackSeg as a robust and efficient solution for practical crack detection tasks. 


**0. Environments.** </br>
The environment installation procedure can be followed by following the steps below:</br>
```
conda create -n UFcrack python=3.8
conda activate UFcrack
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
# Install PyTorch according to your system. Refer to https://pytorch.org/get-started/previous-versions/ for more details.
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Datasets.** </br>

*A. Crack500* </br>
1. Download the Crack500 dataset from [here](https://github.com/fyangneil/pavement-crack-detection). </br>
2. Organize the cropped images into `train`, `validation`, and `test` folders inside the `/data` directory.
3. Run `Dataprepare.py` for data preparation. Output should be `.npy` files </br>

*B. DeepCrack* </br>
1. Download the DeepCrack dataset from [here](https://github.com/yhlleo/DeepCrack).</br>
2. Organize the cropped images into `train` and `test` folders inside the `/data` directory.
3. Run `Dataprepare.py`. Since DeepCrack has no official validation set, use the test set as its validation set. </br>

*C. Prepare your own dataset* </br>
1. The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))
./your_dataset/
  - train
    - images/
      - 0000.png
      - 0001.png
    - masks/
      - 0000.png
      - 0001.png
  - val/
  - test/
  - Dataprepare.py
2. Prepare the training, validation, and test sets accordingly.</br>
3. Run 'Dataprepare.py'. </br>

**2. Train the UltraFastCrackSeg.**
```
python train.py
```
- After trianing, you could obtain the outputs in './results/' </br>
- For the Crack500 dataset, set `'c_list': [32, 64, 72, 96, 128]` in `configs/config_setting.py`.
- For the DeepCrack and Ozgenel datasets, set `'c_list': [32, 48, 64, 72, 96, 128]`.
- Before starting the training, make sure to update the path for `pretrained_path` in `configs/config_setting.py` accordingly.

**3. Test the UltraFastCrackSeg.**  
First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/' </br>



## Acknowledgement
Thanks to [UltraLight VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet), [VM-UNet](https://github.com/JCruan519/VM-UNet) for their outstanding work. We have adopted their codebase as the foundation for our implementation of UltraFastCrackSeg.
