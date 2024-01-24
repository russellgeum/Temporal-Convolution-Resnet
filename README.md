# Temporal-Convolution Resnet (PyTorch)
- [TC-Resnet Official](https://github.com/hyperconnect/TC-ResNet)  
- [TC-Resnet Paper](https://arxiv.org/pdf/1904.03814.pdf)  
  
TC-Resnet is speech deep neural network for KWS task (Hyperconnect)  
Original repository is tensorflow version code  
So, This repository gives you pytorch version code  
### Requirements
```
pytorch == 1.8.0
torchaudio == 0.8.0
or
pytorch == 1.9.0
torchaudio == 0.9.0
tqdm
einops
numba == 0.48
librosa == 0.7.2
```
### Setting enviorments
1. Install conda and setting enviorment
```
bash conda_install.sh
conda create -n tc-resnet python=3.8
```
2. Install some packages
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install numba==0.48
pip install librosa==0.7.2
pip install tqdm 
pip intsall einops
```
3. Or, You can select one-shot option
```
conda env create -f pakages.yml
```
### Preparing dataset
```
bash download.sh
```
### Training model
```
python model_train.py
```
