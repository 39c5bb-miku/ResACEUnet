# ResACEUnet
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14280741.svg)](https://doi.org/10.5281/zenodo.14280741)

<div align=center>
    <img src="ResACEUnet.png" width=100% />
</div>

## Abstract
Detecting fault constitutes a pivotal aspect of seismic interpretation, significantly influencing the outcomes of petroleum and gas exploration. As artificial intelligence advances, Convolutional Neural Network (CNN) has proven effective in detecting faults in seismic interpretation. Nevertheless, the receptive field of convolutional layer within CNN is inherently limited, focusing on extracting local features, which leads to the detection of fewer and discontinuous fault features. In this study, integrating the local feature extraction capabilities of CNN with the global feature extraction prowess of Transformer, we proposed an U-shaped hybrid architecture model (ResACEUnet) to detect fault of three-dimensional (3D) seismic data. In ResACEUnet, we introduced a module called ACE block, which integrates convolution and attention mechanisms. This module enabled the model to simultaneously extract local features and model global contextual information, capturing more accurate fault features. In addition, we utilized a joint loss function composed of binary cross-entropy loss and Dice loss, to tackle the challenge of imbalanced positive and negative samples. The model was trained on a synthetic dataset, with a range of data augmentation techniques were employed to bolster its generalization capabilities and robustness. We implemented our proposed method on the offshore F3 seismic data from the Netherlands, and seismic data from Kerry3D and Parihaka in New Zealand. Compared to conventional CNN models such as Unet and ResUnet, ResACEUnet demonstrated superior capabilities in capturing more features and identifying fault with higher accuracy and continuity.
# Usage
```commandline
fault_detection/
      ├───── datasets/
      │        ├───── 200-20/
      │        │        ├───── train/
      │        │        │         ├───── images  # 200 train data put here
      │        │        │         └───── labels
      │        │        └───── val/
      │        │                  ├───── images  # 20 val data put here
      │        │                  └───── labels
      │ 
      │ 
      └───── ...
```
## Environment Setup
```commandline
conda create -n fault-detection python=3.9
```
```commandline
conda activate fault-detection
```
```commandline
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
```commandline
pip install timm tqdm wandb monai yacs matplotlib seismic_canvas segyio omegaconf scikit-image scikit-learn matplotlib numpy pandas scipy termcolor obispy
```
## Training
Download the training datasets [https://pan.baidu.com/s/1Ga2MNm812T-wcb07j34f8w?pwd=xoqi](https://pan.baidu.com/s/1NjiM6KwRKfMPJJ2wDDGTqA?pwd=x0b0) and place the extracted files in the `datasets`.

In main.py, set the GPU number to be used on line 3.

Hyperparameters are set in configs/config.yaml, which you can modify as needed.

```commandline
python main.py
```
## Testing
Place the data for prediction (in npy format) in the `datasets/test/seismic`.

The results of the prediction will be saved in the `datasets/test/fault`.

```commandline
python predict_3d.py
```
## Visualization
The seismic3D function in `utils/casual.py` can be used to view 3D seismic images, and the fault3D function can be used to view 3D seismic fault images. Modify seismic_path and fault_path as needed.

`utils/show_results.py` can be used to view 3D seismic sections or slices and save them as jpg files. Modify type to represent which dimension of the data, num to represent which face, and file to represent the data name as needed.

`utils/sgy_npy.py` can convert npy files to sgy files. Modify predicted_file as needed.

`utils/readsgy.ipynb` can read sgy data and convert it to npy data.
## Tip
If an error occurs with NaN, it might be due to model initialization issues. Try running the process again or modify the seed and run again.
