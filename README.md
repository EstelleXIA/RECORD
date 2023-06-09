# RECORD
This is the GitHub Repository providing an example code base for "RECIST-guided Consistent Objective Response Evaluation by Deep Learning on Immunotherapy-treated Liver Cancer".

![RECORD](documentation/record_schematic.png)

## Getting started
### Dependencies
This project requires Python 3 (3.7.13) with the following additional packages:
* [PyTorch](https://pytorch.org/) (torch==1.11.0, torchvision==0.12.0) with CUDA support
* [monai](https://pypi.org/project/monai/) (0.9.1)
* [NumPy](https://numpy.org/) (1.21.2)
* [tqdm](https://github.com/tqdm/tqdm) (4.64.0)
* [scikit-learn](https://scikit-learn.org/stable/) (0.22.2)
* [matplotlib](https://matplotlib.org/) (3.5.2)
* [pandas](https://pandas.pydata.org/) (2.0.1)
* [TensorBoard](https://pypi.org/project/tensorboard/) (2.5.1)
* [antspyx](https://pypi.org/project/antspyx/) (0.2.0)
* [connected-components-3d](https://pypi.org/project/connected-components-3d/) (3.10.2)
* [nibabel](https://pypi.org/project/nibabel/) (3.2.2)
* [Optimizers](https://pypi.org/project/Optimizers/) (0.1)
* [scipy](https://pypi.org/project/scipy/) (1.7.3)
* [SimpleITK](https://pypi.org/project/SimpleITK/) (2.2.1)

The numbers in parentheses denote the versions used during development of this project. Other python and package versions may or may not work as intended.

A requirements file is provided in the root of this project and can be used to install the required packages via `pip install -r requirements.txt`. If the process fails, you may need to upgrade setuptools via `pip install --upgrade setuptools`.

### Try it with example data
To get you started, example data and configuration files are provided. The data can be downloaded from [here](https://zenodo.org/record/7955755). Extract `data.zip` and place the contents into the `data` folder like this:
```
data/
|
|---- image/
|---- label_gt/
|---- mask/
```
#### 0. Data description
The `data/` contains demo CT and tumor mask files in the `.nii.gz` format. Filename is named according to `center-patient-time`, i.e, center number of 033002, patient id of 004, and scan time of 20180531. CT scans are end with `_0000`, i.e., `data/img/*_0000.nii.gz`, while ground truth labels are named as `data/label_gt/*.nii.gz`. Each baseline-follow-up CT scan pair has a RECIST response outcome.

| Patient_ID |          Filename          | Label | Set   |
|------------|----------------------------|-------|-------|
| 033002-005 | 033002-005-20180611.nii.gz | /     | TRAIN |
| 033002-005 | 033002-005-20180730.nii.gz | PD    | TRAIN |
| ...        | ...                        | ...   | ...   |
| 033002-004 | 033002-004-20180531.nii.gz | /     | TEST  |
| 033002-004 | 033002-004-20180723.nii.gz | PD    | TEST  |
| ...        | ...                        | ...   | ...   |
| 033002-004 | 033002-004-20210512.nii.gz | SD    | TEST  |

There are some key steps of the RECORD model. The following part would introduce each part in detail.

![Key Steps](documentation/record_steps.png)

#### 1. Get original segmentation model predicted masks
You can try any 3D segmentation model. Here we use two state-of-the-art models, [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/) and [Swin-Unetr](https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d.ipynb). Use `data/image/*.nii.gz` as input to get the predicted masks.

The original predictions should be moved to `data/mask/`.

#### 2. Image and mask registration
Advanced Normalization Tools ([ANTsPy](https://github.com/ANTsX/ANTsPy)) is used to make the baseline-follow-up image pair registered.

Run the following line to register image pairs.
```
python prep/ants_registrate.py --input_img ../data/image/ --input_mask ../data/mask/ --output ../data/transformed_ants/ --maskpath ../data/mask_ants/ --check
```

|  command  | description |
| ------------------- | ------------- |
| `--input_img`  | the input folder of all images |
| `--input_mask`  | the input folder of all predicted masks |
| `--output`  | the forward and inverse transforms (dispplacement fields) |
| `--maskpath`  | the output registered masks |
| `--check`  | to check whether ANTs is successfully performed |


#### 3. Livermask prediction
Liver mask is to remove lesion outside the liver. We use the union of ''[livermask](https://github.com/andreped/livermask)'' and [pretrained nnU-Net abdominal organ segmentation](https://zenodo.org/record/3734294#.ZAGgnHZBw2z).

#### 4. Difference map generation
We subtracted baseline mask from the follow-up mask, and obtained differences on single lesion level by 3D connected component analysis. Then, we labelled extremely large differences as candidate false positives and checked whether they were natural disease progression or model mispredictions with the help of other longitudinal CT scans. 

![Difference Map](documentation/dif_map.png)
Run the following line to get difference map.
```
python prep/dif_map.py --input ../data/mask_ants/ --output ../data/dif_map/ --livermask ../data/liver_ants/
```

|  command  | description |
| ------------------- | ------------- |
| `--input`  | the input folder of registered segmentation model masks |
| `--output`  | the save path of the generated difference map |
| `--livermask`  | optional, if use, the input folder of predicted liver masks |


#### 5. Prepare .json file for RECORD
Prepare the test patient ids in `prep/test.txt`. Then run 
```
python prep/record_json.py
```

#### 6. Start RECORD training
The `data/` folder should be structured as below.
```
data/
|
|---- dif_map/
|---- image/
|---- label_gt/
|---- liver_ants/
|---- mask/
|---- mask_ants/
|---- transformed_ants/
```
Run `bash record_optim.sh`.

|  command  | description |
| ------------------- | ------------- |
| `--json_list`  | the prepared json file in step 5 |
| `--data_dir`  | the root of the json file |
| `--val_every`  | validation per `val_every` epoch |
| `--logdir`  | RECORD log directory |
| `--dif_map_path` | the difference map path |
| `--out_channels` | default=2, background and foreground |
| `--weight` | weights between classification and segmentation tasks |
| `--infer_overlap` | overlap of ROIs in sliding window inference |
| `--probabilistic` | use probablity predicted masks as outputs |


## Contact
If you have any question, please contact this [Email](mailto:Estelle-xyj@sjtu.edu.cn).


## Acknowledgements

