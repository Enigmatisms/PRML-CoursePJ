# PRML-CoursePJ 4
Trying Swin-Transformer 1D / Convolutional Network on chromatin-accessibility prediction (PRML course project).

---

## I. Requirement & Environment

The code is developed on Linux platform and tested on Ubuntu 18.04 LTS and 20.04 LTS. I do not guarantee the reproducibility on Windows and macOS, though on Windows there might just be two minor problems:

- The training procedures are written in shell script (like `train.sh`, `eval.sh`). Some of the shell command might not be available on Windows.
- Path format, on windows: "C:\\" yet on Linux: "/c/", all the "\\" should be replaced by "/"

For environment, I recommend using Anaconda for env setup. The code is mostly tested on Ubuntu 18.04 (computers of my lab), of which the version of python is 3.6. Therefore running the command below should get you going: 

```shell
conda create -n prml_proj python=3.6
conda install --file requirements.txt
```

Also **<u>Note that there is a compulsory requirement: CUDA and GPU should be available</u>**. All the code is written under the assumption that GPU is available, therefore if otherwise, there will be **<u>so much</u>** refactory work to be done. 

For clarity, I'll briefly list the library used here:

| Package / Library name | Version (exact) | Functionality                                        |
| ---------------------- | --------------- | ---------------------------------------------------- |
| torch                  | 1.7.1           | Deep learning framework                              |
| timm                   | 0.6.12          | Pytorch image model: asymmetric loss                 |
| tensorboard            | 2.10.0          | Logging and info recording                           |
| torchmetrics           | 0.8.2           | AUROC                                                |
| numpy                  | 1.19.5          | General purpose data processing                      |
| matplotlib             | 3.3.3           | Visualization framework                              |
| seaborn                | 0.11.2          | Advanced visualization (like violin plotting)        |
| einops                 | 0.3.0           | Pytorch tensor operations (used in swin transformer) |
| scikit-learn           | 0.24.2          | Clustering algorithm                                 |
| scipy                  | 1.5.4           | IO ops and signal processing (in motif mining)       |
| natsort                | 8.2.0           | Customized dataset ops                               |
| tqdm                   | 4.63.1          | Terminal progress bar                                |

CUDA version: 11.x, personally I recommend 11.6 / 11.7 (since they are tested). GPU global memory requirement: > 3GB (easy to meet). For any further question about environment, please contact Qianyue He: he-qy22@mails.tsinghua.edu.cn

## II. Workspace

The structure of the code repo is as follows:

```
.
├── (d) logs/ --- tensorboard log storage (compulsory)
├── (d) model/ --- folder from and to which the models are loaded & stored (compulsory)
├── (d) check_points/ --- check_points folder (compulsory)
├── (f) train_conv.py --- Python main module for convolution NN training (problem 1)
├── (f) motif_mining.py --- Executable file for motif mining (problem 2)
├── (e) spectral_clustering.py --- Python main module for cell clustering (problem 3)
├── (e) train.sh --- Shell executable script for running train_conv.py
├── (e) motif_mine.sh --- Shell executable script for running motif_mining.py
├── (e) eval.sh --- Shell executable script for convolution NN evaluation (AUROC)
├── (e) cl_train.sh --- Shell executable script for contrastive learning MoCo training (problem 3)
├── (d) models --- CNN / Swin Transformer / MoCo-v3 model definition folder
	 ├── (f) seq_pred.py --- CNN model definition
	 └── (f) simple_moco.py	--- MoCo-v3 model definition
├── (d) utils --- Utility functions (including dataset preprocessing)
	 ├── (e) convert_dataset.sh --- Executable file for converting the dataset into format of my proj
	 ├── (f) cosine_anneal.py --- Exponential-decay-cosine-annealing learning rate scheduler
	 ├── (f) opt.py --- argparser: argument setting and parsing
	 ├── (f) train_helper.py --- model related ops helper function
	 ├── (f) utils.py --- Functions converting raw dataset into format of my proj
	 └── (f) dataset.py --- Customized torch dataset
├── (d) data/ --- Dataset converted using my code
	 ├── train_500/
	 ├── train_500_label/
	 ├── test_500/
	 ├── test_500_label/
	 ...
└── (d) data_project4/ --- raw dataset provided by TA
	 ├── train/
	 ├── test/
	 └── celltype.txt
```



---

## III. Setting up workspace

Now you have completed environment setup, it's time you get the workspace ready. I mean:

- Put the training data in the right place
- Pre-process the training data using my code

As you can see from the **II. Workspace**, in the workspace structure part line 22-31, I told you how to place the dataset correctly:

- Extract the raw dataset provided by TAs in the root folder, after that if the extracted folder is not named `data_project4`, please rename it to `data_project4` and make sure the structure is the same as line 29-31 (there can be more things, but the listed ones should not be absent or modified)
- create a folder called `data` to stored the pre-processed data.
- `cd utils/`. Change current directory to `utils/` in your terminal, then: `sudo chmod +x ./convert_dataset.sh`, this should make `convert_dataset.sh` directly executable. Note that if you use conda, you should run `conda activate prml_proj` first.
- Run the script: `./convert_dataset.sh`. The script calls `utils/utils.py` and converts the raw ATCG sequences into one-hot encoding, and stores them as files (uint8, therefore low storage memory requirement) for main module to load. The file will be stored in `data/`. Note that the creation of folders are automatic, you don't have to worry about that.
- After the process is finished, check if all the sequence length has its `train_xxx`, `train_xxx_label`, `test_xxx` and `test_xxx_label`. There should be `seq_00001.dat ...` in each folder.

Okay, everything is done. You should be able to use my code to do the training now. By the way, if anything goes wrong during dataset conversion, you can use the script in `data/` (`data/rm_dataset.sh`) to remove the wrongly generated dataset (if you want to do it by hand, be my guest, it's gonna take you some time).

---

## IV. Running the code

### 4.1 Problem 1 CNN Training

I do not recommend you to use the default parameter setting given by opt, which means: do not just run the following code:

```shell
python3 ./train_conv.py (not recommended)
```

I provided a executable script for you:

```shell
sudo chmod +x ./train.sh			# in case train.sh has no authority to be directly executed
./train.sh
```

You can modify the parameters written in the train.sh, like the name of output file, maximum / minimum learning rate and so on. Note that not all the params are written in train.sh, therefore for some parameters you should go to `utils/opt.py` to set them. If you have no idea what these params are for, just run:

```shell
python3 ./train_conv.py --help
```

For **Swin Transformer**, please checkout commit `cdade2256e10139e798c8` in branch `master`. For TAs who currently can not access the Github repo of mine (since I made it private), I provided the swin transformer version in the sup-materials.

### 4.2 Problem 1 CNN Evaluation

During training, tensorboard logger will be called. Therefore, (after you make sure `logs/` exists in the root dir), go to `logs/` and call a tensorboard visualization process in the terminal:

```shell
tensorboard --logdir ./
```

Tensorboard will start a web page visualization at `https://localhost:6006/`, use your web browser to check it out.

But this is actually not "Evaluation", if you wish to test the outcome of the model on the whole test dataset and visualize AUC distribution, run:

```shell
./eval.sh
```

Basically every procedure you do is consistent with training.

### 4.3 Problem 2 Motif Mining

Run:

```shell
./motif_mine.sh
```

Should work. In the script, if you add `-v`, a visualization function will be called to visualize the Gaussian smoothing and adaptive peak finding algorithm. `--motif_crop_size` can be set in the code and in the shell script.

### 4.4 Problem 3 Cell Clustering

If you want to try MoCo-v3, run:

```shell
./cl_train.sh
```

If `CUDA is not available` pops up in your terminal, please set `CUDA_VISIBLE_DEVICES=0` in `cl_train.sh`. MoCo-v3 requires you to have finished training the CNN (at least you should have a model `.pt` file no matter you trained it yourself or got it from me), loading the `.pt` file and use the weight of the output layer as input. The training is fast (although when used in clustering, the result is not good).

If you do not care about MoCo-v3 and just want to test the clustering result, just run:

```shell
python3 ./spectral_clustering.py
```

Of course, some parameters should be provided, you can use `python3 ./spectral_clustering.py --help` to find out.

---

## V. License

This repo is licensed under Apache License 2.0. Copyright @Qianyue He.

