# Background Modeling via Uncertainty Estimation
Pytorch Implementation of '[Background Modeling via Uncertainty Estimation for Weakly-supervised Action Localization](https://arxiv.org/abs/2006.07006)'

![architecture](https://user-images.githubusercontent.com/16102333/84629197-e0989180-af24-11ea-8377-dfc590e74fbb.png)

## Prerequisites
### Recommended Environment
* Python 3.5
* Pytorch 1.0
* Tensorflow 1.15 (for Tensorboard)

### Depencencies
You can set up the environments by using `$ pip3 install -r requirements.txt`.

### Data Preparation
1. Prepare [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    - We excluded three test videos (270, 1292, 1496) as previous work did.

2. Extract features with two-stream I3D networks
    - We recommend extracting features using [this repo](https://github.com/piergiaj/pytorch-i3d).
    - For convenience, we provide the features we used. You can find them [here](https://drive.google.com/open?id=1NsVN2SPYEcS6sDnN4sfv2cAl0B0I8sp3).
    
3. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.
   
~~~~
├── dataset
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       └── features
           ├── train
               ├── rgb
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
               └── flow
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
           └── test
               ├── rgb
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
               └── flow
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
~~~~

## Usage

### Running
You can easily train and evaluate the model by running the script below.

If you want to try other training options, please refer to `options.py`.

~~~~
$ bash run.sh
~~~~

### Evaulation
The pre-trained model can be found [here](https://drive.google.com/file/d/1EezVHSoL6cs7Hy3MNOs7ihbVG6ErpUeg/view?usp=sharing).
You can evaluate the model by running the command below.

~~~~
$ bash run_eval.sh
~~~~

## References
We note that this repo was built upon our previous model 'Background Suppression Network for Weakly-supervised Temporal Action Localization ' (AAAI 2020). [[paper](https://arxiv.org/abs/1911.09963)] [[code](https://github.com/Pilhyeon/BaSNet-pytorch)]

We also referenced the repos below for the code.

* [STPN](https://github.com/bellos1203/STPN)
* [ActivityNet](https://github.com/activitynet/ActivityNet)

## Citation
If you find this code useful, please cite our paper.

~~~~
@article{lee2020BMUncertainty,
  title={Background Modeling via Uncertainty Estimation for Weakly-supervised Action Localization},
  author={Pilhyeon Lee and Jinglu Wang and Yan Lu and Hyeran Byun},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.07006}
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Pilhyeon Lee (lph1114@yonsei.ac.kr).
