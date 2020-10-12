# Background Modeling via Uncertainty Estimation
### Pytorch Implementation of '[Background Modeling via Uncertainty Estimation for Weakly-supervised Action Localization](https://arxiv.org/abs/2006.07006)'

![architecture](https://user-images.githubusercontent.com/16102333/84629197-e0989180-af24-11ea-8377-dfc590e74fbb.png)

> **Background Modeling via Uncertainty Estimation for Weakly-supervised Action Localization**<br>
> Pilhyeon Lee (Yonsei Univ.), Jinglu Wang (Microsoft Research), Yan Lu (Microsoft Research), Hyeran Byun (Yonsei Univ.)
>
> Paper: https://arxiv.org/abs/2006.07006
>
> **Abstract:** *Weakly-supervised temporal action localization aims to detect intervals of action instances with only video-level action labels for training. A crucial challenge is to separate frames of action classes from remaining, denoted as background frames (i.e., frames not belonging to any action class). Previous methods attempt background modeling by either synthesizing pseudo background videos with static frames or introducing an auxiliary class for background. However, they overlook an essential fact that background frames could be dynamic and inconsistent. Accordingly, we cast the problem of identifying background frames as out-of-distribution detection and isolate it from conventional action classification. Beyond our base action localization network, we propose a module to estimate the probability of being background (i.e., uncertainty [20]), which allows us to learn uncertainty given only video-level labels via multiple instance learning. A background entropy loss is further designed to reject background frames by forcing them to have uniform probability distribution for action classes. Extensive experiments verify the effectiveness of our background modeling and show that our method significantly outperforms state-of-the-art methods on the standard benchmarks - THUMOS'14 and ActivityNet (1.2 and 1.3).*


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
    - For convenience, we provide the features we used. You can find them [here](https://drive.google.com/file/d/1NqaDRo782bGZKo662I0rI_cvpDT67VQU/view?usp=sharing).
    
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
The pre-trained model can be found [here](https://drive.google.com/file/d/1W2LNpX-PO-FJ5gX3Zhrlv5hkH6xJKUrq/view?usp=sharing).
You can evaluate the model by running the command below.

~~~~
$ bash run_eval.sh
~~~~

## References
We note that this repo was built upon our previous model 'Background Suppression Network for Weakly-supervised Temporal Action Localization '. (AAAI 2020) [[paper](https://arxiv.org/abs/1911.09963)] [[code](https://github.com/Pilhyeon/BaSNet-pytorch)]

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
