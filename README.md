# WTAL-Uncertainty-Modeling
### Official Pytorch Implementation of '[Weakly-supervised Temporal Action Localization by Uncertainty Modeling](https://arxiv.org/abs/2006.07006)' (AAAI 2021)

![architecture](https://user-images.githubusercontent.com/16102333/102174520-03f6c600-3ee1-11eb-953b-ffce66d1ccbe.png)

> **Weakly-supervised Temporal Action Localization by Uncertainty Modeling**<br>
> Pilhyeon Lee (Yonsei Univ.), Jinglu Wang (Microsoft Research), Yan Lu (Microsoft Research), Hyeran Byun (Yonsei Univ.)
>
> Paper: https://arxiv.org/abs/2006.07006
>
> **Abstract:** *Weakly-supervised temporal action localization aims to learn detecting temporal intervals of action classes with only video-level labels. To this end, it is crucial to separate frames of action classes from the background frames (i.e., frames not belonging to any action classes). In this paper, we present a new perspective on background frames where they are modeled as out-of-distribution samples regarding their inconsistency. Then, background frames can be detected by estimating the probability of each frame being out-of-distribution, known as uncertainty, but it is infeasible to directly learn uncertainty without frame-level labels. To realize the uncertainty learning in the weakly-supervised setting, we leverage the multiple instance learning formulation. Moreover, we further introduce a background entropy loss to better discriminate background frames by encouraging their in-distribution (action) probabilities to be uniformly distributed over all action classes. Experimental results show that our uncertainty modeling is effective at alleviating the interference of background frames and brings a large performance gain without bells and whistles. We demonstrate that our model significantly outperforms state-of-the-art methods on the benchmarks, THUMOS'14 and ActivityNet (1.2 & 1.3).*


## Prerequisites
### Recommended Environment
* Python 3.6
* Pytorch 1.6
* Tensorflow 1.15 (for Tensorboard)
* CUDA 10.2

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
The pre-trained model can be found [here](https://drive.google.com/file/d/1nStGuSenq-8eCKG9-jpgTMF6LJ8tw7_a/view?usp=sharing).
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
@inproceedings{lee2021WTAL-Uncertainty,
  title={Weakly-supervised Temporal Action Localization by Uncertainty Modeling},
  author={Pilhyeon Lee and Jinglu Wang and Yan Lu and Hyeran Byun},
  booktitle={The 35th AAAI Conference on Artificial Intelligence},
  pages={1854--1862},
  year={2021}
}
~~~~

## Contact
If you have any question or comment, please contact the first author of the paper - Pilhyeon Lee (lph1114@yonsei.ac.kr).
