# Zero-Shot Semantic Segmentation

> 
- Princeton University Senior Thesis
- Advisor: Professor Olga Russakovsky, Princeton University Department of Computer Science
- Part of the [Princeton VisualAI Lab](http://visualai.princeton.edu/people.html) 

## Project Overview

This is the PyTorch implementation of the seenmask zeroshot network (SZN) described in Rohan Doshi's senior thesis "Zero-shot Semantic Segmentation." Please reference this paper (rohan_doshi_senior_thesis.pdf) to understand the code.

## Installation

** Requirements: **  Conda (with Python 3) and Linux

1. Install Conda 

2. Clone repository
```bash
git clone https://github.com/RohanDoshi2018/ZeroshotSemanticSegmentation.git
cd ZeroshotSemanticSegmentation
```

3. Create new conda environment
```bash
conda create --name thesis_env
```

4. Install Dependencies
```bash
conda install pytorch torchvision -c pytorch
pip install pytz pyyaml scipy fcn jupyter tensorflow tensorboardX
```

5. Activate your conda environment
source activate thesis_env

6. Run code
```
./train.py -c 4 -g 0
```

7. [Optional] Run Tensorboard Server. Use Ngrok tunnel to access server remotely.
```
tensorboard --logdir /opt/visualai/rkdoshi/ZeroshotSemanticSegmentation/tb
```
