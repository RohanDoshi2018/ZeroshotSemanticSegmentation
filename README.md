# Zero-Shot Semantic Segmentation

> 
- Princeton University Senior Thesis
- Advisor: Professor Olga Russakovsky, Princeton University Department of Computer Science
- Part of the [Princeton VisualAI Lab](http://visualai.princeton.edu/people.html) 

## Project Overview

We propose a novel computer vision task: zero-shot semantic segmentation. We are building upon the work of [DeViSE](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41473.pdf), but make novel strides by making predictions at the pixel-level as opposed to an image-level.

Our methodology maps mapping pixel-label pairs into a joint-embedding space of visual and semantic embeddings. A visual model maps the pixels in an image to pixel-embeddings, and a semantic model maps the corresponding labels to semantic-embeddings. A mean square loss is used to minimize the distance between visual-embeddings and semantic-embeddings in the joint-embedding space (normally between 20 and 100 dimensions, depending on GPU constraints).

The visual model is adapted from a Fully Convolutional Network (FCN) implemented in PyTorch ([adapted from existing work](https://github.com/wkentaro/pytorch-fcn)); the convolutional layers are pretrained using weights from [VGG-16](https://arxiv.org/abs/1409.1556) on the ImageNet task; we've added additional fully connected layers to map the model to the embedding space.

The semantic model takes a [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) model and reduces pre-trained 300-dimension word embeddings (trained on the Google News corpus of 3B words) via PCA to the joint embedding. These semantic-embeddings are taken as static "ground truths" in the joint-embedding embedding; during train, the visual model tries to map pixel-embeddings to the corresponding semantic-embeddings.

Initially, we are working with the Semantic Boundaries Dataset, a supplemented version of PASCAL-VOC 2011 with ~6000 training and ~2000 validation images. PASCAL-VOC has 20 classes + background. We train the visual model on pixel-label pairs across a subset of 10 classes. And, we test with predictions across all 21 classes. We use nearest neighboring embedding (NNE) to predict labels for each pixel-embedding.


#### What's Been Done

1. I've implemented the visual model. I've baselined my FCN and achieved state of the art performance. I've hit 63% IoU for semantic segmentation on PASCAL VOC 2012 using a traditional softmax loss.

2. I've implemented the semantic model and PCA'ed all of the embeddings from the Word2Vec model from 300 to 20, 50, and 100 dimensions.

3. I've implemented a mean square loss function to minimize the distance between the visual-embeddings and semantic-embeddings. We have a full model now.

4. I've implemented nearest neighboring embeddings for predictions during testing. This may still have bugs.

5. I've done a grid search across various hyper-parameters to deal with numerical instabilities during training (e.g. exploding loss functions). Using a learning rate of 1e-15, after training for ~11 epochs (100,000 images) using an embedding space of 50 dimensions, I've finally stabilized the training loss around 1,000 (as opposed to it growing to infinity).


#### What's Next

1. Once I finish debugging NNE predictions, I need to get a baseline accuracy its accuracy compare it to the baseline softmax loss.

2. I need to continue my hyperparameter search to finetune the model to train with joint-embedding spaces of different sizes.

3. I need to finish the pixel-masking logic in the dataloader so we can baseline the zero-shot learning accuracy.


#### Open Research Questions

Once I finish implementing and baselining my model (see above), I want to explore three broad themes of questions:

1. **Numerical Stability**
What is the optimal size of the joint-embedding space? How can we make the model easier to train without gradients exploding? Should we normalize the norm of pixel-embeddings and semantic embeddings to unit length? Should we use a more intelligent loss function that weights samples by the number of relevant pixels in an image or by class frequency priors? What sort of heuristics can we develop to predict hyper-parameters (e.g. learning rate) for different embedding spaces?
2. **Accuracy** The DeViSE zero-shot accuracy at the image-level were already pretty low; how can we make sure there isn't a significant accuracy drop as we extend to pixel-level of prediction? How can we create richer visual-embeddings that capture more local and global information (e.g. replacing the FCN with [Jia Deng's stacked hourglass network](https://arxiv.org/abs/1603.06937) )? What are some smarter loss functions (e.g. hinge loss)? Can we better incorporate structure into the joint-embedding space using  WordNet hierarchy? Is the nearest NNE approach for predictions too naive? 
3. **Robustness** How does this methodology generalize to different datasets? What does the strengths and weaknesses across different datasets tell us about the strengths and weaknesses of our approach (and, what does visualizing the joint-embedding space across datasets tell us)? How does our model work with different seen-unseen splits between classes during training?

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
source activate thesis

6. Run code
```
python train.py -c 3 --data_dir data;
```

7. [Optional] Run Tensorboard Server. Use Ngrok tunnel to access server remotely.
```
tensorboard --logdir /opt/visualai/rkdoshi/ZeroshotSemanticSegmentation/tb
```

## Zero-Shot Class Split

#### SEEN (10 classes)

'aeroplane',     # class 1 VEHICLES (3x)

'bicycle',       # class 2

'boat',          # class 4

'bird',          # class 3 ANIMALS (4x)

'cat',           # class 8

'cow',           # class 10

'dog',           # class 12

'bottle',        # class 5 HOUSEHOLD OBJECTS (3x)

'chair',         # class 9

'diningtable',   # class 11

#### UNSEEN (10 classes)

'bus',           # class 6 VEHICLES (4x)

'car',           # class 7

'motorbike',     # class 14

'train',         # class 19

'horse',         # class 13 ANIMALS (3x)

'person',        # class 15

'sheep',         # class 17

'potted plant',  # class 16 HOUSEHOLD OBJECTS (3x)

'sofa',          # class 18

'tv/monitor',    # class 20