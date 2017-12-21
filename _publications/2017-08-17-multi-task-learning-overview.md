---
title: "An Overview of Multi-Task Learning in Deep Neural Networks"
collection: publications
permalink: /publications/2017-08-17-multi-task-learning-overview
excerpt: "A long review of the use of DL in robotics"
date: 2017-08-17
paperurl: https://arxiv.org/abs/1706.05098
usemath: false
---

## Older ideas

1. Hard sharing  
One or more layers are shared between tasks.
2. Soft sharing  
One or more layers are constrained between tasks. e.g. with distance measures.

## Newer ideas

1. Deep Relationship Networks.  
In addition to the fully shared layers, the non-shared fully connected layers have priors on them that lets them learn the relationships between tasks.
2. Fully-adaptive feature sharing.  
Starts with a single network that widens during training by grouping similar tasks and giving each group separate branches of the network.
3. Cross-stich networks.  
Learned, linear combinations of the outputs of previous layers from multiple task-specific networks.
4. Low supervision.  
I don't understand this one.
5. Joint multi-task model.  
Another NLP example I don't get.
6. Multi-task loss.  
With the loss for each task weighted by its uncertainty.
7. Tensor factorisation.  
Split the model parameters into shared and task-specific for each layer.
8. Sluice networks.  
The authors genralization of multiple MTL methods.

## Auxiliary tasks

1. Related task.  
Lots of examples, but hard to define 'related'.
2. Adversarial.  
Possible it's easier to define the opposite of the desired loss function?
3. Hints.  
Example is "predict whether an input sentence contains a positive or negative sentiment word as auxiliary tasks for sentiment analysis." Not sure I understand it.
4. Focusing attention.  
Example is predicting lane markings when network might ignore them as being too small a part of the image.
5. Quantization smoothing.  
Use a continuous version of a discrete label as an auxiliary task.
6. Predicting inputs.  
?
7. Using the future to predict the present.
8. Representation learning.  
All auxiliary tasks do this implicity, but it can be made more explicit. One example is an autoencoder objective.
9. What auxiliary tasks are helpful?  
Still little theoretical basis for 'task similarity'.

