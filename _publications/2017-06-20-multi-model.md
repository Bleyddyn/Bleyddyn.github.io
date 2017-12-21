---
title: "One Model To Learn Them All"
collection: publications
permalink: /publications/2017-06-20-multi-model
excerpt: 'A single ML model used for very different tasks.'
date: 2017-06-16
paperurl: https://arxiv.org/abs/1706.05137
usemath: false
---

## Notes

The main question they ask: Can we create a unified deep learning model to solve tasks across multiple domains?

Four main parts to the model:
* Input encoder
* I/O mixer
* Output decoder
* modality inputs (one for each input/output type: text data, images, audio categorical data)

The first three are made up using convolution blocks, mixture of experts, and attention blocks

They trained the model on eight different tasks: ImageNet, COOC image captioning, WSJ speech, WSJ text parsing, and two pairs of language translation tasks. Overall performace was less than SOTA on any given task, but that's before any major attempt at hyperparameter optimization. Training on eight tasks improved performance vs training on a single task, in most cases.

Lots of links in this article to implementation details for the model parts.
