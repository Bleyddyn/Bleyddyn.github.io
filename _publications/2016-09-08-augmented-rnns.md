---
title: "Attention and Augmented Recurrent Neural Networks"
collection: publications
permalink: /publications/2016-09-08-augmented-rnns
excerpt: 'Overview (with references) of attention and several types of augmentation for RNNs.'
date: 2016-09-08
paperurl: https://distill.pub/2016/augmented-rnns/
usemath: true
---

Authors: Olah and Carter

# Neural Turing Machines

Differentially addressable memory (stored as vectors). Can be addressed by content, i.e. similarity to an input vector, or location based, or a combination. Addressing is done not as pointers but as a distribution of 'attention' over all of the memory locations. Memory can be read from and written to.

# Attention

One RNN generate a sequence of outputs, e.g. a representation of an input sequence of characters. A second RNN generates some context at each time step that is combined with the output of the first RNN and a softmax to create a distribution over the output of the first RNN which is then used as the input for that step of the second RNN.

# Adaptive Computation Time

Each time step is broken up into a variable number of computation steps with the output for that time step being a weighted combination of the computation steps.

# Neural Programmer

RNN generates a sequence of operations (or distributions over operations) which are then applied sequentially.
