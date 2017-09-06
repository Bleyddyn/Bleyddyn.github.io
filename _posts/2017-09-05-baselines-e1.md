---
title: 'Experimenting with OpenAIs Baselines code'
date: 2017-09-05
permalink: /posts/2017/09/baselines-e1/
tags:
  - openai
  - baselines
  - model_search
---

I forked Open AI's baseline code and made a few changes. This was my first full run before I started playing around with the model architecture.

Changes from OpenAI's:
* Turned on Logging, including Tensorboard output
* Log rewards
* Add a command line option for setting number of cpus

Code: [Commit used](https://github.com/Bleyddyn/baselines/commit/0439b162e4c03e038884abd2edd1b9a87a6df2cf)

Command line: python3 ./run_atari.py --cpu=3 -m=200

I increased the nsteps parameter to 200 and ran for 200e6 timesteps, which I think offsets my very low number of cpu's.

I save checkpoints every 5000 updates, and also save one after the very last update.

Results of playing Breakout 100 times with the final trained acktr model from above (rewards):

Minimum: 0.000

Maximum: 104.000

Avg/std: 31.470/38.248

Looking at the plot of rewards versus updates, I think I could drop back down to 40e6 timesteps.

![rewards vs updates](/images/posts/baselines-e1-progress.png "Rewards vs Updates")

# Model

I partially reproduced OpenAI's model in Keras so I could output a summary.

Layer (type)          |      Output Shape         |    Param #   
----------------------|---------------------------|--------------
conv2d_1 (Conv2D)     |      (None, 20, 20, 32)   |    8224      
conv2d_2 (Conv2D)     |      (None, 9, 9, 64)     |    32832     
conv2d_3 (Conv2D)     |      (None, 7, 7, 32)     |    18464     
flatten_1 (Flatten)   |      (None, 1568)         |    0         
dense_1 (Dense)       |      (None, 512)          |    803328    
dense_2 (Dense)       |      (None, 4)            |    2052      

Total params: 864,900

Trainable params: 864,900

------
