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

* Minimum: 0.000
* Maximum: 104.000
* Avg/std: 31.470/38.248

Looking at the plot of rewards versus updates, I think I could drop back down to 40e6 timesteps.

![rewards vs updates](/images/blog/2017-09-progress-e1.png "Rewards vs Updates")

The way rewards were originally being logged doesn't make much sense. It looks like Gym considers each point to an episode, by returning done from the step() method. Breakout itself considers 5 points to be a game, so in my later testing I sum scores for each 5 'episodes' and report that.

# Original Model

I partially reproduced OpenAI's model in Keras so I could output a summary even though it didn't include filter size and stride.

Layer (type)          |      Output Shape         |    Param #   | Size    |  Stride |
----------------------|---------------------------|--------------|---------|---------|
conv2d_1 (Conv2D)     |      (None, 20, 20, 32)   |    8224      |   8x8   |    4    |
conv2d_2 (Conv2D)     |      (None, 9, 9, 64)     |    32832     |   4x4   |    2    |
conv2d_3 (Conv2D)     |      (None, 7, 7, 32)     |    18464     |   3x3   |    1    |
flatten_1 (Flatten)   |      (None, 1568)         |    0         |         |         |
dense_1 (Dense)       |      (None, 512)          |    803328    |         |         |
dense_2 (Dense)       |      (None, 4)            |    2052      |         |         |

Total params: 864,900

Trainable params: 864,900

# 3x3 Model

Layer (type)          |      Output Shape         |    Param #   | Size    |  Stride |
----------------------|---------------------------|--------------|---------|---------|
conv2d_1 (Conv2D)     |      (None, 21, 21, 32)   |    1184      |   3x3   |    4    |
conv2d_2 (Conv2D)     |      (None, 10, 10, 64)   |    18496     |   3x3   |    3    |
conv2d_3 (Conv2D)     |      (None, 8, 8, 32)     |    18464     |   3x3   |    1    |
flatten_1 (Flatten)   |      (None, 2048)         |    0         |         |         |
dense_1 (Dense)       |      (None, 512)          |    1049088   |         |         |
dense_2 (Dense)       |      (None, 4)            |    2052      |         |         |

Total params: 1,089,284

Trainable params: 1,089,284

# Testing of checkpoints

I ran checkpointed versions of each model on 10 games and plotted the average reward along with error bars at 1 standard deviation. These plots use the per-game method of figuring reward.

## The original model (8x8, 4x4, 3x3 filters)

![Original Model](/images//blog/2017-09-test_rewards_e1.png "Rewards vs Updates")

## 3x3 Filters

![3x3 Filters](/images/blog/2017-09-test_rewards_e2.png "Rewards vs Updates")
------
