---
title: 'Effect of Dropout Layers in a VAE'
date: 2020-01-14
permalink: /2020-01-14-vae-dropout
excerpt: Dropout layers make VAE output worse
tags:
  - malpi
  - vae
---

# Intro

I had two slightly different versions of my Variational Auto Encoder code, and I have two different datasets, one collected from a track and one from my patio. I noticed that the sampled and reconstructed images from my patio looked much better (clearer, more detail) than the ones from the track. That seemed odd since there are more images in the track dataset so I would have thought that VAE would train better.

Looking through the code the biggest difference I could see was that the patio dataset had dropout turned off, but the track had it turned on with a dropout rate of 0.2. So I merged the code and trained the track dataset with three levels of dropout: None, 0.2 and 0.4.

The run with no dropout was clearly better, both the images and the loss curves.
 
# Methods and Results

## Sampled and Reconstructed Images

### No Dropout

![](/images/blog/2020-01/VAE_no_dropout_v2.png "VAE No Dropout")

### 0.2 Dropout

![](/images/blog/2020-01/VAE_dropout_0.2_v2.png "VAE Dropout 0.2")

### 0.4 Dropout

![](/images/blog/2020-01/VAE_dropout_0.4.png "VAE Dropout 0.4")

## Training and Validation loss curves

### No Dropout

![](/images/blog/2020-01/VAE_no_dropout_loss_v2.png "VAE No Dropout Loss")

### 0.2 Dropout

![](/images/blog/2020-01/VAE_dropout_0.2_loss_v2.png "VAE Dropout 0.2 Loss")

### 0.4 Dropout

![](/images/blog/2020-01/VAE_dropout_0.4_loss.png "VAE Dropout 0.4 Loss")


# Conclusions

Don't use Dropout layers in a VAE.

At least not in the encoder. I have seen example code that has dropout only in the decoder, so maybe that's an option.

# Future

Ever since I read about them ([a more recent example](https://gaborvecsei.github.io/Monte-Carlo-Dropout/)) I've been using SpatialDropout2D layers right after Convolution layers, but maybe I should repeat this experiment with standard Dropout layers.

I could also try with Dropout layers only in the decoder.

Figure out how to get better plots from my Google Colab runs.

---
