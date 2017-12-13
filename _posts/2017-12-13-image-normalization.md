---
title: 'Normalizing image data before training'
date: 2017-12-13
permalink: 2017-12-13-image-normalization
excerpt: I had completely forgotten to normalize the images I'm feeding into MaLPi's network, so I thought I'd try to be a bit more formal about it than my usual.
tags:
  - numpy
  - malpi
---

# Intro

I had completely forgotten to normalize the images I'm feeding into MaLPi's network, so I thought I'd try to be a bit more formal about it than my usual.

Below are results from three different image preprocessing methods, each trained five times with five different weight initializations and five different batch randomizations. Lines are the mean of the five runs, with error bars showing one standard deviation. 'test' in the plots below is actually the results on validation data generated using Keras's validation_split argument to the fit method.

# Methods and Results

First is my initial code with no normalization, just dividing by 255 to get values from 0.0 to 1.0.

```python
# After loading images from multiple directories into the list 'images'
images = np.array(images)
images = images.astype(np.float) / 255.0
```

![No Image Normalization](/images/blog/2017-12-no-norm.png "No Normalization")

Second is centering the mean on zero, separately for each color band.

```python
# After loading images from multiple directories into the list 'images'
images = np.array(images)
images = images.astype(np.float)
images[:,:,:,0] -= np.mean(images[:,:,:,0])
images[:,:,:,1] -= np.mean(images[:,:,:,1])
images[:,:,:,2] -= np.mean(images[:,:,:,2])
```

![Mean Subtraction](/images/blog/2017-12-norm.png "Mean Centered on Zero")

Third is the above plus dividing each color band by its standard deviation.

```python
# After loading images from multiple directories into the list 'images'
images = np.array(images)
images = images.astype(np.float)
images[:,:,:,0] -= np.mean(images[:,:,:,0])
images[:,:,:,1] -= np.mean(images[:,:,:,1])
images[:,:,:,2] -= np.mean(images[:,:,:,2])
images[:,:,:,0] /= np.std(images[:,:,:,0])
images[:,:,:,1] /= np.std(images[:,:,:,1])
images[:,:,:,2] /= np.std(images[:,:,:,2])
```

![With Normalization](/images/blog/2017-12-norm-std.png  "Mean Centered on Zero and divide by the Standard Deviation")

# Conclusions

While I would prefer to focus on validation accuracy as my main metric, I can't because my model apparently isn't generalizing at all. Considering these were all trained on only about seven thousand samples, I'm not surprised.

Go figure, but normalization of the inputs helps. I was a bit surprised that dividing by the standard deviation helped. According to the CS231n notes on [Data Preprocessing](http://cs231n.github.io/neural-networks-2/) it shouldn't help much.

# Future

* Collect more data.
* Repeat all of the above with my LSTM model.

------
