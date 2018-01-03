---
title: 'Normalizing image data before training - GRU version'
date: 2017-12-24
permalink: 2017-12-24-normalization-gru
excerpt: Testing image normalization when using a GRU.
tags:
  - numpy
  - malpi
  - gru
---

# Intro

Repeating my [previous testing](/017-12-21-normalization-lstms), this time using a GRU layer instead of an LSTM.

Below are results from three different image preprocessing methods, each trained five times with five different weight initializations and five different batch randomizations. Lines are the mean of the five runs, with error bars showing one standard deviation. 'test' in the plots below is actually the results on validation data generated using Keras's validation\_split argument to the fit method.

# Methods and Results

First is my initial code with no normalization, just dividing by 255 to get values from 0.0 to 1.0.

```python
# After loading images from multiple directories into the list 'images'
images = np.array(images)
images = images.astype(np.float) / 255.0
```

![No Image Normalization](/images/blog/2017-12/GRU_No_Norm.png "No Normalization")

Second is centering the mean on zero, separately for each color band.

```python
# After loading images from multiple directories into the list 'images'
images = np.array(images)
images = images.astype(np.float)
images[:,:,:,0] -= np.mean(images[:,:,:,0])
images[:,:,:,1] -= np.mean(images[:,:,:,1])
images[:,:,:,2] -= np.mean(images[:,:,:,2])
```

![Mean Subtraction](/images/blog/2017-12/GRU_Norm.png "Mean Centered on Zero")

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

![With Normalization](/images/blog/2017-12/GRU_Norm_std.png  "Mean Centered on Zero and divide by the Standard Deviation")

The code used to run these experiments was [this commit](https://github.com/Bleyddyn/malpi/commit/94cf17c82b90910b2b44f2e82b5c0cca289be47f). I manually changed the lines listed above for the three separate runs and I changed one line in mode\_keras.py to use a GRU layer instead of LSTM. Total number of training samples was approximately 7000.

# Conclusions

Looks very similar to the LSTM tests. Given the GRU has fewer parameters than the LSTM, I will probably switch to using that.

# Models

* [Fully Connected](/2017-12-13-image-normalization)
* [LSTM](/2017-12-21-normalization-lstm)
* [GRU](/2017-12-24-normalization-gru)

# Future

* Retrain the best of these experiments with more data.
* Run speed tests on MaLPi with all three versions.

---
