---
permalink: /mpc-task-list/
title: "MPC ML Task List"
excerpt: "A Model-Predictive-Control style list of ML tasks to do"
author_profile: true
redirect_from: 
  - /mpc-task-list.html
---

This is my very tentative list of features to add to MaLPi. Although they are listed in numbered order, only the first one is 'real'. If/when I finish the first task I'll re-asses the list, possibly moving things around or dropping some or adding new ones.

1. VAE based DonkeyCar pilot
1. Pre-train VAE on images from all but one track
1. Train separate policies on each track with the VAE as an image embedder
1. Switch to RNN policy
1. Train a single policy on multiple tracks/tasks
1. Add a small DNC as a working memory
1. Add an IMU and learn to detect crashes/bumps
1. Add more tasks
    * Cone/Human/dog/etc detector (bounding box as output)
    * Drive to goal with goal given as an image
    * Sketches as inputs (goals) and/or outputs
    * NLP description of a scene or task trajectory as output
1. Switch to more formal multi-task and/or meta-learning and/or lifelong learning
1. Add some form of Aggregate memory that includes all previous experience

Model Predictive Control is a control method based on planning n-steps into the future to find the best path, as currently determined by the model. Take the first step. Then re-plan, take the new first step. Repeat until done.
