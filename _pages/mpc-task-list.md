---
permalink: /mpc-task-list/
title: "MPC ML Task List"
excerpt: "A Model-Predictive-Control style list of ML tasks to do"
author_profile: true
redirect_from: 
  - /mpc-task-list.html
---

This is my very tentative list of features to add to MaLPi. Although they are listed in numbered order, only the first one is 'real'. If/when I finish the first task I'll re-asses the list, possibly moving things around or dropping some or adding new ones.

1. Lane detector -> reward function
	* Add more lane labels to existing data
	* Add support in dagger.py to use existing lane detector to speed up labeling.
    * Run existing detector with dropout turned on, choose n most uncertain images, label them, save to new .npz file, retrain lane detector
1. VAE based DonkeyCar pilot
1. Train on data and on WorldModel
1. Switch to RNN policy
1. Train one model for both UCSD and Home datasets
2. Add location as another auxiliary task.
1. Add DNC
1. Add more tasks
    * Cone/Human/dog/etc detector (bounding box as output)
    * Drive to goal with goal given as an image
    * Sketches as inputs (goals) and/or outputs
    * NLP description of a scene or task trajectory as output
1. Switch to more formal multi-task and/or meta-learning training

Model Predictive Control is a control method based on planning n-steps into the future to find the best path, as currently determined by the model. Take the first step. Then re-plan, take the new first step. Repeat until done.
