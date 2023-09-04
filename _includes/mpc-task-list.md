This is my very tentative list of features to add to MaLPi. Although they are listed in numbered order, only the first one is 'real'. If/when I finish the first task I'll re-asses the list, possibly moving things around or dropping some or adding new ones. A checkmark means I've made some progress on that entry.

1. ✓ Switch to RNN policy
1. Start training with RL in adition to imitation learning
1. ✓ Output a single value with standard deviation from the testing script
1. VAE improvements
    * Add auxiliary outputs: CTE, track #, steering, throttle
    * Collect more training data from all tracks
    * Implement Pre-training paper
1. Add a small DNC as a working memory
1. Add more tasks
    * Cone/Human/dog/etc detector (bounding box as output)
    * Drive to goal with goal given as an image
    * Sketches as inputs (goals) and/or outputs
    * NLP description of a scene or task trajectory as output
1. Add more input and output modalities
    * Add an IMU and learn to detect crashes/bumps
1. Switch to more formal multi-task and/or meta-learning and/or lifelong learning
1. Add some form of Aggregate memory that includes all previous experience

Finished

1. VAE based DonkeyCar pilot
1. Train separate policies on each track with the VAE as an image embedder
1. Switch from fastai to Lightning
1. Train a single policy on multiple tracks

Model Predictive Control is a control method based on planning n-steps into the future to find the best path, as currently determined by the model. Take the first step. Then re-plan, take the new first step. Repeat until done.
