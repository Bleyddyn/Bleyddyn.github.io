---
permalink: /notes/
title: "Miscellaneous Notes"
author_profile: true
redirect_from: 
  - /notes.html
---

# Miscellaneous

The end of [this OpenAI blog post](https://blog.openai.com/baselines-acktr-a2c/) says that their baselines repo includes LSTM implementations. Mabye it's time to abandon my own code and start using one of their baselines.

[Derivative Rules](http://www.mathsisfun.com/calculus/derivatives-rules.html)

[And](https://en.wikipedia.org/wiki/Differentiation_rules)

[Matrix Profiles](http://www.cs.ucr.edu/%7Eeamonn/Matrix_Profile_Tutorial_006.pdf), a method of analyzing time series data. Usefull for pulling features out of the accellerometer data?

[Different Softmax methods](https://arxiv.org/abs/1612.05628)

[Long explanation of RNN's and LSTM's](https://ayearofai.com/rohan-lenny-3-recurrent-neural-networks-10300100899b)

[Tips for Training Recurrent Neural Networks](http://danijar.com/tips-for-training-recurrent-neural-networks/)

[MathJax Tutorial](https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference)

[37 Reasons why your Neural Network is not working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)

[Increasing the Action Gap: New Operators for Reinforcement Learning](https://arxiv.org/abs/1512.04860). This looked interesting, but it wasn't clear enough for me to figure out how to implement it in my own code. I think the idea is to replace the Q-Learning update with one that increases the gap between the optimal action and sub-optimal actions. Their results show better performance on Atari games.

# History
From [this article](http://sdtimes.com/realities-machine-learning-systems/):

"...in 1957, psychologist Frank Rosenblatt invented the perceptron, or an algorithm for supervised learning of binary classifiers."

Find or write up a brief description of the Perceptron and maybe a couple of other important ML advances.


# Classes/Education

## University Reinforcement Learning classes

* [CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)
* [CMU 10703, Spring 2017 Deep Reinforcement Learning and Control](https://katefvision.github.io)
* [Stanford CS234: Reinforcement Learning](http://web.stanford.edu/class/cs234/index.html)
* [David Silver's UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)

## Statistics

From [this reddit comment](https://www.reddit.com/r/MachineLearning/comments/6llhit/d_softmax_interpretation_with_non_1hot_labels/djvsw88/).

I'm not sure about any online courses (I've only done the same two as you) but in regards to books I'd suggest (to be read in this order):

1. An Introduction to Statistical Learning by Hastie et al
1. The Elements of Statistical Learning by Hastie et al
1. Machine Learning A Probabilistic Perspective by Murphy
1. Deep Learning by Goodfellow et al.

You've probably seen these suggested a million times before, but I read through these while I was (and still am!) struggling to get to grips with the maths behind the ML concepts and they cleared up some stuff!

Edit: I also really enjoyed http://u.cs.biu.ac.il/~yogo/nnlp.pdf and https://arxiv.org/pdf/1511.07916 too. Both are more 'tutorials' than books and are both focussed on NLP but are (IMO) incredibly well written (even I could understand them both!) and not too long. Would definitely recommend.


[How to Learn Deep Learning when you're not a CS PhD](https://vimeo.com/214233053)

[A concise introductory course on probabilistic graphical models](https://ermongroup.github.io/cs228-notes/)

[Bayes, SVM, Decision Trees, and Ensembles in sklearn](https://github.com/savan77/Practical-Machine-Learning-With-Python/blob/master/Part%20-%202/Practical%20Machine%20Learning%20With%20Python%20-%20Part%202.ipynb)

[Free electronics textbook](https://www.circuitlab.com/textbook/), work in progress.

[How to Read a Paper](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf)

# [The Myth of a Superhuman AI](https://backchannel.com/the-myth-of-a-superhuman-ai-59282b686c62)

1. Intelligence is not a single dimension, so “smarter than humans” is a meaningless concept.

   I completely agree with the first part, but even in his own text he shows many examples of how "smarter than human's" is not at all meaningless. "AlphaGo is smarter than the best human Go player, possibly smarter than all humans put together." That's completely accurate and not at all meaningless as long as it's understood in the context of playing Go. There are also plenty of mathematical tools for reducing dimensionality so that we should be able to reason about and compare different minds under a high dimensional concept of intelligence.

2. Humans do not have general purpose minds, and neither will AIs.

   It seems to me that this is one of the dimensions of intelligence, with some types of minds more general purpose and others not so much. Right now humans are probably farther along toward general purpose than any other minds we know of. Whether AI's will ever be more general purpose than us is an open question. Or even whether they'll ever be much more than single purpose, e.g. Alpha Go or autonomous vehicles.

3. Emulation of human thinking in other media will be constrained by cost.

   This title seems to be misleading since human thinking itself is also constrained by cost. I don't think I've ever heard of a Machine Learning system that was **more** expensive than the human system it was meant to replace. What would be the point in developing that?

   The section where he discusses this title doesn't actually seem to talk about costs but rather about how similar or dissimilar non-human minds will be from our own.

4. Dimensions of intelligence are not infinite.

   Almost certainly true. However my very limited understanding is that the human brain is many orders of magnitude away from theoretical limits of computation. Computation limit of the mass of a human brain, based on [Bremermann's Limit](https://en.wikipedia.org/wiki/Bremermann%27s_limit): ~2 x 10^50 bits per second. Estimates of the actual computational power of a human brain from [Merkle](http://www.merkle.com/brainLimits.html): 10^13 to 10^16 operations per second. How those two units compare, I'm not sure.

5. Intelligences are only one factor in progress.

   This argument is by far the most persuasive.
