---
title: "Papers To Read"
permalink: /papers-to-read
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

# Sections

* [Papers read](/papers-read)
* [Papers to Read, by Category](#papers-to-read-by-category)
* [Reviews](#reviews)
* [Relevant to MaLPi](#relevant-to-malpi)
* [Not as relevant to MaLPi, but interesting](#not-as-relevant-to-malpi-but-interesting)
* [Autoencoders](#autoencoders)
* [Classes/Education](#classeseducation)
* [Simulators](#simulators)

## Papers to Read, by Category

* [Bayesian Neural Nets](/bayesian-nets)
* [Actor/Critic papers](/actor-critic)
* [Multi-Task papers](/multi-task-learning)

### Reviews

* [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)
* [Continual Lifelong Learning with Neural Networks: A Review](https://arxiv.org/abs/1802.07569)
* [A Survey of Deep Learning Techniques for Mobile Robot Applications](https://arxiv.org/abs/1803.07608)

### Relevant to MaLPi

* [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/)
  * [Arxiv link](https://arxiv.org/abs/1705.05363)
* Learning Atari: An Exploration of the A3C Reinforcement Learning Methods.
  * This paper is from Berkeley class, but I don't have a direct link for it. Google search should work.
* [A Robust Adaptive Stochastic Gradient Method for Deep Learning](https://arxiv.org/abs/1703.00788)
* [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)
  * [PCL implementation](https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py)
* [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732)
* [On Generalized Bellman Equations and Temporal-Difference Learning](https://arxiv.org/abs/1704.04463)
* [Neural Episodic Control](https://arxiv.org/abs/1703.01988)
* [Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440)
  * [Reddit discussion](https://www.reddit.com/r/MachineLearning/comments/6bi6np/d_glearning_taming_the_noise_in_reinforcement/)
* [Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310)
  * Replacing epsilon greedy exploration with a generalized count-based exploration strategy.
* [One-Shot Imitation Learning](https://arxiv.org/abs/1703.07326)
* [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862v2)
* [Recurrent Additive Networks](http://www.kentonl.com/pub/llz.2017.pdf)
  * A simpler type of RNN. Not sure if/where it's been published. Only tested on language tasks?
* [Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1705.06769.pdf)
  * Followup to the auxiliary tasks paper.
* [Non-Markovian Control with Gated End-to-End Memory Policy Networks](https://arxiv.org/abs/1705.10993)
* [Experience Replay Using Transition Sequences](https://arxiv.org/abs/1705.10834)
* [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  * A replacement for RELU activation. Looks fairly simple to implement and try. A quote from the abstract, "...thus, vanishing and exploding gradients are impossible."
* [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551)
  * Rewards for completing tasks given in written instructions.
* [MEC: Memory-efficient Convolution for Deep Neural Network](https://arxiv.org/abs/1706.06873)
* [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389)
  * CNN -> LSTM architecture
* [Neural SLAM](http://arxiv.org/abs/1706.09520v1)
* [Expected Policy Gradients](https://arxiv.org/abs/1706.05374)
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
  * Replace e-greedy or entropy methods of exploration with noisy parameters
* [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732)
* [Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173)
  * [Blog](https://owainevans.github.io/blog/hirl_blog.html)
  * Very nice idea of having a layer between the agent and the environment for preventing disastrous behavior.
  * Initially handled by a human but later by a learned system.
* [Bayesian Neural Networks with Random Inputs for Model Based Reinforcement Learning](https://medium.com/towards-data-science/bayesian-neural-networks-with-random-inputs-for-model-based-reinforcement-learning-36606a9399b4)
  * I read through this once, but don't understand most of it.
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  * From OpenAI.org: "outperforms other online policy gradient methods"
* [Better Exploration with Parameter Noise](https://blog.openai.com/better-exploration-with-parameter-noise/)
  * Looks like I would need [Layer Normalization](https://arxiv.org/abs/1607.06450) first.
* [Guiding Reinforcement Learning Exploration Using Natural Language](https://arxiv.org/abs/1707.08616)
* [Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards](https://arxiv.org/abs/1707.08817)
* [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
* [Decoupled Learning of Environment Characteristics for Safe Exploration](https://arxiv.org/abs/1708.02838)
* [Knowledge Sharing for Reinforcement Learning: Writing a BOOK](https://arxiv.org/abs/1709.01308)
* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
  * Variational or Bayesian Dropout, for use with RNN's.
* [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)
* [The Uncertainty Bellman Equation and Exploration](https://arxiv.org/abs/1709.05380)
* [Lifelong Learning with Dynamically Expandable Networks](https://arxiv.org/abs/1708.01547)
* [Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/abs/1709.10089)
  * Mentions something called Hindsight Experience Replay.
* [Self-supervised Deep Reinforcement Learning with Generalized Computation Graphs for Robot Navigation](https://arxiv.org/abs/1709.10489)
* [Embodied Question Answering](http://embodiedqa.org)
  * [Paper](http://embodiedqa.org/paper.pdf)
* [Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378)
  * For value networks that will be used in a non-episodic way, don't end bootstraping at training episode boundaries.
* [Time-Contrastive Networks: Self-Supervised Learning from Video](https://arxiv.org/abs/1704.06888)
  * [Website](https://sermanet.github.io/imitate/)
* [Reverse Curriculum Generation for Reinforcement Learning Agents](http://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/)
  * This could be very useful when I try to train MaLPi to find its charging station.
  * Paper: [Reverse Curriculum Generation for Reinforcement Learning](http://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf)
  * [Code](https://sites.google.com/view/reversecurriculum)
* [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
* [Ray RLLib](https://t.co/7pfYViDchA)
* [Attention based neural networks](https://towardsdatascience.com/memory-attention-sequences-37456d271992)
* [Expected Policy Gradients for Reinforcement Learning](https://arxiv.org/abs/1801.03326)
* [Model-Based Action Exploration](http://arxiv.org/abs/1801.03954v1)
* [Curiosity-driven reinforcement learning with homeostatic regulation](https://arxiv.org/abs/1801.07440)
* [Regret Minimization for Partially Observable Deep Reinforcement Learning](https://openreview.net/forum?id=BJoBhUUUG)
* [One-shot Imitation from Humans via Domain-Adaptive Meta-Learning](https://arxiv.org/abs/1802.01557)
* [Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization](https://arxiv.org/abs/1802.01569)
* [Temporal Difference Models: Model-Free Deep RL for Model-Based Control](https://arxiv.org/abs/1802.09081), BAIR
* [Reinforcement and Imitation Learning for Diverse Visuomotor Skills](https://arxiv.org/abs/1802.09564)
* [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835)
* [Composable Deep Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/1803.06773), BAIR
* [Recall Traces: Backtracking Models for Efficient Reinforcement Learning](https://arxiv.org/abs/1804.00379), BAIR
* [Universal Planning Networks](https://arxiv.org/abs/1804.00645)
* [Latent Space Policies for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1804.02808), BAIR
  * [ICML link](http://proceedings.mlr.press/v80/haarnoja18a.html)
* [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
  * Running average of weights during training to create an effect similar to ensembling, but at training time instead of run/inference time.
  * [Blog post about implementing it](https://medium.com/@hortonhearsafoo/adding-a-cutting-edge-deep-learning-training-technique-to-the-fast-ai-library-2cd1dba90a49)
  * [PyTorch implementation](https://github.com/fastai/fastai/pull/276/files)
* [Temporal Difference Models: Model-Free Deep RL for Model-Based Control](https://arxiv.org/abs/1802.09081), BAIR
* [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909) BAIR
* [Hierarchical Reinforcement Learning with Deep Nested Agents](https://arxiv.org/abs/1805.07008)
* [Data-Efficient Hierarchical RL](https://sites.google.com/view/efficient-hrl)
* [Variational Inference for Data-Efficient Model Learning in POMDPs](https://arxiv.org/abs/1805.09281)
  * [intro to structured inference networks](http://pyro.ai/examples/dmm.html)
* [Fast Policy Learning through Imitation and Reinforcement](https://arxiv.org/abs/1805.10413)
* [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830)
* [Backplay: "Man muss immer umkehren"](https://arxiv.org/abs/1807.06919)
  * Another curriculum learning paper where they start near the goal and work backwards.
* [Shared Multi-Task Imitation Learning for Indoor Self-Navigation](https://arxiv.org/abs/1808.04503)
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [Learning End-to-end Autonomous Driving using Guided Auxiliary  Supervision](https://arxiv.org/abs/1808.10393)
* [Shared Multi-Task Imitation Learning for Indoor Self-Navigation](https://arxiv.org/abs/1808.04503)
* [Parametrized Deep Q-Networks Learning: Reinforcement Learning with  Discrete-Continuous Hybrid Action Space](https://arxiv.org/abs/1810.06394)
* [CURIOUS: Intrinsically Motivated Multi-Task, Multi-Goal Reinforcement  Learning](https://arxiv.org/abs/1810.06284)
* [GPU-Accelerated Robotic Simulation for Distributed Reinforcement  Learning](https://arxiv.org/abs/1810.05762)
* [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)
* [Learning Hierarchical Information Flow with Recurrent Neural Modules](https://arxiv.org/abs/1706.05744)
* Venkatraman, et al. Predictive state decoders: Encoding the future into recurrent networks. In Proceedings of Advances in Neural Information Processing Systems (NIPS), 2017.
* [At Human Speed: Deep Reinforcement Learning with Action Delay](https://arxiv.org/abs/1810.07286)
* [Closing the Sim-to-Real Loop: Adapting Simulation Randomization with  Real World Experience](https://arxiv.org/abs/1810.05687)
* [Safe Reinforcement Learning with Model Uncertainty Estimates](https://arxiv.org/abs/1810.08700)
* [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/resources/icml17.pdf)
* [Towards Governing Agent's Efficacy: Action-Conditional β-VAE for Deep Transparent Reinforcement Learning](https://arxiv.org/abs/1811.04350)
* [Learned optimizers that outperform SGD on wall-clock and validation loss](https://arxiv.org/abs/1810.10180)
* [Reversible Recurrent Neural Networks](https://arxiv.org/abs/1810.10999)
* [Model-Based Active Exploration](https://arxiv.org/abs/1810.12162)
* [Differentiable MPC for End-to-end Planning and Control](https://arxiv.org/abs/1810.13400)
* [Toward an AI Physicist for Unsupervised Learning](https://arxiv.org/abs/1810.10525)
* Memory-based control with recurrent networks, Heess et al. Meta-learning
* Gu, Holly, Lillicrap ‘16 parallel NAF. Continuous action space Q learning
* [Resilient Computing with Reinforcement Learning on a Dynamical System:  Case Study in Sorting](https://arxiv.org/abs/1809.09261)
* [Constrained Exploration and Recovery from Experience Shaping](https://arxiv.org/abs/1809.08925) 
* [Building a Winning Self-Driving Car in Six Months](https://arxiv.org/abs/1811.01273)
* [QUOTA: The Quantile Option Architecture for Reinforcement Learning](https://arxiv.org/abs/1811.02073) 
* [Efficient Eligibility Traces for Deep Reinforcement Learning](https://arxiv.org/abs/1810.09967)
* [Papers that cite World Models](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=8020027393506054346)
* [Flatland: a Lightweight First-Person 2-D Environment for Reinforcement  Learning](https://arxiv.org/abs/1809.00510)
  * Looks interesting, says code will be available at some point.
* [Guiding Policies with Language via Meta-Learning](https://arxiv.org/abs/1811.07882)
* [Learning Actionable Representations with Goal-Conditioned Policies](https://arxiv.org/abs/1811.07819)
* [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300)
* [Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/abs/1806.03335)
* [An Introduction to Deep Reinforcement Learning](https://arxiv.org/abs/1811.12560)
* Retrieving from a large memory:
  * [The Kanerva Machine: A Generative Distributed Memory](https://arxiv.org/abs/1804.01756)
  * [Followup](https://papers.nips.cc/paper/8149-learning-attractor-dynamics-for-generative-memory.pdf)
  * [Shaping Belief States with Generative Environment Models for RL](https://arxiv.org/abs/1906.09237v2)
* [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224) 
* [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857)
* [Learning To Simulate](https://openreview.net/forum?id=HJgkx2Aqt7&noteId=HJgkx2Aqt7)
* [Adversarial Examples, Uncertainty, and Transfer Testing Robustness in Gaussian Process Hybrid Deep Networks](https://arxiv.org/abs/1707.02476)
* [Self-supervised Learning of Image Embedding for Continuous Control](https://arxiv.org/abs/1901.00943)
* [AlphaStar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/)
  * This blog about DeepMind's StarCraft AI has a large list of potentially useful links.
  * [Original LSTM paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)
  * [Pointer Networks](https://papers.nips.cc/paper/5866-pointer-networks.pdf)
* [The Value Function Polytope in Reinforcement Learning](https://arxiv.org/abs/1901.11524)
* [A Geometric Perspective on Optimal Representations for Reinforcement Learning](https://arxiv.org/abs/1901.11530)
  * Finding better representations. Follow on to previous paper.
* [Task2Vec: Task Embedding for Meta-Learning](https://arxiv.org/abs/1902.03545)
* [Simultaneously Learning Vision and Feature-based Control Policies for Real-world Ball-in-a-Cup](https://arxiv.org/abs/1902.04706)
* [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115)
* [World Discovery Models](https://arxiv.org/abs/1902.07685)
* [From Language to Goals: Inverse Reinforcement Learning for Vision-Based Instruction Following](https://arxiv.org/abs/1902.07742)
* [Continual Learning with Tiny Episodic Memories](https://arxiv.org/abs/1902.10486)
* [Using Natural Language for Reward Shaping in Reinforcement Learning](https://arxiv.org/abs/1903.02020)
* [Assessing Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1810.12282)
* [Inductive transfer with context-sensitive neural networks](https://link.springer.com/content/pdf/10.1007%2Fs10994-008-5088-0.pdf)
  * David Silver, adding context to multi-task learning, 2008.
* [Reinforced Imitation in Heterogeneous Action Space](https://arxiv.org/abs/1904.03438)
* [Reinforcement Learning with Attention that Works: A Self-Supervised Approach](https://arxiv.org/abs/1904.03367)
* Gershman, S.J. and Daw, N.D. (2017) Reinforcement learning and episodic memory in humans and animals: an integrative
framework. Annu. Rev. Psychol. 68, 101–128
* [Meta-learning of Sequential Strategies](https://arxiv.org/abs/1905.03030)
* [SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning](https://arxiv.org/abs/1808.09105)
  * [Code](https://github.com/sharadmv/parasol)
  * [Blog](https://bair.berkeley.edu/blog/2019/05/20/solar/)
  * Based on the [SVAE paper](https://arxiv.org/abs/1603.06277) (in [Autoencoders](#autoencoders))
* [Robustness to Out-of-Distribution Inputs via Task-Aware Generative Uncertainty](https://arxiv.org/abs/1812.10687)
* [Multi-Sample Dropout for Accelerated Training and Better Generalization](https://arxiv.org/abs/1905.09788)
* [Learning Powerful Policies by Using Consistent Dynamics Model](https://arxiv.org/abs/1906.04355)
  * Add an auxiliary task to the learned model that penalizes errors in future predictions.
* [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
  * [Project Page](https://sites.google.com/view/sac-and-applications)
  * Fewer hyperparameters, better sample efficiency.
  * [Learning to Walk via Deep Reinforcement Learning](https://arxiv.org/abs/1812.11103)
    * SAC with a learneable temperature hyperparameter.
* [Learning Dynamics Model in Reinforcement Learning by Incorporating the Long Term Future](https://arxiv.org/abs/1903.01599)
  * Add an auxiliary task to predict the far future.
  * Includes use in imitation learning
* [Unsupervised Learning of Object Keypoints for Perception and Control](https://arxiv.org/abs/1906.11883)
* [Real-Time Freespace Segmentation on Autonomous Robots for Detection of Obstacles and Drop-Offs](https://arxiv.org/abs/1902.00842)
* [Dynamics-aware Embeddings](https://arxiv.org/abs/1908.09357)
* [A Survey on Reproducibility by Evaluating Deep Reinforcement Learning Algorithms on Real-World Robots](https://arxiv.org/abs/1909.03772)
* [Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning](https://arxiv.org/abs/1910.00177)
  * [Blog](https://xbpeng.github.io/projects/AWR/)
* [Stabilizing Off-Policy Reinforcement Learning with Conservative Policy Gradients](https://arxiv.org/abs/1910.01062)
  * [Code](https://github.com/tesslerc/ConservativePolicyGradient)
* [A Mobile Manipulation System for One-Shot Teaching of Complex Tasks in Homes](https://arxiv.org/abs/1910.00127)
* OpenAI's Automatic Domain Randomization on the DonkeyCar simulator?
  * [Blog](https://openai.com/blog/solving-rubiks-cube/)
  * [Paper](https://d4mucfpksywv.cloudfront.net/papers/solving-rubiks-cube.pdf)
  * Start with a single, easy environment in sim. When performance plateaus, increase the range of simulated features. E.g. increase range of friction, or weight of car, or size/color of lane markings.
  * They used Embed + Sum on the inputs so they didn't need to change the policy between sim and real.
  * They used Policy cloning (and DAgger?) to train a new policy from an older one, e.g. if the policy architecture did change. Section 6.4.
* [Generalization in Reinforcement Learning with Selective Noise Injection and Information Bottleneck](https://arxiv.org/abs/1910.12911)
* [Learning to Predict Without Looking Ahead: World Models Without Forward Prediction](https://learningtopredict.github.io/)
  * [Repo](https://github.com/learningtopredict/learningtopredict.github.io/issues), but no code yet.
* [Word2vec to behavior: morphology facilitates the grounding of language in machines](https://arxiv.org/abs/1908.01211)
  * [Code](https://github.com/davidmatthews1uvm/2019-IROS)
* [DeepRacer](https://arxiv.org/abs/1911.01562)
* [CrossNorm: Normalization for Off-Policy TD Reinforcement Learning](https://arxiv.org/abs/1902.05605)
  * Eliminates the need for a target network?
* [Optimizing agent behavior over long time scales by transporting value](https://www.nature.com/articles/s41467-019-13073-w)
  * Looking back over episodic memory
  * [Code](https://github.com/deepmind/deepmind-research/tree/master/tvt)
* [Reinforcement Learning Upside Down: Don't Predict Rewards -- Just Map Them to Actions](https://arxiv.org/abs/1912.02875)
* [Training Agents using Upside-Down Reinforcement Learning](https://arxiv.org/abs/1912.02877)
* [A Simple Randomization Technique for Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1910.05396)
* [Prioritized Sequence Experience Replay](https://arxiv.org/abs/1905.12726)
* [RTFM: Generalising to New Environment Dynamics via Reading](https://openreview.net/forum?id=SJgob6NKvH)
* [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685)
* [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
* [Q-Learning in enormous action spaces via amortized approximate maximization](https://arxiv.org/abs/2001.08116)
* [AMRL: Aggregated Memory For Reinforcement Learning](https://openreview.net/forum?id=Bkl7bREtDr)
  * One of several recent papers on memory.
  * See also: [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
    * Finished reading this. The Future Work section includes ideas about where this could be generalized, e.g. structured knowledge.
* [Network Randomization: A Simple Technique for Generalization in Deep Reinforcement Learning](https://arxiv.org/abs/1910.05396)

### Not as relevant to MaLPi, but interesting

* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [Beating Atari with Natural Language Guided Reinforcement Learning](http://web.stanford.edu/class/cs224n/reports/2762090.pdf)
* [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
* [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)
* [Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)
  * [A Tensorflow implementation](https://github.com/DeNeutoy/bayesian-rnn)
  * [Another TF implementation](https://github.com/mirceamironenco/BayesianRecurrentNN)
* [ML for analyzing unix log files](https://www.google.com/#safe=off&q=unix+log+files+machine+learning+detect+outliers)
ss [Concrete Dropout](https://arxiv.org/abs/1705.07832)
* [Bayesian Reinforcement Learning: A Survey](https://arxiv.org/abs/1609.04436)
* [Beyond Monte Carlo Tree Search: Playing Go with Deep Alternative Neural Network and Long-Term Evaluation](https://arxiv.org/abs/1706.04052)
* [Meta learning Framework for Automated Driving](https://arxiv.org/abs/1706.04038)
* [Representation Learning for Grounded Spatial Reasoning](https://arxiv.org/pdf/1707.03938.pdf)
  * Instruction text -> LSTM -> vectors 1 and 2
  * V1 is used as a kernel in a convolution over the state space object embeddings (hand built?)
  * V2 is used to make a global map representation of the input
  * both outputs are concatenated and input to a CNN to predict the final map value
* [Early Stage Malware Prediction Using Recurrent Neural Networks](https://arxiv.org/abs/1708.03513)
* Hwang J, Jung M, Madapana N, et al. Achieving "synergy" in cognitive behavior of humanoids via deep learning of dynamic visuo-motor-attentional coordination. Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on; Seoul. 2015. p. 817-824.
  * Combined human gesture recogniztion, attention, object detection and grasping.
  * [Arxiv page](https://arxiv.org/abs/1507.02347)
* Deep Mixture Density Network (MDN)
  *  "MDNs combine the benefits of DNNs and GMMs (Gaussian mixture model)  by using the DNN to model the complex relationship between input and output data, but providing probability distributions as output"
  * C. Bishop. Mixture density networks, Tech. Rep. NCRG/94/004, Neural Computing Research Group. Aston University, 1994.
  * H. Zen, A. Senior. Deep mixture density networks for acoustic modeling in statistical parametric speech synthesis, ICASSP, 2014.
* [Anytime Neural Networks via Joint Optimization of Auxiliary Losses](https://arxiv.org/abs/1708.06832)
* [Language Grounding for Robotics accepted papers](https://robonlp2017.github.io/accepted.html)
* Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei- Fei, and Ali Farhadi. Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning. In ICRA, 2017.
* [Augmenting End-to-End Dialog Systems with Commonsense Knowledge](https://arxiv.org/abs/1709.05453)
* [Predictive representations can link model-based reinforcement learning to model-free mechanisms](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768)
* [Neural Task Programming: Learning to Generalize Across Hierarchical Tasks](https://arxiv.org/abs/1710.01813)
* [Understanding Generalization and Stochastic Gradient Descent](https://arxiv.org/abs/1710.06451)
  * Includes how to choose the best batch size for test set accuracy.
* [A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs](http://science.sciencemag.org/content/early/2017/10/25/science.aag2612.full)
* [Generalized Grounding Graphs: A Probabilistic Framework for Understanding Grounded Commands](https://arxiv.org/abs/1712.01097)
* [Peephole: Predicting Network Performance Before Training](https://arxiv.org/abs/1712.03351)
* [Peano-HASEL actuators: Muscle-mimetic, electrohydraulic transducers that linearly contract on activation](http://robotics.sciencemag.org/content/3/14/eaar3276)
* [Hydraulically amplified self-healing electrostatic actuators with muscle-like performance](http://science.sciencemag.org/content/359/6371/61)
* [Unsupervised Low-Dimensional Vector Representations for Words, Phrases and Text that are Transparent, Scalable, and produce Similarity Metrics that are Complementary to Neural Embeddings](https://arxiv.org/abs/1801.01884)
* [Emergent complexity via multi-agent competition](https://arxiv.org/abs/1710.03748)
* [PRNN: Recurrent Neural Network with Persistent Memory](https://arxiv.org/abs/1801.08094)
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
  * Might be useful to classify task descriptions for a multi-task system.
* [Directly Estimating the Variance of the λ-Return Using Temporal-Difference Methods](https://arxiv.org/abs/1801.08287)
* [Scalable Meta-Learning for Bayesian Optimization](https://arxiv.org/abs/1802.02219)
* [Learning to Play with Intrinsically-Motivated Self-Aware Agents](https://arxiv.org/abs/1802.07442)
* [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
* [Accelerated Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1803.02811). Adam Stooke and Pieter Abbeel
* [Learning and Querying Fast Generative Models for Reinforcement Learning](https://arxiv.org/abs/1802.03006)
* [Learning by Playing - Solving Sparse Reward Tasks from Scratch](https://arxiv.org/abs/1802.10567)
* [Selective Experience Replay for Lifelong Learning](https://arxiv.org/abs/1802.10269)
* [Semi-Parametric Topological Memory For Navigation](https://openreview.net/pdf?id=SygwwGbRW)
* [Shifting Mean Activation Towards Zero with Bipolar Activation Functions](https://arxiv.org/abs/1709.04054)
  * Alternative to Batch Norm for normalization
* [Strategic attentive writer for learning macro-actions](https://arxiv.org/abs/1606.04695)
* [The Limits and Potentials of Deep Learning for Robotics](https://arxiv.org/abs/1804.06557)
* [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)
* [Progress & Compress: A scalable framework for continual learning](https://arxiv.org/abs/1805.06370)
* [Deep Curiosity Search: Intra-Life Exploration Improves Performance on  Challenging Deep Reinforcement Learning Problems](https://arxiv.org/abs/1806.00553)
* [Unsupervised Meta-Learning for Reinforcement Learning](https://arxiv.org/abs/1806.04640)
* [Unsupervised Learning by Competing Hidden Units](https://arxiv.org/abs/1806.10181)
* [Adaptive Neural Trees](https://arxiv.org/abs/1807.06699)
  * Combining Decision Trees and neural nets
* [Reliable Uncertainty Estimates in Deep Neural Networks using Noise Contrastive Priors](https://arxiv.org/abs/1807.09289)
* [Papers of the Year](https://kloudstrifeblog.wordpress.com/2017/12/15/my-papers-of-the-year/amp)
* [Wider and Deeper, Cheaper and Faster: Tensorized LSTMs for Sequence Learning](https://arxiv.org/abs/1711.01577)
* [LASER Language-Agnostic SEntence Representations](https://github.com/facebookresearch/LASER)
  * Pre-trained multi-lingual embeddings. Possibly useful for task description embedding.
* [Building Machines That Learn and Think Like People](https://arxiv.org/abs/1604.00289)
* [Learning to Understand Goal Specifications by Modelling Reward](https://arxiv.org/abs/1806.01946)
* [Investigating Generalisation in Continuous Deep Reinforcement Learning](https://arxiv.org/abs/1902.07015)
* [Hyperbolic Discounting and Learning over Multiple Horizons](https://arxiv.org/abs/1902.06865)
  * Also useful as an auxiliary task.
* [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)
* [Stiffness: A New Perspective on Generalization in Neural Networks](https://arxiv.org/abs/1901.09491)
* [IndyLSTMs: Independently Recurrent LSTMs](https://arxiv.org/abs/1903.08023)
* [Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://arxiv.org/abs/1903.08254)
* [Diagnosing Bottlenecks in Deep Q-learning Algorithms](https://arxiv.org/abs/1902.10250)
* [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/abs/1904.05160)
  * Another use of memory, this time for situations that don't occur enough to train on.
* [Human Visual Understanding for Cognition and Manipulation -- A primer for the roboticist](https://arxiv.org/abs/1905.05272)
* [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/abs/1906.05909)
  * Replacing convolutions with attention in vision models.
* [Learning the Arrow of Time](https://openreview.net/pdf?id=SkevntbkJB)
* [Improving the robustness of ImageNet classifiers using elements of human visual cognition](https://arxiv.org/abs/1906.08416)
  * Episodic memory and shape based representations.
* [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)
* [Metalearned Neural Memory](https://arxiv.org/abs/1907.09720)
* [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)
* [Hierarchical Decision Making by Generating and Following Natural Language Instructions](https://arxiv.org/abs/1906.00744)
* [AC-Teach: A Bayesian Actor-Critic Method for Policy Learning with an Ensemble of Suboptimal Teachers](https://arxiv.org/abs/1909.04121)
  * [Blog](https://ai.stanford.edu/blog/acteach/)
* [Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/abs/1904.04998)
* [Teacher algorithms for curriculum learning of Deep RL in continuously parameterized environments](https://arxiv.org/abs/1910.07224)
* [A Review of Robot Learning for Manipulation: Challenges, Representations, and Algorithms](https://arxiv.org/abs/1907.03146)
* [Weakly Supervised Disentanglement with Guarantees](https://arxiv.org/abs/1910.09772)
* [Using a Logarithmic Mapping to Enable Lower Discount Factors in Reinforcement Learning](https://arxiv.org/abs/1906.00572)
* [Regularization Matters in Policy Optimization](https://arxiv.org/abs/1910.09191)
* [Meta-Learning without Memorization](https://arxiv.org/abs/1912.03820)

## Autoencoders

* [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). The original VAE paper.
* [Density Estimation: A Neurotically In-Depth Look At Variational Autoencoders](http://ruishu.io/2018/03/14/vae/)
* [Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)
* [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)
* [Disentangling Factors of Variation Using Few Labels](https://arxiv.org/abs/1905.01258)
* [Composing graphical models with neural networks for structured representations and fast inference](https://arxiv.org/abs/1603.06277)
* [Balancing Reconstruction Quality and Regularisation in ELBO for VAEs](https://arxiv.org/abs/1909.03765)

## Classes/Education

* [CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)
* [CMU 10703, Spring 2017 Deep Reinforcement Learning and Control](https://katefvision.github.io)
* [Stanford CS234: Reinforcement Learning](http://web.stanford.edu/class/cs234/index.html)
* [David Silver's UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
* [Deep Learning (DLSS) and Reinforcement Learning (RLSS) Summer School, Montreal 2017 (videos)](http://videolectures.net/deeplearning2017_montreal/)
* [Deep RL Bootcamp (Aug 2017, Berkeley](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [Theories of Deep Learning (STATS 385), Stanford 2017](https://stats385.github.io)
* [Bayesian Deeplearning](http://bayesiandeeplearning.org)
* [CS20: TensorFlow for Deep Learning Research](https://cs20.stanford.edu)
* [A list of fifteen more classes](https://sky2learn.com/deep-learning-reinforcement-learning-online-courses-and-tutorials-theory-and-applications.html) (Some overlap)
* [Hierarchical RL Workshop](https://sites.google.com/view/hrlnips2017)
  * Includes lectures by David Silver
* [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) (15 hours, lessons, videos, exercises)
* [CS294-158 Deep Unsupervised Learning Spring 2019](https://sites.google.com/view/berkeley-cs294-158-sp19/home)
* [Testing and Debugging in Machine Learning](https://developers.google.com/machine-learning/testing-debugging/) (~4 hours)
* [Metacademy](https://metacademy.org/browse)
  * Lists of subjects and prerequisites for ML.
* [Mathematics for Machine Learning](https://mml-book.com/)
  * A free textbook.

## Simulators

* [HoME: a Household Multimodal Environment](https://arxiv.org/abs/1711.11017)
  * [Github](https://github.com/HoME-Platform/home-platform), [Website](https://home-platform.github.io)
* [MINOS: Multimodal Indoor Simulator](https://minosworld.github.io)
* [CHALET: Cornell House Agent Learning Environment](https://arxiv.org/abs/1801.07357)

---
