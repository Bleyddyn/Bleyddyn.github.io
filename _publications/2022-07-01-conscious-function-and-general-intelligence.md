---
title: "On the link between conscious function and general intelligence in humans and machines"
collection: publications
permalink: /publications/2022-07-01-conscious-function-and-general-intelligence
excerpt: "Access consciousness and it's relation to general intelligence"
date: 2022-07-01
paperurl: https://openreview.net/pdf?id=LTyqvLEv5b
usemath: true
---


Authors: Juliani, Arulkumaran, Sasai, Kanai

Transactions on Machine Learning Research (07/2022)

# Definition

"Unlike “phenomenal consciousness” which corresponds to any subjective experience, regardless of how subtle, “access consciousness” corresponds to information one is aware of experiencing, and can thus report."

# Current Theories

## GWT

The Global Workspace Theory provides an account of the function of attention, working memory, and information sharing between brain modules in humans and other mammals, and has garnered a body of neuroscientific work which supports it.

The workspace is defined as a common representational space of fixed capacity where various modules within the brain can share information with other modules.
this workspace has been associated with the frontal and parietal cortex and functionally connected to both attentional control and working memory. Attentional control can be interpreted as the specific policy for admitting behaviorally-relevant information into the workspace at the exclusion of irrelevant information. Working memory can be seen as the capacity to maintain information within the workspace for an extended period of time, as well as to manipulate the contents of the workspace to meet behavioral goals

## IGT

The Information Generation Theory provides an account of the internal generation of trajectories of coherent experience in humans, and is based on clinical and research findings concerning the neural basis of consciousness.

Information Generator: information which is consciously accessible is not simply first-order sensory input, but is rather always the result of a generative model within the brain, which during waking experience is most often conditioned on sensory input. This generative model is also capable of information generation which is contrary to the sensory input, or even completely disconnected from it, as is the case when dreaming. As such, the generation process confers the ability to not only predict the sensory world, but to simulate unseen events in that world. According to IGT, it is this simulation which corresponds to what is consciously accessible.

## AST

Attention Schema Theory provides an account of flexible adaptation to novel contexts by an attentional system, and is based on a number of pieces of evidence from cognitive neuroscience literature.

It is not attention which makes conscious access possible, but rather the capacity to represent a high-level model of the attentional process. In this view, the content of consciousness corresponds to the information present in the brain’s model of attention, not the information being attended to itself. According to the proponents of AST, consciousness, including the feeling of having a subjective experience, is simply the result of this mental model, which they refer to as awareness, and distinguish from attention. The functional role of the attention schema is then as a higher level controller responsible for monitoring and adapting the dynamics of the attentional process based on the particular behavioral needs of the organism at a given time.

# Miscellaneous Quotes

Cognitive Map: the “cognitive map” used to ground navigation in the world

"we define intelligence as the ability to quickly acquire novel skills with little direct experience, knowledge, or structural priors related to the skills being acquired. Put another way, if two agents can learn to solve a new task to the same proficiency level, but one solves it with less experience, knowledge, or in-built structural advantage than the other agent, then we say that the former agent is the more intelligent of the two."

"mathematically formalized as the speed of skill acquisition divided by the amount of experience, knowledge, and structural priors needed in order to acquire those skills"

"mental time travel is the ability to project oneself into the past or future and actively participate in a series of imagined events within the projection."

"By creating systems which can utilize selective attention, represent multi-modal structured information in a domain-agnostic fashion (as in GWT), generate spatially and temporally coherent imagined trajectories of experience (as in IGT), and adapt the attention policy governing information access to the problem at hand (as in AST), mental time travel as described by Tulving can be made possible."


## Current approaches to generalization in AI:

RL, Model-based RL, inductive biases: convolutional layers, sparsity, modularity; attention, adaptive processing, abstraction (deeper layers represent more abstract representations), multi-modality, causality and counter-factual reasoning, Meta-learning.

### Meta-learning:

* metric-based: new datapoint is related to a weighted average of previously seen datapoints from an episodic memory
* optimization-based: train a model that can be efficiently updated with just a few training examples
* memory/model-based: sub-set of parameters that update quickly in the presence of new data. RNN's are an example


## Four levels of experience generation

* Direct Experience: Current trajectory, current env and task
  * On-Policy RL
* Replay: seen trajectories, seen envs and tasks
  * Off-Policy RL
* Preplay: possible trajectories, seen envs and tasks
  * World Models, especially if a search algorithm is used to avoid using the current policy. Must be able to generalize beyound observed states/actions.
* Mental time travel: possible trajectories, envs and tasks


Increasing intelligence would be needed to perform mental time travel in increasingly complex worlds: 2D grid worlds, Atari games, the real world


How the three models could work together to provide access conciousness:

"how functional theories of consciousness work together to support mental time travel and other intelligent behavior. Sensory and motor information is mediated by a process of internal infor- mation generation. This process of information generation is then modulated by a cognitive map, which ensures temporal coherency of generated information. Subsets of this generated information are attended to and thus maintained and manipulated within the global workspace. The content and dynamics of the global workspace are attenuated by an attentional schema modeling those same dynamics."

---
