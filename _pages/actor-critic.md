---
layout: archive
title: "Actor/Critic Papers"
permalink: /actor-critic/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

* [Sample Efficient Actor-Critic With Experience Replay](https://arxiv.org/abs/1611.01224)
  * [ACER implementation](https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/acer.py)
* [The Reactor: A Sample-Efficient Actor-Critic Architecture](https://arxiv.org/abs/1704.04651)
  * Also compares time stacked inputs versus LSTMs in section 3.3.
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
  * The A3C paper.
* [A Survey of Actor-Critic Reinforcement Learning: Standard and Natural Policy Gradients](https://pdfs.semanticscholar.org/145a/42e83ec142a125da3ad845ee95027ef702e5.pdf)
  * 2010, maybe?
* [ON ACTOR-CRITIC ALGORITHMS](http://www.mit.edu/~jnt/Papers/J094-03-kon-actors.pdf)
  * 2003
* [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)
  * This algorithm is in OpenAI's baseline repo.
  * It's a natural gradient actor critic method ([Natural Gradients](http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/)).
* [Hindsight Experience Replay](https://arxiv.org/pdf/1707.01495.pdf)
* [SOFT ACTOR-CRITIC: OFF-POLICY MAXIMUM ENTROPY DEEP REINFORCEMENT LEARNING WITH A STOCHASTIC ACTOR](https://openreview.net/pdf?id=HJjvxl-Cb)
  * [arxiv](https://arxiv.org/abs/1801.01290)
* [Mean Actor Critic](https://arxiv.org/abs/1709.00503)

---
