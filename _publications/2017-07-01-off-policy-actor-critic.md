---
title: "Off-Policy Actor-Critic"
collection: publications
permalink: /publications/2017-07-01-off-policy-actor-critic
excerpt: "Off-Policy AC with linear state features. Includes elegibility traces."
date: 2013-06-20
paperurl: https://arxiv.org/abs/1205.4839
usemath: true
---

## Notes

Action-value methods like Greedy-GQ and Q-Learning have three limitations:

1. Policies are deterministic.
2. Finding max action value is difficult for larger action spaces.
3. Small changes in the action value function can produce large changes in behavior.

Policy gradient methods, including Actor-Critic avoid those limitations, but they were all on-policy before this paper.

### The Off-PAC algorithm:

Initialize the vectors $e_v$, $e_u$, and w to zero

Initialize the vectors v and u arbitrarily

Initialize the state s

For each step:
<div style="margin-left:4em;">
<style>
p.algo {
    margin-bottom: 0.5em;
}
</style>
  <p class="algo">Choose an action, a, according to $b(·\|s)$</p>

  <p class="algo">Observe resultant reward, r, and next state, s′</p>

  <p class="algo">$\delta \leftarrow r + \gamma(s′)v^Tx_{s′} − v^Tx_s$</p>

  <p class="algo">$ρ \leftarrow \pi_u(a\|s) / b(a\|s)$</p>

  <p class="algo">Update the critic using the $GTD(\lambda)$  algorithm:</p>

  <p class="algo" style="margin-left:4em;">$e_v \leftarrow ρ(x_s + \gamma(s) \lambda e_v)$</p>
  <p class="algo" style="margin-left:4em;">$v \leftarrow v + \alpha_v  [\delta e_v − \gamma(s′)(1 − \lambda)(w^Te_v)x_s]$</p>
  <p class="algo" style="margin-left:4em;">$w \leftarrow w + \alpha_w  [\delta e_v − (w^Tx_s)x_s] $</p>

  <p class="algo">Update the actor:</p>

  <p class="algo" style="margin-left:4em;">$e_u \leftarrow ρ [ \frac{\nabla_u \pi_u (a\|s)}{\pi_u(a\|s)} + \gamma(s) \lambda e_u]$</p>
  <p class="algo" style="margin-left:4em;">$u \leftarrow u + \alpha_u \delta e_u$</p>

  <p class="algo">$s \leftarrow  s′$</p>
</div>


They only talk about using elegibility traces with a linear combination of state features so I have no idea how well they would work with a neural network. I'm also not sure what the w weights are. They aren't mentioned anywhere in the paper.
