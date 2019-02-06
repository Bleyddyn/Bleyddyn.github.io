---
title: 'Stable Baselines Algorithms'
date: 2019-02-03
permalink: /2019-02-03-baselines-algorithms
excerpt: Table of algorithms in the Stable Baselines repo
tags:
  - baselines
  - RL
---

# Intro

Stable Baselines (<a href="https://stable-baselines.readthedocs.io/en/master/index.html">Docs</a>) is a cleaned up and easier to use version of OpenAI's baseline Reinforcement Learning algorithms. They support multiple RL algorithms (PPO, DQN, etc) each of which supports some sub-set of features. The docs, however, don't include a single table where you can see what all the algorithms support in one place. The table below shows them all at a glance, making it easier to decide which algorithms you can or can't use based on recurrance, continuous actions, multi-processing, etc.

# Algorithms

<style>
td.centered {
    text-align: center;
}
td.extra_space {
    padding-left: 2em;
}
th.centered {
    text-align: center;
}
th.fixed {
    width: 5em;
}
th.extra_space {
    padding-left: 2em;
}
</style>


<table>
    <tr><th>Algorithm</th>
        <th class="centered extra_space">Recurrent</th><th>Multi-Processing</th><th>Replay Buffer</th>
            <th colspan="4" class="centered">Action Spaces</th><th colspan="4" class="centered">Observation Spaces</th></tr>
    <tr><th></th><th></th><th></th><th></th>
            <th class="fixed extra_space">Discrete</th><th class="fixed">Box</th><th class="fixed">MultiDiscrete</th><th class="fixed">MultiBinary</th>
            <th class="fixed extra_space">Discrete</th><th class="fixed">Box</th><th class="fixed">MultiDiscrete</th><th class="fixed">MultiBinary</th>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/a2c.html">A2C</a></td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/acer.html">ACER</a></td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
        <td class="centered extra_space">✔️</td><td class="centered">❌</td><td class="centered">❌</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/acktr.html">ACKTR</a></td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">❌</td><td class="centered">❌</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html">DDPG</a></td>
        <td class="centered extra_space">❌</td><td class="centered">❌</td><td class="centered">✔️</td>
        <td class="centered extra_space">❌</td><td class="centered">✔️</td><td class="centered">❌</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/dqn.html">DQN</a></td>
        <td class="centered extra_space">❌</td><td class="centered">❌</td><td class="centered">✔️</td>
        <td class="centered extra_space">✔️</td><td class="centered">❌</td><td class="centered">❌</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/gail.html">GAIL</a></td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️ (MPI)</td><td class="centered">❌</td>
        <td class="centered extra_space">❌</td><td class="centered">✔️</td><td class="centered">❌</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/ppo1.html">PPO1</a></td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️ (MPI)</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html">PPO2</a></td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/sac.html">SAC</a></td>
        <td class="centered extra_space">❌</td><td class="centered">❌</td><td class="centered">✔️</td>
        <td class="centered extra_space">❌</td><td class="centered">✔️</td><td class="centered">❌</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
    <tr><td><a href="https://stable-baselines.readthedocs.io/en/master/modules/trpo.html">TRPO</a></td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️ (MPI)</td><td class="centered">❌</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
        <td class="centered extra_space">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td><td class="centered">✔️</td>
    </tr>
</table>
<div>
<h2>Notes</h2>
<ol>
<li>DDPG does not support stable_baselines.common.policies because it uses q-value instead of value estimation</li>
<li>DQN does not support stable_baselines.common.policies</li>
<li>PPO2 is the implementation OpenAI made for GPU. For multiprocessing, it uses vectorized environments compared to PPO1 which uses MPI</li>
<li>SAC does not support stable_baselines.common.policies because it uses double q-values and value estimation</li>
<li><a href="https://stable-baselines.readthedocs.io/en/master/modules/her.html">HER</a> (Hindsight Experience Replay) is not refactored yet.</li>
</ol>

Edit 1: add Replay Buffer.


---
