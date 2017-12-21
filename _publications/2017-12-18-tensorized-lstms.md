---
title: "Tensorizing LSTMs"
collection: publications
permalink: /publications/2017-12-18-tensorized-lstms
excerpt: "Tensorizing LSTMs to make them wider and deeper without adding parameters and with minimal extra compute costs."
date: 2017-12-18
paperurl: https://arxiv.org/abs/1711.01577
usemath: true
---
Warning! Work in progress.

First paper: [Wider and Deeper, Cheaper and Faster: Tensorized LSTMs for Sequence Learning](https://arxiv.org/abs/1711.01577)

> we introduce a way to both widen and deepen the LSTM whilst keeping the parameter number and runtime largely unchanged.

I wanted to quote this directly because it sums up why I'm interested in these papers.

Their three novel contributions are:

* Tensorize hidden state vectors into higher dimensional tensors.
* Merge RNN deep computations into its temporal computations.
* Integrate a new memory cell convolution when extending the previous two to LSTMs.


More papers, some of which probably aren't relevant:
* [Learning Compact Recurrent Neural Networks with Block-Term Tensor Decomposition](https://arxiv.org/abs/1712.05134). BTD is a combination of:
  * CP decomposition: J. D. Carroll and J.-J. Chang. Analysis of individual dif- ferences in multidimensional scaling via an n-way gener- alization of eckart-young decomposition. Psychometrika, 35(3):283–319, 1970.
  * Tucker decomposition: L.R. Tucker. Some mathematical notes on three-mode factor analysis. Psychometrika, 31(3):279–311, 1966.
* [Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569) (referenced from Wider and Deeper...)
* Luca Bertinetto, João F Henriques, Jack Valmadre, Philip Torr, and Andrea Vedaldi. Learning feed-forward one-shot learners. In NIPS, 2016
* Misha Denil, Babak Shakibi, Laurent Dinh, Nando de Freitas, et al. Predicting parameters in deep learning.  In NIPS, 2013.
* Timur Garipov, Dmitry Podoprikhin, Alexander Novikov, and Dmitry Vetrov. Ultimate tensorization: compressing convolutional and fc layers alike. In NIPS Workshop, 2016.
* Ozan Irsoy and Claire Cardie. Modeling compositionality with multiplicative recurrent neural networks. In ICLR, 2015.
* Ben Krause, Liang Lu, Iain Murray, and Steve Renals. Multiplicative lstm for sequence modelling. In ICLR Workshop, 2017.
* Alexander Novikov,Dmitrii Podoprikhin,Anton Osokin, and Dmitry P Vetrov. Tensorizing neural networks. In NIPS, 2015. https://arxiv.org/abs/1509.06569
* Ilya Sutskever, James Martens, and Geoffrey E Hinton. Generating text with recurrent neural networks. In ICML, 2011.
* Graham W Taylor and Geoffrey E Hinton. Factored conditional restricted boltzmann machines for modeling motion style. In ICML, 2009.
* Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, and Ruslan Salakhutdinov. On multiplicative integration with recurrent neural networks. In NIPS, 2016.
* A. Novikov, D. Podoprikhin, A. Osokin, and D. P. Vetrov. Tensorizing neural networks. In Advances in Neural Infor- mation Processing Systems, pages 442–450, 2015.
* A. Tjandra, S. Sakti, and S. Nakamura. Compressing recurrent neural network with tensor train. arXiv preprint arXiv:1705.08052, 2017.
* Y. Yang, D. Krompass, and V. Tresp. Tensor-train recurrent neural networks for video classification. arXiv preprint arXiv:1707.01786, 2017.
* R. Yu, S. Zheng, A. Anandkumar, and Y. Yue. Long-term forecasting using tensor-train rnns. arXiv preprint arXiv:1711.00073, 2017.
---
