# CaPAN

This repo is the official PyTorch implementation of "CaPAN: Class-aware Prototypical Adversarial Networks for Unsupervised Domain Adaptation".

<div align=center><img src="./figures/Figure1.png" width="100%"></div>

## Abstract

Adversarial domain adaptation has achieved impressive performances for unsupervised domain adaptation (UDA). However, existing adversarial UDA methods often rely on multiple domain discriminators to capture diverse patterns, which limit their scalability and resulting in dispersed features. To address these issues, we propose a novel method called Class-aware Prototypical Adversarial Network (CaPAN), which efficiently extracts transferable and discriminative features. Specifically, our class-aware adversarial learning employs a single multi-class discriminator to capture various patterns, aligning class-level features. Furthermore, to enhance the discriminative ability of our model, we introduce a prototypical domain discriminator to enhance the discriminatively of the learned features by aligning target sample towards prototypes (centers of each class), resulting in a more compact feature space. Extensive experiments validate the effectiveness of CaPAN, which can also be integrated as a regularization technique for existing methods to further improve their performance.

## Acknowledgment
Some codes is mainly based on following repositories. Thanks for their authors.
* [SCDA](https://github.com/BIT-DA/SCDA.git)
* [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library.git)

