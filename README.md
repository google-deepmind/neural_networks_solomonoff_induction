# Learning Universal Predictors

This repository provides an implementation of our paper [Learning Universal Predictors](https://arxiv.org/abs/2401.14953).

> Meta-learning has emerged as a powerful approach to train neural networks to learn new tasks quickly from limited data.
Broad exposure to different tasks leads to versatile representations enabling general problem solving.
But, what are the limits of meta-learning?
In this work, we explore the potential of amortizing the most powerful universal predictor, namely Solomonoff Induction (SI), into neural networks via leveraging meta-learning to its limits.
We use Universal Turing Machines (UTMs) to generate training data used to expose networks to a broad range of patterns.
We provide theoretical analysis of the UTM data generation processes and meta-training protocols.
We conduct comprehensive experiments with neural architectures (e.g. LSTMs, Transformers) and algorithmic data generators of varying complexity and universality.
Our results suggest that UTM data is a valuable resource for meta-learning, and that it can be used to train neural networks capable of learning universal prediction strategies.

It is based on [JAX](https://jax.readthedocs.io) and [Haiku](https://dm-haiku.readthedocs.io) and contains all code, datasets, and models necessary to reproduce the paper's results.


## Content

```
.
├── data
|   ├── chomsky_data_generator.py - Chomsky Task Source, for a single task.
|   ├── ctw_data_generator.py.    - Variable-order Markov Source.
|   ├── data_generator.py         - Main abstract class for our data generators.
|   ├── meta_data_generator.py    - Sampling from multiple generators.
|   ├── utm_data_generator.py     - BrainPhoque UTM Source, from randomly sampled programs.
|   └── utms.py                   - UTM interface and implementation of BrainPhoque.
├── models
|   ├── ctw.py                    - CTW (Willems, 1995)
|   └── transformer.py            - Decoder-only Transformer (Vaswani, 2017).
├── README.md
├── requirements.txt              - Dependencies
└── train.py                      - Script to train a neural model.
```


## Installation

Clone the source code into a local directory:
```bash
git clone https://github.com/google-deepmind/neural_networks_solomonoff_induction.git
cd neural_networks_solomonoff_induction
```

`pip install -r requirements.txt` will install all required dependencies.
This is best done inside a [conda environment](https://www.anaconda.com/).
To that end, install [Anaconda](https://www.anaconda.com/download#downloads).
Then, create and activate the conda environment:
```bash
conda create --name nnsi
conda activate nnsi
```

Install `pip` and use it to install all the dependencies:
```bash
conda install pip
pip install -r requirements.txt
```

If you have a GPU available (highly recommended for fast training), then you can install JAX with CUDA support.
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that the jax version must correspond to the existing CUDA installation you wish to use (CUDA 12 in the example above).
Please see the [JAX documentation](https://github.com/google/jax#installation) for more details.


## Citing This Work

```bibtex
@article{grau2024learning,
  author       = {Jordi Grau{-}Moya and
                  Tim Genewein and
                  Marcus Hutter and
                  Laurent Orseau and
                  Gr{\'{e}}goire Del{\'{e}}tang and
                  Elliot Catt and
                  Anian Ruoss and
                  Li Kevin Wenliang and
                  Christopher Mattern and
                  Matthew Aitchison and
                  Joel Veness},
  title        = {Learning Universal Predictors},
  journal      = {arXiv:2401.14953},
  year         = {2024},
}
```


## License and Disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
