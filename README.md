# Neural Networks and Solomonoff Induction

This repository provides an implementation of our paper Neural Networks and Solomonoff Induction.

>   Solomonoff Induction (SI) is the most powerful universal predictor given unlimited computational resources. Naive SI approximations are challenging and require running vast amount of programs for extremely long. Here we explore an alternative path to SI consisting in meta-training neural networks on universal data sources.
We generate the training data by feeding random programs to Universal Turing Machines (UTMs) and guarantee convergence in the limit to various SI variants (under certain assumptions). We provide novel results on how a non-uniform distribution over programs still maintain the universality property. Experimentally, we investigate the effect neural network architectures (i.e. LSTMs, Transformers, etc.) and sizes on their performance on algorithmic data, crucial for SI. First, we consider variable-order Markov sources where the Bayes-optimal predictor is the well-known Context Tree Weighting (CTW) algorithm.
Second, we evaluate on challenging algorithmic tasks on Chomsky hierarchy that require different memory structures. Finally, we test on the UTM domain following our theoretical results.  We show that scaling network size always improves performance on all tasks, Transformers outperforming all others, even achieving optimality on par with CTW. Promisingly, large Transformers and LSTMs trained on UTM data exhibit transfer to the other domains.

It is based on [JAX](https://jax.readthedocs.io) and [Haiku](https://dm-haiku.readthedocs.io) and contains all code, datasets, and models necessary to reproduce the paper's results.


## Content

```
.
├── models
|   ├── ctw.py                    - CTW (Willems, 1995)
|   └── transformer.py            - Decoder-only Transformer (Vaswani, 2017).
├── data
|   ├── data_generator.py         - Main abstract class for our data generators.
|   ├── ctw_data_generator.py.    - Variable-order Markov Source.
|   ├── utms.py                   - UTM interface and implementation of BrainPhoque.
|   ├── utm_data_generator.py     - BrainPhoque UTM Source, from randomly sampled programs.
|   ├── chomsky_data_generator.py - Chomsky Task Source, for a single task.
|   └── meta_data_generator.py    - Sampling from multiple generators.
├── README.md
├── requirements.txt              - Dependencies
└── train.py                      - Script to train a neural model.
```

`data` contains all data generators. They all inherit the abstract class `DataGenerator`, defined in `data/data_generator.py`.

`models` contains all the neural models we use, written in [jax](https://github.com/google/jax) and [haiku](https://github.com/deepmind/dm-haiku), two open source libraries, and models used for evaluation,
such as CTW.


## Installation

Clone the source code into a local directory:
```bash
git clone https://github.com/deepmind/neural_networks_solomonoff_induction.git
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
@article{grau2023neural,
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
  year         = {2023}
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
