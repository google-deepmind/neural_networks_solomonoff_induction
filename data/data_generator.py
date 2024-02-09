# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data generators used in meta-learning."""

import abc
from collections.abc import Mapping
from typing import Any

import jaxtyping as jtp
import numpy as np


# The parameters used within a trajectory (e.g. for i.i.d. distributions these
# will coincide with the CategoricalProbs, for CTW these are the trees with
# their corresponding leaf-parameters).
Params = Any
# The samples, one-hot sequences.
Sequences = jtp.UInt8[jtp.Array, "B T F"]
# The probabilities used to sample the categorical samples.
CategoricalProbs = jtp.Float32[jtp.Array, "B T F"]


class DataGenerator(abc.ABC):
  """Abstract class for trajectory generators.

  Example use with a dirichlet prior, to sample 10 binary sequences:
    1. Sample the parameters for the sequences with sample_params(10). That
      gives us some bernoulli parameters p_1 to p_10.
    2. Sample the sequences from the bernoulli parameters. That returns
      the binary sequences AND the categorical probabilities at each timestep,
      which are equal to p_i for sequence i, along the sequence. In that case,
      the parameters coincide with the categorical probabilities as it is IID!
  """

  def __init__(
      self,
      batch_size: int,
      seq_length: int,
      rng: int | np.random.Generator,
  ):
    self._rng = np.random.default_rng(rng)
    self._batch_size = batch_size
    self._seq_length = seq_length

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def seq_length(self) -> int:
    return self._seq_length

  @property
  @abc.abstractmethod
  def feature_size(self) -> int:
    """Returns the size of the categorical distribution."""

  @abc.abstractmethod
  def sample_params(
      self,
      sample_size: int,
  ) -> Params:
    """Returns a set of parameters.

    They correspond to the parameters that remain fixed within a trajectory.
    For instance, for a simple IID categorical distribution, the parameters
    are the probabilities we use to sample from the categorical distribution.
    For the CTW data generator, that will be the context trees used to sample
    the sequences.

    Args:
      sample_size: The number of meta parameters to sample.
    """

  @abc.abstractmethod
  def sample_from_params(
      self,
      params: Params,
  ) -> tuple[Sequences, CategoricalProbs, dict[str, Any]]:
    """Returns sampled sequences and probabilities from a set of parameters.

    The sequences are one-hot encoded samples from CategoricalProbs. These are
    different from the passed parameters, which are the same along a single
    sequence (while the categorical probabilities are specific to the position
    within the sequence). If there is no switching though, and you use an IID
    distribution, then the passed params are equal to the returned categorical
    probabilities, with an extra time dimension.

    Args:
      params: The parameters, specific for each sequence. The first dimension
        must be equal to self._batch_size.

    Returns:
      sequences: A tensor of one-hot encoded sequences. See type for shape.
      categorical_probs: A tensor of the same shape as 'sequences' above, but
        containing probabilities (floats) used to sample the 'sequences' tensor.
      logs_dict: Extra information.
    """

  def sample(self) -> tuple[Sequences, Mapping[str, Any]]:
    """Samples a batch with randomly sampled true parameters.

    Returns:
      contexts: None.
      sequences: Sequences of tokens, see type.
      log_dict: Auxiliary logs, like the categorical probabilities along the
        sequence.
    """
    params = self.sample_params(sample_size=self._batch_size)
    sequences, categorical_probs, extra = self.sample_from_params(params=params)
    log_dict = {
        "categorical_probs": categorical_probs,
        "params": params,
    }
    log_dict.update(extra)
    return sequences, log_dict

  def sample_dummy(
      self, batch_size: int
  ) -> tuple[Sequences, Mapping[str, Any]]:
    """Returns dummy data and logs (to initialize parameters or check specs)."""
    sequences, _ = self.sample()
    sequences = sequences[:batch_size]
    return sequences, dict()
