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

"""A meta-generator, sampling data from a distribution of generators.

This is used in particular in the paper to sample from multiple Chomsky tasks.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from neural_networks_solomonoff_induction.data import data_generator as dg_lib


class MetaDataGenerator(dg_lib.DataGenerator):
  """A generator composed of a list of generators.

  The resulting feature size is the biggest one found from the generators.

  The distribution over generators is uniform at the moment.
  """

  def __init__(
      self,
      batch_size: int,
      seq_length: int,
      rng: int | np.random.Generator,
      generators: Sequence[dg_lib.DataGenerator],
      logs_to_concatenate: Sequence[str] | None = None,
  ):
    """Constructor.

    Args:
      batch_size: The batch size.
      seq_length: The sequence length.
      rng: numpy random number generator or a seed.
      generators: The generators to combine.
      logs_to_concatenate: If the generators generate logs that we want to
        record in the output of this class, then this variable specifies a list
        of keys (and values) that we want to extract from the generator logs.
    """
    super().__init__(batch_size, seq_length, rng)
    self._generators = generators
    self._num_generators = len(self._generators)
    self._feature_sizes = [gen.feature_size for gen in self._generators]
    self._max_feature_size = max(self._feature_sizes)
    self._logs_to_concatenate = logs_to_concatenate
    self._gen_batch_sizes = [gen.batch_size for gen in self._generators]
    self._check_batch_size_and_seq_length()

  def _check_batch_size_and_seq_length(self):
    """We force the batch size of generators to add up to self._batch_size."""
    if np.sum(self._gen_batch_sizes) != self._batch_size:
      raise ValueError(
          'The sum of batch sizes should add to the provided batch size for'
          f' the MetaDataGenerator which is {self._batch_size} but found'
          f'{np.sum(self._gen_batch_sizes)}.'
      )
    for gen in self._generators:
      if gen.seq_length != self._seq_length:
        raise ValueError('Sequence lengths of all generators must be the same.')

  @property
  def feature_size(self) -> int:
    return self._max_feature_size

  def sample_params(self, sample_size: int) -> dg_lib.Params:
    """Returns the parameters used to sample the sequences.

    The returned params is a list composed of the params that each generator
    returns.
    Args:
      sample_size: The sample size in total for all generators.
    """
    if sample_size != self._batch_size:
      raise ValueError(
          'Sample_size should be equal to the batch size, which is'
          f' {self._batch_size}, but found {sample_size}.'
      )
    all_params = []
    for gen, batch_size in zip(self._generators, self._gen_batch_sizes):
      all_params.append(gen.sample_params(batch_size))
    return all_params

  def sample_from_params(
      self, params: dg_lib.Params
  ) -> tuple[dg_lib.Sequences, dg_lib.CategoricalProbs, dict[str, Any]]:
    seqs = []
    probs_list = []

    # Initializing logs dict.
    outer_logs = {}
    if self._logs_to_concatenate is not None:
      for key in self._logs_to_concatenate:
        outer_logs[key] = []

    for (gen, gen_param) in zip(self._generators, params):
      seq, probs, logs = gen.sample_from_params(gen_param)
      seqs.append(seq)
      probs_list.append(probs)
      if self._logs_to_concatenate is not None:
        for key in self._logs_to_concatenate:
          outer_logs[key].append(logs[key])

    if self._logs_to_concatenate is not None:
      for key in self._logs_to_concatenate:
        outer_logs[key] = np.concatenate(outer_logs[key], axis=0)

    return (
        np.concatenate(seqs, axis=0),
        np.concatenate(probs_list, axis=0),
        outer_logs,
    )
