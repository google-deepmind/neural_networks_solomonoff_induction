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

"""Context Tree Weighting data generator."""

from collections.abc import Mapping
import copy
from typing import Any, Sequence

import jaxtyping as jtp
import numpy as np

from data import data_generator as dg_lib


# Probability of spawning childrens from a node when creating CTW tree.
_SPAWN_PROB = 0.5

_FEATURE_SIZE = 2  # The size is 2 because we have one-hot encoded bits.
_BETA_PARAM = 0.5  # Leafs use a Beta to sample the thetas.

ContextLengths = jtp.UInt8[jtp.Array, 'B T']
TreeHashMap = dict[str, float]


def generate_tree(
    tree_hashmap: TreeHashMap,
    rng: np.random.Generator,
    current_context: str = '',
    max_level: int = 10,
) -> None:
  """Fills the tree hashmap with the contexts and associated thetas.

  Args:
    tree_hashmap: A dictionary {context: theta}.
    rng: The numpy rng to generate random number.
    current_context: The current context, used for recursivity.
    max_level: A constraint on the maximum depth of the tree.
  """
  level = len(current_context)
  should_make_children = rng.uniform() < _SPAWN_PROB
  if should_make_children and level < max_level:
    for next_bit in ['0', '1']:
      child_context = current_context + next_bit
      generate_tree(tree_hashmap, rng, child_context, max_level)
  else:
    theta = rng.beta(_BETA_PARAM, _BETA_PARAM)
    tree_hashmap[current_context] = theta


def _find_theta_for_seq(
    seq: str, tree_hashmap: TreeHashMap
) -> tuple[float, int]:
  """Returns theta and context length in the tree matching with the sequence."""
  for i in range(len(seq)):
    context = seq[-(i + 1) :]
    if context in tree_hashmap:
      return tree_hashmap[context], len(context)
  return tree_hashmap[''], 0


class CTWGenerator(dg_lib.DataGenerator):
  """A CTW data generator of binary data.

  This creates a tree of CTWNodes using the generate_tree function.
  In addition it provides with the sample method to actually sample from the
  tree following the DataGenerator base class.
  """

  def __init__(
      self,
      batch_size: int,
      seq_length: int,
      rng: int | np.random.Generator,
      max_depth: int = 5,
      with_contexts: bool = False,
  ):
    """Constructor.

    Args:
      batch_size: Size of first dimension of the outputs. batch_size here has
        two different meanings depending on the self._with_contexts variable.
        When self._with_contexts = True then this method samples one sequence
        and, from it, generates a batch of contexts and sequences. In this case
        batch_size is equal to seq_length (otherwise we throw a ValueError).
        When self._with_contexts = False, then every element of the batch is an
        independent sequence of seq_length, and we return contexts = None.
      seq_length: Size of second dimension of the outputs. The length of the
        sequence. See above: batch_size for more info.
      rng: numpy random number generator or a seed.
      max_depth: The maximum depth of the tree.
      with_contexts: If True, the sample() method takes a single sequence of
        data with shape (1, T, F), where T is the sequence length, and builds a
        batch of contexts with shape (T, T-1, F) and targets with shape (T, 1,
        F). This means that sample() must be called with batch_size ==
        seq_length, otherwise you get errors. If False, contexts are empty, and
        targets are of shape (B, T, F) where B is the batch_size.
    """
    super().__init__(batch_size, seq_length, rng)
    self._max_depth = max_depth
    self._with_contexts = with_contexts

  @property
  def feature_size(self) -> int:
    return _FEATURE_SIZE

  def sample_params(self, sample_size: int) -> list[TreeHashMap]:
    """Returns sample_size CTW trees."""
    params = []
    for _ in range(sample_size):
      tree_hashmap = {'': self._rng.beta(_BETA_PARAM, _BETA_PARAM)}
      generate_tree(
          tree_hashmap,
          rng=self._rng,
          max_level=self._max_depth,
      )
      params.append(copy.deepcopy(tree_hashmap))
    return params

  def _sample_one(
      self,
      seq_length: int,
      meta_params: TreeHashMap,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples one stream of binary data of length seq_length.

    Args:
      seq_length: The length of the sampled sequence.
      meta_params: The tree hashmap of the form {context: theta}.

    Returns:
      A tuple of numpy arrays: the sampled sequence, the parameters used to
      sample the sequence and the context lengths.
    """
    sequence, thetas, context_lengths = [], [], []
    str_sequence = ''

    # Buffer of random values to avoid resampling at each timestep.
    random_values = self._rng.uniform(size=seq_length)
    for t in range(seq_length):
      context = str_sequence[-self._max_depth :]
      # Padding with zeros if it does not fill the tree depth.
      if len(context) < self._max_depth:
        context = '0' * (self._max_depth - len(context)) + context

      # Find the right theta in the hashmap.
      theta, context_length = _find_theta_for_seq(
          seq=context,
          tree_hashmap=meta_params,
      )
      thetas.append(theta)
      context_lengths.append(context_length)

      # Sample an observation using the buffer of random values.
      observation = int(random_values[t] > 1.0 - theta)
      str_sequence += str(observation)
      sequence.append(observation)

    current_seq = np.asarray(sequence, dtype=np.uint8)
    current_seq = np.eye(_FEATURE_SIZE, dtype=np.uint8)[current_seq]
    thetas = np.asarray(thetas, dtype=np.float32)
    thetas = np.stack([1 - thetas, thetas], axis=1)

    context_lengths = np.asarray(context_lengths, dtype=np.int32)
    return current_seq, thetas, context_lengths

  def sample_from_params(
      self, params: list[TreeHashMap]
  ) -> tuple[dg_lib.Sequences, dg_lib.CategoricalProbs, dict[str, Any]]:
    """Returns samples, categorical probs, and context lengths at each sample."""
    samples = []
    for single_params in params:
      samples.append(self._sample_one(self._seq_length, single_params))
    sequences, categorical_probs, context_lengths = zip(*samples)
    return (
        np.stack(sequences),
        np.stack(categorical_probs),
        {'context_lengths': np.array(context_lengths)},
    )

  def _get_tree_depths(self, params: Sequence[TreeHashMap]) -> np.ndarray:
    """Compute maximum tree depth for each sequence in the batch."""
    return np.asarray(
        [np.max([len(p) for p in tree.keys()]) for tree in params]
    )

  def sample(self) -> tuple[dg_lib.Sequences, Mapping[str, Any]]:
    """Returns a batch of data and auxiliary logs from CTW trees."""
    sequences, log_dict = super().sample()
    tree_depths = self._get_tree_depths(log_dict['params'])
    log_dict['tree_depths'] = tree_depths
    return sequences, log_dict
