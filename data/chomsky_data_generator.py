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

"""Data generator for a single Chomsky task.

Tasks taken from the paper "Neural Networks and the Chomsky Hierarchy".
Arxiv: https://arxiv.org/abs/2207.02098
Github: https://github.com/google-deepmind/neural_networks_chomsky_hierarchy
"""

from typing import Any

import jax
from experiments import constants as chomsky_constants
import numpy as np

from data import data_generator as dg_lib


# Delimiters are allocated in the last two feature dims of the one-hot vectors.
_NUM_DELIMITERS = 2
# We use the second last dimension (in feature space) for the input delimiter.
_INPUT_DEL_IDX = -2
# We use the last dimension (in feature space) for the output delimiter.
_OUTPUT_DEL_IDX = -1

# For now we only allow the tasks with binary input and output.
BINARY_TASKS = [
    'even_pairs',
    'parity_check',
    'reverse_string',
    'duplicate_string',
    'odds_first',
    'compute_sqrt',
]

ORDERED_TASKS = [
    'even_pairs',  # Regular.
    'modular_arithmetic',
    'parity_check',
    'cycle_navigation',
    'stack_manipulation',  # Context free.
    'reverse_string',
    'modular_arithmetic_brackets',
    'solve_equation',
    'duplicate_string',  # Context sensitive.
    'missing_duplicate_string',
    'odds_first',
    'binary_addition',
    'binary_multiplication',
    'compute_sqrt',
    'bucket_sort',
]

# Assumed indices and symbols:
# 0: 0
# 1: 1
# 2: 2
# 3: 3
# 4: 4
# 5: +
# 6: *
# 7: -
# 8: (
# 9: )
# 10: x
# 11: =
# 12: POP
# 13: PUSH 0
# 14: PUSH 1
# 15: . (delimiter 1)
# 16: : (delimiter 2)


FEATURE_MAP_WITHOUT_DELIMITERS = {
    'even_pairs': (0, 1),
    'modular_arithmetic': (0, 1, 2, 3, 4, 5, 7, 6),
    'parity_check': (0, 1),
    'cycle_navigation': (0, 1, 2, 3, 4),
    'stack_manipulation': (0, 1, 12, 13, 14),
    'reverse_string': (0, 1),
    'modular_arithmetic_brackets': (0, 1, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11),
    'solve_equation': (0, 1, 2, 3, 4, 5, 7, 8, 9),
    'duplicate_string': (0, 1),
    'missing_duplicate_string': (0, 1, 2, 3),
    'odds_first': (0, 1),
    'binary_addition': (0, 1, 5),
    'binary_multiplication': (0, 1, 6),
    'compute_sqrt': (0, 1),
    'bucket_sort': (0, 1, 2, 3, 4),
}


class ChomskyDataGenerator(dg_lib.DataGenerator):
  """Data generator for a single Chomsky task.

  This generator samples an input of some random length, computes the
  output according to a Chomsky task, does this multiple times concatenating
  everything into one string.

  **Warning!** it is assumed that inputs are sampled randomly with 0.5 prob.
  (since we consider only binary inputs). Since computation is deterministic
  outputs have probability 1.

  There are 2 symbol delimiters, one right after the input string (.) and
  another right after a full input-output example (:).

  Example with 'reverse_string' task:
    No delimiters:
      Substring 1:  baab (coming from ba -> ab)
      Substring 2: aaabbaaa (coming from aaab -> baaa)
      Final string: baabaaabbaa
    With delimiters:
      Substring 1: ba.ab
      Substring 2: aaab.baaa
      Final string: ba.ab:aaab.baaa:
  """

  def __init__(
      self,
      task_str: str,
      max_input_length: int,
      use_delimiters: bool,
      *args,
      expand_feature_size: int = 0,
      **kwargs,
  ):
    """Constructor.

    Args:
      task_str: The task to use. See ORDERED_TASKS.
      max_input_length: The maximum length of the inputs.
      use_delimiters: Whether to include delimiters after inputs and outputs.
      *args: Extra arguments to pass to base class.
      expand_feature_size: The desired size of the feature size without counting
        the delimiters (which would add 2 more). This integer should be either 0
        (no modification to the task feature size) or bigger than the task
        feature size. For example, for a binary task of feature size 2, if
        expand_feature_size is 4, then the final feature size will be 4 and the
        binary symbols will be one-hot-encoded to the first two feature
        dimensions (the last two dimensions will exist but unused).
      **kwargs: Extra kwargs to pass to base class.
    """
    super().__init__(*args, **kwargs)

    self._task_str = task_str
    self._task = chomsky_constants.TASK_BUILDERS[task_str]()
    self._expand_feature_size = expand_feature_size
    self._max_raw_feature_size = max(
        self._task.input_size, self._task.output_size
    )

    if 0 < self._expand_feature_size <= self._max_raw_feature_size:
      raise ValueError(
          f'The provided expand_feature_size is {expand_feature_size}, but it'
          ' should be either 0 or bigger than the task max feature size, which '
          f'is {self._max_raw_feature_size}.'
      )
    elif self._expand_feature_size > self._max_raw_feature_size:
      self._max_raw_feature_size = self._expand_feature_size

    self._max_input_length = max_input_length
    self._use_delimiters = use_delimiters

    if self._use_delimiters:
      self._input_delimiter = np.zeros((self._batch_size, 1, self.feature_size))
      self._input_delimiter[:, :, _INPUT_DEL_IDX] = 1
      self._output_delimiter = np.zeros(
          (self._batch_size, 1, self.feature_size)
      )
      self._output_delimiter[:, :, _OUTPUT_DEL_IDX] = 1
      self._input_delimiter_probs = self._build_input_delimiter_probs()

    self._input_cat_probs = self._build_input_categorical_probs()

  @property
  def feature_size(self) -> int:
    return self._max_raw_feature_size + _NUM_DELIMITERS * int(
        self._use_delimiters
    )

  def _build_input_delimiter_probs(self) -> np.ndarray:
    """Returns the probability of the delimiter being outputed.

    To do so, we compute the remaining length at each step (input_len + 1 - i)
    and then, compute the uniform probability 1/remaining_length. That is
    exactly the probability of the next token being the delimiter. Since at the
    minimum input length is 1, then at time step zero the probabiliy of the next
    token being the delimiter is 0.0 (that's why we start with 0). As we
    approach the end the probability increases rapidly.

    This is only useful for analysis later.
    """
    input_len = self._max_input_length
    probs = [0.0] + [1 / (input_len + 1 - i) for i in range(1, input_len + 1)]
    return np.array(probs)

  def _build_input_categorical_probs(self) -> np.ndarray:
    """Returns categorical probs of the input tokens.

    It is assumed a uniform distribution over the available input tokens.
    This is useful for analysis.
    """
    feature_len = self._task.input_size  # Raw input size.
    input_probs = np.zeros(
        (self._batch_size, self._max_input_length + 1, self.feature_size)
    )
    if self._use_delimiters:
      delimiter_probs = self._build_input_delimiter_probs()
      delimiter_tiled = np.tile(delimiter_probs, (self._batch_size, 1))
      input_probs[:, :, _INPUT_DEL_IDX] = delimiter_tiled

      remainders = 1 - delimiter_probs
      input_probs_single = np.array([r / feature_len for r in remainders])
    else:
      input_probs_single = 1 / feature_len * np.ones(self._max_input_length + 1)

    tiled = np.tile(
        input_probs_single,
        (self._batch_size, 1),
    )
    for i in range(feature_len):
      input_probs[:, :, i] = tiled
    return input_probs

  def _add_zeros_to_feature_dim(
      self, tensor: np.ndarray, num_zeros: int = 1
  ) -> np.ndarray:
    zeros = np.zeros((self._batch_size, tensor.shape[1], num_zeros))
    return np.concatenate([tensor, zeros], axis=-1)

  def _build_categorical_probs(
      self, s_batch: dict[str, np.ndarray]
  ) -> np.ndarray:
    """Returns categorical probs assuming random inputs and input lengths."""
    input_len = s_batch['input'].shape[1]
    if input_len <= self._input_cat_probs.shape[1]:
      input_probs = self._input_cat_probs[:, :input_len, :]
    else:
      raise ValueError(
          'The inputs are bigger than the precomputed categoracal probs.'
      )
    num_pad_tokens = self.feature_size - s_batch['output'].shape[-1]
    output_probs = np.pad(
        s_batch['output'],
        ((0, 0), (0, 0), (0, num_pad_tokens)),
        'constant',
        constant_values=0,
    )
    return np.concatenate([input_probs, output_probs], axis=1)

  def sample_params(self, sample_size: int) -> dg_lib.Params:
    # There are no params yet.
    return None

  def sample_from_params(
      self, params: dg_lib.Params
  ) -> tuple[dg_lib.Sequences, dg_lib.CategoricalProbs, dict[str, Any]]:
    del params
    batches = []
    probs_list = []
    input_locs = []
    output_delim = []
    total_time_len = 0

    def _sample_randint(low: int, high: int) -> int:
      """Sample random-uniformly from the interval [low, high)."""
      # This uses the internal numpy rng, whose seed is controlled during
      # initialization. Each call will advance the rng state.
      return self._rng.integers(low=low, high=high)

    while True:
      input_seq_length = _sample_randint(low=1, high=self._max_input_length + 1)
      # Convert the internal rng (with controlled seed) to jax rng.
      rand_int = _sample_randint(
          low=np.iinfo(np.int64).min,
          high=np.iinfo(np.int64).max,
      )
      rng = jax.random.PRNGKey(rand_int)
      s_batch = self._task.sample_batch(rng, self._batch_size, input_seq_length)
      s_batch['input'] = np.array(s_batch['input'])
      s_batch['output'] = np.array(s_batch['output'])

      # This standarizes shapes for some tasks.
      if len(s_batch['output'].shape) == 2 and len(s_batch['input'].shape) == 3:
        s_batch['output'] = np.expand_dims(s_batch['output'], axis=1)

      if self._expand_feature_size:
        size_diff = self._expand_feature_size - self._task.input_size
        s_batch['input'] = self._add_zeros_to_feature_dim(
            s_batch['input'], size_diff
        )

        size_diff = self._expand_feature_size - self._task.output_size
        s_batch['output'] = self._add_zeros_to_feature_dim(
            s_batch['output'], size_diff
        )

      if self._use_delimiters:
        s_batch['input'] = self._add_zeros_to_feature_dim(
            s_batch['input'], _NUM_DELIMITERS
        )
        s_batch['output'] = self._add_zeros_to_feature_dim(
            s_batch['output'], _NUM_DELIMITERS
        )

        s_batch['input'] = np.concatenate(
            [s_batch['input'], self._input_delimiter], axis=1
        )

        s_batch['output'] = np.concatenate(
            [s_batch['output'], self._output_delimiter], axis=1
        )

      inputs_ones = np.ones(s_batch['input'].shape[:2], dtype=np.int8)
      outputs_zeros = np.zeros(s_batch['output'].shape[:2], dtype=np.int8)

      s_output_delim = outputs_zeros
      s_output_delim[:, -1] = 1
      s_output_delim = np.concatenate(
          [np.zeros_like(inputs_ones), s_output_delim], axis=1
      )
      s_input_locs = np.concatenate([inputs_ones, outputs_zeros], axis=1)

      s_batch_joined = np.concatenate(
          [s_batch['input'], s_batch['output']], axis=1
      )

      # Now we compute the 'params' used to generate the small sequences.
      s_probs = self._build_categorical_probs(s_batch)

      # We append the small sequences to a list.
      probs_list.append(s_probs)
      batches.append(s_batch_joined)
      input_locs.append(s_input_locs)
      output_delim.append(s_output_delim)

      # When we have enough small sequences, we concatenate them together.
      total_time_len += s_batch_joined.shape[1]
      if total_time_len >= self._seq_length:
        sequences = np.concatenate(batches, axis=1)[:, : self._seq_length, :]
        probs = np.concatenate(probs_list, axis=1)[:, : self._seq_length, :]
        input_locs = np.concatenate(input_locs, axis=1)[:, : self._seq_length]
        output_delim = np.concatenate(output_delim, axis=1)[
            :, : self._seq_length
        ]

        return (
            sequences,
            probs,
            {'input_locations': input_locs, 'output_delimiters': output_delim},
        )
