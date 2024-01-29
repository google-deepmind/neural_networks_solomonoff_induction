# Copyright 2023 DeepMind Technologies Limited
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

"""Data generator for UTMs, used to sample sequences from programs."""

import enum
from typing import Any, Dict, Sequence

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from data import data_generator as dg_lib
from data import utms


class Tokenizer(enum.IntEnum):
  """Method to use to convert an output string to a list of integers."""

  ASCII = 0  # We use the ascii value of each output character.
  SEQ_POSITION = 1  # One-hot encoding in the alphabet_size.


class UTMDataGenerator(dg_lib.DataGenerator):
  """Samples programs and runs them on the UTM."""

  def __init__(
      self,
      batch_size: int,
      seq_length: int,
      rng: int | np.random.Generator,
      utm: utms.UniversalTuringMachine,
      memory_size: int,
      maximum_steps: int,
      tokenizer: Tokenizer,
      maximum_program_length: int = 100,
  ):
    """Initializes the UTM data generator.

    Args:
      batch_size: The number of sequences to sample.
      seq_length: The length of the sequences to sample. Only output sequences
        with length at least seq_length are returned.
      rng: numpy random number generator or a seed.
      utm: Used to run and sample programs.
      memory_size: Size of the memory to use in the UTM. It is the number of
        cells that the machine has access to. The size of each cell (1 bit, 1
        byte, 1 kilobyte?) is defined by the machine itself. See the passed UTM
        docstring for details on how it's used internally.
      maximum_steps: Maximum number of steps the UTM will be able to run after
        which it will send a TimeoutError.
      tokenizer: The tokenizer that maps the outputs of the UTM (characters) to
        integers.
      maximum_program_length: Maximum length of the sampled programs.
    """
    super().__init__(batch_size, seq_length, rng)
    self._utm = utm
    self._memory_size = memory_size
    self._maximum_steps = maximum_steps
    self._tokenizer = tokenizer
    self._maximum_program_length = maximum_program_length

    self._token_position: dict[str, int] | None = None
    if tokenizer == Tokenizer.SEQ_POSITION:
      self._token_position = {}
      for index, token in enumerate(range(utm.alphabet_size)):
        self._token_position[token] = index

  @property
  def feature_size(self) -> int:
    """Returns the feature size (last dimension of the sampled sequences).

    Raises:
      NotImplementedError if the tokenizer is not in {LIST_POSITION, ASCII}.
    """
    match self._tokenizer:
      case Tokenizer.SEQ_POSITION:
        return self._utm.alphabet_size
      case Tokenizer.ASCII:
        return 128

  def sample_params(self, sample_size: int) -> Sequence[str]:
    """Samples parameters used to generate the sequences. See base class.

    Args:
      sample_size: Number of parameters to sample.

    Returns:
      The programs to be run on the UTM. Parameters are programs for this
      data generator.
    """
    programs = []
    for _ in range(sample_size):
      programs.append(
          self._utm.sample_program(self._maximum_program_length, self._rng)
      )

    return programs

  def sample_from_params(
      self,
      params: Sequence[str],
  ) -> tuple[dg_lib.Sequences, dg_lib.CategoricalProbs, Dict[str, Any]]:
    """Samples sequences and their categorical probabilities from programs.

    We run the programs on the UTM, retrieve the outputs, and pad them with
    self._padding_token to match self._seq_length.

    Args:
      params: Programs to run on the UTM.

    Returns:
      * A tensor of one-hot sequences, of shape (B, T, F) where B is equal to
        len(params), T is equal to self._seq_length and F is equal to
        self.feature_size.
      * A tensor of categorical probabilities, here equal to the sequences
        themselves.

    Raises:
      NotImplementedError if the tokenizer is not in
      {SEQ_POSITION, ASCII}.
    """
    outputs = []
    results = []
    masks = []
    for program in params:
      result = self._utm.run_program(
          program=program,
          memory_size=self._memory_size,
          maximum_steps=self._maximum_steps,
          max_output_length=self._seq_length,
      )
      output = result['output']
      # We keep only the sequences that are long enough.
      # result['output_length'] = len(result['output']) unless some padding
      # is used in the UTM.
      mask = np.zeros(self._seq_length, dtype=np.uint8)
      if result['output_length'] != self._seq_length:
        padding_length = self._seq_length - result['output_length']
        output += '\x00' * padding_length
        mask[-padding_length:] = 1
      assert len(output) == self._seq_length
      match self._tokenizer:
        case Tokenizer.ASCII:
          # We convert the output to a buffer of bytes, using a utf-8
          # encoding.
          buffer = bytes(output, 'utf-8')
          # Numpy converts the ascii values to uint8 values.
          output = np.frombuffer(buffer, dtype=np.uint8)
        case Tokenizer.SEQ_POSITION:
          assert self._token_position is not None  # For pytype.
          output = np.asarray(
              [self._token_position[ord(token)] for token in output]
          )
      outputs.append(output)
      results.append(result)
      masks.append(mask)

    loss_mask = jnp.asarray(masks)
    output = jnp.asarray(outputs)
    output = jnn.one_hot(output, num_classes=self.feature_size, dtype=jnp.uint8)
    # In this case, the probabilities for the categorical distribution are
    # delta, meaning there is only one possible next token. They are equal to
    # the one hot output itself.
    categorical_probs = jnp.copy(output)
    log_dict = {'results': results, 'loss_mask': loss_mask}
    return output, categorical_probs, log_dict
