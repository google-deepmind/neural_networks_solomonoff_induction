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

"""A Context Tree Weighting predictor.

Pure conditional implementation following Christopher Mattern's thesis:
https://www.db-thueringen.de/receive/dbt_mods_00027239

This implementation supports any alphabet size (not only binary), and computes
conditional probabilities for all tokens (not only 1).
The multi-alphabet update formula comes from the paper Sequential Weighting
Algorithms for Multi-Alphabet Sources.
https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=0d2a4ed4f1b5b1eef92e725089f2b53a7afbc6d5

This implementation is quite optimized for pure python, but does not exploit
the modern GPUs, especially to process batches of sequences.
"""

import copy
import dataclasses
import enum
from typing import Any

import jaxtyping as jtp
import numpy as np

from neural_networks_solomonoff_induction.data import data_generator as dg_lib

# The predictions are log-probabilities (natural logarithm) for the passed
# sequences. It can either be marginal log-probabilities (i.e. log P(s) for all
# sequences s in the batch), or full conditionals (i.e. log P(token | s_<t) for
# all sequence s, time t and token in the alphabet).
Marginals = jtp.Float32[jtp.Array, '*B']
Conditionals = jtp.Float32[jtp.Array, '*B T F']
Predictions = Marginals | Conditionals


def _to_marginals(
    predictions: Conditionals,
    sequences: dg_lib.Sequences,
) -> Marginals:
  """Converts a conditional array to a marginals array."""
  sequences = np.argmax(sequences, axis=-1)
  true_predictions = np.take_along_axis(
      predictions, sequences[..., None], axis=-1
  )
  true_predictions = true_predictions[..., 0]  # Shape (B, T).
  return np.sum(true_predictions, axis=1)  # Shape (B,).


# Maximum absolute value that a log can take.
MAX_LOG: float = 30.0
# Default maximum capacity of the context table, i.e. the maximum number of
# contexts CTW can learn.
DEFAULT_MAX_CAPACITY = 2**27


@dataclasses.dataclass
class Node:
  """A node of the CTW tree."""

  # Counts contains the current counts for each tokens seen during training.
  counts: np.ndarray  # Shape (num_tokens,).
  # Sum of the counts above. Avoids recomputing the sum every time.
  counts_sum: float
  # A cache of the KT probabilities, in case we don't update the counts.
  prob_kt: np.ndarray | None = None
  # Beta is the weight used for the mixture.
  log_beta: float = np.log(1 / 2)


# This table contains pairs (hash_context, node), which makes the process of
# retrieving nodes even faster than a recursive tree of nodes.
Table = dict[int, Node]


def _hash_array(array: np.ndarray) -> int:
  """Returns the hash for an array."""
  return hash(np.asarray(array).data.tobytes())


def _get_node(
    context: np.ndarray,
    table: Table,
    num_tokens: int,
    max_capacity: int,
) -> Node:
  """Returns a node of the CTW tree, based on the context."""
  hash_context = _hash_array(context)
  if hash_context not in table:
    if len(table) < max_capacity:
      table[hash_context] = Node(
          counts=np.full((num_tokens,), fill_value=1 / 2),
          counts_sum=num_tokens / 2,
          prob_kt=np.full((num_tokens,), fill_value=1 / num_tokens),
      )
    else:
      raise MemoryError('Max capacity reached!')
  return table[hash_context]


def _update_tree(
    table: Table,
    context: np.ndarray,
    true_token: int,
    num_tokens: int,
    true_prob_only: bool = False,
    train_mode: bool = False,
    max_capacity: int = DEFAULT_MAX_CAPACITY,
) -> np.ndarray | float:
  """Updates the CTW tree.

  Args:
    table: The hash table that contains all the information we need about the
      tree.
    context: The list of tokens to use as context.
    true_token: The final token that we observe in the sequence.
    num_tokens: Number of possible tokens.
    true_prob_only: Whether to compute the conditional probabilities of all the
      tokens, or only the true one.
    train_mode: Whether to update the counts with the observed tokens, and
      update the weights for the mixture (beta).
    max_capacity: Maximum capacity of the table.

  Returns:
    The final conditional probability of the true token, or all possible tokens,
    depending on 'true_prob_only'.
  """
  node = _get_node(context, table, num_tokens, max_capacity)

  # We now go up the path we just underwent, and update all the probabilities.
  # The conditional probability for a leaf node is just the KT.
  if true_prob_only:
    last_prob_cond = node.prob_kt[true_token]
  else:
    last_prob_cond = np.copy(node.prob_kt)

  if train_mode:
    node.counts[true_token] += 1
    node.counts_sum += 1
    # Recompute prob_kt after update, but last_prob_cond is not updated because
    # of the copy.
    node.prob_kt = node.counts / node.counts_sum

  for _ in range(len(context)):
    # We shorten the context by one at each level.
    context = context[1:]
    node = _get_node(context, table, num_tokens, max_capacity)

    # The main update rule: beta * cond_d+1 + (1 - beta) * KT.
    beta = np.exp(node.log_beta)
    prob_kt = node.prob_kt[true_token] if true_prob_only else node.prob_kt
    prob_cond = beta * last_prob_cond + (1 - beta) * prob_kt

    if train_mode:
      # Update beta with the rule beta *= CTW_d+1 / CTW_d, in log space.
      if true_prob_only:
        node.log_beta += np.log(last_prob_cond) - np.log(prob_cond)
      else:
        node.log_beta += np.log(last_prob_cond[true_token]) - np.log(
            prob_cond[true_token]
        )
      # Clip to avoid overflow.
      node.log_beta = -MAX_LOG if node.log_beta < -MAX_LOG else node.log_beta
      node.log_beta = 0 if node.log_beta > 0 else node.log_beta

      # Update the counts.
      node.counts[true_token] += 1
      node.counts_sum += 1
      node.prob_kt = node.counts / node.counts_sum

    # We keep the conditional prob around for the update rule (CTW_d+1).
    last_prob_cond = prob_cond
  return last_prob_cond


def _build_context(
    base_context: np.ndarray,
    index: int,
    sequence: np.ndarray,
    depth: int,
    pad_with_zeros: bool = False,
) -> np.ndarray:
  """Returns a (maybe padded) context for CTW."""
  # We look D-timesteps in the past.
  min_index = max(index - depth + len(base_context), 0)
  context = np.concatenate([base_context, sequence[min_index:index]])
  if len(context) < depth and pad_with_zeros:
    # If there is not enough context, we pad with zeros.
    context = np.pad(context, pad_width=(depth - index, 0))
  return context


# CTW only accepts contexts that are sequences of one-hot tokens, exactly as
# bp.Sequences (see base_predictor file for details on these).
CTWContexts = jtp.Int8[jtp.Array, 'B T F']


class FirstTokensMethod(enum.Enum):
  """Method to use when dealing with the first tokens (less than the depth)."""

  # Artificially 'pad' the sequence with (depth - 1) zero tokens. These tokens
  # do not update any KT estimators, but allow us to go down the whole tree
  # right from the beginning.
  PAD_WITH_ZEROS = enum.auto()
  # Build the tree along the way, i.e. slowly add depth to the tree, token by
  # token.
  BUILD_TREE = enum.auto()


class CTWPredictor:
  """Context tree weighting predictor."""

  def __init__(
      self,
      depth: int,
      batched_update: bool = False,
      max_capacity: int = DEFAULT_MAX_CAPACITY,
      first_tokens_method: FirstTokensMethod = FirstTokensMethod.PAD_WITH_ZEROS,
  ) -> None:
    """Initializes the predictor.

    Args:
      depth: The depth of the CTW tree, which corresponds to the maximum length
        of the contexts it considers.
      batched_update: This option modifies the behavior of the update method. If
        True, it assumes each element in a batch is an independent sequence and,
        therefore, the parameters are reset after consuming each one of them. If
        False, it assumes that each element within a batch corresponds to a
        single long sequence and parameters are maintained across batch elements
        (note that histories for some tokens, specially at the beginning of the
        time axis, can be messed up in this case).
      max_capacity: Maximum capacity of the table.
      first_tokens_method: How to deal with the first tokens in the sequence.
    """
    self._depth = depth
    self._batched_update = batched_update
    self._max_capacity = max_capacity
    self._first_tokens_method = first_tokens_method

  @property
  def depth(self) -> int:
    """Returns the depth of the tree."""
    return self._depth

  def predict(
      self,
      params: Table,
      sequences: dg_lib.Sequences,
      contexts: CTWContexts | None = None,
      return_only_marginals: bool = False,
  ) -> Predictions:
    """Unrolls CTW on a batch of sequences. See main _unroll method."""
    predictions, _, _ = self._unroll(
        params=params,
        sequences=sequences,
        contexts=contexts,
        return_only_marginals=return_only_marginals,
        train_mode=False,
    )
    return predictions, None

  def update(
      self,
      params: Table,
      sequences: dg_lib.Sequences,
      contexts: CTWContexts | None = None,
      return_only_marginals: bool = False,
  ) -> tuple[Predictions, None, Table, None, dict[str, Any]]:
    """Update the model's parameters. See main _unroll method."""
    predictions, new_params, loss = self._unroll(
        params=params,
        sequences=sequences,
        contexts=contexts,
        return_only_marginals=return_only_marginals,
        train_mode=True,
    )
    return predictions, None, new_params, None, {'loss': loss}

  def _unroll(
      self,
      params: Table,
      sequences: dg_lib.Sequences,
      contexts: CTWContexts | None = None,
      return_only_marginals: bool = False,
      train_mode: bool = False,
  ) -> tuple[Predictions, Table, float]:
    """Unrolls CTW on a batch of sequences, and returns predictions and params."""
    if contexts is not None:
      num_tokens = max(sequences.shape[-1], contexts.shape[-1])
    else:
      num_tokens = sequences.shape[-1]

    np_sequences = np.array(sequences, dtype=np.uint8)
    int_sequences = np.argmax(np_sequences, axis=-1)
    if contexts is None:
      # Create some empty contexts if it's None.
      contexts = np.empty((sequences.shape[0], 0), dtype=np.int32)
    else:
      contexts = np.array(contexts)
      contexts = np.argmax(contexts, axis=-1)

    all_conditional_probs = []
    # We process the batch sequence by sequence.
    for base_context, sequence in zip(contexts, int_sequences):
      conditional_probs = []
      if self._batched_update:
        used_params = copy.deepcopy(params)
      else:
        used_params = params

      for index, end_token in enumerate(sequence):
        context = _build_context(
            base_context=base_context,
            index=index,
            sequence=sequence,
            depth=self._depth,
            pad_with_zeros=(
                self._first_tokens_method == FirstTokensMethod.PAD_WITH_ZEROS
            ),
        ).astype(np.int32)
        conditional_prob = _update_tree(
            table=used_params,
            num_tokens=num_tokens,
            context=context,
            true_token=end_token,
            train_mode=train_mode,
            true_prob_only=return_only_marginals,
            max_capacity=self._max_capacity,
        )
        conditional_probs.append(conditional_prob)
      all_conditional_probs.append(conditional_probs)
    predictions = np.log(np.array(all_conditional_probs))

    if return_only_marginals:
      # In this case, predictions is an array of size (B, L) and we must sum
      # the log probs over the length dimension.
      marginals = np.sum(predictions, axis=-1)
    else:
      marginals = _to_marginals(predictions, sequences)

    loss = -np.mean(marginals)

    if return_only_marginals:
      predictions = marginals
    return predictions, used_params, loss
