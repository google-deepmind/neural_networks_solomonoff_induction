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

"""Trains a neural model on some data generated from the data/ folder."""

import functools
from typing import Any

from absl import app
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import tree

from data import data_generator as dg_lib
from data import utm_data_generator as utm_dg_lib
from data import utms as utms_lib
from models import transformer


def _make_loss_fn(model: hk.Transformed) -> Any:
  """Returns the loss function for update_parameters."""

  def loss_fn(
      params: hk.Params,
      sequences: dg_lib.Sequences,
      mask: jax.Array,
  ) -> jnp.float32:
    """Returns the loss for the model and the last state.

    Args:
      params: The parameters of the model, usually a neural network.
      sequences: The sequences to evaluate, see type.
      mask: A binary array, True (1's) denote where to skip computing the loss.
    """
    conditionals = model.apply(
        params=params,
        targets=sequences,
        rng=None,
    )
    true_conditionals = jnp.take_along_axis(
        conditionals, sequences[..., None], axis=-1
    )[..., 0]
    true_conditionals = jnp.multiply(mask, true_conditionals)
    marginals = jnp.sum(true_conditionals, axis=1)  # Shape (B,).
    return -jnp.mean(marginals)

  return loss_fn


@functools.partial(
    jax.jit, static_argnames=('optimizer', 'grad_fn', 'normalize_gradients')
)
def _update_parameters(
    params: hk.Params,
    opt_state: optax.OptState,
    sequences: jax.Array,
    mask: jax.Array,
    grad_fn: Any,
    optimizer: optax.GradientTransformation,
    normalize_gradients: bool = True,
) -> tuple[hk.Params, optax.OptState, dict[str, Any]]:
  """Returns updated params and extra logs (like loss, last state etc).

  Backpropagation is done on the whole sequence. The whole function is jitted.

  Args:
    params: The current parameters of the network.
    opt_state: The optimizer state.
    sequences: The input of sequences to evaluate. See base_predictor.py.
    mask: A binary array, True (1's) denote where to skip computing the loss.
    grad_fn: A gradient function, which takes some parameters, a random seed,
      the data to compute the gradient on, and an initial state for the
      predictor. It returns the gradient of the parameters for this batch of
      data, and extra values.
    optimizer: An optax optimizer.
    normalize_gradients: Whether to divide the gradients by the length of the
      sequences, or keep them as is. Using this option guarantees to have the
      same scale across various sequence lengths, and therefore tasks.
  """
  loss, grad = grad_fn(params, sequences, mask)
  if normalize_gradients:
    length_sequence = float(sequences.shape[1])
    grad = tree.map_structure(lambda x: x / length_sequence, grad)
  updates, new_opt_state = optimizer.update(grad, opt_state)
  new_params = optax.apply_updates(params, updates)

  log_dict = {
      'loss': loss,
      'grad_norm_unclipped': optax.global_norm(grad),
  }

  return new_params, new_opt_state, log_dict


def train_transformer_decoder(
    data_generator: dg_lib.DataGenerator,
    training_steps: int,
    log_every: int,
    batch_size: int = 128,
    use_tqdm: bool = True,
) -> tuple[hk.Params, float]:
  """Trains a neural network on some synthetic data.

  We train a decoder-only transformer on batches, minimizing the log-loss
  objective. The exact architecture can be modified using the TransformerConfig
  object (defined in models/transformer.py)

  Args:
    data_generator: Used to generate batches of data to train on.
    training_steps: Number of batches to train on.
    log_every: How often to log the loss. If negative or 0, no log at all.
    batch_size: The number of sequences in a batch.
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The final loss, and final parameters.
  """
  config = transformer.TransformerConfig(vocab_size=data_generator.feature_size)
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )

  # Initialize parameters.
  dummy_batch, _ = data_generator.sample_dummy(batch_size)
  # Transform one-hots to integer tokens.
  dummy_batch = np.argmax(dummy_batch, axis=-1)
  rng = jax.random.PRNGKey(0)
  params = model.init(rng, dummy_batch)

  # Make gradient function.
  loss_fn = _make_loss_fn(model)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

  # Make optimizer, to apply the gradients.
  optimizer = optax.adam(learning_rate=1e-4)
  opt_state = optimizer.init(params)

  logging.info('Initialization done, starting training...')
  last_loss = 0.0
  default_mask = lambda x: np.ones(x.shape[:2], dtype=bool)
  for step in tqdm.trange(training_steps, disable=not use_tqdm):
    batch, log_dict = data_generator.sample()
    # Transform one-hots to integer tokens.
    batch = np.argmax(batch, axis=-1)
    if 'loss_mask' in log_dict:
      loss_mask = log_dict['loss_mask']
    else:
      loss_mask = default_mask(batch)
    logging.info('Batch fetched.')

    params, opt_state, logs = _update_parameters(
        params=params,
        opt_state=opt_state,
        sequences=batch,
        grad_fn=grad_fn,
        optimizer=optimizer,
        mask=loss_mask,
    )
    if log_every > 0 and step % log_every == 0:
      logging.info(
          'Step %f, Loss %f, Grad norm %f',
          step,
          logs['loss'],
          logs['grad_norm_unclipped'],
      )
    last_loss = logs['loss']

  return params, last_loss


def main(_) -> None:
  """Trains a model and save the parameters to a file."""
  utm = utms_lib.BrainPhoqueUTM()
  data_generator = utm_dg_lib.UTMDataGenerator(
      batch_size=32,
      seq_length=256,
      rng=1,
      utm=utm,
      memory_size=10,
      maximum_steps=100,
      tokenizer=utm_dg_lib.Tokenizer.ASCII,
      maximum_program_length=100,
  )

  params, loss = train_transformer_decoder(
      data_generator=data_generator,
      training_steps=100,
      log_every=10,
  )
  logging.info('Final loss: %f', loss)

  np.savez('params.npz', **params)
  logging.info('Parameters saved in file params.npz')


if __name__ == '__main__':
  app.run(main)
