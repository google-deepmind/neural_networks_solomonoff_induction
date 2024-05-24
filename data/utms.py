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

"""Simple universal turing machines, used for generating data.

This file contains the interface for the UTMs (an abstract class), and the code
for the BrainPhoque machine (an implementation of the interface), operating on
bits. The goal is to generate data from programs sampled from the Solomonoff
prior.
"""

import abc
import ast
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


# Result type for run_program.
RunResult = Mapping[str, Any]


class UniversalTuringMachine(abc.ABC):
  """An abstract class for our universal Turing machines."""

  @property
  @abc.abstractmethod
  def program_tokens(self) -> Sequence[str]:
    """Returns the tokens that can be used in a program."""

  @property
  @abc.abstractmethod
  def alphabet_size(self) -> int:
    """Returns the size of the alphabet of the data and outputs."""

  @abc.abstractmethod
  def run_program(
      self,
      program: str,
      memory_size: int,
      maximum_steps: int,
      max_output_length: int,
  ) -> Mapping[str, int | str]:
    """Returns an output string from a program run on the machine.

    Args:
      program: The program to run on the UTM. It is a string of concatenated
        tokens from `self.program_tokens`.
      memory_size: The size of the memory to use.
      maximum_steps: Maximum number of steps after which the function will stop.
      max_output_length: Maximum length of the output sequence.

    Raises:
      IncorrectProgramError: If the input program is incorrect (see examples
        in exception definition below).
    """

  def sample_program(
      self,
      length: int,
      rng: np.random.Generator,
  ) -> str:
    """Returns a program of a given length, with uniformly sampled tokens.

    Args:
      length: The length of the program which will be sampled.
      rng: The numpy random generator to use, or the integer seed. Allows to
        reuse the same generator for multiple usage.
    """
    return ''.join(rng.choice(self.program_tokens, length))


class IncorrectProgramError(Exception):
  """A program cannot be run on the UTM.

  Examples:
    * Some tokens are not part of `utm.program_tokens`.
    * Bad indentation (in Python for instance).
  """


class ProgramSampler(abc.ABC):
  """Abstract base class for sampling programs within a UTM."""

  def __init__(self, rng: np.random.Generator):
    self._rng = rng
    self._tokens = None  # Must be initialized by calling set_tokens().

  def set_tokens(self, tokens: Sequence[str]) -> None:
    """Initializes the sampler."""
    self._tokens = tokens

  @abc.abstractmethod
  def get_sample(self, program: str) -> str:
    """Returns a single sample."""

  @abc.abstractmethod
  def program_ln_loss(self, program: str) -> float:
    """Returns the ln loss of a program."""


class FastSampler(ProgramSampler):
  """Like numpy choice to sample one element at a time uniformly, but faster.

  Batches the samples but returns one sample at a time.

  The samples are i.i.d. so preemptions are not an issue.
  """

  def __init__(self, rng: np.random.Generator):
    super().__init__(rng)
    self._samples = []
    self._sample_idx = 0

  def get_sample(self, program: str) -> str:
    """Returns a single sample."""
    if self._sample_idx == len(self._samples):
      self._samples = self._rng.choice(self._tokens, 100)
      self._sample_idx = 0
    sample = self._samples[self._sample_idx]
    self._sample_idx += 1
    return sample

  def program_ln_loss(self, program: str) -> float:
    if self._tokens is None:
      raise ValueError(
          'set_tokens() must be called before calling program_ln_loss'
      )
    return len(program) * np.log(len(self._tokens))


class MCSampler(ProgramSampler):
  """Markov chain sampler for BrainPhoque programs only.

  This sampler uses a k-order Markov model to predict the sequence of
  instructions of the program.
  The Markov model is loaded from a file as well as the set of tokens.
  """

  def __init__(
      self,
      rng: np.random.Generator,
      filename: str = 'ctx2_filtered.pyd',
      alpha: float = 0.5,
  ):
    """Initializes the sampler.

    Args:
      rng: The random number generator.
      filename: Name of the file containing the counts data. The data is a
        dictionary dict[key, value], where the key can be 'tokens' or
        'counts_dict'. The value for 'tokens' is an (ordered) string of tokens.
        The value for 'counts_dict' is an array of integers. An index in counts
        corresponds to an instruction, while the integer value corresponds to
        the number of times this instruction has been observed to follow the
        corresponding context. After loading, the counts are used to build a
        probability distribution over the instructions, per context.
      alpha: Pseudo-count used for turning the counts into probability
        distributions. alpha=0.5 corresponds to the KT predictor.

    Raises:
      IOError: If the file does not exist.
      KeyError: If the data file does not contain the correct keys.
    """
    super().__init__(rng)

    with open(filename, 'r') as file:
      data_str = str(file.read())

    data = ast.literal_eval(data_str)
    counts = data['counts_dict']
    self._tokens = data['tokens']
    self._token_indices = {ch: idx for idx, ch in enumerate(self._tokens)}

    self._ctx_len = max(map(len, counts.keys()))
    # Turn the counts into distributions by normalizing them.
    self._distributions = {}
    for ctx, dist in counts.items():
      s = np.sum(dist) + alpha * len(dist)
      self._distributions[ctx] = [(x + alpha) / s for x in dist]
    # Call this here also, to avoid having to call it in tests.

  def set_tokens(self, tokens: Sequence[str]) -> None:
    """Ignored. Tokens are set from file data.

    Note that the tokens from the data file may differ from the tokens
    passed to this method.

    Args:
      tokens: Ignored.
    """
    # Do not remove this method. It needs to override the default behaviour
    # to avoid losing the tokens that are loaded from file.

  def get_token_indices(self) -> Mapping[str, int]:
    return self._token_indices

  def get_sample(self, program: str) -> str:
    """Returns a random instruction to extend the program."""
    # Note: prog[-2:] works fine even if prog is empty.
    ctx = program[-self._ctx_len :]
    dist = self._distributions.get(ctx, None)
    instruction = self._rng.choice(self._tokens, p=dist)
    return instruction

  # WARNING: The probability of the short program may be lower
  # than the probability of the long program that has been sampled!
  def sample_program(self, length: int) -> str:
    """Returns a random program of the given length."""
    program = ''
    for _ in range(length):
      program += self.get_sample(program)
    return program

  def program_ln_loss(self, program: str) -> float:
    """Returns the ln loss of a program."""
    ln_loss = 0.0
    ctx_len = self._ctx_len
    for i, instruction in enumerate(program):
      # Get the distribution over next instructions for the current context.
      if i < ctx_len:
        ctx = program[:i]  # Just take whatever suffix we can get.
      else:
        ctx = program[i - ctx_len : i]
      dist = self._distributions.get(ctx, None)
      if dist is not None:
        instruction_idx = self._token_indices[instruction]
        ln_loss += -np.log(dist[instruction_idx])
      else:
        # Unknown context, uniform probabilities.
        # -1 because { is already counted with [, and is not really a separate
        # instruction from a sampling POV.
        ln_loss += np.log(len(self._tokens) - 1)
    return ln_loss


class BrainPhoqueUTM(UniversalTuringMachine):
  """BrainPhoque (slight variant of BrainFuck) universal turing machine.

  Reference: https://en.wikipedia.org/wiki/Brainfuck
  This machine only uses 7 program characters, described in the link above. The
  machine uses a tape memory, filled with integers within the range
  {first_data_int, ... last_data_int}, which can be incremented or
  decremented with the '+' and '-' instructions, with wrap-around when reaching
  the boundaries of the interval.
  Moving the reading head on the data tape is done with the '>' and
  '<' instructions. Loops are created using the '[' and ']' instructions.
  Finally, outputs are returned with the instruction '.'.
  Note that we omit the ',' which is used to read from the input.
  See the reference above for more details.

  BrainPhoque (BP) differs slightly from BrainFuck (BF).

  Why it differs:
  Programs for normalized Solomonoff induction are assumed to be infinite.
  But suppose we want to sample a BF program and run it for T=100000 steps.
  We know that the program size cannot be more than T, so we could sample
  programs of length T, but that's very wasteful as most programs will have
  loops. We want a more agnostic way to sample arbitrary programs.

  How it differs:
  Consider the evaluation of a BF program, and imagine that the instructions
  are sampled as the program is being evaluated. Suppose we sample a '[', and
  the data value is 1, then we need to sample the first instruction of the '[]'
  block, evaluate it and so on. However, if the data value is 0, we need to jump
  to right after the corresponding ']'; but the block does not exist yet. Since
  the block may have arbitrary length once sampled. One idea is to leave a
  placeholder for the block and fill-in the block when the evaluator actually
  enters the block. But this requires inserting elements in the middle of an
  array, which is expensive. A similar idea is to build the program as a tree,
  but then jumping becomes more complicated and less efficient.
  Instead, in order to keep a single array that only needs to be extended at the
  end, we change the semantics of BF slightly — though this not change the
  evaluation of the program.
  The BrainPhoque (BP) UTM executes the same instructions as BF. The main
  difference with BF is that code is sometimes written at a different location.
  More precisely, consider a BF block: A[B]C, where A, B and C are sequences of
  instructions. We call B the _body_ of the block, and C the _continuation_ of
  the block. In BrainPhoque, when encountering a '[', the meaning is the same as
  for BF, but when encountering a '{' the meaning is reversed, and the sequence
  is written A{C]…B where … means that there may be other instructions
  in-between. The block body B is pushed at a later point in the program, and
  the jump table is updated accordingly.
  Since newly sampled instructions are written only at the end of the program,
  we can use a single (extensible) array to generate the BP program.
  An additional property is that every sampled instruction is immediately
  evaluated, avoiding waste.

  Note about the probabilities of the output strings generated by the
  programs (bear with me, this is a little subtle):
  One may think of using the prior weight of the generated program to
  obtain a lower bound on the output sequence generated by this program.
  However, this is pretty loose compared to the cumulative weight of all
  the programs that output the same sequence (and possibly more) --- which
  is the actual quantity we care about.
  Instead, every program on our machine is assumed to be infinite(!). The
  BrainFreak machine ensures that every program is valid.
  The prior of a given program is A^{-n} with n->oo for a program alphabet
  of size A.
  The program are not prefix-free, but the sum of the prior weights is still
  (exactly) 1.
  Let Q be the set of all programs, let w_q be the prior weight of a
  program q.
  Suppose we run program p for T steps and we obtain output s (≤ T).
  Let N be the number of different instructions of the program p that have
  been evaluated during the T steps (N ≤ T).
  Then the Solomonoff mixture probability for s is
  M(s) = sum_{q in Q:U(q)=s*} w_q  ≥  A^{-N}

  Note that 'utm.run_program' returns a shortened program of length at most N.
  Actually, a few additional simplifications are performed on the program
  (see `short_program` in the result dictionary of `run_program`) to further
  increase the lower bound on the Solomonoff probability of the sequence.

  BP can read BF programs *iff* the first time any '[' is encountered the
  data value is not 0, that is, the evaluator immediately enters the body —
  otherwise the '[' should be replaced with '{' and the blocks should be
  moved to the end of the program.
  """

  def __init__(
      self,
      sampler: ProgramSampler,
      alphabet_size: int = 9,
      print_trace: bool = False,
      shorten_program: bool = False,
  ):
    """Constructor.

    Args:
      sampler: ProgramSampler used to generate programs. Can be FastSampler, or
        MCSampler, etc. This is not an optional argument, to force the user to
        pass an RNG explicitly
      alphabet_size: The size of the data and output alphabets.
      print_trace: Prints debugging information as the program is evaluated.
      shorten_program: If False, the keys `short_program` and `short_ln_loss` of
        the result dictionary of `run_program` have values `None`.
    """
    self._sampler = sampler
    self._alphabet_size = alphabet_size
    self._print_trace = print_trace
    self._shorten_program = shorten_program
    sampler.set_tokens(tokens=self.program_tokens)

  @property
  def alphabet_size(self) -> int:
    """Returns the size of the data and output alphabets."""
    return self._alphabet_size

  @property
  def program_tokens(self) -> Sequence[str]:
    """Returns the tokens that can be used to write a BrainPhoque program.

    Note that '{' is not included because when sampling a program an open
    bracket '[' is turned automatically into '{' if the data value is 0,
    and vice versa.
    """
    return ['+', '-', '>', '<', '[', ']', '.']

  @property
  def program_valid_tokens(self) -> Sequence[str]:
    """Returns the tokens that can appear in a BrainPhoque program."""
    return ['+', '-', '>', '<', '[', '{', ']', '.']

  def run_program(
      self,
      program: str,
      memory_size: int,
      maximum_steps: int,
      max_output_length: int,
  ) -> RunResult:
    """Returns the output of a program on the BrainPhoque UTM.

    If program is not '', the program is evaluated for at most `maximum_steps`.
    The 'status' key of the return dictionary can then be 'HALTED' (all
    instructions of the programs have been evaluated), 'TIMEOUT' (the
    number of evaluation steps has reached `maximum_steps`), or 'OUTPUT_LIMIT'
    (if the number of output characters reaches `max_output_length`).
    If program is '', a new program is sampled and evaluated at the same time,
    and the status values can only be 'TIMEOUT' or 'OUTPUT_LIMIT', but not
    'HALTED'.

    Some of the keys of the returned dictionary:
    * 'status': 'HALTED', 'OUTPUT_LIMIT', or 'TIMEOUT'.
    * 'program': the original program if given (possibly shorter if not all
      instructions could be evaluated), or the sampled program.
    * 'short_program': Equivalent to 'program', but after applying a few
      simplifications to shorten it. Evaluating `short_program` is at least as
      fast as evaluating `program`.
    * 'short_ln_loss': Upper bound on the log-loss of the Solomonoff predictor
    for the output sequence, based on the (shortened) sampled program

    Args:
      program: The program to evaluate, or '' to sample and evaluate at the same
        time.
      memory_size: The size of the (tape) memory to use. If the boundaries are
        reached by the memory pointer, we cycle back to the other end.
      maximum_steps: See base class.
      max_output_length: maximum length of the output sequence.

    Raises:
      IncorrectProgramError: If some tokens of the program are not in
        `self.program_tokens`.
    """
    # The index of the pointer moving on the program string.
    program_index = 0
    # The memory tape, initialised with all zeros. Can only contain integers
    # between 0 and alphabet size - 1.
    memory = [0] * memory_size
    # The index of the pointer moving on the memory tape.
    memory_index = 0
    # The output is a string.
    output = ''
    # We count the number of steps to stop if it's too high.
    num_steps = 0
    # Dictionary of places to jump to on brackets.
    jumps = {}
    unmatched_brackets = []
    # Program to construct either from `program`, or by sampling instructions.
    program2 = ''
    # Keep track of the instructions that are sampled after the last print `.`
    # evaluation.
    first_sampled_idx_after_print = len(program)

    def make_result(status: str) -> RunResult:
      # This function is nested because it requires access to many of the
      # local variables of the enclosing function, and passing so many arguments
      # wouldn't be more readable or more efficient.
      # Using class fields for all these temporary variables would be rather
      # clunky too.
      """Returns the dictionary to be returned by `run_program`.

      See the docstring for `run_program`.

      Args:
        status: The reason why the program has finished.
      """
      # Reduce the program to an equivalent program (apart from smaller size
      # computation steps), by removing unnecessary brackets, self-cancelling
      # pairs of instructions (e.g., '+-'), and instructions that are evaluated
      # for the first time after the last print `.` operations. See below for
      # details.
      # This takes linear time with the length of the original program.
      # The resulting program is never slower than the original one.
      #   For example, one could reduce '+[..]' to '+[.]' which is semantically
      #   equivalent, but twice as slow due to the time required by the
      #   brackets.
      if self._shorten_program:
        short_program = []
        # No need to include {'[': ']} because infinite loops that are evaluated
        # are removed automatically using `first_sampled_idx_after_print`, while
        # infinite loops that are skipped cannot be generated, since the open
        # bracket would be '{' and the body would not be generated, and the
        # hanging '{' would later be removed by the code below.
        cancelling_ids = {'+': '-', '-': '+', '<': '>', '>': '<'}
        for idx, instruction in enumerate(program2):
          if idx == first_sampled_idx_after_print:
            # Remove all instructions that are sampled after the last evaluation
            # of a print `.` instruction. The program remains consistent with
            # the output. This gives a better lower-bound on the prediction
            # probability of Solomonoff Induction.
            # In particular, this removes all infinite loops that use the print
            # operation.
            # Due to the property that newly sampled instructions are always
            # appended (at the end) to the program, it's just a matter of
            # keeping track the index of the first sampled operation after a `.`
            # is evaluated.
            break
          if instruction == '[' and idx in unmatched_brackets:
            # Remove '[' that are in unmatched_brackets, but not those that are
            # not in jumps! (it just means they don't have a continuation).
            continue
          if instruction == '{' and idx not in jumps:
            # Remove '{' that are not in jumps (they have no body), but not
            # those that are in unmatched (it just means the body hasn't been
            # finished when the program stopped, but they are still in jumps).
            continue
          if instruction == ']' and jumps[idx] == idx:
            # Remove ']' that jump to themselves (= skip).
            continue
          if (
              short_program
              and cancelling_ids.get(instruction, '') == short_program[-1]
          ):
            # Remove self-cancelling sequences such as '<>', '+-'.
            # Remove the last instruction, and don't append the new one.
            short_program.pop()
          else:
            short_program.append(instruction)
        short_program = ''.join(short_program)
        # Upper bound on the Solomonoff ln-loss. Note that we use
        # program_tokens and not program_valid_tokens because when sampling
        # we don't need to sample '{' (only '[').
        short_ln_loss = self._sampler.program_ln_loss(short_program)
        long_ln_loss = self._sampler.program_ln_loss(program2)
      else:
        short_program = None
        short_ln_loss = None
        long_ln_loss = None

      return {
          'status': status,
          'alphabet_size': self._alphabet_size,
          'num_steps': num_steps,
          'memory_index': memory_index,
          'output': output,
          'output_length': len(output),
          'program': program2,
          'short_program': short_program,
          # Upper bound on the (natural) log-loss of Solomonoff induction
          # for the generated output, based on `short_program`.
          'short_ln_loss': short_ln_loss,
          'long_ln_loss': long_ln_loss,
      }

    # Program evaluation loop, possibly with program generation at the same
    # time.
    while True:
      mem = memory[memory_index]
      new_instruction = False
      if program_index == len(program2):
        # We need to extend the program by one instruction.
        if not program:
          instruction = self._sampler.get_sample(program2)
        elif program_index == len(program):
          return make_result('HALTED')  # perhaps should be END_OF_PROGRAM ?
        else:
          instruction = program[program_index]
        # Fix the opening brackets.
        if instruction == '[' and mem == 0:
          instruction = '{'
        elif instruction == '{' and mem != 0:
          instruction = '['
        program2 += instruction
        new_instruction = True

      command = program2[program_index]
      if self._print_trace:
        print(
            'program:',
            program2,
            'num_steps:',
            num_steps,
            'program_index:',
            program_index,
            'command:',
            command,
            'memory_index:',
            memory_index,
            'mem:',
            mem,
            'jumps:',
            jumps,
            'unmatched_brackets:',
            unmatched_brackets,
        )

      match command:
        case '+':
          # Increment data cell value with wrap-around
          memory[memory_index] = (mem + 1) % self._alphabet_size
        case '-':
          # Decrement data cell value with wrap-around.
          memory[memory_index] = (mem - 1) % self._alphabet_size
        case '.':
          # Output command: we append to the output string.
          # We use a string for convenience, but it should be view as a
          # bytearray instead.
          output += chr(mem)
          # Reset the pointer to the first instruction that's sampled after
          # the last print `.`.
          first_sampled_idx_after_print = len(program2)
        case '<':
          # Move left on the tape.
          memory_index = (memory_index - 1) % memory_size
        case '>':
          # Move right on the tape.
          memory_index = (memory_index + 1) % memory_size
        case '[':
          if new_instruction:  # mem != 0 necessarily
            # We need to enter the block but the body of the block does not
            # exist yet, so we need to generate it here (at the end of the
            # current program)
            unmatched_brackets.append(program_index)
          elif mem == 0:
            # We need to skip the block
            if program_index not in jumps:
              # The body of the block has been generated, and we have jumped
              # back to the opening bracket of the block.
              # But the continuation of the block has not been generated yet, so
              # we need to do so now, at the end of the program. Now we also
              # know where to jump to, so we fill in the jumps table.
              jumps[program_index] = len(program2) - 1
              program_index = len(program2) - 1
            else:
              # The continuation of the block already exists, so we jump to
              # there.
              program_index = jumps[program_index]
          else:  # mem == 1
            # We enter the block, which already exists, and is next to
            # program_index, so nothing to do.
            pass
        case '{':
          if new_instruction:  # necessarily mem == 0
            # Since mem == 0, this block is skipped for now.
            # Hence we don't need to know what instructions the body of the
            # block will be made of, and for now we only need to construct
            # the sequence of instructions that come after the block (the
            # continuation of the block).
            # So we need to request more new instructions, which are placed
            # right after this bracket (by contrast to '[').
            # We do not register '{' as an unmatched opening bracket yet, and
            # we will do so only once we need to generate the body of this
            # block, that is, the next time we visit this '{' and jump[here]
            # is not set.
            pass
          elif mem != 0:
            # We need to enter the block.
            if program_index in jumps:
              # The block position is known and is at jumps[here].
              program_index = jumps[program_index]
            else:
              # The block position is not yet known, which means the block
              # has not been constructed yet, and we need to do so now, at the
              # end of the program.
              block_pos = len(program2) - 1
              jumps[program_index] = block_pos
              # Now that the block is being generated, we can wait for the
              # matching closing bracket to be produced.
              unmatched_brackets.append(program_index)
              program_index = block_pos
          else:
            # Not a new instruction, mem == 0, we need to move the pointer to
            # the continuation of the block, which is right after the current
            # instruction, so there's actually nothing to do.
            pass
        case ']':
          # Unconditional jump. Wikipedia mentions that we could check the
          # condition here too to avoid one step, but treating it as
          # unconditional makes the logic simpler and avoids more bugs (since
          # we have complicated the logic somewhat).
          if new_instruction:
            if unmatched_brackets:
              # We just generated a closing bracket, and there are some
              # opening bracket hanging, so we match these two.
              open_pos = unmatched_brackets.pop()
              jumps[program_index] = open_pos
              # Jump to there. -1 to ensure the next instruction is the
              # opening bracket.
              program_index = open_pos - 1
            else:
              # This is an unmatched closing bracket, we just skip it.
              jumps[program_index] = program_index
          else:
            # -1 to ensure the next instruction is the opening bracket.
            program_index = jumps[program_index] - 1
        case _:
          raise IncorrectProgramError(
              f'Character {command} is not recognized. All '
              'characters in the input program must be part of the set ('
              f'{",".join(self.program_tokens)}).',
          )
      program_index += 1
      num_steps += 1
      if num_steps == maximum_steps:
        return make_result('TIMEOUT')
      if len(output) >= max_output_length:
        return make_result('OUTPUT_LIMIT')
