"""Defines the Adalead explorer class."""
import random
import numpy as np
import pandas as pd
#import flexs
import abc
import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Optional, Tuple
from requests import get
import tqdm
###
from helpers.test_inf_BIND import get_reward
import sys
from typing import Any, List, Union
SEQUENCES_TYPE = Union[List[str], np.ndarray]
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.from_config import get_wildtype_smile
from analysis.iterative_pruning import get_fold_increase, print_status



######


"""Utility functions for manipulating sequences."""

AAS = "ILVAGMFYWEDQNHCRKSTP"
"""str: Amino acid alphabet for proteins (length 20 - no stop codon)."""

RNAA = "UGCA"
"""str: RNA alphabet (4 base pairs)."""

DNAA = "TGCA"
"""str: DNA alphabet (4 base pairs)."""

BA = "01"
"""str: Binary alphabet '01'."""


def construct_mutant_from_sample(
    pwm_sample: np.ndarray, one_hot_base: np.ndarray
) -> np.ndarray:
    """Return one hot mutant, a utility function for some explorers."""
    one_hot = np.zeros(one_hot_base.shape)
    one_hot += one_hot_base
    i, j = np.nonzero(pwm_sample)  # this can be problematic for non-positive fitnesses
    one_hot[i, :] = 0
    one_hot[i, j] = 1
    return one_hot


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    """
    Return the one-hot representation of a sequence string according to an alphabet.

    Args:
        sequence: Sequence string to convert to one_hot representation.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        One-hot numpy array of shape `(len(sequence), len(alphabet))`.

    """
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def generate_single_mutants(wt: str, alphabet: str) -> List[str]:
    """Generate all single mutants of `wt`."""
    sequences = [wt]
    for i in range(len(wt)):
        tmp = list(wt)
        for j in range(len(alphabet)):
            tmp[i] = alphabet[j]
            sequences.append("".join(tmp))
    return sequences


def generate_random_sequences(length: int, number: int, alphabet: str) -> List[str]:
    """Generate random sequences of particular length."""
    return [
        "".join([random.choice(alphabet) for _ in range(length)]) for _ in range(number)
    ]


def generate_random_mutant(sequence: str, mu: float, alphabet: str) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.

    So the expected value of the total number of mutations is `len(sequence) * mu`.

    Args:
        sequence: Sequence that will be mutated from.
        mu: Probability of mutation per residue.
        alphabet: Alphabet string.

    Returns:
        Mutant sequence string.

    """
    mutant = []
    for s in sequence:
        if random.random() < mu:
            mutant.append(random.choice(alphabet))
        else:
            mutant.append(s)
    return "".join(mutant)

######
class Landscape(abc.ABC):
    """
    Base class for all landscapes and for `flexs.Model`.

    Attributes:
        cost (int): Number of sequences whose fitness has been evaluated.
        name (str): A human-readable name for the landscape (often contains
            parameter values in the name) which is used when logging explorer runs.

    """

    def __init__(self, name: str):
        """Create Landscape, setting `name` and setting `cost` to zero."""
        self.cost = 0
        self.name = name

    @abc.abstractmethod
    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        pass

    def get_fitness(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        """
        Score a list/numpy array of sequences.

        This public method should not be overriden – new landscapes should
        override the private `_fitness_function` method instead. This method
        increments `self.cost` and then calls and returns `_fitness_function`.

        Args:
            sequences: A list/numpy array of sequence strings to be scored.

        Returns:
            Scores for each sequence.

        """
        self.cost += len(sequences)
        return self._fitness_function(sequences)

class LandscapeMy(Landscape):
    def __init__(self, name: str):
        super().__init__(name)

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        return np.array( [get_reward(seq,smile) for seq in sequences] )

#####

class Model(Landscape, abc.ABC):
    """
    Base model class. Inherits from `flexs.Landscape` and adds an additional
    `train` method.

    """

    @abc.abstractmethod
    def train(self, sequences: SEQUENCES_TYPE, labels: List[Any]):
        """
        Train model.

        This function is called whenever you would want your model to update itself
        based on the set of sequences it has measurements for.

        """
        pass

class LandscapeAsModel(Model):
    """
    This simple class wraps a `flexs.Landscape` in a `flexs.Model` to allow running
    experiments against a perfect model.

    This class's `_fitness_function` simply calls the landscape's `_fitness_function`.
    """

    def __init__(self, landscape: Landscape):
        """
        Create a `flexs.Model` out of a `flexs.Landscape`.

        Args:
            landscape: Landscape to wrap in a model.

        """
        super().__init__(f"LandscapeAsModel={landscape.name}")
        self.landscape = landscape

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        return self.landscape._fitness_function(sequences)

    def train(self, sequences: SEQUENCES_TYPE, labels: List[Any]):
        """No-op."""
        pass
######
class Explorer(abc.ABC):
    """
    Abstract base explorer class.

    Run explorer through the `run` method. Implement subclasses
    by overriding `propose_sequences` (do not override `run`).
    """

    def __init__(
        self,
        model: Model,
        name: str,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        log_file: Optional[str] = None,
    ):
        """
        Create an Explorer.

        Args:
            model: Model of ground truth that the explorer will use to help guide
                sequence proposal.
            name: A human-readable name for the explorer (may include parameter values).
            rounds: Number of rounds to run for (a round consists of sequence proposal,
                ground truth fitness measurement of proposed sequences, and retraining
                the model).
            sequences_batch_size: Number of sequences to propose for measurement from
                ground truth per round.
            model_queries_per_batch: Number of allowed "in-silico" model evaluations
                per round.
            starting_sequence: Sequence from which to start exploration.
            log_file: .csv filepath to write output.

        """
        self.model = model
        self.name = name

        self.rounds = rounds
        self.sequences_batch_size = sequences_batch_size
        self.model_queries_per_batch = model_queries_per_batch
        self.starting_sequence = starting_sequence

        self.wildtype_reward = get_reward(starting_sequence,smile) # cla added
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outfile = f"adaLead_{complexName}_{timestamp}.csv" # cla added

        self.log_file = log_file
        if self.log_file is not None:
            dir_path, filename = os.path.split(self.log_file)
            os.makedirs(dir_path, exist_ok=True)

        if model_queries_per_batch < sequences_batch_size:
            warnings.warn(
                "`model_queries_per_batch` should be >= `sequences_batch_size`"
            )

    @abc.abstractmethod
    def propose_sequences(
        self, measured_sequences_data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propose a list of sequences to be measured in the next round.

        This method will be overriden to contain the explorer logic for each explorer.

        Args:
            measured_sequences_data: A pandas dataframe of all sequences that have been
            measured by the ground truth so far. Has columns "sequence",
            "true_score", "model_score", and "round".

        Returns:
            A tuple containing the proposed sequences and their scores
                (according to the model).

        """
        pass

    def _log(
        self,
        sequences_data: pd.DataFrame,
        metadata: Dict,
        current_round: int,
        verbose: bool,
        round_start_time: float,
    ) -> None:
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                # First write metadata
                json.dump(metadata, f)
                f.write("\n")

                # Then write pandas dataframe
                sequences_data.to_csv(f, index=False)

        if verbose:
            print(
                f"round: {current_round}, top: {sequences_data['true_score'].max()}, "
                f"time: {time.time() - round_start_time:02f}s"
            )
            #print(sequences_data)
            # print(sequences_data['true_score'].astype(float).idxmax()) # return always 0 !
            idx = np.argmax(sequences_data['true_score'].astype(float).to_list())
            sequence = sequences_data['sequence'].iloc[idx]  
            print_status(sequence,current_round, self.starting_sequence, smile, self.wildtype_reward, total_importance=None, outfile=self.outfile) # cla added


    def run(
        self, landscape: Landscape, verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the exporer.

        Args:
            landscape: Ground truth fitness landscape.
            verbose: Whether to print output or not.

        """
        self.model.cost = 0

        # Metadata about run that will be used for logging purposes
        metadata = {
            "run_id": datetime.now().strftime("%H:%M:%S-%m/%d/%Y"),
            "exp_name": self.name,
            "model_name": self.model.name,
            "landscape_name": landscape.name,
            "rounds": self.rounds,
            "sequences_batch_size": self.sequences_batch_size,
            "model_queries_per_batch": self.model_queries_per_batch,
        }

        # Initial sequences and their scores
        sequences_data = pd.DataFrame(
            {
                "sequence": self.starting_sequence,
                "model_score": np.nan,
                "true_score": landscape.get_fitness([self.starting_sequence]),
                "round": 0,
                "model_cost": self.model.cost,
                "measurement_cost": 1,
            }
        )
        self._log(sequences_data, metadata, 0, verbose, time.time())

        # For each round, train model on available data, propose sequences,
        # measure them on the true landscape, add to available data, and repeat.
        range_iterator = range if verbose else tqdm.trange
        for r in range_iterator(1, self.rounds + 1):
            round_start_time = time.time()
            self.model.train(
                sequences_data["sequence"].to_numpy(),
                sequences_data["true_score"].to_numpy(),
            )

            seqs, preds = self.propose_sequences(sequences_data)
            true_score = landscape.get_fitness(seqs)

            if len(seqs) > self.sequences_batch_size:
                warnings.warn(
                    "Must propose <= `self.sequences_batch_size` sequences per round"
                )

            sequences_data = sequences_data.append(
                pd.DataFrame(
                    {
                        "sequence": seqs,
                        "model_score": preds,
                        "true_score": true_score,
                        "round": r,
                        "model_cost": self.model.cost,
                        "measurement_cost": len(sequences_data) + len(seqs),
                    }
                )
            )
            self._log(sequences_data, metadata, r, verbose, round_start_time)

        return sequences_data, metadata
    
######
class Adalead(Explorer):
    """
    Adalead explorer.

    Algorithm works as follows:
        Initialize set of top sequences whose fitnesses are at least
            (1 - threshold) of the maximum fitness so far
        While we can still make model queries in this batch
            Recombine top sequences and append to parents
            Rollout from parents and append to mutants.

    """

    def __init__(
        self,
        model: Model,
        rounds: int,
        sequences_batch_size: int,
        model_queries_per_batch: int,
        starting_sequence: str,
        alphabet: str,
        mu: int = 1,
        recomb_rate: float = 0,
        threshold: float = 0.05,
        rho: int = 0,
        eval_batch_size: int = 20,
        log_file: Optional[str] = None,
    ):
        """
        Args:
            mu: Expected number of mutations to the full sequence (mu/L per position).
            recomb_rate: The probability of a crossover at any position in a sequence.
            threshold: At each round only sequences with fitness above
                (1-threshold)*f_max are retained as parents for generating next set of
                sequences.
            rho: The expected number of recombination partners for each recombinant.
            eval_batch_size: For code optimization; size of batches sent to model.

        """
        name = f"Adalead_mu={mu}_threshold={threshold}"

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )
        self.threshold = threshold
        self.recomb_rate = recomb_rate
        self.alphabet = alphabet
        self.mu = mu  # number of mutations per *sequence*.
        self.rho = rho
        self.eval_batch_size = eval_batch_size

    def _recombine_population(self, gen):
        # If only one member of population, can't do any recombining
        if len(gen) == 1:
            return gen

        random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if random.random() < self.recomb_rate:
                    switch = not switch

                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])

            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `sequences_batch_size` sequences for evaluation."""
        measured_sequence_set = set(measured_sequences["sequence"])

        # Get all sequences within `self.threshold` percentile of the top_fitness
        top_fitness = measured_sequences["true_score"].max()
        top_inds = measured_sequences["true_score"] >= top_fitness * (
            1 - np.sign(top_fitness) * self.threshold
        )

        parents = np.resize(
            measured_sequences["sequence"][top_inds].to_numpy(),
            self.sequences_batch_size,
        )

        sequences = {}
        previous_model_cost = self.model.cost
        while self.model.cost - previous_model_cost < self.model_queries_per_batch:
            # generate recombinant mutants
            for i in range(self.rho):
                parents = self._recombine_population(parents)

            for i in range(0, len(parents), self.eval_batch_size):
                # Here we do rollouts from each parent (root of rollout tree)
                roots = parents[i : i + self.eval_batch_size]
                root_fitnesses = self.model.get_fitness(roots)

                nodes = list(enumerate(roots))

                while (
                    len(nodes) > 0
                    and self.model.cost - previous_model_cost + self.eval_batch_size
                    < self.model_queries_per_batch
                ):
                    child_idxs = []
                    children = []
                    while len(children) < len(nodes):
                        idx, node = nodes[len(children) - 1]

                        child = generate_random_mutant(
                            node,
                            self.mu * 1 / len(node),
                            self.alphabet,
                        )

                        # Stop when we generate new child that has never been seen
                        # before
                        if (
                            child not in measured_sequence_set
                            and child not in sequences
                        ):
                            child_idxs.append(idx)
                            children.append(child)

                    # Stop the rollout once the child has worse predicted
                    # fitness than the root of the rollout tree.
                    # Otherwise, set node = child and add child to the list
                    # of sequences to propose.
                    fitnesses = self.model.get_fitness(children)
                    sequences.update(zip(children, fitnesses))

                    nodes = []
                    for idx, child, fitness in zip(child_idxs, children, fitnesses):
                        if fitness >= root_fitnesses[idx]:
                            nodes.append((idx, child))

        if len(sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_batch` is small, try "
                "making `eval_batch_size` smaller"
            )

        # We propose the top `self.sequences_batch_size` new sequences we have generated
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.sequences_batch_size : -1]

        return new_seqs[sorted_order], preds[sorted_order]

    
if __name__ == "__main__":
    #complexName = "3ebp"
    #complexName = "3PRS"
    #complexName = "2BRB"
    complexName = "1bcu"
    #complexName = "3dxg"
    #complexName = "1p1q"
    #complexName = "1o0H"
    #
    wildtype, smile = get_wildtype_smile(config_file=None, complexName=complexName) 

    # Define a model
    landscape = LandscapeMy(name="LandscapeMy")
    model = LandscapeAsModel(landscape)

    # Define an explorer
    explorer = Adalead(
        model=model,
        rounds=60,
        sequences_batch_size=10,  # Number of sequences to propose per round
        model_queries_per_batch=100,
        starting_sequence=wildtype,
        alphabet=AAS,
        mu=1,
        recomb_rate=0.5,
        threshold=0.05,
        rho=1,
        eval_batch_size=1,
    )

    # Run the explorer
    sequences_data, metadata = explorer.run(landscape, verbose=True)
    print(sequences_data.head())
    print(metadata)