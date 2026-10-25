"""Prompt-grouped train/validation/test splits and frozen held-out reporting."""
import random
from dataclasses import dataclass

from .executor import Task
from .fitness import Evaluator
from .genome import Genome


@dataclass
class TaskSplit:
    train: list[Task]
    validation: list[Task]
    test: list[Task]


def split_tasks(tasks: list[Task], *, validation_fraction: float = 0.2,
                test_fraction: float = 0.2, seed: int = 0,
                chronological: bool = False) -> TaskSplit:
    """Keep duplicate prompts in one partition; reject conflicting references.

    For temporal tasks supply chronological=True and already time-sorted tasks.
    Splitting is by unique prompt; proportions may differ when groups repeat.
    """
    if not (0 < validation_fraction < 1 and 0 < test_fraction < 1 and
            validation_fraction + test_fraction < 1):
        raise ValueError("fractions must be positive and sum to less than one")
    groups: dict[str, list[Task]] = {}
    for task in tasks:
        if task.prompt in groups and groups[task.prompt][0].reference != task.reference:
            raise ValueError("duplicate prompt has conflicting references")
        groups.setdefault(task.prompt, []).append(task)
    keys = list(groups)
    if chronological and len(keys) != len(tasks):
        raise ValueError("chronological splits require unique prompts to preserve ordering")
    if not chronological:
        random.Random(seed).shuffle(keys)
    nval, ntest = max(1, int(len(keys) * validation_fraction)), max(1, int(len(keys) * test_fraction))
    ntrain = len(keys) - nval - ntest
    if ntrain < 1:
        raise ValueError("at least three unique prompts are required")
    def expand(part):
        return [task for key in part for task in groups[key]]
    return TaskSplit(expand(keys[:ntrain]), expand(keys[ntrain:ntrain+nval]), expand(keys[ntrain+nval:]))


@dataclass
class HeldOutScore:
    genome: Genome
    accuracy: float
    mean_cost: float
    sample_accuracies: list[float]


def evaluate_frozen(genomes: list[Genome], evaluator: Evaluator, *,
                    search_tasks: list[Task], samples: int = 1) -> list[HeldOutScore]:
    """Report candidates without selecting/re-sorting them on test performance.

    search_tasks must include every training/validation task previously consulted.
    Sample IDs separate cache entries; hosted APIs need not provide seeded RNGs.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    if {t.prompt for t in search_tasks} & {t.prompt for t in evaluator.tasks}:
        raise ValueError("held-out tasks overlap search/validation prompts")
    results = []
    for genome in genomes:
        scores = [evaluator.score(genome, sample=s) for s in range(samples)]
        results.append(HeldOutScore(genome.copy(), sum(a for a, _ in scores) / samples,
                                    sum(c for _, c in scores) / samples, [a for a, _ in scores]))
    return results
