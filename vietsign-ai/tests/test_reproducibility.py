import os
import random

from vietsign.utils.reproducibility import set_seed


def test_set_seed_sets_python_random_and_hash_seed() -> None:
    set_seed(42)
    first = random.random()

    set_seed(42)
    second = random.random()

    assert os.environ["PYTHONHASHSEED"] == "42"
    assert first == second

