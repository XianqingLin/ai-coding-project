import sys

sys.path.insert(0, ".")

from main import run
from utils import new_name


def test_run():
    assert run() == 10


def test_new_name():
    assert new_name(3) == 6


if __name__ == "__main__":
    test_run()
    test_new_name()
    print("[PASS] rename tests")
