import sys

sys.path.insert(0, ".")

from calculator import add


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


if __name__ == "__main__":
    test_add()
    print("[PASS] calculator tests")
