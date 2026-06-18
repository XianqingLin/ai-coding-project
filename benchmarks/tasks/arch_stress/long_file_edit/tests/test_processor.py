import sys

sys.path.insert(0, ".")

from data_processor import process_data


def test_process_data():
    assert process_data([1, 2, 3]) == [2, 4, 6]
    assert process_data([]) == []
    assert process_data([0, 5]) == [0, 10]


def test_helpers_unchanged():
    from data_processor import helper_01, helper_20

    assert helper_01(0) == 1
    assert helper_20(0) == 20


if __name__ == "__main__":
    test_process_data()
    test_helpers_unchanged()
    print("[PASS] processor tests")
