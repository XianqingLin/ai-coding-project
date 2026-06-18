import sys

sys.path.insert(0, ".")


def test_output():
    with open("output.txt", "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello, World!\n", f"unexpected content: {content!r}"


if __name__ == "__main__":
    test_output()
    print("[PASS] output tests")
