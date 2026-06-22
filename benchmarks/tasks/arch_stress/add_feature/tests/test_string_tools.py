import sys

sys.path.insert(0, ".")

from string_tools import is_palindrome, reverse


def test_reverse():
    assert reverse("abc") == "cba"
    assert reverse("") == ""


def test_is_palindrome():
    assert is_palindrome("aba") is True
    assert is_palindrome("abc") is False
    assert is_palindrome("") is True
    assert is_palindrome("a") is True


if __name__ == "__main__":
    test_reverse()
    test_is_palindrome()
    print("[PASS] string_tools tests")
