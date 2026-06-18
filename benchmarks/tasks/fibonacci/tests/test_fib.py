"""斐波那契函数测试."""

import sys

sys.path.insert(0, ".")

import py_compile

try:
    py_compile.compile("fib.py", doraise=True)
except Exception as e:
    print(f"[FAIL] 语法错误: {e}")
    sys.exit(1)

from fib import fib

assert fib(0) == 0
assert fib(1) == 1
assert fib(2) == 1
assert fib(10) == 55
assert fib(20) == 6765

print("[PASS] fibonacci 测试全部通过")
