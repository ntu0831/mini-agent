"""Calculator safety and correctness tests."""

from __future__ import annotations

from mini_agent import calculator


def test_basic_arithmetic() -> None:
    assert calculator("(25 + 15) * 3 - 8") == "112"
    assert calculator("15 × 8 + 32 ÷ 4") == "128"
    assert calculator("2^10") == "1024"
    assert calculator("7 // 2") == "3"
    assert calculator("7 % 2") == "1"


def test_functions_and_constants() -> None:
    assert calculator("sqrt(16) + 2 ** 3") == "12"
    assert calculator("abs(-3) * round(2.6)") == "9"
    assert calculator("floor(3.9) + ceil(1.1)") == "5"
    assert calculator("log(e)") == "1"


def test_unicode_operators() -> None:
    assert calculator("3 × 4 - 2 ÷ 2") == "11"


def test_rejects_code_execution() -> None:
    assert "Error" in calculator("__import__('os').system('ls')")
    assert "Error" in calculator("open('/etc/passwd').read()")
    assert "Error" in calculator("[1, 2, 3]")
    assert "Error" in calculator("(lambda: 1)()")


def test_rejects_dangerous_or_invalid_input() -> None:
    assert "Error" in calculator("")
    assert "Error" in calculator("1/0")
    assert "Error" in calculator("2 ** 100000")
    assert "Error" in calculator("x" * 600)
    assert "Error" in calculator("1 +")


def test_unknown_name_is_rejected() -> None:
    assert "Error" in calculator("os")
