"""Smoke test: validates that the test toolchain works in CI."""


def test_toolchain_is_alive() -> None:
    assert 1 + 1 == 2
