import rampgate_docker


def test_version() -> None:
    assert rampgate_docker.__version__ == "0.1.0"
