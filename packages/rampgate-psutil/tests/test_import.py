import rampgate_psutil


def test_version() -> None:
    assert rampgate_psutil.__version__ == "0.1.0.dev0"
