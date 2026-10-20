import rampgate_nvml


def test_version() -> None:
    assert rampgate_nvml.__version__ == "0.1.0.dev0"
