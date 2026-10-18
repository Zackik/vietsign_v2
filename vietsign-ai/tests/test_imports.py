def test_basic_imports() -> None:
    import vietsign
    from vietsign import cli
    from vietsign.utils import device, logging, reproducibility

    assert vietsign.__version__
    assert cli.build_parser()
    assert device.select_device("cpu").type == "cpu"
    assert logging.get_logger("tests")
    assert reproducibility.set_seed

