from pathlib import Path

from vietsign.config import ProjectConfig, load_config


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert isinstance(config, ProjectConfig)
    assert config.seed == 42
    assert config.data.root_dir == Path("data")
    assert config.preprocessing.image_size == (224, 224)
    assert config.features.pose_backend == "mediapipe"
    assert config.training.learning_rate > 0
    assert config.device.preference == "auto"

