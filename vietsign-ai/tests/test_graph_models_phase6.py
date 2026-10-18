from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vietsign.models.graph import graph_for_features, vietsign55_graph, visualize_graph
from vietsign.models.graph_models import CTRGCNClassifier, STGCNClassifier
from vietsign.models.registry import InputSpec, build_model
from vietsign.training.benchmark import run_synthetic_smoke_benchmark
from vietsign.training.checkpoint import CheckpointCompatibilityError, validate_model_compatibility
from vietsign.training.config import InputFeatureConfig, RecognitionTrainingConfig


def _batch(t: int = 7, j: int = 55, c: int = 4):
    coordinates = torch.randn((2, t, j, c))
    valid_mask = torch.ones((2, t, j), dtype=torch.bool)
    temporal_mask = torch.ones((2, t), dtype=torch.bool)
    temporal_mask[0, -2:] = False
    return coordinates, valid_mask, temporal_mask


def test_vietsign55_graph_node_contract_and_edges() -> None:
    graph = vietsign55_graph()
    edge_pairs = {(edge.source, edge.target) for edge in graph.edges}

    assert graph.node_count == 55
    assert graph.node_names[0] == "left_hand:wrist"
    assert graph.node_names[20] == "left_hand:pinky_tip"
    assert graph.node_names[21] == "right_hand:wrist"
    assert graph.node_names[41] == "right_hand:pinky_tip"
    assert graph.node_names[42] == "pose:nose"
    assert (0, 1) in edge_pairs
    assert (17, 18) in edge_pairs
    assert (21, 22) in edge_pairs
    assert (38, 39) in edge_pairs
    assert (0, 52) in edge_pairs
    assert (21, 53) in edge_pairs
    assert (52, 50) in edge_pairs
    assert (50, 47) in edge_pairs
    assert (47, 48) in edge_pairs


def test_subgraph_selection_hands_only_and_pose_only() -> None:
    hands = graph_for_features(InputFeatureConfig(pose=False))
    pose = graph_for_features(InputFeatureConfig(left_hand=False, right_hand=False, pose=True))

    assert hands.node_count == 42
    assert all(name.startswith(("left_hand", "right_hand")) for name in hands.node_names)
    assert pose.node_count == 13
    assert pose.node_names[0] == "pose:nose"
    assert all(name.startswith("pose") for name in pose.node_names)


@pytest.mark.parametrize("model_cls", [STGCNClassifier, CTRGCNClassifier])
def test_graph_model_forward_variable_t_masks_and_velocity_channels(model_cls) -> None:
    graph = vietsign55_graph()
    model = model_cls(graph=graph, input_channels=7, num_classes=3, channels=(8, 8), dropout=0.0)
    coordinates, valid_mask, temporal_mask = _batch(t=9, c=7)

    logits = model(coordinates, valid_mask, temporal_mask)

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()


def test_graph_model_rejects_all_missing_sequence() -> None:
    graph = vietsign55_graph()
    model = STGCNClassifier(graph=graph, input_channels=4, num_classes=2, channels=(8,), dropout=0.0)
    coordinates = torch.zeros((1, 4, 55, 4))
    valid_mask = torch.zeros((1, 4, 55), dtype=torch.bool)
    temporal_mask = torch.ones((1, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="all-missing"):
        model(coordinates, valid_mask, temporal_mask)


def test_model_registry_builds_all_models() -> None:
    for model_type in ("temporal_transformer", "stgcn", "ctrgcn"):
        config = RecognitionTrainingConfig.model_validate(
            {
                "dataset": {"config": "configs/data/vsl400.yaml"},
                "model": {"type": model_type, "channels": [8], "embedding_dim": 8, "num_layers": 1, "num_heads": 2},
            }
        )
        model = build_model(config, num_classes=2, input_spec=InputSpec(joint_count=55, channel_count=4))
        assert model.introspect().__dict__["class_count"] == 2


def test_checkpoint_compatibility_rejects_cross_model() -> None:
    graph = vietsign55_graph()
    stgcn = STGCNClassifier(graph=graph, input_channels=4, num_classes=2, channels=(8,), dropout=0.0)
    ctrgcn = CTRGCNClassifier(graph=graph, input_channels=4, num_classes=2, channels=(8,), dropout=0.0)
    checkpoint = {"model_metadata": stgcn.introspect().__dict__}

    with pytest.raises(CheckpointCompatibilityError):
        validate_model_compatibility(checkpoint, ctrgcn.introspect().__dict__)


def test_graph_visualization_generation(tmp_path: Path) -> None:
    path = visualize_graph(vietsign55_graph(), tmp_path / "vietsign55.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_synthetic_benchmark_report_generation(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        """
seed: 42
run:
  name: smoke
  output_dir: outputs/runs
dataset:
  config: configs/data/vsl400.yaml
  cache_root: data/cache/keypoints
model:
  type: stgcn
  channels: [8]
  dropout: 0.0
training:
  device: cpu
  epochs: 1
  batch_size: 2
early_stopping:
  enabled: false
""",
        encoding="utf-8",
    )

    report = run_synthetic_smoke_benchmark([baseline], output_dir=tmp_path / "benchmark")

    assert "NON-VSL" in report["label"]
    assert Path(report["comparison_csv"]).exists()
    assert Path(report["comparison_json"]).exists()
    assert report["results"][0]["model"] == "stgcn"
