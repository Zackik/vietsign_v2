"""Canonical VietSignAI skeleton graph definitions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from vietsign.training.config import InputFeatureConfig
from vietsign.training.input import selected_joint_indices
from vietsign.vision.constants import CANONICAL_JOINT_NAMES


@dataclass(frozen=True)
class GraphEdge:
    source: int
    target: int
    edge_type: str = "anatomical"


@dataclass(frozen=True)
class SkeletonGraph:
    name: str
    version: str
    node_names: tuple[str, ...]
    edges: tuple[GraphEdge, ...]

    @property
    def node_count(self) -> int:
        return len(self.node_names)

    def adjacency(self, *, include_self: bool = True) -> np.ndarray:
        matrix = np.zeros((self.node_count, self.node_count), dtype=np.float32)
        for edge in self.edges:
            matrix[edge.source, edge.target] = 1.0
            matrix[edge.target, edge.source] = 1.0
        if include_self:
            np.fill_diagonal(matrix, 1.0)
        degree = matrix.sum(axis=1, keepdims=True).clip(min=1.0)
        return matrix / degree

    def induced_subgraph(self, selected_indices: np.ndarray) -> SkeletonGraph:
        selected = [int(index) for index in selected_indices.tolist()]
        remap = {old: new for new, old in enumerate(selected)}
        edges = tuple(
            GraphEdge(remap[edge.source], remap[edge.target], edge.edge_type)
            for edge in self.edges
            if edge.source in remap and edge.target in remap
        )
        return SkeletonGraph(
            name=f"{self.name}_subgraph",
            version=self.version,
            node_names=tuple(self.node_names[index] for index in selected),
            edges=edges,
        )


def vietsign55_graph() -> SkeletonGraph:
    edges: list[GraphEdge] = []
    edges.extend(_hand_edges(offset=0))
    edges.extend(_hand_edges(offset=21))
    edges.extend(
        GraphEdge(source, target)
        for source, target in (
            (0, 52),
            (21, 53),
            (52, 50),
            (50, 47),
            (53, 51),
            (51, 48),
            (47, 48),
            (47, 54),
            (48, 54),
            (47, 45),
            (48, 46),
            (45, 42),
            (46, 42),
            (45, 43),
            (46, 44),
        )
    )
    return SkeletonGraph(
        name="vietsign55",
        version="vietsign55.v1",
        node_names=CANONICAL_JOINT_NAMES,
        edges=tuple(edges),
    )


def graph_for_features(features: InputFeatureConfig) -> SkeletonGraph:
    return vietsign55_graph().induced_subgraph(selected_joint_indices(features))


def visualize_graph(graph: SkeletonGraph, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((760, 1100, 3), 255, dtype=np.uint8)
    positions = _node_positions(graph.node_names)
    for edge in graph.edges:
        start = positions[edge.source]
        end = positions[edge.target]
        cv2.line(canvas, start, end, (110, 110, 110), 1, cv2.LINE_AA)
    for index, name in enumerate(graph.node_names):
        color = _node_color(name)
        cv2.circle(canvas, positions[index], 5, color, -1, cv2.LINE_AA)
        if index in {0, 20, 21, 41, 42, 47, 48, 52, 53}:
            cv2.putText(
                canvas,
                str(index),
                (positions[index][0] + 7, positions[index][1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        f"{graph.name} nodes={graph.node_count} edges={len(graph.edges)}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)
    return path


def _hand_edges(offset: int) -> list[GraphEdge]:
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    return [
        GraphEdge(offset + source, offset + target)
        for chain in chains
        for source, target in pairwise(chain)
    ]


def _node_positions(node_names: tuple[str, ...]) -> dict[int, tuple[int, int]]:
    positions = {}
    for index, name in enumerate(node_names):
        if name.startswith("left_hand"):
            positions[index] = _hand_position(name, origin=(260, 380), scale=34)
        elif name.startswith("right_hand"):
            positions[index] = _hand_position(name, origin=(840, 380), scale=34)
        else:
            positions[index] = _pose_position(name)
    return positions


def _hand_position(name: str, *, origin: tuple[int, int], scale: int) -> tuple[int, int]:
    joint = name.split(":", 1)[1]
    lookup = {
        "wrist": (0, 0),
        "thumb_cmc": (-1, -1),
        "thumb_mcp": (-2, -2),
        "thumb_ip": (-3, -3),
        "thumb_tip": (-4, -4),
        "index_mcp": (-1, -2),
        "index_pip": (-1, -3),
        "index_dip": (-1, -4),
        "index_tip": (-1, -5),
        "middle_mcp": (0, -2),
        "middle_pip": (0, -3),
        "middle_dip": (0, -4),
        "middle_tip": (0, -5),
        "ring_mcp": (1, -2),
        "ring_pip": (1, -3),
        "ring_dip": (1, -4),
        "ring_tip": (1, -5),
        "pinky_mcp": (2, -2),
        "pinky_pip": (2, -3),
        "pinky_dip": (2, -4),
        "pinky_tip": (2, -5),
    }
    x, y = lookup[joint]
    return origin[0] + x * scale, origin[1] + y * scale


def _pose_position(name: str) -> tuple[int, int]:
    joint = name.split(":", 1)[1]
    lookup = {
        "nose": (550, 130),
        "left_eye": (520, 110),
        "right_eye": (580, 110),
        "left_ear": (490, 130),
        "right_ear": (610, 130),
        "left_shoulder": (430, 260),
        "right_shoulder": (670, 260),
        "left_elbow": (360, 330),
        "right_elbow": (740, 330),
        "left_wrist": (300, 410),
        "right_wrist": (800, 410),
        "left_hip": (470, 520),
        "right_hip": (630, 520),
    }
    return lookup[joint]


def _node_color(name: str) -> tuple[int, int, int]:
    if name.startswith("left_hand"):
        return (60, 120, 230)
    if name.startswith("right_hand"):
        return (60, 180, 90)
    return (210, 80, 80)
