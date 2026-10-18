"""Graph convolutional recognition models for canonical keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from vietsign.models.graph import SkeletonGraph


@dataclass(frozen=True)
class GraphModelIntrospection:
    model_name: str
    input_joints: int
    input_channels: int
    class_count: int
    channels: tuple[int, ...]
    dropout: float
    graph_name: str
    graph_version: str
    graph_nodes: int
    graph_edges: int
    parameter_count: int


class STGCNClassifier(nn.Module):
    """Spatial-temporal graph convolutional classifier over canonical joints."""

    def __init__(
        self,
        *,
        graph: SkeletonGraph,
        input_channels: int,
        num_classes: int,
        channels: tuple[int, ...] = (64, 64, 128, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("ST-GCN requires at least one channel block.")
        self.graph = graph
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.dropout = dropout
        adjacency = torch.from_numpy(graph.adjacency(include_self=True))
        self.register_buffer("adjacency", adjacency, persistent=False)
        blocks = []
        in_channels = input_channels
        for index, out_channels in enumerate(channels):
            stride = 2 if index > 0 and channels[index - 1] != out_channels else 1
            blocks.append(STGCNBlock(in_channels, out_channels, dropout=dropout, stride=stride))
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(
        self,
        coordinates: torch.Tensor,
        valid_mask: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits [B,num_classes] from input [B,T,J,C]."""

        x, temporal_mask = _prepare_graph_input(coordinates, valid_mask, temporal_mask)
        adjacency = self.adjacency.to(device=x.device, dtype=x.dtype)
        for block in self.blocks:
            previous_t = x.shape[2]
            x = block(x, adjacency)
            temporal_mask = _downsample_temporal_mask(temporal_mask, previous_t, x.shape[2])
        pooled = _masked_global_pool(x, temporal_mask)
        return self.classifier(pooled)

    def introspect(self) -> GraphModelIntrospection:
        return GraphModelIntrospection(
            model_name="stgcn",
            input_joints=self.graph.node_count,
            input_channels=self.input_channels,
            class_count=self.num_classes,
            channels=self.channels,
            dropout=self.dropout,
            graph_name=self.graph.name,
            graph_version=self.graph.version,
            graph_nodes=self.graph.node_count,
            graph_edges=len(self.graph.edges),
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
        )


class CTRGCNClassifier(STGCNClassifier):
    """CTR-GCN style classifier with channel-wise topology refinement."""

    def __init__(
        self,
        *,
        graph: SkeletonGraph,
        input_channels: int,
        num_classes: int,
        channels: tuple[int, ...] = (64, 64, 128, 256),
        dropout: float = 0.2,
        relation_reduction: int = 8,
    ) -> None:
        nn.Module.__init__(self)
        if not channels:
            raise ValueError("CTR-GCN requires at least one channel block.")
        self.graph = graph
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.dropout = dropout
        self.relation_reduction = relation_reduction
        adjacency = torch.from_numpy(graph.adjacency(include_self=True))
        self.register_buffer("adjacency", adjacency, persistent=False)
        blocks = []
        in_channels = input_channels
        for index, out_channels in enumerate(channels):
            stride = 2 if index > 0 and channels[index - 1] != out_channels else 1
            blocks.append(
                CTRGCNBlock(
                    in_channels,
                    out_channels,
                    dropout=dropout,
                    stride=stride,
                    relation_reduction=relation_reduction,
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def introspect(self) -> GraphModelIntrospection:
        return GraphModelIntrospection(
            model_name="ctrgcn",
            input_joints=self.graph.node_count,
            input_channels=self.input_channels,
            class_count=self.num_classes,
            channels=self.channels,
            dropout=self.dropout,
            graph_name=self.graph.name,
            graph_version=self.graph.version,
            graph_nodes=self.graph.node_count,
            graph_edges=len(self.graph.edges),
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
        )


class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float, stride: int = 1) -> None:
        super().__init__()
        self.graph_conv = SpatialGraphConvolution(in_channels, out_channels)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                stride=(stride, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = _residual(in_channels, out_channels, stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.activation(self.temporal(self.graph_conv(x, adjacency)) + self.residual(x))


class CTRGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float,
        stride: int = 1,
        relation_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.graph_conv = ChannelTopologyRefinement(in_channels, out_channels, relation_reduction)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                stride=(stride, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        self.residual = _residual(in_channels, out_channels, stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.activation(self.temporal(self.graph_conv(x, adjacency)) + self.residual(x))


class SpatialGraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("nctv,vw->nctw", x, adjacency)
        return self.projection(x)


class ChannelTopologyRefinement(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relation_reduction: int) -> None:
        super().__init__()
        hidden = max(out_channels // relation_reduction, 4)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.query = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.key = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.relation = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        base = torch.einsum("nctv,vw->nctw", value, adjacency)
        query = self.query(x).mean(dim=2)
        key = self.key(x).mean(dim=2)
        relation = torch.tanh(query.unsqueeze(-1) - key.unsqueeze(-2))
        refined = self.relation(relation).mean(dim=1)
        adaptive = torch.softmax(refined, dim=-1)
        adaptive_value = torch.einsum("nctv,nvw->nctw", value, adaptive)
        return base + self.alpha * adaptive_value


def _prepare_graph_input(
    coordinates: torch.Tensor,
    valid_mask: torch.Tensor,
    temporal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.nan_to_num(coordinates.float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid = valid_mask.bool()
    temporal = temporal_mask.bool()
    if valid.sum() == 0:
        raise ValueError("Graph model received an all-missing keypoint sequence.")
    x = x * valid.unsqueeze(-1).to(x.dtype)
    return x.permute(0, 3, 1, 2).contiguous(), temporal


def _masked_global_pool(x: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
    weights = temporal_mask[:, None, :, None].to(x.dtype)
    x = x * weights
    denom = (weights.sum(dim=(2, 3)) * x.shape[-1]).clamp_min(1.0)
    return x.sum(dim=(2, 3)) / denom


def _downsample_temporal_mask(mask: torch.Tensor, previous_t: int, current_t: int) -> torch.Tensor:
    if previous_t == current_t:
        return mask
    indices = torch.linspace(0, previous_t - 1, steps=current_t, device=mask.device).round().long()
    return mask.index_select(1, indices)


def _residual(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    if in_channels == out_channels and stride == 1:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
        nn.BatchNorm2d(out_channels),
    )
