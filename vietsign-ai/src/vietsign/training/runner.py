"""Training, evaluation, and prediction workflows."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from vietsign.continuous.decoding import CTCGreedyDecoder
from vietsign.continuous.metrics import aggregate_gloss_metrics, gloss_edit_stats
from vietsign.continuous.vocabulary import GlossVocabulary
from vietsign.data.config import load_dataset_config
from vietsign.evaluation.metrics import MetricsResult, compute_metrics, topk_predictions
from vietsign.models.continuous import ContinuousSignRecognizer
from vietsign.models.registry import InputSpec, build_model, model_metadata, model_modality
from vietsign.models.translation import GlossToVietnameseTransformer
from vietsign.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    validate_model_compatibility,
)
from vietsign.training.config import RecognitionTrainingConfig
from vietsign.training.continuous import (
    ContinuousKeypointDataset,
    collate_continuous_batch,
    load_continuous_samples,
)
from vietsign.training.dataset import (
    CachedKeypointDataset,
    build_class_mapping,
    load_samples_for_training,
    signer_overlap_report,
    split_samples,
)
from vietsign.training.input import collate_keypoint_batch, load_npz_safely, prepare_keypoint_arrays
from vietsign.training.multimodal import (
    MultimodalVSLDataset,
    collate_multimodal_batch,
    load_keypoint_arrays_for_prediction,
)
from vietsign.training.rgb import (
    RGBVideoDataset,
    collate_rgb_batch,
    load_rgb_clip,
    rgb_cache_fingerprint,
    rgb_cache_paths,
    save_rgb_embedding_cache,
)
from vietsign.translation.dataset import (
    GlossTranslationDataset,
    build_translation_vocabularies,
    collate_translation_batch,
    load_translation_samples,
)
from vietsign.translation.metrics import (
    aggregate_translation_metrics,
    chrf,
    exact_match,
    sentence_bleu,
)
from vietsign.translation.normalization import glosses_to_text, normalize_gloss_sequence
from vietsign.translation.vocabulary import TextVocabulary
from vietsign.utils.device import select_device
from vietsign.utils.reproducibility import set_seed


def train_model(config: RecognitionTrainingConfig, *, resume: str | Path | None = None) -> dict[str, Any]:
    if config.task.type == "gloss_translation" or config.model.type in {"gloss_transformer", "pretrained_seq2seq"}:
        return train_translation_model(config, resume=resume)
    if config.task.type == "continuous_recognition" or config.model.type == "ctc_recognizer":
        return train_continuous_model(config, resume=resume)
    if model_modality(config.model.type) == "multimodal":
        return train_multimodal_model(config, resume=resume)
    if model_modality(config.model.type) == "rgb":
        return train_rgb_model(config, resume=resume)
    set_seed(config.seed)
    samples = load_samples_for_training(config)
    class_mapping = build_class_mapping(samples)
    train_samples, val_samples, _ = split_samples(samples)
    if not train_samples:
        raise ValueError("Training split is empty.")
    if not val_samples:
        raise ValueError("Validation split is empty.")
    signer_report = signer_overlap_report(samples)
    if config.evaluation.fail_on_signer_overlap and signer_report["has_overlap"]:
        raise ValueError(f"Signer overlap detected: {signer_report['overlap']}")

    run_dir = _create_run_dir(config)
    _write_effective_config(run_dir, config, signer_report)
    train_dataset = CachedKeypointDataset(
        train_samples,
        cache_root=config.dataset.cache_root,
        label_to_id=class_mapping,
        config=config,
    )
    val_dataset = CachedKeypointDataset(
        val_samples,
        cache_root=config.dataset.cache_root,
        label_to_id=class_mapping,
        config=config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_keypoint_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_keypoint_batch,
    )
    first_batch = next(iter(train_loader))
    model = build_model_from_batch(config, first_batch, num_classes=len(class_mapping))
    device = _torch_device(config.training.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = _build_scheduler(config, optimizer)
    start_epoch = 0
    best_metric = float("-inf")
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location=device)
        validate_model_compatibility(checkpoint, model.introspect().__dict__)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(config, train_samples, class_mapping, device))
    history_path = run_dir / "history.csv"
    patience = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(start_epoch, config.training.epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )
        val_metrics = evaluate_loader(model, val_loader, criterion, device=device, class_mapping=class_mapping)
        if scheduler is not None:
            if config.scheduler.type == "plateau":
                scheduler.step(val_metrics.loss if val_metrics.loss is not None else train_loss)
            else:
                scheduler.step()
        metric_value = _metric_for_early_stopping(val_metrics, config.early_stopping.metric)
        is_best = metric_value > best_metric
        if is_best:
            best_metric = metric_value
            patience = 0
        else:
            patience += 1

        metadata = model_metadata(model)
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config.model_dump(mode="json"),
            class_mapping=class_mapping,
            best_metric=best_metric,
            model_metadata=metadata,
        )
        if is_best:
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config.model_dump(mode="json"),
                class_mapping=class_mapping,
                best_metric=best_metric,
                model_metadata=metadata,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "top1_accuracy": val_metrics.top1_accuracy,
            "top5_accuracy": val_metrics.top5_accuracy,
            "macro_f1": val_metrics.macro_f1,
            "weighted_f1": val_metrics.weighted_f1,
        }
        history_rows.append(row)
        _write_history(history_path, history_rows)
        if config.early_stopping.enabled and patience >= config.early_stopping.patience:
            break

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "num_classes": len(class_mapping),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "model": model_metadata(model),
        "signer_report": signer_report,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def train_continuous_model(config: RecognitionTrainingConfig, *, resume: str | Path | None = None) -> dict[str, Any]:
    del resume
    set_seed(config.seed)
    samples = _load_continuous_samples_from_config(config)
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split in {"validation", "val"}]
    vocabulary = GlossVocabulary.build([sample.glosses for sample in samples], blank_token=config.ctc.blank_token)
    if not train_samples or not val_samples:
        raise ValueError("Continuous training and validation splits must be non-empty.")
    run_dir = _create_run_dir(config)
    train_loader = DataLoader(
        ContinuousKeypointDataset(
            train_samples,
            cache_root=config.dataset.cache_root,
            vocabulary=vocabulary,
            config=config,
        ),
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_continuous_batch,
    )
    val_loader = DataLoader(
        ContinuousKeypointDataset(
            val_samples,
            cache_root=config.dataset.cache_root,
            vocabulary=vocabulary,
            config=config,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_continuous_batch,
    )
    first_batch = next(iter(train_loader))
    model = ContinuousSignRecognizer(
        joint_count=int(first_batch["coordinates"].shape[2]),
        input_channels=int(first_batch["coordinates"].shape[3]),
        vocab_size=len(vocabulary.id_to_token),
        blank_index=vocabulary.blank_index,
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    )
    device = _torch_device(config.training.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    best_ger = float("inf")
    history_rows = []
    for epoch in range(config.training.epochs):
        train_loss = _run_continuous_epoch(model, train_loader, optimizer, device=device, config=config)
        val_report = evaluate_continuous_loader(model, val_loader, vocabulary, device=device, config=config)
        is_best = val_report["metrics"]["gloss_error_rate"] < best_ger
        best_ger = min(best_ger, val_report["metrics"]["gloss_error_rate"])
        metadata = model.introspect().__dict__
        metadata["gloss_vocabulary"] = vocabulary.to_dict()
        metadata["task_type"] = "continuous_recognition"
        for name in ("last.pt", "best.pt" if is_best else "last.pt"):
            save_checkpoint(
                run_dir / name,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                config=config.model_dump(mode="json"),
                class_mapping=vocabulary.token_to_id,
                best_metric=-best_ger,
                model_metadata=metadata,
            )
        row = {"epoch": epoch, "train_loss": train_loss, "val_ger": val_report["metrics"]["gloss_error_rate"]}
        history_rows.append(row)
        _write_history(run_dir / "history.csv", history_rows)
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "task": "continuous_recognition",
        "vocab_size": len(vocabulary.id_to_token),
        "blank_index": vocabulary.blank_index,
        "best_ger": best_ger,
        "model": model.introspect().__dict__,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_translation_model(config: RecognitionTrainingConfig, *, resume: str | Path | None = None) -> dict[str, Any]:
    del resume
    if config.model.type == "pretrained_seq2seq":
        raise ValueError("pretrained_seq2seq training is not implemented; use an explicit pretrained adapter later.")
    if config.model.type != "gloss_transformer":
        raise ValueError("Gloss translation training requires model.type=gloss_transformer.")
    set_seed(config.seed)
    samples = _load_translation_samples_from_config(config)
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split in {"validation", "val"}]
    if not train_samples or not val_samples:
        raise ValueError("Translation training and validation splits must be non-empty.")
    source_vocab, target_vocab = build_translation_vocabularies(samples)
    run_dir = _create_run_dir(config)
    _write_effective_config(run_dir, config, {"task": "gloss_translation"})
    train_loader = DataLoader(
        GlossTranslationDataset(
            train_samples,
            source_vocabulary=source_vocab,
            target_vocabulary=target_vocab,
            max_source_length=config.translation.max_source_length,
            max_target_length=config.translation.max_target_length,
        ),
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_translation_batch,
    )
    val_loader = DataLoader(
        GlossTranslationDataset(
            val_samples,
            source_vocabulary=source_vocab,
            target_vocabulary=target_vocab,
            max_source_length=config.translation.max_source_length,
            max_target_length=config.translation.max_target_length,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch,
    )
    model = _build_translation_model(config, source_vocab, target_vocab)
    device = _torch_device(config.training.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    best_chrf = float("-inf")
    history_rows = []
    for epoch in range(config.training.epochs):
        train_loss = _run_translation_epoch(model, train_loader, optimizer, device=device, config=config)
        val_report = evaluate_translation_loader(
            model,
            val_loader,
            source_vocab,
            target_vocab,
            device=device,
            config=config,
        )
        metric = float(val_report["metrics"]["chrf"])
        is_best = metric > best_chrf
        best_chrf = max(best_chrf, metric)
        metadata = _translation_metadata(model, source_vocab, target_vocab, config)
        for name in ("last.pt", "best.pt" if is_best else "last.pt"):
            save_checkpoint(
                run_dir / name,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                config=config.model_dump(mode="json"),
                class_mapping=source_vocab.token_to_id,
                best_metric=best_chrf,
                model_metadata=metadata,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_report["metrics"]["loss"],
            "val_bleu": val_report["metrics"]["bleu"],
            "val_chrf": val_report["metrics"]["chrf"],
            "val_exact_match": val_report["metrics"]["exact_match"],
        }
        history_rows.append(row)
        _write_history(run_dir / "history.csv", history_rows)
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "task": "gloss_translation",
        "label": _translation_result_label(config),
        "source_vocab_size": len(source_vocab.id_to_token),
        "target_vocab_size": len(target_vocab.id_to_token),
        "best_chrf": best_chrf,
        "model": model.introspect().__dict__,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

def train_multimodal_model(config: RecognitionTrainingConfig, *, resume: str | Path | None = None) -> dict[str, Any]:
    set_seed(config.seed)
    samples = load_samples_for_training(config)
    class_mapping = build_class_mapping(samples)
    train_samples, val_samples, _ = split_samples(samples)
    if not train_samples or not val_samples:
        raise ValueError("Training and validation splits must be non-empty.")
    signer_report = signer_overlap_report(samples)
    run_dir = _create_run_dir(config)
    _write_effective_config(run_dir, config, signer_report)
    train_loader = DataLoader(
        MultimodalVSLDataset(
            train_samples,
            cache_root=config.dataset.cache_root,
            label_to_id=class_mapping,
            config=config,
            training=True,
        ),
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_multimodal_batch,
    )
    val_loader = DataLoader(
        MultimodalVSLDataset(
            val_samples,
            cache_root=config.dataset.cache_root,
            label_to_id=class_mapping,
            config=config,
            training=False,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
    )
    first_batch = next(iter(train_loader))
    model = build_model_from_batch(config, first_batch, num_classes=len(class_mapping))
    device = _torch_device(config.training.device)
    model.to(device)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)
    start_epoch = 0
    best_metric = float("-inf")
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location=device)
        validate_model_compatibility(checkpoint, model_metadata(model))
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))
    criterion = torch.nn.CrossEntropyLoss()
    history_rows: list[dict[str, Any]] = []
    patience = 0
    for epoch in range(start_epoch, config.training.epochs):
        train_loss = _run_multimodal_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.training.gradient_clip_norm,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        )
        val_metrics = evaluate_multimodal_loader(
            model,
            val_loader,
            criterion,
            device=device,
            class_mapping=class_mapping,
        )
        metric_value = _metric_for_early_stopping(val_metrics, config.early_stopping.metric)
        is_best = metric_value > best_metric
        best_metric = max(best_metric, metric_value)
        patience = 0 if is_best else patience + 1
        metadata = model_metadata(model)
        metadata["effective_batch_size"] = config.training.batch_size * config.training.gradient_accumulation_steps
        for name in ("last.pt", "best.pt" if is_best else "last.pt"):
            save_checkpoint(
                run_dir / name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config.model_dump(mode="json"),
                class_mapping=class_mapping,
                best_metric=best_metric,
                model_metadata=metadata,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "top1_accuracy": val_metrics.top1_accuracy,
            "top5_accuracy": val_metrics.top5_accuracy,
            "macro_f1": val_metrics.macro_f1,
            "weighted_f1": val_metrics.weighted_f1,
        }
        history_rows.append(row)
        _write_history(run_dir / "history.csv", history_rows)
        if config.early_stopping.enabled and patience >= config.early_stopping.patience:
            break
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "num_classes": len(class_mapping),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "model": model_metadata(model),
        "signer_report": signer_report,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def train_rgb_model(config: RecognitionTrainingConfig, *, resume: str | Path | None = None) -> dict[str, Any]:
    set_seed(config.seed)
    samples = load_samples_for_training(config)
    class_mapping = build_class_mapping(samples)
    train_samples, val_samples, _ = split_samples(samples)
    if not train_samples:
        raise ValueError("Training split is empty.")
    if not val_samples:
        raise ValueError("Validation split is empty.")
    signer_report = signer_overlap_report(samples)
    if config.evaluation.fail_on_signer_overlap and signer_report["has_overlap"]:
        raise ValueError(f"Signer overlap detected: {signer_report['overlap']}")

    run_dir = _create_run_dir(config)
    _write_effective_config(run_dir, config, signer_report)
    train_dataset = RGBVideoDataset(train_samples, label_to_id=class_mapping, config=config, training=True)
    val_dataset = RGBVideoDataset(val_samples, label_to_id=class_mapping, config=config, training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_rgb_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_rgb_batch,
    )
    model = build_model(
        config,
        num_classes=len(class_mapping),
        input_spec=InputSpec(joint_count=0, channel_count=3),
    )
    device = _torch_device(config.training.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = _build_scheduler(config, optimizer)
    start_epoch = 0
    best_metric = float("-inf")
    if resume is not None:
        checkpoint = load_checkpoint(resume, map_location=device)
        validate_model_compatibility(checkpoint, model_metadata(model))
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", best_metric))

    criterion = torch.nn.CrossEntropyLoss(weight=_class_weights(config, train_samples, class_mapping, device))
    history_path = run_dir / "history.csv"
    patience = 0
    history_rows: list[dict[str, Any]] = []
    for epoch in range(start_epoch, config.training.epochs):
        train_loss = _run_rgb_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=config.training.gradient_clip_norm,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        )
        val_metrics = evaluate_rgb_loader(model, val_loader, criterion, device=device, class_mapping=class_mapping)
        if scheduler is not None:
            scheduler.step(val_metrics.loss if config.scheduler.type == "plateau" else None)
        metric_value = _metric_for_early_stopping(val_metrics, config.early_stopping.metric)
        is_best = metric_value > best_metric
        if is_best:
            best_metric = metric_value
            patience = 0
        else:
            patience += 1

        metadata = model_metadata(model)
        metadata["effective_batch_size"] = (
            config.training.batch_size * config.training.gradient_accumulation_steps
        )
        save_checkpoint(
            run_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config.model_dump(mode="json"),
            class_mapping=class_mapping,
            best_metric=best_metric,
            model_metadata=metadata,
        )
        if is_best:
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config.model_dump(mode="json"),
                class_mapping=class_mapping,
                best_metric=best_metric,
                model_metadata=metadata,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.loss,
            "top1_accuracy": val_metrics.top1_accuracy,
            "top5_accuracy": val_metrics.top5_accuracy,
            "macro_f1": val_metrics.macro_f1,
            "weighted_f1": val_metrics.weighted_f1,
        }
        history_rows.append(row)
        _write_history(history_path, history_rows)
        if config.early_stopping.enabled and patience >= config.early_stopping.patience:
            break

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "num_classes": len(class_mapping),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "model": model_metadata(model),
        "signer_report": signer_report,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_checkpoint(
    config: RecognitionTrainingConfig,
    *,
    checkpoint_path: str | Path,
    split: str = "validation",
) -> dict[str, Any]:
    if config.task.type == "gloss_translation" or config.model.type in {"gloss_transformer", "pretrained_seq2seq"}:
        return evaluate_translation_checkpoint(config, checkpoint_path=checkpoint_path, split=split)
    if config.task.type == "continuous_recognition" or config.model.type == "ctc_recognizer":
        return evaluate_continuous_checkpoint(config, checkpoint_path=checkpoint_path, split=split)
    if model_modality(config.model.type) == "multimodal":
        return evaluate_multimodal_checkpoint(config, checkpoint_path=checkpoint_path, split=split)
    if model_modality(config.model.type) == "rgb":
        return evaluate_rgb_checkpoint(config, checkpoint_path=checkpoint_path, split=split)
    samples = load_samples_for_training(config)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_mapping = checkpoint["class_mapping"]
    selected = _samples_for_split(samples, split)
    if not selected:
        raise ValueError(f"Split '{split}' is empty.")
    dataset = CachedKeypointDataset(
        selected,
        cache_root=config.dataset.cache_root,
        label_to_id=class_mapping,
        config=config,
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_keypoint_batch)
    first_batch = next(iter(loader))
    model = build_model_from_batch(config, first_batch, num_classes=len(class_mapping))
    validate_model_compatibility(checkpoint, model.introspect().__dict__)
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_loader(model, loader, criterion, device=device, class_mapping=class_mapping)
    report = {
        "split": split,
        "samples": len(selected),
        "classes": len(class_mapping),
        "metrics": metrics.__dict__,
        "signer_report": signer_overlap_report(samples),
    }
    run_dir = Path(checkpoint_path).parent
    confusion_matrix_path = run_dir / "confusion_matrix.png"
    _write_confusion_matrix_png(
        confusion_matrix_path,
        metrics.confusion_matrix,
        class_names=[label for label, _ in sorted(class_mapping.items(), key=lambda item: item[1])],
    )
    report["confusion_matrix_png"] = str(confusion_matrix_path)
    output_path = run_dir / "evaluation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def evaluate_continuous_checkpoint(
    config: RecognitionTrainingConfig,
    *,
    checkpoint_path: str | Path,
    split: str = "validation",
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    vocabulary = GlossVocabulary.from_dict(checkpoint["model_metadata"]["gloss_vocabulary"])
    selected = [
        sample
        for sample in _load_continuous_samples_from_config(config)
        if sample.split in ({"validation", "val"} if split == "validation" else {split})
    ]
    loader = DataLoader(
        ContinuousKeypointDataset(
            selected,
            cache_root=config.dataset.cache_root,
            vocabulary=vocabulary,
            config=config,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_continuous_batch,
    )
    first_batch = next(iter(loader))
    model = ContinuousSignRecognizer(
        joint_count=int(first_batch["coordinates"].shape[2]),
        input_channels=int(first_batch["coordinates"].shape[3]),
        vocab_size=len(vocabulary.id_to_token),
        blank_index=vocabulary.blank_index,
        embedding_dim=config.model.embedding_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    )
    validate_model_compatibility(checkpoint, model.introspect().__dict__)
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    report = evaluate_continuous_loader(model, loader, vocabulary, device=device, config=config)
    run_dir = Path(checkpoint_path).parent
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_continuous_predictions_csv(run_dir / "predictions.csv", report["predictions"])
    return report


def evaluate_translation_checkpoint(
    config: RecognitionTrainingConfig,
    *,
    checkpoint_path: str | Path,
    split: str = "validation",
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    source_vocab = TextVocabulary.from_dict(checkpoint["model_metadata"]["source_vocabulary"])
    target_vocab = TextVocabulary.from_dict(checkpoint["model_metadata"]["target_vocabulary"])
    selected = [
        sample
        for sample in _load_translation_samples_from_config(config)
        if sample.split in ({"validation", "val"} if split == "validation" else {split})
    ]
    if not selected:
        raise ValueError(f"Translation split '{split}' is empty.")
    loader = DataLoader(
        GlossTranslationDataset(
            selected,
            source_vocabulary=source_vocab,
            target_vocabulary=target_vocab,
            max_source_length=config.translation.max_source_length,
            max_target_length=config.translation.max_target_length,
        ),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch,
    )
    model = _build_translation_model(config, source_vocab, target_vocab)
    validate_model_compatibility(checkpoint, model.introspect().__dict__)
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    report = evaluate_translation_loader(
        model,
        loader,
        source_vocab,
        target_vocab,
        device=device,
        config=config,
    )
    report["evaluation_mode"] = "oracle_gloss"
    run_dir = Path(checkpoint_path).parent
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_translation_predictions_csv(run_dir / "translations.csv", report["predictions"])
    return report

def evaluate_multimodal_checkpoint(
    config: RecognitionTrainingConfig,
    *,
    checkpoint_path: str | Path,
    split: str = "validation",
) -> dict[str, Any]:
    samples = load_samples_for_training(config)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_mapping = checkpoint["class_mapping"]
    selected = _samples_for_split(samples, split)
    dataset = MultimodalVSLDataset(
        selected,
        cache_root=config.dataset.cache_root,
        label_to_id=class_mapping,
        config=config,
        training=False,
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)
    first_batch = next(iter(loader))
    model = build_model_from_batch(config, first_batch, num_classes=len(class_mapping))
    validate_model_compatibility(checkpoint, model_metadata(model))
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_multimodal_loader(model, loader, criterion, device=device, class_mapping=class_mapping)
    report = {"split": split, "samples": len(selected), "classes": len(class_mapping), "metrics": metrics.__dict__}
    run_dir = Path(checkpoint_path).parent
    confusion_matrix_path = run_dir / "confusion_matrix.png"
    _write_confusion_matrix_png(
        confusion_matrix_path,
        metrics.confusion_matrix,
        class_names=[label for label, _ in sorted(class_mapping.items(), key=lambda item: item[1])],
    )
    report["confusion_matrix_png"] = str(confusion_matrix_path)
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def evaluate_rgb_checkpoint(
    config: RecognitionTrainingConfig,
    *,
    checkpoint_path: str | Path,
    split: str = "validation",
) -> dict[str, Any]:
    samples = load_samples_for_training(config)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_mapping = checkpoint["class_mapping"]
    selected = _samples_for_split(samples, split)
    if not selected:
        raise ValueError(f"Split '{split}' is empty.")
    dataset = RGBVideoDataset(selected, label_to_id=class_mapping, config=config, training=False)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_rgb_batch)
    model = build_model(config, num_classes=len(class_mapping), input_spec=InputSpec(0, 3))
    validate_model_compatibility(checkpoint, model_metadata(model))
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate_rgb_loader(model, loader, criterion, device=device, class_mapping=class_mapping)
    report = {
        "split": split,
        "samples": len(selected),
        "classes": len(class_mapping),
        "metrics": metrics.__dict__,
        "signer_report": signer_overlap_report(samples),
    }
    run_dir = Path(checkpoint_path).parent
    confusion_matrix_path = run_dir / "confusion_matrix.png"
    _write_confusion_matrix_png(
        confusion_matrix_path,
        metrics.confusion_matrix,
        class_names=[label for label, _ in sorted(class_mapping.items(), key=lambda item: item[1])],
    )
    report["confusion_matrix_png"] = str(confusion_matrix_path)
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict_cached_features(
    config: RecognitionTrainingConfig,
    *,
    features_path: str | Path,
    checkpoint_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_mapping = checkpoint["class_mapping"]
    id_to_label = {value: key for key, value in class_mapping.items()}
    arrays = load_npz_safely(features_path)
    inputs, valid_mask, temporal_mask, valid_ratio = prepare_keypoint_arrays(
        arrays,
        feature_config=config.features,
        sequence_config=config.sequence,
        minimum_valid_joint_ratio=config.data_quality.minimum_valid_joint_ratio,
        reject_empty=True,
    )
    batch = {
        "coordinates": torch.from_numpy(inputs).unsqueeze(0),
        "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0),
        "temporal_mask": torch.from_numpy(temporal_mask).unsqueeze(0),
    }
    model = build_model_from_batch(config, batch, num_classes=len(class_mapping))
    validate_model_compatibility(checkpoint, model.introspect().__dict__)
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(
            batch["coordinates"].to(device),
            batch["valid_mask"].to(device),
            batch["temporal_mask"].to(device),
        )
    top = topk_predictions(logits.cpu(), id_to_label=id_to_label, k=top_k)[0]
    return {
        "label": top[0]["label"],
        "label_id": top[0]["label_id"],
        "confidence": top[0]["confidence"],
        "top_k": top,
        "valid_joint_ratio": valid_ratio,
    }

def predict_video(
    config: RecognitionTrainingConfig,
    *,
    video_path: str | Path,
    checkpoint_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    if model_modality(config.model.type) != "rgb":
        raise ValueError("predict_video requires an RGB model config.")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_mapping = checkpoint["class_mapping"]
    id_to_label = {value: key for key, value in class_mapping.items()}
    frames, metadata = load_rgb_clip(
        video_path,
        video_config=config.video,
        preprocessing=config.rgb,
        training=False,
    )
    batch = torch.from_numpy(frames).unsqueeze(0)
    model = build_model(config, num_classes=len(class_mapping), input_spec=InputSpec(0, 3))
    validate_model_compatibility(checkpoint, model_metadata(model))
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device), None)
    top = topk_predictions(logits.cpu(), id_to_label=id_to_label, k=top_k)[0]
    return {
        "label": top[0]["label"],
        "label_id": top[0]["label_id"],
        "confidence": top[0]["confidence"],
        "top_k": top,
        "video_metadata": metadata,
        "note": _rgb_prediction_note(config),
    }

def predict_multimodal(
    config: RecognitionTrainingConfig,
    *,
    video_path: str | Path,
    keypoints_path: str | Path,
    checkpoint_path: str | Path,
    metadata_path: str | Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    if model_modality(config.model.type) != "multimodal":
        raise ValueError("predict_multimodal requires a multimodal config.")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_mapping = checkpoint["class_mapping"]
    id_to_label = {value: key for key, value in class_mapping.items()}
    pixel_values, rgb_metadata = load_rgb_clip(
        video_path,
        video_config=config.video,
        preprocessing=config.rgb,
        training=False,
    )
    metadata_file = Path(metadata_path) if metadata_path else Path(keypoints_path).with_suffix(".json")
    coordinates, valid_mask, temporal_mask, pose_metadata = load_keypoint_arrays_for_prediction(
        keypoints_path,
        metadata_file,
        config,
    )
    if pose_metadata.get("source_video") and Path(pose_metadata["source_video"]).name != Path(video_path).name:
        raise ValueError("RGB video and keypoint cache source_video do not match.")
    batch = {
        "pixel_values": torch.from_numpy(pixel_values).unsqueeze(0),
        "coordinates": coordinates,
        "valid_mask": valid_mask,
        "temporal_mask": temporal_mask,
    }
    model = build_model_from_batch(config, batch, num_classes=len(class_mapping))
    validate_model_compatibility(checkpoint, model_metadata(model))
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(config.training.device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(
            pixel_values=batch["pixel_values"].to(device),
            coordinates=batch["coordinates"].to(device),
            valid_mask=batch["valid_mask"].to(device),
            temporal_mask=batch["temporal_mask"].to(device),
        )
    top = topk_predictions(logits.cpu(), id_to_label=id_to_label, k=top_k)[0]
    return {
        "label": top[0]["label"],
        "label_id": top[0]["label_id"],
        "confidence": top[0]["confidence"],
        "top_k": top,
        "rgb_metadata": rgb_metadata,
        "pose_metadata": pose_metadata,
        "note": "Prediction semantics require a genuinely trained real-VSL multimodal checkpoint.",
    }

def predict_continuous(
    *,
    features_path: str | Path,
    checkpoint_path: str | Path,
    config: RecognitionTrainingConfig | None = None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    vocabulary = GlossVocabulary.from_dict(checkpoint["model_metadata"]["gloss_vocabulary"])
    runtime_config = config or RecognitionTrainingConfig.model_validate(checkpoint["config"])
    arrays = load_npz_safely(features_path)
    inputs, valid_mask, temporal_mask, _ratio = prepare_keypoint_arrays(
        arrays,
        feature_config=runtime_config.features,
        sequence_config=runtime_config.sequence,
        minimum_valid_joint_ratio=runtime_config.data_quality.minimum_valid_joint_ratio,
        reject_empty=runtime_config.data_quality.reject_empty,
    )
    model = ContinuousSignRecognizer(
        joint_count=inputs.shape[1],
        input_channels=inputs.shape[2],
        vocab_size=len(vocabulary.id_to_token),
        blank_index=vocabulary.blank_index,
        embedding_dim=runtime_config.model.embedding_dim,
        num_layers=runtime_config.model.num_layers,
        num_heads=runtime_config.model.num_heads,
        dropout=runtime_config.model.dropout,
    )
    validate_model_compatibility(checkpoint, model.introspect().__dict__)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(inputs).unsqueeze(0),
            torch.from_numpy(valid_mask).unsqueeze(0),
            torch.from_numpy(temporal_mask).unsqueeze(0),
        )
    decoder = CTCGreedyDecoder(vocabulary)
    decoded = decoder.decode(logits, input_lengths=torch.tensor([int(temporal_mask.sum())]))[0]
    return {
        "gloss_sequence": decoded,
        "decoder": "ctc_greedy",
        "note": "Gloss output is not natural Vietnamese translation.",
    }


def translate_gloss(
    *,
    gloss: str | list[str],
    checkpoint_path: str | Path,
    config: RecognitionTrainingConfig | None = None,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    runtime_config = config or RecognitionTrainingConfig.model_validate(checkpoint["config"])
    if runtime_config.decoding.type != "greedy":
        raise NotImplementedError("Beam search translation decoding is not implemented in Phase 10.")
    source_vocab = TextVocabulary.from_dict(checkpoint["model_metadata"]["source_vocabulary"])
    target_vocab = TextVocabulary.from_dict(checkpoint["model_metadata"]["target_vocabulary"])
    glosses = normalize_gloss_sequence(gloss, uppercase=runtime_config.translation.uppercase_glosses)
    source_ids = source_vocab.encode(glosses)
    source_tensor = torch.tensor([source_ids], dtype=torch.long)
    source_mask = source_tensor.ne(source_vocab.pad_index)
    model = _build_translation_model(runtime_config, source_vocab, target_vocab)
    validate_model_compatibility(checkpoint, model.introspect().__dict__)
    model.load_state_dict(checkpoint["model_state"])
    device = _torch_device(runtime_config.training.device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        generated = model.greedy_decode(
            source_tensor.to(device),
            source_attention_mask=source_mask.to(device),
            max_new_tokens=runtime_config.decoding.max_new_tokens,
        ).cpu()[0].tolist()
    decoded_ids = _strip_generated_ids(generated, target_vocab)
    tokens = target_vocab.decode(decoded_ids, skip_special=True)
    return {
        "source_gloss": glosses_to_text(glosses),
        "source_token_ids": source_ids,
        "translation": " ".join(tokens),
        "generated_token_ids": decoded_ids,
        "decoder": "greedy",
        "note": "Synthetic translation checkpoints do not imply real VSL translation quality.",
    }


def translate_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path,
    config: RecognitionTrainingConfig | None = None,
) -> dict[str, Any]:
    rows = []
    with Path(input_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            gloss_text = row.get("predicted_glosses") or row.get("source_gloss") or row.get("gloss_sequence")
            if not gloss_text:
                raise ValueError("Translation input CSV requires predicted_glosses, source_gloss, or gloss_sequence.")
            prediction = translate_gloss(gloss=gloss_text, checkpoint_path=checkpoint_path, config=config)
            rows.append({**row, "source_gloss": prediction["source_gloss"], "translation": prediction["translation"]})
    if rows:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {"translated": len(rows), "output": str(output_path)}


def recognize_and_translate(
    *,
    features_path: str | Path,
    cslr_checkpoint: str | Path,
    translation_checkpoint: str | Path,
    cslr_config: RecognitionTrainingConfig | None = None,
    translation_config: RecognitionTrainingConfig | None = None,
) -> dict[str, Any]:
    recognition = predict_continuous(
        features_path=features_path,
        checkpoint_path=cslr_checkpoint,
        config=cslr_config,
    )
    translation = translate_gloss(
        gloss=recognition["gloss_sequence"],
        checkpoint_path=translation_checkpoint,
        config=translation_config,
    )
    return {
        "recognized_glosses": recognition["gloss_sequence"],
        "translation": translation["translation"],
        "recognition": recognition,
        "translation_result": translation,
        "note": "Preserves CSLR output so recognition errors are visible before translation.",
    }

def extract_rgb_embeddings(config: RecognitionTrainingConfig) -> dict[str, Any]:
    if model_modality(config.model.type) != "rgb":
        raise ValueError("RGB embedding extraction requires an RGB model config.")
    samples = load_samples_for_training(config)
    class_mapping = build_class_mapping(samples)
    model = build_model(config, num_classes=len(class_mapping), input_spec=InputSpec(0, 3))
    device = _torch_device(config.training.device)
    model.to(device)
    model.eval()
    fingerprint = rgb_cache_fingerprint(config)
    results = []
    failures = 0
    started = time.perf_counter()
    for sample in samples:
        npz_path, metadata_path = rgb_cache_paths(config.rgb.cache_root, sample)
        try:
            frames, frame_metadata = load_rgb_clip(
                sample.video_path,
                video_config=config.video,
                preprocessing=config.rgb,
                training=False,
            )
            with torch.no_grad():
                embedding = model.encode(torch.from_numpy(frames).unsqueeze(0).to(device)).cpu().numpy()[0]
            save_rgb_embedding_cache(
                npz_path,
                metadata_path,
                embedding=embedding,
                sample=sample,
                metadata={
                    "fingerprint": fingerprint,
                    "model_identity": {
                        "type": config.model.type,
                        "pretrained_model": config.model.pretrained_model,
                        "pretrained": config.model.pretrained,
                        "freeze_backbone": config.model.freeze_backbone,
                    },
                    "preprocessing": config.rgb.model_dump(mode="json"),
                    "frame_sampling": config.video.model_dump(mode="json"),
                    "frame_metadata": frame_metadata,
                    "embedding_shape": list(embedding.shape),
                },
            )
            results.append({"sample_id": sample.sample_id, "status": "cached", "path": str(npz_path)})
        except Exception as error:  # noqa: BLE001
            failures += 1
            results.append({"sample_id": sample.sample_id, "status": "failed", "error": str(error)})
    return {
        "schema_version": "rgb_embeddings.v1",
        "processed": len(results) - failures,
        "failed": failures,
        "cache_root": str(config.rgb.cache_root),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }


def build_model_from_batch(
    config: RecognitionTrainingConfig,
    batch: dict[str, Any],
    *,
    num_classes: int,
) -> torch.nn.Module:
    coordinates = batch["coordinates"]
    return build_model(
        config,
        num_classes=num_classes,
        input_spec=InputSpec(
            joint_count=int(coordinates.shape[2]),
            channel_count=int(coordinates.shape[3]),
        ),
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    class_mapping: dict[str, int],
) -> MetricsResult:
    model.eval()
    logits_list = []
    labels_list = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["coordinates"].to(device),
                batch["valid_mask"].to(device),
                batch["temporal_mask"].to(device),
            )
            labels = batch["labels"].to(device)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            batches += 1
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    id_to_label = {value: key for key, value in class_mapping.items()}
    class_names = [id_to_label[index] for index in range(len(id_to_label))]
    return compute_metrics(
        torch.cat(logits_list),
        torch.cat(labels_list),
        class_names=class_names,
        loss=total_loss / max(batches, 1),
    )

def evaluate_multimodal_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    class_mapping: dict[str, int],
) -> MetricsResult:
    model.eval()
    logits_list = []
    labels_list = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                pixel_values=batch["pixel_values"].to(device),
                coordinates=batch["coordinates"].to(device),
                valid_mask=batch["valid_mask"].to(device),
                temporal_mask=batch["temporal_mask"].to(device),
                modality_mask={key: value.to(device) for key, value in batch["modality_mask"].items()},
            )
            labels = batch["labels"].to(device)
            total_loss += float(criterion(logits, labels).item())
            batches += 1
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    id_to_label = {value: key for key, value in class_mapping.items()}
    class_names = [id_to_label[index] for index in range(len(id_to_label))]
    return compute_metrics(
        torch.cat(logits_list),
        torch.cat(labels_list),
        class_names=class_names,
        loss=total_loss / max(batches, 1),
    )

def evaluate_rgb_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    class_mapping: dict[str, int],
) -> MetricsResult:
    model.eval()
    logits_list = []
    labels_list = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["pixel_values"].to(device), batch.get("temporal_mask"))
            labels = batch["labels"].to(device)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            batches += 1
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    id_to_label = {value: key for key, value in class_mapping.items()}
    class_names = [id_to_label[index] for index in range(len(id_to_label))]
    return compute_metrics(
        torch.cat(logits_list),
        torch.cat(labels_list),
        class_names=class_names,
        loss=total_loss / max(batches, 1),
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["coordinates"].to(device),
            batch["valid_mask"].to(device),
            batch["temporal_mask"].to(device),
        )
        loss = criterion(logits, batch["labels"].to(device))
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)

def _run_rgb_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None,
    gradient_accumulation_steps: int,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    optimizer.zero_grad(set_to_none=True)
    accumulation = max(gradient_accumulation_steps, 1)
    for batch_index, batch in enumerate(loader):
        logits = model(batch["pixel_values"].to(device), batch.get("temporal_mask"))
        loss = criterion(logits, batch["labels"].to(device)) / accumulation
        loss.backward()
        if (batch_index + 1) % accumulation == 0 or (batch_index + 1) == len(loader):
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.item() * accumulation)
        batches += 1
    return total / max(batches, 1)

def _run_multimodal_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None,
    gradient_accumulation_steps: int,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    accumulation = max(gradient_accumulation_steps, 1)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        logits = model(
            pixel_values=batch["pixel_values"].to(device),
            coordinates=batch["coordinates"].to(device),
            valid_mask=batch["valid_mask"].to(device),
            temporal_mask=batch["temporal_mask"].to(device),
            modality_mask={key: value.to(device) for key, value in batch["modality_mask"].items()},
        )
        loss = criterion(logits, batch["labels"].to(device)) / accumulation
        loss.backward()
        if (batch_index + 1) % accumulation == 0 or (batch_index + 1) == len(loader):
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.item() * accumulation)
        batches += 1
    return total / max(batches, 1)

def _run_continuous_epoch(
    model: ContinuousSignRecognizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    config: RecognitionTrainingConfig,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model.ctc_loss(
            batch["coordinates"].to(device),
            batch["valid_mask"].to(device),
            batch["temporal_mask"].to(device),
            batch["targets"].to(device),
            batch["target_lengths"].to(device),
            zero_infinity=config.ctc.zero_infinity,
        )
        loss.backward()
        if config.training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.gradient_clip_norm)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def evaluate_continuous_loader(
    model: ContinuousSignRecognizer,
    loader: DataLoader,
    vocabulary: GlossVocabulary,
    *,
    device: torch.device,
    config: RecognitionTrainingConfig,
) -> dict[str, Any]:
    model.eval()
    decoder = CTCGreedyDecoder(vocabulary)
    pairs = []
    predictions = []
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            coordinates = batch["coordinates"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            temporal_mask = batch["temporal_mask"].to(device)
            logits = model(coordinates, valid_mask, temporal_mask)
            loss = model.ctc_loss(
                coordinates,
                valid_mask,
                temporal_mask,
                batch["targets"].to(device),
                batch["target_lengths"].to(device),
                zero_infinity=config.ctc.zero_infinity,
            )
            total_loss += float(loss.item())
            batches += 1
            decoded = decoder.decode(logits.cpu(), input_lengths=batch["input_lengths"])
            for sample_id, reference, hypothesis in zip(batch["sample_ids"], batch["reference_glosses"], decoded):
                stats = gloss_edit_stats(reference, hypothesis)
                pairs.append((reference, hypothesis))
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "reference_glosses": " ".join(reference),
                        "predicted_glosses": " ".join(hypothesis),
                        "substitutions": stats.substitutions,
                        "deletions": stats.deletions,
                        "insertions": stats.insertions,
                        "ger": stats.ger,
                    }
                )
    metrics = aggregate_gloss_metrics(pairs)
    metrics["ctc_loss"] = total_loss / max(batches, 1)
    return {
        "task": "continuous_recognition",
        "label": "SYNTHETIC CSLR SOFTWARE VALIDATION - NOT VSL PERFORMANCE",
        "metrics": metrics,
        "predictions": predictions,
    }


def evaluate_translation_loader(
    model: GlossToVietnameseTransformer,
    loader: DataLoader,
    source_vocabulary: TextVocabulary,
    target_vocabulary: TextVocabulary,
    *,
    device: torch.device,
    config: RecognitionTrainingConfig,
) -> dict[str, Any]:
    del source_vocabulary
    model.eval()
    predictions = []
    pairs = []
    total_loss = 0.0
    batches = 0
    logits_nan = False
    logits_inf = False
    with torch.no_grad():
        for batch in loader:
            source_ids = batch["source_input_ids"].to(device)
            source_mask = batch["source_attention_mask"].to(device)
            target_ids = batch["target_input_ids"].to(device)
            target_mask = batch["target_attention_mask"].to(device)
            labels = batch["target_labels"].to(device)
            logits = model(
                source_ids,
                target_ids,
                source_attention_mask=source_mask,
                target_attention_mask=target_mask,
            )
            logits_nan = logits_nan or bool(torch.isnan(logits).any())
            logits_inf = logits_inf or bool(torch.isinf(logits).any())
            loss = torch.nn.CrossEntropyLoss(ignore_index=target_vocabulary.pad_index)(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )
            total_loss += float(loss.item())
            batches += 1
            generated = model.greedy_decode(
                source_ids,
                source_attention_mask=source_mask,
                max_new_tokens=config.decoding.max_new_tokens,
            )
            for sample_id, source_gloss, reference, token_ids in zip(
                batch["sample_ids"],
                batch["source_glosses"],
                batch["target_texts"],
                generated.cpu().tolist(),
            ):
                decoded_ids = _strip_generated_ids(token_ids, target_vocabulary)
                prediction = " ".join(target_vocabulary.decode(decoded_ids, skip_special=True))
                pairs.append((reference, prediction))
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "source_gloss": source_gloss,
                        "reference_vietnamese": reference,
                        "predicted_vietnamese": prediction,
                        "bleu": sentence_bleu(reference, prediction),
                        "chrf": chrf(reference, prediction),
                        "exact_match": exact_match(reference, prediction),
                    }
                )
    metrics = aggregate_translation_metrics(pairs)
    metrics["loss"] = total_loss / max(batches, 1)
    metrics["logits_nan"] = logits_nan
    metrics["logits_inf"] = logits_inf
    return {
        "task": "gloss_translation",
        "label": _translation_result_label(config),
        "metrics": metrics,
        "predictions": predictions,
    }


def _translation_result_label(config: RecognitionTrainingConfig) -> str:
    dataset_config = str(config.dataset.config).lower()
    if any(name in dataset_config for name in ("parallel_vie_vsl", "vsl_gh")):
        return (
            "REAL GLOSS-TO-VIETNAMESE TEXT TRANSLATION DATA - "
            "NOT WEBCAM/VISUAL VSL PERFORMANCE"
        )
    return "SYNTHETIC TRANSLATION SOFTWARE VALIDATION - NOT REAL VSL PERFORMANCE"


def _rgb_prediction_note(config: RecognitionTrainingConfig) -> str:
    dataset_config = str(config.dataset.config).lower()
    if "visignlanguage" in dataset_config:
        return (
            "Real ViSignLanguage-Video checkpoint; label semantics are Vietnamese "
            "class recognition, not full sentence-level VSL translation."
        )
    return "Prediction semantics require a genuinely trained real-VSL checkpoint."


def _run_translation_epoch(
    model: GlossToVietnameseTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    config: RecognitionTrainingConfig,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(
            batch["source_input_ids"].to(device),
            batch["target_input_ids"].to(device),
            batch["target_labels"].to(device),
            source_attention_mask=batch["source_attention_mask"].to(device),
            target_attention_mask=batch["target_attention_mask"].to(device),
        )
        loss.backward()
        if config.training.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.gradient_clip_norm)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def _build_translation_model(
    config: RecognitionTrainingConfig,
    source_vocabulary: TextVocabulary,
    target_vocabulary: TextVocabulary,
) -> GlossToVietnameseTransformer:
    return GlossToVietnameseTransformer(
        source_vocab_size=len(source_vocabulary.id_to_token),
        target_vocab_size=len(target_vocabulary.id_to_token),
        pad_index=target_vocabulary.pad_index,
        bos_index=target_vocabulary.bos_index,
        eos_index=target_vocabulary.eos_index,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        encoder_layers=config.model.encoder_layers,
        decoder_layers=config.model.decoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        max_length=max(config.translation.max_source_length, config.translation.max_target_length) + 4,
    )


def _translation_metadata(
    model: GlossToVietnameseTransformer,
    source_vocabulary: TextVocabulary,
    target_vocabulary: TextVocabulary,
    config: RecognitionTrainingConfig,
) -> dict[str, Any]:
    metadata = model.introspect().__dict__
    metadata.update(
        {
            "task_type": "gloss_translation",
            "source_vocabulary": source_vocabulary.to_dict(),
            "target_vocabulary": target_vocabulary.to_dict(),
            "tokenizer_identity": "whitespace",
            "translation_config": config.translation.model_dump(mode="json"),
            "decoding_config": config.decoding.model_dump(mode="json"),
        }
    )
    return metadata

def _build_optimizer(config: RecognitionTrainingConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.model.type != "multimodal":
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    groups = []
    rgb_params = [parameter for parameter in model.rgb_encoder.parameters() if parameter.requires_grad]
    pose_params = [parameter for parameter in model.pose_encoder.parameters() if parameter.requires_grad]
    fusion_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(("rgb_encoder.", "pose_encoder."))
    ]
    if rgb_params:
        groups.append({"params": rgb_params, "lr": config.optimizer.rgb_lr or config.training.learning_rate})
    if pose_params:
        groups.append({"params": pose_params, "lr": config.optimizer.pose_lr or config.training.learning_rate})
    if fusion_params:
        groups.append({"params": fusion_params, "lr": config.optimizer.fusion_lr or config.training.learning_rate})
    return torch.optim.AdamW(groups, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)


def _load_continuous_samples_from_config(config: RecognitionTrainingConfig):
    dataset_config = load_dataset_config(config.dataset.config)
    source = dataset_config.dataset
    if source.metadata_csv is None:
        raise ValueError("Continuous recognition config requires dataset.metadata_csv.")
    return load_continuous_samples(source.metadata_csv, dataset=source.name)


def _load_translation_samples_from_config(config: RecognitionTrainingConfig):
    dataset_config = load_dataset_config(config.dataset.config)
    source = dataset_config.dataset
    if source.metadata_csv is None:
        raise ValueError("Gloss translation config requires dataset.metadata_csv.")
    return load_translation_samples(
        source.metadata_csv,
        dataset=source.name,
        lowercase_vietnamese=config.translation.lowercase_vietnamese,
        uppercase_glosses=config.translation.uppercase_glosses,
    )


def _write_continuous_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_translation_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _strip_generated_ids(token_ids: list[int], vocabulary: TextVocabulary) -> list[int]:
    stripped = []
    for token_id in token_ids:
        if token_id in {vocabulary.bos_index, vocabulary.pad_index}:
            continue
        if token_id == vocabulary.eos_index:
            break
        stripped.append(token_id)
    return stripped


def _torch_device(preference: str) -> torch.device:
    runtime = select_device(preference)
    return torch.device(runtime.type)


def _build_scheduler(config: RecognitionTrainingConfig, optimizer: torch.optim.Optimizer):
    if config.scheduler.type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    if config.scheduler.type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    return None


def _class_weights(
    config: RecognitionTrainingConfig,
    samples: list,
    class_mapping: dict[str, int],
    device: torch.device,
) -> torch.Tensor | None:
    if config.loss.class_weighting == "none":
        return None
    counts = torch.zeros((len(class_mapping),), dtype=torch.float32)
    for sample in samples:
        counts[class_mapping[sample.label]] += 1
    return (counts.sum() / counts.clamp_min(1.0)).to(device)


def _metric_for_early_stopping(metrics: MetricsResult, name: str) -> float:
    if name == "macro_f1":
        return metrics.macro_f1
    if name == "top1_accuracy":
        return metrics.top1_accuracy
    if name == "loss":
        return -(metrics.loss if metrics.loss is not None else float("inf"))
    raise ValueError(f"Unsupported early stopping metric: {name}")


def _create_run_dir(config: RecognitionTrainingConfig) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = config.run.output_dir / f"{config.run.name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_effective_config(
    run_dir: Path,
    config: RecognitionTrainingConfig,
    signer_report: dict[str, Any],
) -> None:
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_commit = result.stdout.strip()
    except Exception:  # noqa: BLE001
        git_commit = "uncommitted"
    payload = config.model_dump(mode="json")
    payload["runtime"] = {
        "torch_version": str(torch.__version__),
        "git_commit": str(git_commit),
        "signer_report": signer_report,
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _samples_for_split(samples: list, split: str) -> list:
    if split == "validation":
        return [sample for sample in samples if sample.split in {"validation", "val"}]
    return [sample for sample in samples if sample.split == split]


def _write_confusion_matrix_png(
    path: Path,
    confusion_matrix: list[list[int]],
    *,
    class_names: list[str],
) -> None:
    """Write a small confusion matrix PNG without requiring a GUI."""

    import cv2  # type: ignore
    import numpy as np

    matrix = np.asarray(confusion_matrix, dtype=np.float32)
    if matrix.size == 0:
        matrix = np.zeros((1, 1), dtype=np.float32)
    scale = 255.0 / max(float(matrix.max()), 1.0)
    image = (matrix * scale).astype(np.uint8)
    cell_size = 48
    image = cv2.resize(
        image,
        (max(1, image.shape[1]) * cell_size, max(1, image.shape[0]) * cell_size),
        interpolation=cv2.INTER_NEAREST,
    )
    image = cv2.applyColorMap(image, cv2.COLORMAP_VIRIDIS)
    for row_index, row in enumerate(confusion_matrix):
        for column_index, value in enumerate(row):
            cv2.putText(
                image,
                str(value),
                (column_index * cell_size + 10, row_index * cell_size + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    label_panel = np.full((40, image.shape[1], 3), 255, dtype=np.uint8)
    title = "Confusion matrix: " + ", ".join(class_names[:6])
    cv2.putText(label_panel, title[:120], (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([label_panel, image]))
