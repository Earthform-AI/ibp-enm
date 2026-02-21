"""Training infrastructure for GNCA protein archetype classification.

Provides:
- :class:`GNCATrainer` — single-split training loop with early stopping
- :func:`cross_validate_gnca` — k-fold stratified CV matching ibp_enm conventions
- :class:`TrainResult` / :class:`CVResult` — structured result objects

The trainer is deliberately simple and transparent: no mixed precision,
no gradient accumulation, no distributed training.  The priority is
reliable evaluation of whether graph NCA dynamics add value to protein
classification in a small-data regime (200 proteins, 5 classes).

Usage
-----
>>> from ibp_enm.gnca import GNCAConfig, GNCAClassifier
>>> from ibp_enm.gnca_trainer import cross_validate_gnca
>>> from ibp_enm.graph_data import corpus_to_dataset
>>> from ibp_enm.benchmark import LARGE_CORPUS
>>>
>>> dataset = corpus_to_dataset(LARGE_CORPUS)
>>> result = cross_validate_gnca(dataset, n_folds=5)
>>> print(result.summary())

Requires
--------
``pip install ibp-enm[gnca]``
"""

from __future__ import annotations

import copy
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .gnca import GNCAClassifier, GNCAConfig
from .graph_data import ARCHETYPE_NAMES, ARCHETYPE_TO_IDX, NUM_CLASSES

__all__ = [
    "GNCATrainer",
    "TrainResult",
    "CVResult",
    "cross_validate_gnca",
]


# ═══════════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrainResult:
    """Results from a single training run."""
    train_acc: float = 0.0
    val_acc: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0
    time_s: float = 0.0

    # Per-epoch history
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    train_acc_history: List[float] = field(default_factory=list)
    val_acc_history: List[float] = field(default_factory=list)

    # Per-class metrics (at best epoch)
    per_class_acc: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None

    # Predictions at best epoch
    val_preds: List[int] = field(default_factory=list)
    val_labels: List[int] = field(default_factory=list)
    val_pdb_ids: List[str] = field(default_factory=list)


@dataclass
class CVResult:
    """Results from k-fold cross-validation."""
    fold_results: List[TrainResult] = field(default_factory=list)
    config: Optional[GNCAConfig] = None
    n_folds: int = 0
    dataset_size: int = 0
    time_s: float = 0.0

    @property
    def mean_val_acc(self) -> float:
        accs = [r.val_acc for r in self.fold_results]
        return float(np.mean(accs)) if accs else 0.0

    @property
    def std_val_acc(self) -> float:
        accs = [r.val_acc for r in self.fold_results]
        return float(np.std(accs)) if len(accs) > 1 else 0.0

    @property
    def mean_train_acc(self) -> float:
        accs = [r.train_acc for r in self.fold_results]
        return float(np.mean(accs)) if accs else 0.0

    @property
    def per_class_mean_acc(self) -> Dict[str, float]:
        """Mean per-class accuracy across folds."""
        all_classes = set()
        for r in self.fold_results:
            all_classes.update(r.per_class_acc.keys())
        result = {}
        for cls in sorted(all_classes):
            vals = [r.per_class_acc.get(cls, 0.0) for r in self.fold_results
                    if cls in r.per_class_acc]
            result[cls] = float(np.mean(vals)) if vals else 0.0
        return result

    @property
    def confusion_matrix_total(self) -> Optional[np.ndarray]:
        """Sum confusion matrices across folds."""
        mats = [r.confusion_matrix for r in self.fold_results
                if r.confusion_matrix is not None]
        if not mats:
            return None
        return sum(mats)

    def summary(self) -> str:
        lines = [
            f"GNCA {self.n_folds}-Fold Cross-Validation",
            "=" * 50,
            f"Dataset size:   {self.dataset_size}",
            f"Val accuracy:   {self.mean_val_acc:.1%} ± {self.std_val_acc:.1%}",
            f"Train accuracy: {self.mean_train_acc:.1%}",
            f"Total time:     {self.time_s:.1f}s",
            "",
            "Per-fold accuracy:",
        ]
        for i, r in enumerate(self.fold_results):
            lines.append(
                f"  Fold {i+1}: train={r.train_acc:.1%}  val={r.val_acc:.1%}  "
                f"(best@{r.best_epoch}, {r.time_s:.0f}s)"
            )

        # Per-class
        pcm = self.per_class_mean_acc
        if pcm:
            lines.append("")
            lines.append("Per-class mean accuracy:")
            for cls, acc in pcm.items():
                lines.append(f"  {cls:20s}: {acc:.1%}")

        # Confusion matrix
        cm = self.confusion_matrix_total
        if cm is not None:
            lines.append("")
            lines.append("Confusion matrix (rows=true, cols=pred):")
            header = "         " + " ".join(f"{n[:5]:>6s}" for n in ARCHETYPE_NAMES)
            lines.append(header)
            for i, name in enumerate(ARCHETYPE_NAMES):
                row = " ".join(f"{int(cm[i,j]):6d}" for j in range(NUM_CLASSES))
                lines.append(f"  {name[:7]:>7s} {row}")

        if self.config:
            lines.append("")
            lines.append(f"Config: state_dim={self.config.state_dim}, "
                         f"hidden={self.config.hidden_dim}, "
                         f"T=[{self.config.t_min},{self.config.t_max}], "
                         f"update_p={self.config.update_prob}, "
                         f"wd={self.config.weight_decay}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dictionary."""
        return {
            "n_folds": self.n_folds,
            "dataset_size": self.dataset_size,
            "mean_val_acc": self.mean_val_acc,
            "std_val_acc": self.std_val_acc,
            "mean_train_acc": self.mean_train_acc,
            "time_s": self.time_s,
            "per_class_mean_acc": self.per_class_mean_acc,
            "folds": [
                {
                    "train_acc": r.train_acc,
                    "val_acc": r.val_acc,
                    "best_epoch": r.best_epoch,
                    "total_epochs": r.total_epochs,
                    "time_s": r.time_s,
                    "per_class_acc": r.per_class_acc,
                    "val_preds": r.val_preds,
                    "val_labels": r.val_labels,
                    "val_pdb_ids": r.val_pdb_ids,
                }
                for r in self.fold_results
            ],
            "confusion_matrix": (
                self.confusion_matrix_total.tolist()
                if self.confusion_matrix_total is not None
                else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════

class GNCATrainer:
    """Training loop for GNCAClassifier with early stopping.

    Parameters
    ----------
    config : GNCAConfig
        Hyperparameters.
    device : str or torch.device
        Training device.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        config: Optional[GNCAConfig] = None,
        device: str = "auto",
        verbose: bool = True,
    ):
        self.config = config or GNCAConfig()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.verbose = verbose

    def train(
        self,
        train_data: List,
        val_data: List,
    ) -> Tuple[GNCAClassifier, TrainResult]:
        """Train a GNCA model on the given split.

        Parameters
        ----------
        train_data : list of Data
            Training graphs.
        val_data : list of Data
            Validation graphs.

        Returns
        -------
        model : GNCAClassifier
            Best model (by val loss).
        result : TrainResult
        """
        c = self.config
        device = self.device

        # ── Build model ─────────────────────────────────────────
        model = GNCAClassifier(c).to(device)
        if self.verbose:
            print(f"  Model: {model.count_parameters():,} params on {device}")

        # ── Data loaders ────────────────────────────────────────
        train_loader = DataLoader(
            train_data, batch_size=c.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_data, batch_size=max(c.batch_size, len(val_data)), shuffle=False
        )

        # ── Optimiser ───────────────────────────────────────────
        optimizer = torch.optim.Adam(
            model.parameters(), lr=c.lr, weight_decay=c.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=c.lr_factor,
            patience=c.lr_patience, verbose=False,
        )

        # ── Class weights for imbalanced data ───────────────────
        label_counts = Counter(int(d.y.item()) for d in train_data)
        total = sum(label_counts.values())
        weights = torch.zeros(c.n_classes, device=device)
        for cls_idx in range(c.n_classes):
            cnt = label_counts.get(cls_idx, 1)
            weights[cls_idx] = total / (c.n_classes * cnt)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # ── Training loop ───────────────────────────────────────
        result = TrainResult()
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        t0 = time.time()

        for epoch in range(1, c.epochs + 1):
            # Train
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = criterion(logits, batch.y.squeeze())
                loss.backward()

                # Gradient clipping (stability for deep unrolling)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == batch.y.squeeze()).sum().item()
                epoch_total += batch.num_graphs

            train_loss = epoch_loss / max(epoch_total, 1)
            train_acc = epoch_correct / max(epoch_total, 1)

            # Validate
            val_loss, val_acc, val_preds, val_labels, val_pdb_ids = self._evaluate(
                model, val_loader, criterion, device
            )

            scheduler.step(val_loss)

            # Record history
            result.train_loss_history.append(train_loss)
            result.val_loss_history.append(val_loss)
            result.train_acc_history.append(train_acc)
            result.val_acc_history.append(val_acc)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                result.best_epoch = epoch
                result.train_acc = train_acc
                result.val_acc = val_acc
                result.train_loss = train_loss
                result.val_loss = val_loss
                result.val_preds = val_preds
                result.val_labels = val_labels
                result.val_pdb_ids = val_pdb_ids
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (epoch % 25 == 0 or epoch == 1 or
                                  patience_counter == 0):
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch:3d}: "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.1%}  "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.1%}  "
                    f"lr={lr:.1e}  "
                    f"{'*' if patience_counter == 0 else ''}"
                )

            if patience_counter >= c.early_stop_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best @ {result.best_epoch})")
                break

        result.total_epochs = epoch
        result.time_s = time.time() - t0

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Per-class accuracy and confusion matrix
        result.per_class_acc, result.confusion_matrix = self._per_class_metrics(
            result.val_preds, result.val_labels
        )

        return model, result

    @staticmethod
    def _evaluate(
        model: GNCAClassifier,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float, List[int], List[int], List[str]]:
        """Evaluate model on a DataLoader."""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        all_pdb_ids = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y.squeeze())
                total_loss += loss.item() * batch.num_graphs
                preds = logits.argmax(dim=-1)
                total_correct += (preds == batch.y.squeeze()).sum().item()
                total_samples += batch.num_graphs
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch.y.squeeze().cpu().tolist())

                # Extract pdb_ids from batch
                if hasattr(batch, "pdb_id"):
                    if isinstance(batch.pdb_id, (list, tuple)):
                        all_pdb_ids.extend(batch.pdb_id)
                    else:
                        all_pdb_ids.extend([str(batch.pdb_id)])

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy, all_preds, all_labels, all_pdb_ids

    @staticmethod
    def _per_class_metrics(
        preds: List[int], labels: List[int]
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Compute per-class accuracy and confusion matrix."""
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for p, l in zip(preds, labels):
            if 0 <= l < NUM_CLASSES and 0 <= p < NUM_CLASSES:
                cm[l, p] += 1

        per_class = {}
        for i, name in enumerate(ARCHETYPE_NAMES):
            total = cm[i].sum()
            if total > 0:
                per_class[name] = float(cm[i, i]) / total
            # Skip classes with 0 validation samples

        return per_class, cm


# ═══════════════════════════════════════════════════════════════════
# Cross-Validation
# ═══════════════════════════════════════════════════════════════════

def cross_validate_gnca(
    dataset: List,
    n_folds: int = 5,
    config: Optional[GNCAConfig] = None,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = True,
) -> CVResult:
    """Stratified k-fold cross-validation for GNCA.

    Parameters
    ----------
    dataset : list of Data
        PyG graph data objects with ``.y`` labels.
    n_folds : int
        Number of folds.
    config : GNCAConfig or None
        Hyperparameters (defaults used if None).
    device : str
        ``"auto"``, ``"cuda"``, or ``"cpu"``.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    CVResult
    """
    config = config or GNCAConfig()
    rng = np.random.RandomState(seed)

    # ── Stratified fold assignment ──────────────────────────────
    class_buckets: Dict[int, List[int]] = {}
    for idx, data in enumerate(dataset):
        label = int(data.y.item())
        class_buckets.setdefault(label, []).append(idx)

    # Shuffle within each class
    for label in class_buckets:
        rng.shuffle(class_buckets[label])

    # Assign fold indices
    fold_indices: List[List[int]] = [[] for _ in range(n_folds)]
    for label in sorted(class_buckets.keys()):
        indices = class_buckets[label]
        for i, idx in enumerate(indices):
            fold_indices[i % n_folds].append(idx)

    # Shuffle within each fold
    for fold in fold_indices:
        rng.shuffle(fold)

    # ── Run folds ───────────────────────────────────────────────
    cv_result = CVResult(config=config, n_folds=n_folds, dataset_size=len(dataset))
    t0 = time.time()

    for fold_idx in range(n_folds):
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"Fold {fold_idx + 1}/{n_folds}")
            print(f"{'─' * 50}")

        # Split
        val_indices = set(fold_indices[fold_idx])
        train_data = [dataset[i] for i in range(len(dataset)) if i not in val_indices]
        val_data = [dataset[i] for i in fold_indices[fold_idx]]

        if verbose:
            train_labels = Counter(int(d.y.item()) for d in train_data)
            val_labels = Counter(int(d.y.item()) for d in val_data)
            print(f"  Train: {len(train_data)} ({dict(train_labels)})")
            print(f"  Val:   {len(val_data)} ({dict(val_labels)})")

        # Train
        # Use a different seed per fold for model init reproducibility
        torch.manual_seed(seed + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + fold_idx)

        trainer = GNCATrainer(config=config, device=device, verbose=verbose)
        _, fold_result = trainer.train(train_data, val_data)
        cv_result.fold_results.append(fold_result)

        if verbose:
            print(f"  → Fold {fold_idx+1} val_acc={fold_result.val_acc:.1%} "
                  f"(train={fold_result.train_acc:.1%}, best@{fold_result.best_epoch})")

    cv_result.time_s = time.time() - t0

    if verbose:
        print(f"\n{'=' * 50}")
        print(cv_result.summary())

    return cv_result
