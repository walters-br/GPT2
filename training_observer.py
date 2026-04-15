"""
EECE 4520 - Milestone 4: Design Patterns
Observer Pattern — Training Event Notifications

Defines a TrainingSubject that fires events during the training loop,
and several concrete TrainingObserver implementations that react to
those events (logging, checkpointing, early stopping).

train.py registers observers on a TrainingSubject instance and calls
notify() at each log/checkpoint interval instead of scattering print
statements and torch.save calls throughout the loop.

Pattern roles:
  Subject (observable) : TrainingSubject
  Observer (abstract)  : TrainingObserver
  Concrete observers   : ConsoleLogObserver, CheckpointObserver,
                         EarlyStoppingObserver
"""

from __future__ import annotations

import os
import math
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Event payload — passed to every observer on each notify() call
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingEvent:
    """Snapshot of training state at the moment an event is fired."""
    step:       int
    epoch:      int
    train_loss: float
    val_loss:   float | None = None
    model:      Any          = field(default=None, repr=False)
    extra:      dict         = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Observer
# ─────────────────────────────────────────────────────────────────────────────

class TrainingObserver(ABC):
    """
    Abstract base class for objects that react to training events.

    Subclasses implement on_event() to define their specific behaviour
    (logging, saving checkpoints, triggering early-stop, etc.).
    """

    @abstractmethod
    def on_event(self, event: TrainingEvent) -> None:
        """Called by the subject whenever a training event is fired."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Subject (Observable)
# ─────────────────────────────────────────────────────────────────────────────

class TrainingSubject:
    """
    Maintains a list of TrainingObserver subscribers and notifies them
    when training milestones occur (log interval, checkpoint interval, etc.).

    The training loop only calls subject.notify(event); all side-effects
    (printing, saving, early-stop logic) live in the observers.
    """

    def __init__(self):
        self._observers: list[TrainingObserver] = []

    # ── Subscription management ──────────────────────────────────────────────

    def attach(self, observer: TrainingObserver) -> None:
        """Register a new observer."""
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: TrainingObserver) -> None:
        """Unregister an existing observer."""
        self._observers.remove(observer)

    # ── Event dispatch ───────────────────────────────────────────────────────

    def notify(self, event: TrainingEvent) -> None:
        """Broadcast a TrainingEvent to all attached observers."""
        for observer in self._observers:
            observer.on_event(event)


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Observer 1 — Console Logger
# ─────────────────────────────────────────────────────────────────────────────

class ConsoleLogObserver(TrainingObserver):
    """
    Prints a formatted loss summary to stdout on every event.

    Replaces the inline print() calls that were previously scattered
    throughout the training loop.
    """

    def on_event(self, event: TrainingEvent) -> None:
        val_str = (
            f"  val_loss={event.val_loss:.4f}"
            if event.val_loss is not None
            else ""
        )
        print(
            f"[epoch {event.epoch:>2} | step {event.step:>6}]"
            f"  train_loss={event.train_loss:.4f}{val_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Observer 2 — Checkpoint Saver
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointObserver(TrainingObserver):
    """
    Saves a model checkpoint to disk every `save_every` steps.

    Replaces the inline torch.save() logic that was previously embedded
    in the training loop, keeping persistence concerns out of train.py.
    """

    def __init__(self, ckpt_dir: str = "checkpoints", save_every: int = 200):
        self.ckpt_dir   = ckpt_dir
        self.save_every = save_every
        os.makedirs(ckpt_dir, exist_ok=True)

    def on_event(self, event: TrainingEvent) -> None:
        if event.step % self.save_every != 0:
            return
        if event.model is None:
            return

        path = os.path.join(
            self.ckpt_dir, f"ckpt_step_{event.step:06d}.pt"
        )
        torch.save(
            {
                "step":        event.step,
                "epoch":       event.epoch,
                "train_loss":  event.train_loss,
                "val_loss":    event.val_loss,
                "model_state": event.model.state_dict(),
            },
            path,
        )
        print(f"  ✓ Checkpoint saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Concrete Observer 3 — Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStoppingObserver(TrainingObserver):
    """
    Monitors validation loss and raises StopIteration when improvement
    stalls for `patience` consecutive events.

    The training loop catches StopIteration to break cleanly:

        try:
            for step, (x, y) in enumerate(loader):
                ...
                subject.notify(event)
        except StopIteration:
            print("Early stopping triggered.")
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self._best_loss  = math.inf
        self._bad_steps  = 0

    def on_event(self, event: TrainingEvent) -> None:
        if event.val_loss is None:
            return

        if event.val_loss < self._best_loss - self.min_delta:
            self._best_loss = event.val_loss
            self._bad_steps = 0
        else:
            self._bad_steps += 1
            if self._bad_steps >= self.patience:
                print(
                    f"  Early stopping: val_loss has not improved by "
                    f"{self.min_delta} for {self.patience} events. "
                    f"Best: {self._best_loss:.4f}"
                )
                raise StopIteration
