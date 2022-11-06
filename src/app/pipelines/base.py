from __future__ import annotations

from abc import ABC, abstractmethod


class Pipeline(ABC):
    """Base class for the implementation of a pipeline stage."""

    @abstractmethod
    def run(self) -> None:
        """Implement a run method for a given pipeline stage."""
