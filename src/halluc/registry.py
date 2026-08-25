"""Minimal name -> factory registry.

One registry per extensible axis (datasets, generators, judges, features, detectors).
Adding a new entry is a decorator on the implementation; nothing else in the pipeline
needs to change.
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._entries:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._entries[name] = factory
            return factory

        return decorator

    def create(self, name: str, **kwargs) -> T:
        if name not in self._entries:
            raise KeyError(
                f"unknown {self.kind} {name!r}; available: {sorted(self._entries)}"
            )
        return self._entries[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._entries))
