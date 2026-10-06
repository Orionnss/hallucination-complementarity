"""Load modules from an upstream checkout without running its package side effects.

Upstream repositories are written as applications, not libraries: their package
`__init__` files import the whole project (configs, data loaders, API clients) and pin a
dependency stack that conflicts with ours. Installing them is not possible, and copying
their code is not permitted when they carry no licence.

So the needed modules are imported straight from the checkout, with two controls:

* **Identity.** The checkout must be at the pinned commit, and every file the adapter
  imports must match its recorded sha256. A local edit to an upstream file therefore
  fails loudly instead of changing the method.
* **Isolation.** Package `__init__` files are not executed. Parent packages are created
  as empty namespace modules, and modules that are imported only for code paths the
  adapter never calls (path helpers, config loaders) are replaced by stubs that raise if
  they are ever used. The functions that compute the method run exactly as written.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_DIR = REPO_ROOT / "original-repos"


class UpstreamMismatch(RuntimeError):
    """The checkout is not the pinned code."""


@dataclass(frozen=True)
class UpstreamSpec:
    name: str
    url: str
    commit: str
    #: Repo-relative path -> sha256 of every file whose code the adapter runs.
    files: dict[str, str]
    #: Packages to create empty, so their `__init__` is not executed.
    packages: tuple[str, ...]
    #: Module name -> attribute names that are stubbed out (raise on use).
    stubs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return UPSTREAM_DIR / self.name


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(spec: UpstreamSpec) -> None:
    root = spec.path
    if not root.is_dir():
        raise UpstreamMismatch(
            f"{root} not found. Clone {spec.url} there and check out {spec.commit}."
        )
    if (root / ".git").exists():
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if head != spec.commit:
            raise UpstreamMismatch(f"{root} is at {head}, pinned commit is {spec.commit}")
    for rel, expected in spec.files.items():
        got = _sha256(root / rel)
        if got != expected:
            raise UpstreamMismatch(f"{rel} sha256 {got} != pinned {expected}")


def _stub(module_name: str, attr: str):
    def unavailable(*args, **kwargs):
        raise RuntimeError(
            f"{module_name}.{attr} is stubbed: the adapter does not support this upstream "
            "code path"
        )

    return type(attr, (), {"__init__": unavailable}) if attr[:1].isupper() else unavailable


def load_upstream(spec: UpstreamSpec, modules: list[str]) -> dict[str, types.ModuleType]:
    """Verify the checkout, then import `modules` from it. Idempotent."""
    verify(spec)
    for pkg in spec.packages:
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__path__ = [str(spec.path / pkg.replace(".", "/"))]
            sys.modules[pkg] = module
    for module_name, attrs in spec.stubs.items():
        if module_name not in sys.modules:
            module = types.ModuleType(module_name)
            for attr in attrs:
                setattr(module, attr, _stub(module_name, attr))
            sys.modules[module_name] = module
    # No sys.path entry: submodules resolve through the parent packages' `__path__`, so
    # nothing else in the checkout becomes importable by accident.
    return {name: importlib.import_module(name) for name in modules}
