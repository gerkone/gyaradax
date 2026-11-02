"""Pytree containers: precomputed coefficients (:class:`GKPre`) and the run state (:class:`GKState`)."""

from collections.abc import ItemsView, KeysView
from typing import Any, Protocol
from dataclasses import dataclass

import jax
import jax.numpy as jnp


class Precompute(Protocol):
    """Read-only mapping interface shared by :class:`GKPre` and plain dicts.

    Values are heterogeneous: JAX arrays, nested dicts and static metadata
    (``nl_m*``, ``ixzero``, ``iyzero``, ``nsp``).
    """

    def __contains__(self, key: object) -> bool: ...

    def __getitem__(self, key: str) -> Any: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def items(self) -> ItemsView[str, Any]: ...

    def keys(self) -> KeysView[str]: ...


@jax.tree_util.register_pytree_node_class
class GKPre:
    """Precomputed coefficients, as returned by :func:`gyaradax.operators.precompute.linear_precompute`.

    A dict-like pytree: arrays (also inside nested dicts) are leaves, while
    FFT sizes (``nl_m*``), ``ixzero``, ``iyzero``, ``nsp`` and other non-array
    values are static auxiliary data, so they stay Python ints under ``jit``.
    """

    def __init__(self, items: dict[str, Any]) -> None:
        self._items: dict[str, Any] = items

    def tree_flatten(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        leaves = []
        leaf_keys = []
        aux = {}
        for k, v in self._items.items():
            if k.startswith("nl_m") or k in ("ixzero", "iyzero", "nsp"):
                aux[k] = v
            elif isinstance(v, dict):
                # flatten dict values into leaves so traced arrays stay out of aux
                for dk, dv in sorted(v.items()):
                    leaves.append(dv)
                    leaf_keys.append(f"{k}.{dk}")
            elif hasattr(v, "shape") and hasattr(v, "dtype"):
                # jax arrays AND abstract ShapeDtypeStruct (jax.eval_shape)
                leaves.append(v)
                leaf_keys.append(k)
            else:
                aux[k] = v
        return tuple(leaves), {"leaf_keys": tuple(leaf_keys), "aux": aux}

    @classmethod
    def tree_unflatten(cls, metadata: dict[str, Any], leaves: tuple[Any, ...]) -> "GKPre":
        items: dict[str, Any] = {}
        for key, val in zip(metadata["leaf_keys"], leaves):
            if "." in key:
                parent, child = key.split(".", 1)
                if parent not in items:
                    items[parent] = {}
                items[parent][child] = val
            else:
                items[key] = val
        items.update(metadata["aux"])
        return cls(items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._items.get(key, default)

    def items(self) -> ItemsView[str, Any]:
        return self._items.items()

    def keys(self) -> KeysView[str]:
        return self._items.keys()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKState:
    """Time-stepping state carried between blocks.

    Attributes:
        time: Simulation time.
        step: Step counter.
        accumulated_norm_factor: Product of the per-ky renormalisation factors of linear runs.
        window_start_amp: Per-ky mode amplitude at the start of the growth-rate window.
        last_growth_rate: Per-ky growth rate of the last completed window.
    """

    time: jnp.ndarray
    step: jnp.ndarray
    accumulated_norm_factor: jnp.ndarray
    window_start_amp: jnp.ndarray
    last_growth_rate: jnp.ndarray

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], None]:
        return tuple(vars(self).values()), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, leaves: tuple[jnp.ndarray, ...]) -> "GKState":
        return cls(*leaves)
