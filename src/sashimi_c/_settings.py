"""CDM-owned settings; no physical schema or calibration is selected by ITAMAE."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from ._concentration import TabulatedConcentration

_DEFAULTS = {
    "accretion": {
        "prescription": "yang2011",
        "model": 3,
        "mass_nodes": 500,
        "redshift_step": 0.01,
        "host_history_nodes": 200,
    },
    "concentration": {
        "prescription": "correa2015",
        "relation": None,
        "scatter_dex": 0.128,
        "quadrature_nodes": 5,
    },
    "stripping": {
        "prescription": "cdm",
        "solver": "pert2_shanks",
        "interpolation_nodes": 64,
        "profile_change": True,
        "solver_options": {},
    },
    "disruption": {"prescription": "truncation", "ct_threshold": 0.0},
}
_SOLVERS = {
    "picard_table",
    "odeint",
    "pert0",
    "pert1",
    "pert2",
    "pert2_shanks",
    "pert3",
}


def thaw(value):
    """Return a detached, JSON-compatible view of a resolved specification."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def finite_scalar(value: Any, name: str, *, minimum=0.0, strict=False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not boolean.")
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real numeric scalar.")
    result = float(array)
    if not np.isfinite(result) or result < minimum or (strict and result == minimum):
        raise ValueError(
            f"{name} must be finite and {'greater than' if strict else 'at least'} {minimum}."
        )
    return result


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _solver_options(options, solver):
    if not isinstance(options, Mapping):
        raise TypeError("stripping.solver_options must be a mapping.")
    if solver != "odeint" and options:
        raise ValueError("solver_options are supported only by the odeint solver.")
    allowed = {
        "rtol",
        "atol",
        "h0",
        "hmax",
        "hmin",
        "mxstep",
        "mxhnil",
        "mxordn",
        "mxords",
    }
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported odeint options: {sorted(map(str, unknown))}.")
    resolved = {}
    for name, value in options.items():
        if name.startswith("mx"):
            resolved[name] = integer(
                value, name, 1 if name in {"mxordn", "mxords"} else 0
            )
            maximum = {"mxordn": 12, "mxords": 5}.get(name)
            if maximum is not None and resolved[name] > maximum:
                raise ValueError(f"{name} must not exceed {maximum}.")
        else:
            # h0 is signed: evolution is toward decreasing redshift.
            resolved[name] = finite_scalar(
                value,
                name,
                minimum=-np.inf if name == "h0" else 0.0,
                strict=name in {"rtol", "atol"},
            )
    if resolved.get("h0", 0.0) > 0:
        raise ValueError(
            "h0 must be nonpositive for evolution toward decreasing redshift."
        )
    if resolved.get("hmax", 0.0) > 0 and resolved.get("hmin", 0.0) > resolved["hmax"]:
        raise ValueError("hmin must not exceed a positive hmax.")
    return resolved


def resolve(previous=None, **overrides):
    values = thaw(_DEFAULTS if previous is None else previous)
    for group, changes in overrides.items():
        if group not in values:
            raise TypeError(f"Unknown settings group {group!r}.")
        if changes is None:
            continue
        if not isinstance(changes, Mapping):
            raise TypeError(f"{group} settings must be a mapping.")
        unknown = changes.keys() - values[group].keys()
        if unknown:
            raise ValueError(f"Unknown {group} settings: {sorted(map(str, unknown))}.")
        for key, value in changes.items():
            if key == "solver_options":
                if not isinstance(value, Mapping):
                    raise TypeError("solver_options must be a mapping.")
                # Empty mapping clears options; nonempty mappings partially override them.
                values[group][key] = {**values[group][key], **value} if value else {}
            else:
                values[group][key] = value
    for group in values:
        if (
            not isinstance(values[group]["prescription"], str)
            or values[group]["prescription"] != _DEFAULTS[group]["prescription"]
        ):
            raise ValueError(
                f"Unsupported {group} prescription: {values[group]['prescription']!r}."
            )
    a, c, s, d = (values[key] for key in _DEFAULTS)
    if c["relation"] is not None and type(c["relation"]) is not TabulatedConcentration:
        raise TypeError(
            "concentration.relation must be an immutable TabulatedConcentration or None."
        )
    a["model"] = integer(a["model"], "accretion.model")
    if a["model"] not in (1, 2, 3):
        raise ValueError("accretion.model must be 1, 2, or 3.")
    for group, key, minimum in (
        (a, "mass_nodes", 2),
        (a, "host_history_nodes", 1),
        (c, "quadrature_nodes", 1),
        (s, "interpolation_nodes", 2),
    ):
        group[key] = integer(group[key], key, minimum)
    a["redshift_step"] = finite_scalar(a["redshift_step"], "redshift_step", strict=True)
    c["scatter_dex"] = finite_scalar(c["scatter_dex"], "scatter_dex")
    d["ct_threshold"] = finite_scalar(d["ct_threshold"], "ct_threshold")
    if not isinstance(s["solver"], str) or s["solver"] not in _SOLVERS:
        raise ValueError(f"Unsupported stripping solver: {s['solver']!r}.")
    if not isinstance(s["profile_change"], (bool, np.bool_)):
        raise TypeError("profile_change must be boolean.")
    s["profile_change"] = bool(s["profile_change"])
    s["solver_options"] = _solver_options(s["solver_options"], s["solver"])
    return freeze(values)
