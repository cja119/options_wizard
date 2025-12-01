"""
Base class definitions for data objects
"""

import pickle
from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Union, get_type_hints

PRIMITIVES = (int, float, str, bool, type(None))


class Serializable:
    def to_dict(self):
        out = {}
        for f in fields(self):
            out[f.name] = self._encode(getattr(self, f.name))
        return out

    @classmethod
    def from_dict(cls, d):
        hints = get_type_hints(cls)  # resolves forward refs into real types
        kwargs = {}
        for f in fields(cls):
            expected = hints.get(f.name, f.type)
            kwargs[f.name] = cls._decode(d.get(f.name), expected)
        return cls(**kwargs)

    @staticmethod
    def _encode(value):
        if isinstance(value, PRIMITIVES):
            return value
        if is_dataclass(value):
            # nested Serializable
            return value.to_dict()
        if isinstance(value, list):
            return [Serializable._encode(v) for v in value]
        if isinstance(value, dict):
            return {k: Serializable._encode(v) for k, v in value.items()}
        return pickle.dumps(value)

    @staticmethod
    def _unwrap_type(t):
        """Extract concrete Serializable type from Union[T, None] or direct type."""
        if isinstance(t, type):
            return t

        origin = get_origin(t)
        if origin is Union:
            for arg in get_args(t):
                if arg is not type(None) and isinstance(arg, type):
                    return arg
        return None

    @staticmethod
    def _decode(value, expected_type):
        target = Serializable._unwrap_type(expected_type)

        # reconstruct Serializable subclass
        if target is not None and issubclass(target, Serializable):
            # value may be a string, dict, or primitive
            return target.from_dict(value)

        # primitive
        if isinstance(value, PRIMITIVES):
            return value

        # list
        if isinstance(value, list):
            inner = None
            if hasattr(expected_type, "__args__"):
                inner = expected_type.__args__[0]
            return [Serializable._decode(v, inner) for v in value]

        # dict
        if isinstance(value, dict):
            inner = None
            if hasattr(expected_type, "__args__") and len(expected_type.__args__) == 2:
                inner = expected_type.__args__[1]   # dict[str, VALUE_TYPE]
            return {k: Serializable._decode(v, inner) for k, v in value.items()}

        # fallback pickle
        try:
            return pickle.loads(value)
        except Exception:
            return value
