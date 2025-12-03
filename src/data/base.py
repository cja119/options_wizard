"""
Base class definitions for data objects
"""

import pickle
from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Union, get_type_hints, Dict
from functools import lru_cache

PRIMITIVES = (int, float, str, bool, type(None))


class Serializable:
    def to_dict(self):
        out = {}
        for f in fields(self):
            out[f.name] = self._encode(getattr(self, f.name))
        return out

    @staticmethod
    @lru_cache(maxsize=None)
    def cached_hints(tp):
        return get_type_hints(tp)

    @classmethod
    def from_dict(cls, d):
        hints = Serializable.cached_hints(cls)  # resolves forward refs into real types
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
        if value is None:
            return None

        target = Serializable._unwrap_type(expected_type)

        # reconstruct Serializable subclass; allow subclasses to provide a hook
        if target is not None and issubclass(target, Serializable):
            hook = getattr(target, "from_serialized_dict", None)
            if callable(hook):
                return hook(value)
            return target.from_dict(value)

        # primitive
        if isinstance(value, PRIMITIVES):
            return value

        # list
        if isinstance(value, list):
            inner = None
            if hasattr(expected_type, "__args__"):
                origin = get_origin(expected_type)
                args = get_args(expected_type)
                if origin is list and args:
                    inner = args[0]
            return [Serializable._decode(v, inner) for v in value]

        # dict
        if isinstance(value, dict):
            inner = None
            origin = get_origin(expected_type)
            args = get_args(expected_type)
            if origin in (dict, Dict) and len(args) == 2:
                inner = args[1]  # dict[key_type, value_type]
            elif origin is Union:
                for arg in args:
                    if arg is type(None):
                        continue
                    arg_origin = get_origin(arg)
                    arg_args = get_args(arg)
                    if arg_origin in (dict, Dict) and len(arg_args) == 2:
                        inner = arg_args[1]
                        break
            return {k: Serializable._decode(v, inner) for k, v in value.items()}

        # fallback pickle
        try:
            return pickle.loads(value)
        except Exception:
            return value
