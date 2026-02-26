"""
Base class definitions for data objects
"""

import pickle
from dataclasses import fields, is_dataclass
from typing import get_origin, get_args, Union, get_type_hints, Dict
from functools import lru_cache
import pprint
import structlog

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

    def to_log_dict(self):
        """
        Like to_dict(), but ensures dict KEYS are also encoded (and stringified if needed),
        so logs are always readable and don't rely on nested __repr__.
        """
        return self._log_encode(self)

    def to_log_str(self, *, width: int = 120, sort_dicts: bool = True) -> str:
        return pprint.pformat(self.to_log_dict(), width=width, sort_dicts=sort_dicts)

    def log_debug(
        self,
        *,
        tick: str | None = None,
        prefix: str | None = None,
        logger=None,
    ) -> None:
        logger = logger or structlog.get_logger(type(self).__module__)
        body = self.to_log_str()
        log_kwargs = {"tick": tick} if tick else {}
        if prefix:
            logger.debug(prefix, body=body, **log_kwargs)
        else:
            logger.debug("Serializable debug", body=body, **log_kwargs)

    @staticmethod
    def _log_encode(value):
        """
        Log-safe encoder:
        - expands nested Serializables/dataclasses to dicts
        - encodes list/tuple/set recursively
        - encodes dict keys too (stringifies non-primitive keys)
        - shows pickled bytes as a short tag instead of raw bytes spam
        """
        # primitives
        if isinstance(value, PRIMITIVES):
            return value

        # bytes (your pickle fallback output) -> don't dump raw bytes into logs
        if isinstance(value, (bytes, bytearray)):
            return f"<pickled:{len(value)} bytes>"

        # nested dataclass / Serializable
        if is_dataclass(value):
            # use your normal serialization path
            d = value.to_dict() if hasattr(value, "to_dict") else {f.name: getattr(value, f.name) for f in fields(value)}
            return Serializable._log_encode(d)

        # list-like
        if isinstance(value, (list, tuple, set)):
            return [Serializable._log_encode(v) for v in value]

        # dict: encode keys AND values
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                kk = Serializable._log_encode(k)
                # keys must be hashable; if we turned it into a dict/list, stringify it
                if not isinstance(kk, (str, int, float, bool, type(None))):
                    kk = pprint.pformat(kk, width=80, sort_dicts=True)
                out[kk] = Serializable._log_encode(v)
            return out

        # fallback: for unknown objects, show something readable
        return str(value)
