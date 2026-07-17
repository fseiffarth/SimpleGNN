"""Stable canonical hashing of node-label vocabularies (spec 18, Part A).

Every label producer can attach a per-label int64 hash computed from the
label's structural signature alone, so that the same structural role hashes
to the same value in any dataset. The hashes never touch the training hot
path; they are only used at dataset-boundary crossings (vocabulary build,
weight export/import).
"""
import hashlib
import struct

import numpy as np

# bump on ANY encoding change -> old vocabularies invalid
HASH_SCHEMA_VERSION = 1
# hash slot for the -1 invalid label
RESERVED_INVALID = -2 ** 63
# hash slot for the max_labels "other" bucket
RESERVED_CAPPED = -2 ** 63 + 1

_RESERVED_HASHES = frozenset((RESERVED_INVALID, RESERVED_CAPPED))


def _encode_part(part, out: bytearray) -> None:
    """Append a type-tagged, length-prefixed byte encoding of part to out."""
    if part is None:
        out += b'n'
    elif isinstance(part, bool):
        out += b'b\x01' if part else b'b\x00'
    elif isinstance(part, (int, np.integer)):
        value = int(part)
        encoded = value.to_bytes((value.bit_length() + 8) // 8, 'little', signed=True)
        out += b'i' + len(encoded).to_bytes(4, 'little') + encoded
    elif isinstance(part, float):
        out += b'f' + struct.pack('<d', part)
    elif isinstance(part, str):
        encoded = part.encode('utf-8')
        out += b's' + len(encoded).to_bytes(4, 'little') + encoded
    elif isinstance(part, bytes):
        out += b'y' + len(part).to_bytes(4, 'little') + part
    elif isinstance(part, (tuple, list)):
        out += b't' + len(part).to_bytes(4, 'little')
        for item in part:
            _encode_part(item, out)
    else:
        raise TypeError(f"stable_hash cannot encode values of type {type(part).__name__}")


def stable_hash(*parts) -> int:
    """
    blake2b(digest_size=8) over a length-prefixed, versioned byte encoding of
    parts (ints -> variable-length signed LE with length prefix, str -> utf-8,
    bytes passthrough, tuple/list -> recursive with type+length prefix).
    Returns the digest as a signed int64 bit-pattern. Deliberately NOT Python
    hash() (salted per process) and not repr() of unsorted containers.
    """
    out = bytearray()
    _encode_part(tuple(parts), out)
    digest = hashlib.blake2b(bytes(out), digest_size=8).digest()
    return int.from_bytes(digest, 'little', signed=True)


def hash_vocabulary(signatures: dict, label_kind: str, params: tuple) -> np.ndarray:
    """
    Map an original-label-id -> signature dict to an original-label-id -> int64
    hash array. Every hash commits to
    (HASH_SCHEMA_VERSION, label_kind, params, signature) so e.g. wl_3 can never
    alias wl_4. Raises on a within-vocabulary collision (two distinct
    signatures, same hash) -- exact detection, the signatures are in hand here.
    Ids in [0, max_id] that are absent from signatures get RESERVED_INVALID.
    """
    size = max(signatures) + 1 if signatures else 0
    hashes = np.full(size, RESERVED_INVALID, dtype=np.int64)
    signature_by_hash = {}
    for label_id, signature in signatures.items():
        if label_id < 0:
            raise ValueError(f"negative label id {label_id} in {label_kind} vocabulary")
        value = stable_hash(HASH_SCHEMA_VERSION, label_kind, params, signature)
        if value in _RESERVED_HASHES:
            raise ValueError(
                f"hash of {label_kind} signature {signature!r} collides with a reserved hash slot")
        previous = signature_by_hash.setdefault(value, signature)
        if previous != signature:
            raise ValueError(
                f"hash collision in {label_kind} vocabulary: signatures {previous!r} and "
                f"{signature!r} both map to {value}")
        hashes[label_id] = value
    return hashes
