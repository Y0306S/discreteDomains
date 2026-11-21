"""Protein sequence features used as interpretable experience roles."""

from __future__ import annotations

from typing import Dict, Sequence

HYDRO = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
POS = set("KRH")
NEG = set("DE")


def sequence_features(seq: str) -> Dict[str, float]:
    s = "".join(c for c in str(seq).upper() if c.isalpha())
    n = max(len(s), 1)
    hydro = sum(HYDRO.get(c, 0.0) for c in s) / n
    charge = sum(1 for c in s if c in POS) - sum(1 for c in s if c in NEG)
    cys = sum(1 for c in s if c == "C")
    arom = sum(1 for c in s if c in "FWY") / n
    return {
        "length": float(len(s)),
        "hydrophobicity": float(hydro),
        "charge": float(charge),
        "cys_count": float(cys),
        "aromatic_frac": float(arom),
    }


def length_bin(length: float) -> str:
    if length < 18:
        return "short"
    if length < 28:
        return "medium"
    return "long"


def charge_bin(charge: float) -> str:
    if charge <= -2:
        return "anionic"
    if charge >= 4:
        return "cationic"
    return "neutralish"


def extra_from_seq(seq: str) -> Dict[str, object]:
    feat = sequence_features(seq)
    return {
        **feat,
        "length_bin": length_bin(feat["length"]),
        "charge_bin": charge_bin(feat["charge"]),
        "cys_bin": "none" if feat["cys_count"] <= 0 else ("one" if feat["cys_count"] == 1 else "multi"),
    }


PROTEIN_ROLE_HINTS = (
    ("hydrophobicity", "hydrophobicity"),
    ("charge", "charge"),
    ("cys", "cysteine"),
    ("length", "sequence_length"),
    ("aromatic", "aromatic"),
    ("motif", "motif"),
    ("z_", "latent_coord"),
)
"""Protein sequence features used as interpretable experience roles."""
