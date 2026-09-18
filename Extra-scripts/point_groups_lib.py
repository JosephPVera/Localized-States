#!/usr/bin/env python3
# Written by Joseph P.Vera
# 2026-08

"""
Models molecular symmetry point groups together with:

  1. Their irreducible representations (irreps), e.g.
         C_3v -> {A_1, A_2, E}
         T_d  -> {A_1, A_2, E, T_1, T_2}

  2. The symmetry operations (conjugacy classes) of each group and the
     corresponding character table, e.g. T_d:

         table = {
             "A1": [1, 1, 1, 1, 1],
             "A2": [1, 1, 1, -1, -1],
             "E":  [2, -1, 2, 0, 0],
             "T1": [3, 0, -1, 1, -1],
             "T2": [3, 0, -1, -1, 1],
         }
         order = ["E", "C3", "C2", "S4", "sigma_d"]

     That same pattern (classes + characters per irrep) is applied here
     to all the other point groups.

Data compiled from standard character tables (see e.g.
https://www.webqc.org/symmetry.php).

Convention for E, T, G, H (degenerate representations)
-----------------------------------------------------------
For the NON-abelian groups (C_nv, D_n, D_nh, D_nd, T_d, O_h, I_h, ...)
the E, T, G, H representations are genuine irreps of dimension > 1; the
sum of (dimension)^2 over all irreps exactly matches the order of the
group.

For the purely ABELIAN cyclic groups (C_n with n>=3, and also T) the
"E" that appears here is the "combined real" form used in practice (the
same one your script already uses for C_3v/T_d): it is the sum of a
pair of complex-conjugate 1D irreps, presented as a single row with
real characters 2*cos(k*2*pi/n). This is the standard convention used
in spectroscopy/vibrational analysis (Atkins, Cotton), and that is why
for those groups the sum of (dimension)^2 does NOT reproduce the order
of the group. It is a practical simplification, not a strict
irreducible decomposition.

Contents
---------
- PointGroup            : a point group + its irreps + classes + characters
- PointGroupTable       : the full catalog, with lookup/normalization
- create_point_group    : convenience factory function
- get_irreps            : returns just the list of irreps
- get_classes           : returns just the list of classes (operations)
- get_character_table   : returns the irrep -> characters dict
- list_point_groups     : lists all available groups
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple, Union

# Irrational constants used in groups with C5/C10 (phi) and C8/C4 (sqrt2),
# C12 (sqrt3) -- computed instead of hand-typing decimals.
PHI = (1 + 5 ** 0.5) / 2          # 2*cos(36 deg)  ~ 1.618033988749895
IPHI = PHI - 1                    # 2*cos(72 deg)  ~ 0.618033988749895 (=1/phi)
SQRT2 = 2 ** 0.5                  # ~ 1.4142135623730951
SQRT3 = 3 ** 0.5                  # ~ 1.7320508075688772

Number = Union[int, float]
ClassSpec = Tuple[str, Union[int, str]]   # (class_label, multiplicity)

# Main class: a concrete point group
@dataclass
class PointGroup:
    """
    Represents a point group: its name, its symmetry operations
    (conjugacy classes, with multiplicity) and the character table of
    each irreducible representation.
    """

    name: str
    classes: List[ClassSpec] = field(default_factory=list)
    characters: Dict[str, List[Number]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validates that the character table is consistent: each irrep must
        have exactly one character per declared class. This prevents
        silent typo errors when extending the catalog by hand.
        """
        n_classes = len(self.classes)
        for irrep, chars in self.characters.items():
            if len(chars) != n_classes:
                raise ValueError(
                    f"Inconsistent character table in '{self.name}': "
                    f"irrep '{irrep}' has {len(chars)} characters, "
                    f"but the group declares {n_classes} classes."
                )

    # irreps is derived from characters, preserves insertion order
    @property
    def irreps(self) -> List[str]:
        return list(self.characters.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self.irreps)

    def __len__(self) -> int:
        return len(self.irreps)

    def __contains__(self, irrep: str) -> bool:
        return irrep in self.characters

    def __repr__(self) -> str:
        return f"{self.name}: {{{', '.join(self.irreps)}}}"

    @property
    def class_labels(self) -> List[str]:
        """Just the class labels, in order, e.g. ['E','C3','C2',...]."""
        return [label for label, _ in self.classes]

    @property
    def order(self):
        """Order of the group (total number of operations). None if continuous (C_inf_v, D_inf_h)."""
        total = 0
        for _, mult in self.classes:
            if isinstance(mult, str):
                return None  # continuous group (infinite multiplicity)
            total += mult
        return total

    def character(self, irrep: str) -> List[Number]:
        """Characters of a given irrep, in the same order as class_labels."""
        return self.characters[irrep]

    def dimension(self, irrep: str) -> Number:
        """
        Dimension of the irrep: its character under the identity
        operation E (by convention, the first class in the table).
        """
        return self.characters[irrep][0]

    def reduce_representation(self, chi_reducible: List[Number]) -> Dict[str, Number]:
        """
        Reduces a reducible representation into its irreducible
        components using the standard reduction formula:

            n_i = (1/h) * sum_c  g_c * chi_red(c) * chi_i(c)

        where h is the order of the group, g_c the multiplicity (size)
        of class c, chi_red(c) the character of the reducible
        representation in that class, and chi_i(c) the character of
        irrep i in that class.

        chi_reducible must have one value per class, in the same order
        as class_labels/classes.

        Returns a dict {irrep: n_i}. Values very close to an integer
        (within 1e-6, due to floating-point rounding errors) are
        returned as int.
        """
        h = self.order
        if h is None:
            raise ValueError(
                f"Cannot reduce a representation in a continuous group ({self.name})."
            )
        if len(chi_reducible) != len(self.classes):
            raise ValueError(
                f"chi_reducible must have {len(self.classes)} values (one per class), "
                f"got {len(chi_reducible)}."
            )

        multiplicities = [mult for _, mult in self.classes]
        result: Dict[str, Number] = {}
        for irrep, chars in self.characters.items():
            n_i = sum(
                g * xr * xi for g, xr, xi in zip(multiplicities, chi_reducible, chars)
            ) / h
            rounded = round(n_i)
            result[irrep] = rounded if abs(n_i - rounded) < 1e-6 else n_i
        return result

    def print_character_table(self) -> None:
        """
        Prints the character table in the same style used in
        irr-ps.py (columns = classes, rows = irreps).
        """
        header = f"{'':6s}" + "".join(f"{lbl:>10s}" for lbl in self.class_labels)
        print(f"\nCharacter table of {self.name}")
        print(header)
        for irrep, chars in self.characters.items():
            row = f"{irrep:6s}" + "".join(f"{c:>10.3f}" if isinstance(c, float)
                                           else f"{c:>10}" for c in chars)
            print(row)


# Point group catalog
class PointGroupTable:
    """
    Static catalog: point group name -> (classes, character table).
    Use the classmethods instead of touching _DATA directly.
    """

    _DATA: Dict[str, Dict] = {

        # ================= Non-axial groups =================
        "C_1": {
            "classes": [("E", 1)],
            "characters": {"A": [1]},
        },
        "C_s": {
            "classes": [("E", 1), ("sigma_h", 1)],
            "characters": {
                "A'":  [1, 1],
                "A''": [1, -1],
            },
        },
        "C_i": {
            "classes": [("E", 1), ("i", 1)],
            "characters": {
                "Ag": [1, 1],
                "Au": [1, -1],
            },
        },

        # ================= C_n =================
        "C_2": {
            "classes": [("E", 1), ("C2", 1)],
            "characters": {"A": [1, 1], "B": [1, -1]},
        },
        "C_3": {
            "classes": [("E", 1), ("C3", 2)],
            "characters": {"A": [1, 1], "E": [2, -1]},
        },
        "C_4": {
            "classes": [("E", 1), ("C4", 2), ("C2", 1)],
            "characters": {
                "A": [1, 1, 1],
                "B": [1, -1, 1],
                "E": [2, 0, -2],
            },
        },
        "C_5": {
            "classes": [("E", 1), ("C5", 2), ("C5^2", 2)],
            "characters": {
                "A":  [1, 1, 1],
                "E1": [2, IPHI, -PHI],
                "E2": [2, -PHI, IPHI],
            },
        },
        "C_6": {
            "classes": [("E", 1), ("C6", 2), ("C3", 2), ("C2", 1)],
            "characters": {
                "A":  [1, 1, 1, 1],
                "B":  [1, -1, 1, -1],
                "E1": [2, 1, -1, -2],
                "E2": [2, -1, -1, 2],
            },
        },

        # ================= C_nv =================
        "C_2v": {
            "classes": [("E", 1), ("C2", 1), ("sigma_v(xz)", 1), ("sigma_v'(yz)", 1)],
            "characters": {
                "A1": [1, 1, 1, 1],
                "A2": [1, 1, -1, -1],
                "B1": [1, -1, 1, -1],
                "B2": [1, -1, -1, 1],
            },
        },
        "C_3v": {
            "classes": [("E", 1), ("C3", 2), ("sigma_v", 3)],
            "characters": {
                "A1": [1, 1, 1],
                "A2": [1, 1, -1],
                "E":  [2, -1, 0],
            },
        },
        "C_4v": {
            "classes": [("E", 1), ("C4", 2), ("C2", 1), ("sigma_v", 2), ("sigma_d", 2)],
            "characters": {
                "A1": [1, 1, 1, 1, 1],
                "A2": [1, 1, 1, -1, -1],
                "B1": [1, -1, 1, 1, -1],
                "B2": [1, -1, 1, -1, 1],
                "E":  [2, 0, -2, 0, 0],
            },
        },
        "C_5v": {
            "classes": [("E", 1), ("C5", 2), ("C5^2", 2), ("sigma_v", 5)],
            "characters": {
                "A1": [1, 1, 1, 1],
                "A2": [1, 1, 1, -1],
                "E1": [2, IPHI, -PHI, 0],
                "E2": [2, -PHI, IPHI, 0],
            },
        },
        "C_6v": {
            "classes": [("E", 1), ("C6", 2), ("C3", 2), ("C2", 1), ("sigma_v", 3), ("sigma_d", 3)],
            "characters": {
                "A1": [1, 1, 1, 1, 1, 1],
                "A2": [1, 1, 1, 1, -1, -1],
                "B1": [1, -1, 1, -1, 1, -1],
                "B2": [1, -1, 1, -1, -1, 1],
                "E1": [2, 1, -1, -2, 0, 0],
                "E2": [2, -1, -1, 2, 0, 0],
            },
        },

        # ================= C_nh =================
        "C_2h": {
            "classes": [("E", 1), ("C2", 1), ("i", 1), ("sigma_h", 1)],
            "characters": {
                "Ag": [1, 1, 1, 1],
                "Bg": [1, -1, 1, -1],
                "Au": [1, 1, -1, -1],
                "Bu": [1, -1, -1, 1],
            },
        },
        "C_3h": {
            "classes": [("E", 1), ("C3", 2), ("sigma_h", 1), ("S3", 2)],
            "characters": {
                "A'":  [1, 1, 1, 1],
                "E'":  [2, -1, 2, -1],
                "A''": [1, 1, -1, -1],
                "E''": [2, -1, -2, 1],
            },
        },
        "C_4h": {
            "classes": [("E", 1), ("C4", 2), ("C2", 1), ("i", 1), ("S4", 2), ("sigma_h", 1)],
            "characters": {
                "Ag": [1, 1, 1, 1, 1, 1],
                "Bg": [1, -1, 1, 1, -1, 1],
                "Eg": [2, 0, -2, 2, 0, -2],
                "Au": [1, 1, 1, -1, -1, -1],
                "Bu": [1, -1, 1, -1, 1, -1],
                "Eu": [2, 0, -2, -2, 0, 2],
            },
        },
        "C_5h": {
            "classes": [("E", 1), ("C5", 2), ("C5^2", 2), ("sigma_h", 1), ("S5", 2), ("S5^3", 2)],
            "characters": {
                "A'":   [1, 1, 1, 1, 1, 1],
                "E1'":  [2, IPHI, -PHI, 2, IPHI, -PHI],
                "E2'":  [2, -PHI, IPHI, 2, -PHI, IPHI],
                "A''":  [1, 1, 1, -1, -1, -1],
                "E1''": [2, IPHI, -PHI, -2, -IPHI, PHI],
                "E2''": [2, -PHI, IPHI, -2, PHI, -IPHI],
            },
        },
        "C_6h": {
            "classes": [("E", 1), ("C6", 2), ("C3", 2), ("C2", 1), ("i", 1),
                        ("S3", 2), ("S6", 2), ("sigma_h", 1)],
            "characters": {
                "Ag":  [1, 1, 1, 1, 1, 1, 1, 1],
                "Bg":  [1, -1, 1, -1, 1, -1, 1, -1],
                "E1g": [2, 1, -1, -2, 2, 1, -1, -2],
                "E2g": [2, -1, -1, 2, 2, -1, -1, 2],
                "Au":  [1, 1, 1, 1, -1, -1, -1, -1],
                "Bu":  [1, -1, 1, -1, -1, 1, -1, 1],
                "E1u": [2, 1, -1, -2, -2, -1, 1, 2],
                "E2u": [2, -1, -1, 2, -2, 1, 1, -2],
            },
        },

        # ================= D_n =================
        "D_2": {
            "classes": [("E", 1), ("C2(z)", 1), ("C2(y)", 1), ("C2(x)", 1)],
            "characters": {
                "A":  [1, 1, 1, 1],
                "B1": [1, 1, -1, -1],
                "B2": [1, -1, 1, -1],
                "B3": [1, -1, -1, 1],
            },
        },
        "D_3": {
            "classes": [("E", 1), ("C3", 2), ("C2", 3)],
            "characters": {
                "A1": [1, 1, 1],
                "A2": [1, 1, -1],
                "E":  [2, -1, 0],
            },
        },
        "D_4": {
            "classes": [("E", 1), ("C4", 2), ("C2", 1), ("C2'", 2), ("C2''", 2)],
            "characters": {
                "A1": [1, 1, 1, 1, 1],
                "A2": [1, 1, 1, -1, -1],
                "B1": [1, -1, 1, 1, -1],
                "B2": [1, -1, 1, -1, 1],
                "E":  [2, 0, -2, 0, 0],
            },
        },
        "D_5": {
            "classes": [("E", 1), ("C5", 2), ("C5^2", 2), ("C2", 5)],
            "characters": {
                "A1": [1, 1, 1, 1],
                "A2": [1, 1, 1, -1],
                "E1": [2, IPHI, -PHI, 0],
                "E2": [2, -PHI, IPHI, 0],
            },
        },
        "D_6": {
            "classes": [("E", 1), ("C6", 2), ("C3", 2), ("C2", 1), ("C2'", 3), ("C2''", 3)],
            "characters": {
                "A1": [1, 1, 1, 1, 1, 1],
                "A2": [1, 1, 1, 1, -1, -1],
                "B1": [1, -1, 1, -1, 1, -1],
                "B2": [1, -1, 1, -1, -1, 1],
                "E1": [2, 1, -1, -2, 0, 0],
                "E2": [2, -1, -1, 2, 0, 0],
            },
        },

        # ================= D_nh =================
        "D_2h": {
            "classes": [("E", 1), ("C2(z)", 1), ("C2(y)", 1), ("C2(x)", 1),
                        ("i", 1), ("sigma(xy)", 1), ("sigma(xz)", 1), ("sigma(yz)", 1)],
            "characters": {
                "Ag":  [1, 1, 1, 1, 1, 1, 1, 1],
                "B1g": [1, 1, -1, -1, 1, 1, -1, -1],
                "B2g": [1, -1, 1, -1, 1, -1, 1, -1],
                "B3g": [1, -1, -1, 1, 1, -1, -1, 1],
                "Au":  [1, 1, 1, 1, -1, -1, -1, -1],
                "B1u": [1, 1, -1, -1, -1, -1, 1, 1],
                "B2u": [1, -1, 1, -1, -1, 1, -1, 1],
                "B3u": [1, -1, -1, 1, -1, 1, 1, -1],
            },
        },
        "D_3h": {
            "classes": [("E", 1), ("C3", 2), ("C2", 3), ("sigma_h", 1), ("S3", 2), ("sigma_v", 3)],
            "characters": {
                "A1'":  [1, 1, 1, 1, 1, 1],
                "A2'":  [1, 1, -1, 1, 1, -1],
                "E'":   [2, -1, 0, 2, -1, 0],
                "A1''": [1, 1, 1, -1, -1, -1],
                "A2''": [1, 1, -1, -1, -1, 1],
                "E''":  [2, -1, 0, -2, 1, 0],
            },
        },
        "D_4h": {
            "classes": [("E", 1), ("C4", 2), ("C2", 1), ("C2'", 2), ("C2''", 2),
                        ("i", 1), ("S4", 2), ("sigma_h", 1), ("sigma_v", 2), ("sigma_d", 2)],
            "characters": {
                "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "A2g": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
                "B1g": [1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
                "B2g": [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
                "Eg":  [2, 0, -2, 0, 0, 2, 0, -2, 0, 0],
                "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                "A2u": [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
                "B1u": [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
                "B2u": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                "Eu":  [2, 0, -2, 0, 0, -2, 0, 2, 0, 0],
            },
        },
        "D_5h": {
            "classes": [("E", 1), ("C5", 2), ("C5^2", 2), ("C2", 5),
                        ("sigma_h", 1), ("S5", 2), ("S5^3", 2), ("sigma_v", 5)],
            "characters": {
                "A1'":  [1, 1, 1, 1, 1, 1, 1, 1],
                "A2'":  [1, 1, 1, -1, 1, 1, 1, -1],
                "E1'":  [2, IPHI, -PHI, 0, 2, IPHI, -PHI, 0],
                "E2'":  [2, -PHI, IPHI, 0, 2, -PHI, IPHI, 0],
                "A1''": [1, 1, 1, 1, -1, -1, -1, -1],
                "A2''": [1, 1, 1, -1, -1, -1, -1, 1],
                "E1''": [2, IPHI, -PHI, 0, -2, -IPHI, PHI, 0],
                "E2''": [2, -PHI, IPHI, 0, -2, PHI, -IPHI, 0],
            },
        },
        "D_6h": {
            "classes": [("E", 1), ("C6", 2), ("C3", 2), ("C2", 1), ("C2'", 3), ("C2''", 3),
                        ("i", 1), ("S3", 2), ("S6", 2), ("sigma_h", 1), ("sigma_d", 3), ("sigma_v", 3)],
            "characters": {
                "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "A2g": [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                "B1g": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                "B2g": [1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
                "E1g": [2, 1, -1, -2, 0, 0, 2, 1, -1, -2, 0, 0],
                "E2g": [2, -1, -1, 2, 0, 0, 2, -1, -1, 2, 0, 0],
                "A1u": [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
                "A2u": [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
                "B1u": [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1],
                "B2u": [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1],
                "E1u": [2, 1, -1, -2, 0, 0, -2, -1, 1, 2, 0, 0],
                "E2u": [2, -1, -1, 2, 0, 0, -2, 1, 1, -2, 0, 0],
            },
        },

        # ================= D_nd =================
        "D_2d": {
            "classes": [("E", 1), ("S4", 2), ("C2", 1), ("C2'", 2), ("sigma_d", 2)],
            "characters": {
                "A1": [1, 1, 1, 1, 1],
                "A2": [1, 1, 1, -1, -1],
                "B1": [1, -1, 1, 1, -1],
                "B2": [1, -1, 1, -1, 1],
                "E":  [2, 0, -2, 0, 0],
            },
        },
        "D_3d": {
            "classes": [("E", 1), ("C3", 2), ("C2", 3), ("i", 1), ("S6", 2), ("sigma_d", 3)],
            "characters": {
                "A1g": [1, 1, 1, 1, 1, 1],
                "A2g": [1, 1, -1, 1, 1, -1],
                "Eg":  [2, -1, 0, 2, -1, 0],
                "A1u": [1, 1, 1, -1, -1, -1],
                "A2u": [1, 1, -1, -1, -1, 1],
                "Eu":  [2, -1, 0, -2, 1, 0],
            },
        },
        "D_4d": {
            "classes": [("E", 1), ("S8", 2), ("C4", 2), ("S8^3", 2), ("C2", 1),
                        ("C2'", 4), ("sigma_d", 4)],
            "characters": {
                "A1": [1, 1, 1, 1, 1, 1, 1],
                "A2": [1, 1, 1, 1, 1, -1, -1],
                "B1": [1, -1, 1, -1, 1, 1, -1],
                "B2": [1, -1, 1, -1, 1, -1, 1],
                "E1": [2, SQRT2, 0, -SQRT2, -2, 0, 0],
                "E2": [2, 0, -2, 0, 2, 0, 0],
                "E3": [2, -SQRT2, 0, SQRT2, -2, 0, 0],
            },
        },
        "D_5d": {
            "classes": [("E", 1), ("C5", 2), ("C5^2", 2), ("C2", 5),
                        ("i", 1), ("S10^3", 2), ("S10", 2), ("sigma_d", 5)],
            "characters": {
                "A1g": [1, 1, 1, 1, 1, 1, 1, 1],
                "A2g": [1, 1, 1, -1, 1, 1, 1, -1],
                "E1g": [2, IPHI, -PHI, 0, 2, IPHI, -PHI, 0],
                "E2g": [2, -PHI, IPHI, 0, 2, -PHI, IPHI, 0],
                "A1u": [1, 1, 1, 1, -1, -1, -1, -1],
                "A2u": [1, 1, 1, -1, -1, -1, -1, 1],
                "E1u": [2, IPHI, -PHI, 0, -2, -IPHI, PHI, 0],
                "E2u": [2, -PHI, IPHI, 0, -2, PHI, -IPHI, 0],
            },
        },
        "D_6d": {
            "classes": [("E", 1), ("S12", 2), ("C6", 2), ("S4", 2), ("C3", 2),
                        ("S12^5", 2), ("C2", 1), ("C2'", 6), ("sigma_d", 6)],
            "characters": {
                "A1": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "A2": [1, 1, 1, 1, 1, 1, 1, -1, -1],
                "B1": [1, -1, 1, -1, 1, -1, 1, 1, -1],
                "B2": [1, -1, 1, -1, 1, -1, 1, -1, 1],
                "E1": [2, SQRT3, 1, 0, -1, -SQRT3, -2, 0, 0],
                "E2": [2, 1, -1, -2, -1, 1, 2, 0, 0],
                "E3": [2, 0, -2, 0, 2, 0, -2, 0, 0],
                "E4": [2, -1, -1, 2, -1, -1, 2, 0, 0],
                "E5": [2, -SQRT3, 1, 0, -1, SQRT3, -2, 0, 0],
            },
        },

        # ================= S_n (n par) =================
        "S_4": {
            "classes": [("E", 1), ("S4", 2), ("C2", 1)],
            "characters": {
                "A": [1, 1, 1],
                "B": [1, -1, 1],
                "E": [2, 0, -2],
            },
        },
        "S_6": {
            "classes": [("E", 1), ("C3", 2), ("i", 1), ("S6", 2)],
            "characters": {
                "Ag": [1, 1, 1, 1],
                "Eg": [2, -1, 2, -1],
                "Au": [1, 1, -1, -1],
                "Eu": [2, -1, -2, 1],
            },
        },
        "S_8": {
            "classes": [("E", 1), ("S8", 2), ("C4", 2), ("S8^3", 2), ("C2", 1)],
            "characters": {
                "A":  [1, 1, 1, 1, 1],
                "B":  [1, -1, 1, -1, 1],
                "E1": [2, SQRT2, 0, -SQRT2, -2],
                "E2": [2, 0, -2, 0, 2],
                "E3": [2, -SQRT2, 0, SQRT2, -2],
            },
        },

        # ================= Cubic groups =================
        "T": {
            "classes": [("E", 1), ("C3", 4), ("C3^2", 4), ("C2", 3)],
            "characters": {
                "A": [1, 1, 1, 1],
                "E": [2, -1, -1, 2],
                "T": [3, 0, 0, -1],
            },
        },
        "T_d": {
            "classes": [("E", 1), ("C3", 8), ("C2", 3), ("S4", 6), ("sigma_d", 6)],
            "characters": {
                "A1": [1, 1, 1, 1, 1],
                "A2": [1, 1, 1, -1, -1],
                "E":  [2, -1, 2, 0, 0],
                "T1": [3, 0, -1, 1, -1],
                "T2": [3, 0, -1, -1, 1],
            },
        },
        "T_h": {
            "classes": [("E", 1), ("C3", 4), ("C3^2", 4), ("C2", 3),
                        ("i", 1), ("S6", 4), ("S6^5", 4), ("sigma_h", 3)],
            "characters": {
                "Ag": [1, 1, 1, 1, 1, 1, 1, 1],
                "Eg": [2, -1, -1, 2, 2, -1, -1, 2],
                "Tg": [3, 0, 0, -1, 3, 0, 0, -1],
                "Au": [1, 1, 1, 1, -1, -1, -1, -1],
                "Eu": [2, -1, -1, 2, -2, 1, 1, -2],
                "Tu": [3, 0, 0, -1, -3, 0, 0, 1],
            },
        },
        "O": {
            "classes": [("E", 1), ("C4", 6), ("C2", 3), ("C3", 8), ("C2'", 6)],
            "characters": {
                "A1": [1, 1, 1, 1, 1],
                "A2": [1, -1, 1, 1, -1],
                "E":  [2, 0, 2, -1, 0],
                "T1": [3, 1, -1, 0, -1],
                "T2": [3, -1, -1, 0, 1],
            },
        },
        "O_h": {
            "classes": [("E", 1), ("C3", 8), ("C2", 6), ("C4", 6), ("C2''", 3),
                        ("i", 1), ("S4", 6), ("S6", 8), ("sigma_h", 3), ("sigma_d", 6)],
            "characters": {
                "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "A2g": [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
                "Eg":  [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],
                "T1g": [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
                "T2g": [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],
                "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                "A2u": [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
                "Eu":  [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
                "T1u": [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
                "T2u": [3, 0, 1, -1, -1, -3, 1, 0, 1, -1],
            },
        },

        # ================= Icosahedral groups =================
        "I": {
            "classes": [("E", 1), ("C5", 12), ("C5^2", 12), ("C3", 20), ("C2", 15)],
            "characters": {
                "A":  [1, 1, 1, 1, 1],
                "T1": [3, PHI, IPHI, 0, -1],
                "T2": [3, IPHI, PHI, 0, -1],
                "G":  [4, -1, -1, 1, 0],
                "H":  [5, 0, 0, -1, 1],
            },
        },
        "I_h": {
            "classes": [("E", 1), ("C5", 12), ("C5^2", 12), ("C3", 20), ("C2", 15),
                        ("i", 1), ("S10", 12), ("S10^3", 12), ("S6", 20), ("sigma", 15)],
            "characters": {
                "Ag":  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "T1g": [3, PHI, IPHI, 0, -1, 3, PHI, IPHI, 0, -1],
                "T2g": [3, IPHI, PHI, 0, -1, 3, IPHI, PHI, 0, -1],
                "Gg":  [4, -1, -1, 1, 0, 4, -1, -1, 1, 0],
                "Hg":  [5, 0, 0, -1, 1, 5, 0, 0, -1, 1],
                "Au":  [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                "T1u": [3, PHI, IPHI, 0, -1, -3, -PHI, -IPHI, 0, 1],
                "T2u": [3, IPHI, PHI, 0, -1, -3, -IPHI, -PHI, 0, 1],
                "Gu":  [4, -1, -1, 1, 0, -4, 1, 1, -1, 0],
                "Hu":  [5, 0, 0, -1, 1, -5, 0, 0, 1, -1],
            },
        },

        # ================= Linear (continuous) groups =================
        # Multiplicity "inf" = infinitely many rotations/reflections of
        # the continuous axis; the characters depend on the angle phi
        # (formulas, not fixed numbers), as is usual for C_inf_v and D_inf_h.
        "C_inf_v": {
            "classes": [("E", 1), ("2C(phi)", "inf"), ("sigma_v", "inf")],
            "characters": {
                "A1(Sigma+)": [1, 1, 1],
                "A2(Sigma-)": [1, 1, -1],
                "E1(Pi)":     [2, "2cos(phi)", 0],
                "E2(Delta)":  [2, "2cos(2phi)", 0],
                "E3(Phi)":    [2, "2cos(3phi)", 0],
            },
        },
        "D_inf_h": {
            "classes": [("E", 1), ("2C(phi)", "inf"), ("sigma_v", "inf"),
                        ("i", 1), ("2S(phi)", "inf"), ("C2'", "inf")],
            "characters": {
                "Sigma_g+": [1, 1, 1, 1, 1, 1],
                "Sigma_g-": [1, 1, -1, 1, 1, -1],
                "Pi_g":     [2, "2cos(phi)", 0, 2, "-2cos(phi)", 0],
                "Delta_g":  [2, "2cos(2phi)", 0, 2, "2cos(2phi)", 0],
                "Sigma_u+": [1, 1, 1, -1, -1, -1],
                "Sigma_u-": [1, 1, -1, -1, -1, 1],
                "Pi_u":     [2, "2cos(phi)", 0, -2, "2cos(phi)", 0],
                "Delta_u":  [2, "2cos(2phi)", 0, -2, "-2cos(2phi)", 0],
            },
        },
    }

    # Normalization / lookup
    @staticmethod
    def _normalize(name: str) -> str:
        """
        Normalizes a user-given point group name to the internal
        format, e.g. "c3v" -> "C_3v", "Td" -> "T_d",
        "D4h" -> "D_4h".
        """
        raw = name.strip().replace(" ", "").replace("_", "")
        if not raw:
            raise ValueError("Empty point group name.")

        lower = raw.lower()
        if lower in ("cinfv", "cv", "cinfinityv"):
            return "C_inf_v"
        if lower in ("dinfh", "dinfinityh"):
            return "D_inf_h"

        letter = raw[0].upper()
        rest = raw[1:]

        if rest == "":
            return letter  # T, O, I

        digits = ""
        i = 0
        while i < len(rest) and rest[i].isdigit():
            digits += rest[i]
            i += 1
        suffix = rest[i:].lower()  # "v", "h", "d", ...

        if digits:
            return f"{letter}_{digits}{suffix}"
        return f"{letter}_{suffix}"  # Cs, Ci, Td, Th, Oh, Ih

    @classmethod
    def get(cls, name: str) -> PointGroup:
        """Returns a PointGroup object for the given group name."""
        key = cls._normalize(name)
        if key not in cls._DATA:
            raise KeyError(
                f"Unknown point group '{name}' (normalized to '{key}'). "
                f"Use list_point_groups() to see the available groups."
            )
        entry = cls._DATA[key]
        return PointGroup(
            name=key,
            classes=list(entry["classes"]),
            characters={irrep: list(chars) for irrep, chars in entry["characters"].items()},
        )

    @classmethod
    def list_groups(cls) -> List[str]:
        """Sorted list of all available point groups."""
        return sorted(cls._DATA.keys())

    @classmethod
    def search_by_irrep(cls, irrep: str) -> List[str]:
        """Point groups that contain an irrep with that label."""
        return [name for name, entry in cls._DATA.items() if irrep in entry["characters"]]

    @classmethod
    def add_group(cls, name: str, classes: List[ClassSpec],
                   characters: Dict[str, List[Number]]) -> None:
        """Registers (or overwrites) a point group in the catalog."""
        key = cls._normalize(name)
        cls._DATA[key] = {"classes": list(classes), "characters": dict(characters)}

# Convenience functions
def create_point_group(name: str) -> PointGroup:
    """Factory function: builds a PointGroup by name, e.g. 'C_3v'."""
    return PointGroupTable.get(name)


def get_irreps(name: str) -> List[str]:
    """Returns just the list of irreducible representations of a group."""
    return create_point_group(name).irreps


def get_classes(name: str) -> List[ClassSpec]:
    """Returns the symmetry operation classes (label, multiplicity)."""
    return create_point_group(name).classes


def get_character_table(name: str) -> Dict[str, List[Number]]:
    """Returns the {irrep: [characters...]} dict of a point group."""
    return create_point_group(name).characters


def list_point_groups() -> List[str]:
    """Returns the names of all point groups in the catalog."""
    return PointGroupTable.list_groups()

# self-test
if __name__ == "__main__":
    examples = ["C_3v", "T_d", "D_4h", "O_h", "C2v", "S8", "Ih"]

    print("Examples:")
    for ex in examples:
        pg = create_point_group(ex)
        print(f"  {ex:8s} -> {pg}  (order={pg.order})")

    print(f"\nTotal point groups in the catalog: {len(list_point_groups())}")
    print("All groups:", ", ".join(list_point_groups()))

    print("\nGroups that contain an 'E' irrep:")
    print(PointGroupTable.search_by_irrep("E"))

    # Full character table, same style as irr-ps.py
    create_point_group("T_d").print_character_table()
    create_point_group("C_3v").print_character_table()

    # Example of reducing a reducible representation in T_d,
    # typical of a vibrational mode analysis (Gamma_total).
    td = create_point_group("T_d")
    gamma_total = [15, 0, -1, -1, 3]  # characters in [E, C3, C2, S4, sigma_d]
    print("\nReduction of Gamma_total in T_d:")
    print(td.reduce_representation(gamma_total))
