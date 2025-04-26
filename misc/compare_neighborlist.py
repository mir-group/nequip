import argparse
from ase.io import iread
from nequip.utils.test import compare_neighborlists


def main():
    parser = argparse.ArgumentParser(
        description="Compares neighborlist implementations (ase, matscipy, vesin) on an ASE-readable file."
    )

    parser.add_argument(
        "-fname",
        help="name of input structure file in a format ASE knows (e.g. with an explicit extension)",
    )

    parser.add_argument(
        "-nl1",
        help="first neighborlist method",
    )

    parser.add_argument(
        "-nl2",
        help="second neighborlist method",
    )

    parser.add_argument(
        "-r_max",
        help="cutoff radius",
    )
    args = parser.parse_args()

    nl_kwargs = {"r_max": float(getattr(args, "r_max"))}
    for atoms in iread(getattr(args, "fname")):
        compare_neighborlists(
            atoms_or_data=atoms,
            nl1=getattr(args, "nl1"),
            nl2=getattr(args, "nl2"),
            **nl_kwargs,
        )


if __name__ == "__main__":
    main()
