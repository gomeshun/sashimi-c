"""Installed entry point for explicit, provenance-bearing boost tables."""

from ._boost import generate_boost_tables, generation_cli


def main():
    """Generate the explicitly configured annihilation hierarchy tables.

    Notes
    -----
    This entry point generates recursive annihilation tables from an explicit JSON
    grid/configuration and records their physical assumptions, source identity and file
    hashes. Existing result directories are not overwritten.
    """
    generation_cli(prompt_cusps=False)


__all__ = ["generate_boost_tables", "main"]


if __name__ == "__main__":
    main()
