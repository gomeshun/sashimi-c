"""Installed entry point for explicit, provenance-bearing boost tables."""

from ._boost import generate_boost_tables, generation_cli


def main():
    """Read the requested grid and compute the existing observable formulas."""
    generation_cli(prompt_cusps=True)


__all__ = ["generate_boost_tables", "main"]


if __name__ == "__main__":
    main()
