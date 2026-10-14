"""Installed entry point for explicit, provenance-bearing boost tables."""

from ._boost import generate_boost_tables, generation_cli


def main():
    r"""Generate the explicitly configured prompt-cusp annihilation hierarchy tables.

    Notes
    -----
    The prompt-cusp prescription is Ando et al., Sections 2.1-2.3. Its sharp-k mapping
    alpha=1.8 is a large-mass top-hat matching choice, explicitly not a direct microhalo-
    simulation calibration.

    This entry point generates recursive annihilation tables from an explicit JSON
    grid/configuration and records their physical assumptions, source identity and file
    hashes. Existing result directories are not overwritten.

    References
    ----------

    .. [cite-prompt] Ando, Moro-Gonzalez & Li (2026), Prompt cusps in hierarchical dark matter halos: Implications for annihilation boost, Sections 2.1-2.3.
        https://arxiv.org/abs/2601.19863
    """
    generation_cli(prompt_cusps=True)


__all__ = ["generate_boost_tables", "main"]


if __name__ == "__main__":
    main()
