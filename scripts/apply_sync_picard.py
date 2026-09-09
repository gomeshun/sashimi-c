from pathlib import Path


def insert_after(lines, needle, additions, label):
    try:
        index = lines.index(needle)
    except ValueError as exc:
        raise RuntimeError(f"missing expected line for {label}: {needle!r}") from exc
    lines[index + 1:index + 1] = additions


def replace_line(lines, old, new, label):
    try:
        index = lines.index(old)
    except ValueError as exc:
        raise RuntimeError(f"missing expected line for {label}: {old!r}") from exc
    lines[index] = new


def patch_sashimi_c():
    path = Path("sashimi_c.py")
    lines = path.read_text().splitlines()

    insert_after(
        lines,
        "from numpy.polynomial.hermite import hermgauss",
        ["from picard_tidal_stripping import PicardTidalStrippingTable"],
        "Picard import",
    )
    insert_after(
        lines,
        "        self.n_z_interp  = n_z_interp",
        ["        self._picard_tables = {}"],
        "Picard cache initialization",
    )
    insert_after(
        lines,
        "        self._M0 = value",
        [
            "        if hasattr(self, \"_picard_tables\"):",
            "            self._picard_tables.clear()",
        ],
        "Picard cache invalidation",
    )
    try:
        index = lines.index("    def msolve(self,m, z):")
    except ValueError as exc:
        raise RuntimeError("missing msolve insertion point") from exc
    helper = [
        "    def _get_picard_table(self, z_final):",
        "        \"\"\"Return a cached x3 Picard table for the requested final redshift.\"\"\"",
        "        key = float(z_final)",
        "        table = self._picard_tables.get(key)",
        "        if table is None:",
        "            table = PicardTidalStrippingTable(self, z_final=key)",
        "            self._picard_tables[key] = table",
        "        return table",
        "",
        "",
        "    def subhalo_mass_stripped_picard_table(self, ma, za, z):",
        "        \"\"\"Calculate tidal mass loss with the precomputed Picard table.\"\"\"",
        "        return self._get_picard_table(z).mass(ma, za)",
        "",
        "",
    ]
    lines[index:index] = helper

    try:
        index = lines.index("        match method:")
    except ValueError as exc:
        raise RuntimeError("missing stripping dispatch") from exc
    lines[index + 1:index + 1] = [
        "            case \"picard_table\":",
        "                return self.subhalo_mass_stripped_picard_table(ma,za,z)",
    ]

    ode_doc = "            - \"odeint\" : use odeint to solve the differential equation."
    insertions = 0
    index = 0
    while index < len(lines):
        if lines[index] == ode_doc:
            lines.insert(
                index,
                "            - \"picard_table\" : use the precomputed third-order Picard table.",
            )
            insertions += 1
            index += 1
        index += 1
    if insertions == 0:
        raise RuntimeError("missing stripping method documentation")

    path.write_text("\n".join(lines) + "\n")


def patch_migration():
    path = Path("sashimi_c_itamae_migration.py")
    lines = path.read_text().splitlines()

    insert_after(
        lines,
        "_STRIPPING_METHODS = (",
        ["    \"picard_table\","],
        "migration Picard method registration",
    )

    try:
        index = lines.index("        metadata = build_migration_metadata(")
    except ValueError as exc:
        raise RuntimeError("missing metadata construction") from exc
    block = [
        "        picard_table_settings = None",
        "        if method == \"picard_table\":",
        "            table = solver._get_picard_table(redshift)",
        "            picard_table_settings = {",
        "                \"n_iterations\": int(table.n_iterations),",
        "                \"n_z_acc\": int(table.n_z_acc),",
        "                \"n_log_ratio\": int(table.n_log_ratio),",
        "                \"log10_ratio_min\": float(table.log10_ratio_min),",
        "                \"log10_ratio_max\": float(table.log10_ratio_max),",
        "                \"n_integration\": int(table.n_integration),",
        "            }",
        "",
    ]
    lines[index:index] = block

    insert_after(
        lines,
        "                \"default_stripping_method\": _DEFAULT_STRIPPING_METHOD,",
        ["                \"picard_table_settings\": picard_table_settings,"],
        "Picard provenance metadata",
    )
    path.write_text("\n".join(lines) + "\n")


def patch_pyproject():
    path = Path("pyproject.toml")
    lines = path.read_text().splitlines()
    insert_after(
        lines,
        '  "prompt_cusps",',
        ['  "picard_tidal_stripping",'],
        "wheel module list",
    )
    path.write_text("\n".join(lines) + "\n")


def patch_readme():
    path = Path("README.md")
    text = path.read_text()
    heading = "### Tidal-stripping solver compatibility"
    if heading in text:
        return
    section = """

### Tidal-stripping solver compatibility

The historical and migration defaults remain `method=\"pert2_shanks\"`. The
precomputed third-order Picard solver synchronized from `main` is available as
an explicit `method=\"picard_table\"` option. This synchronization does **not**
adopt the proposed default change from PR #5; changing the public scientific
default requires separate PHY-C review.

Historical migration fixtures continue to use repository revision
`9f6713b686805645da459e99522e2049e7dea793`, `method=\"pert2_shanks\"`,
`ct_th=0.0`, and `physics_mode=\"legacy\"` for strict reproduction. Catalog
metadata records the selected stripping method and, for Picard runs, the table
iteration/grid settings.
"""
    path.write_text(text.rstrip() + section + "\n")


if __name__ == "__main__":
    patch_sashimi_c()
    patch_migration()
    patch_pyproject()
    patch_readme()
