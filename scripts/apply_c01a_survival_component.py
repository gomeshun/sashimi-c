from pathlib import Path

path = Path("sashimi_c_itamae_migration.py")
text = path.read_text()

old_import = """from sashimi_c import (
    TidalStrippingSolver,
    halo_model,
    subhalo_observables,
    subhalo_properties,
)

_Base = TypeVar(\"_Base\", bound=type)
"""
new_import = """from sashimi_c import (
    TidalStrippingSolver,
    halo_model,
    subhalo_observables,
    subhalo_properties,
)
from sashimi_c_itamae_components import TruncationThresholdSurvival

_Base = TypeVar(\"_Base\", bound=type)
"""

old_survival = """        def survival(batch, initial, evolved, context):
            return evolved[\"c_t\"] > ct_th

        def columns(batch, initial, evolved, survival_masks, context):
"""
new_survival = """        survival_component = TruncationThresholdSurvival(ct_threshold=ct_th)

        def columns(batch, initial, evolved, survival_masks, context):
"""

old_pipeline = """            survival=survival,
"""
new_pipeline = """            survival=survival_component.select,
"""

for old, new, label in (
    (old_import, new_import, "component import"),
    (old_survival, new_survival, "survival closure"),
    (old_pipeline, new_pipeline, "pipeline survival callback"),
):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} match, found {count}")
    text = text.replace(old, new)

path.write_text(text)
