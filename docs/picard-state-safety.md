# Picard host-state safety

This narrow backport of [C PR #22](https://github.com/gomeshun/sashimi-c/pull/22)
([upstream 9f7559c](https://github.com/gomeshun/sashimi-c/commit/9f7559c0b192401620647f10e6383d2b4e7738cc))
adds host/background identity checks to the migration branch's existing table.
It does not adopt the upstream solver rewrite, interpolation, fallbacks or defaults.

An unchanged solver reuses its table. Changing tracked scalar settings, numeric
arrays, instance callables, cosmology adapter identity/scalar state, or the
provider's explicit identity clears the solver-owned table cache on the next
Picard request. A separately retained table instead raises a `ValueError`
containing `stale`; reconstruct it with the current host. Older serialized tables
without identity are likewise rejected, never silently relabelled.

The process-local fingerprint is not a persistent scientific identifier. Custom
providers with hidden state, mutable closures or nested dependencies must supply
`picard_physics_key()` returning an immutable, equality-comparable value covering
those dependencies. It must change whenever those inputs change. Arbitrary
external state is not automatically introspected. The default immutable
`NativeFlatLCDM` adapter is supported, as is replacing that adapter.

This only protects Picard-table reuse. It does not change the mutation semantics
of the separate perturbative epsilon interpolation caches. Construct a fresh
solver when changing physical settings for those methods.

The native C solver supplies this hook for its supported immutable concentration
relation. Its content-addressed identifier participates in the fingerprint;
replacing a relation with different data or switching to/from the standard
Correa relation invalidates retained tables. An equivalent immutable payload
keeps the same identity. Subclasses overriding the hook must include this base
identity along with any additional hidden dependencies.
