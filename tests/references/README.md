# Independent corrected C reference

B-all is generated from frozen C 9f6713b686805645da459e99522e2049e7dea793,
using the two isolated patches for backend-consistent gravity and a 50-digit
principal Lambert-W NFW inverse. Its separate reference process imports neither
the product nor ITAMAE. Source/data/environment/configuration/patch hashes and
all parameters are in the sidecar; full patches and the immutable export runner
are maintained in sashimi-family/validation/references/sashimi-c.

The current product is compared at rtol 5e-12 across the full catalog; this is
separate from the preserved historical consistent observable golden at 5e-10.
No existing fixture was regenerated or relabeled. The reference is validation
input, not a product runtime dependency.
