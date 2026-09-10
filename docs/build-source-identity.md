# Source identity when rebuilding distributions

A source archive retains its embedded revision even when unpacked inside an
unrelated Git repository. An export without embedded revision does not borrow
the enclosing repository's HEAD. A recognized source checkout must have its
own Git root exactly at the package root.

Explicit build revision variables remain available for controlled exports,
but malformed values and values that conflict with the archive/checkout fail.
The installed runtime continues to use the embedded provenance without reading
these environment overrides. A source-archive-to-wheel rebuild requires no
revision injection.

Five new boundary cases failed before this change; all seven source identity
cases now pass. This addresses build identity only, with no physical changes.
Final release evidence must still build from clean pinned sources and record
artifact hashes; the build identity does not certify an uncommitted worktree.
