# Spike: rough slab under beam illumination (scoping only)

**Status: parked, 6 Oct 2026.** This branch has no code. `slab-scoping.tex`
(compiled to `.pdf`, draft 1, 24 Sep 2026) works out how to extend the BIE to
a film bounded by two rough surfaces that extend to infinity, following
Maradudin et al., Ann. Phys. 203, 255 (1990).

Main points:
- Each surface contributes a copy of the existing 2×2 block, and the two
  surfaces are coupled through the slab. The building blocks are the ones the
  package already has.
- The new difficulty is truncation. Once roughness couples into guided modes,
  those modes carry power to the cut ends.
- Validation anchors: the flat slab (closed form, no guided-mode excitation)
  and a small-roughness perturbation result.

The near-field maps of elongated ellipses made for this study are what exposed
the interior-field defect (`../interior-field-fix.md`), which was fixed in v0.8.1.

The roadmap direction since then is half-space (v0.9) → multilayer via R(q)
(`v1.1-layered`). That route treats the background as layered and does not use
two rough truncated surfaces, so this note is background reading for it, not a
plan.
