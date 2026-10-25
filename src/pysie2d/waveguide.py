"""Particles over an N-layer stack: what single particles and clusters share.

:class:`_StackResult` carries the stack branch of every observable a result
offers — far field, cross-sections, the near field inside and below the stack,
the incident field and the mode amplitudes — written once against a list of
boundaries, so :class:`~pysie2d.solver.ScatterResult` and
:class:`~pysie2d.cluster.ClusterScatterResult` cannot disagree (slab-spec §1.3).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import layered
from .background import HalfSpace, Stack, _effective
from .fields import (
    _layer_field,
    _stack_background_field,
    _stack_downward_cross_section,
    _stack_downward_far_field,
    _stack_planes,
    _upward_cross_sections,
    _upward_far_field,
)


class _Source:
    """A boundary tuple seen as the ``src`` geometry of the layered GEMMs."""

    def __init__(self, b: tuple) -> None:
        self.f, self.g, self.df, self.dg, self.delt = b[:5]
        self.n_pts = self.f.size


class _StackResult:
    """Stack behaviour shared by the single-particle and cluster results.

    The host provides ``background``, ``wavelength``, ``wnum_bg``, ``_pol``,
    ``_n_clad``, ``excitation``-style ``_excitation_name``, ``angle``,
    ``source`` and ``_boundaries()``.
    """

    background: HalfSpace | Stack | None
    wavelength: float
    angle: float
    source: tuple[float, float] | None

    @property
    def _bg(self) -> HalfSpace | Stack | None:
        """The background this result runs on after the T6 normalisation."""
        return _effective(self.background, self._n_clad, self.wavelength)

    def _stack_rel(self) -> layered.StackRel:
        assert isinstance(self.background, Stack)
        return self.background.eps_rel(self._n_clad, self.wavelength)

    def _stack_r(self) -> Callable[[np.ndarray], np.ndarray]:
        lay, d, e_bot, _ = self._stack_rel()
        k, pol = self.wnum_bg, self._pol
        return lambda q: layered.stack_r(q, k, lay, d, e_bot, pol)

    def _stack_far_field(self, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
        """Upward amplitude with the stack's R, about the origin."""
        _, _, e_bot, z_top = self._stack_rel()
        return _upward_far_field(
            n_angles,
            self._boundaries(),
            self.wnum_bg,
            e_bot,
            self._pol,
            z_top,
            r=self._stack_r(),
        )

    def _stack_cross_sections(self, n_angles: int) -> dict[str, float]:
        """``c_sca_up``, ``c_abs`` and ``c_sca_down`` (real λ only, D18)."""
        rel = self._stack_rel()
        c = _upward_cross_sections(
            n_angles,
            self._boundaries(),
            self.wnum_bg,
            rel[2],
            self._pol,
            rel[3],
            r=self._stack_r(),
        )
        if np.imag(self.wavelength) == 0.0:
            c["c_sca_down"] = _stack_downward_cross_section(
                self._boundaries(), self.wnum_bg, rel, self._pol
            )
        return c

    def _stack_far_field_down(self, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
        """``a_down`` below the stack; ``NoFarFieldError`` for a lossy bottom."""
        return _stack_downward_far_field(
            n_angles, self._boundaries(), self.wnum_bg, self._stack_rel(), self._pol
        )

    def _stack_eval_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Scattered field over a stack: cover, layers and bottom (R19).

        Cover: the host's representation with the stack's reflection. Layers:
        :func:`~pysie2d.fields._layer_field`. Bottom: the transmitted blocks
        with the stack's τ. 1-D arrays.

        Raises:
            ValueError: A point on an interface plane.
        """
        rel = self._stack_rel()
        planes = _stack_planes(rel)
        if np.any(np.isin(z, planes)):
            raise ValueError(
                "observation point exactly on an interface plane of the stack: "
                "step off it (the field is continuous there)"
            )
        above, below = z > planes[0], z < planes[-1]
        inside = ~above & ~below
        field = np.zeros(x.size, dtype=complex)
        if above.any():
            field[above] = self._cover_field(x[above], z[above])
        b = self._boundaries()
        if inside.any():
            field[inside] = _layer_field(
                x[inside], z[inside], b, self.wnum_bg, self._pol, rel
            )
        if below.any():
            tr = layered.Transmission.build_stack(
                self._pol,
                self.wnum_bg,
                rel,
                np.concatenate([bi[0] for bi in b]),
                np.concatenate([bi[1] for bi in b]),
                x[below],
                z[below],
            )
            pts = layered.Points(x[below], z[below])
            for bi in b:
                m1, m2 = tr.blocks(pts, _Source(bi))
                nn = bi[0].size
                field[below] -= m1 @ bi[5][:nn] + m2 @ bi[5][nn:]
        return field

    def _stack_cover_reflection(
        self, x: np.ndarray, z: np.ndarray
    ) -> layered.StackReflection:
        """Reflected context for the boundary nodes and cover points ``(x, z)``."""
        b = self._boundaries()
        refl = layered.StackReflection.build(
            self._pol,
            self.wnum_bg,
            self._stack_rel(),
            np.concatenate([bi[0] for bi in b] + [x]),
            np.concatenate([bi[1] for bi in b] + [z]),
        )
        assert refl is not None
        return refl

    def _stack_background(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Unmasked background field over the stack."""
        return _stack_background_field(
            x,
            z,
            self.wnum_bg,
            self._pol,
            self._excitation_name,
            self.angle,
            self.source,
            self._stack_rel(),
        )
