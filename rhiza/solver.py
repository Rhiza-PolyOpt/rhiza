"""Backend-independent solution and extraction lifecycle.

Solver owns one native worker and presents normalized reports, resource
management, and moment queries through a common public interface.
"""

import numpy as np

from rhiza.backends.mosek import build_mosek
from rhiza.backends.scs import build_scs
from rhiza.diagnostics import (
    PipelineError,
    SolveError,
    format_summary,
    to_python_report,
)
from rhiza.tools.parsing import cast_exact_int, cast_real_scalar
from rhiza.tools.extraction import (
    extract_clique_moments,
    extract_moment_diagnostics,
    extract_means,
    extract_moments,
    reconstruct_clique_functionals,
    cast_clique_ids,
    cast_moment_spec,
    cast_variable_names,
)


################################################################################
# Solver
################################################################################


class Solver:
    """Build, solve, and extract one finalized Rhiza SDPR. It is the public
    runtime boundary that selects a backend, normalizes solve reports, and
    exposes moment extraction."""

    def __init__(
        self,
        sdpr,
        *,
        backend='SCS',
        gap_tol=None,
        feas_tol=None,
        iter_limit=None,
        config=None,
    ):
        """Build one native worker for a finalized SDPR.

        Backend selection and generic tolerances are frozen here so solving and
        extraction always refer to the same native problem.

        Inputs:
            sdpr:
                SDPR. Finalized relaxation recipe.
            backend:
                String. Backend name.
            gap_tol:
                Float or None. Requested relative primal-dual gap.
            feas_tol:
                Float or None. Requested feasibility accuracy.
            iter_limit:
                Integer or None. Backend iteration limit.
            config:
                Mapping or None. Native backend settings.
        """
        # Normalize the backend before resolving its accuracy scale. A first-
        # order SCS solve and an interior-point MOSEK solve need different useful
        # defaults even though they share one public tolerance vocabulary.
        backend = str(backend).upper()
        default_gap_tols = {
            'SCS': 1e-5,
            'MOSEK': 1e-8,
        }
        if backend not in default_gap_tols:
            raise SolveError("Backend must be 'SCS' or 'MOSEK'.")
        gap_tol = (
            default_gap_tols[backend]
            if gap_tol is None
            else cast_real_scalar(gap_tol, "gap_tol", SolveError)
        )
        feas_tol = (
            gap_tol
            if feas_tol is None
            else cast_real_scalar(feas_tol, "feas_tol", SolveError)
        )
        iter_limit = (
            None if iter_limit is None
            else cast_exact_int(iter_limit, "iter_limit", SolveError)
        )
        try:
            config = {} if config is None else dict(config)
        except (TypeError, ValueError) as error:
            raise SolveError("config must be a mapping.") from error

        if not getattr(sdpr, 'finalized', False):
            raise PipelineError(
                "Finalize the SDPR before constructing a Solver."
            )
        if not np.isfinite(gap_tol) or gap_tol < 0.0:
            raise SolveError("gap_tol must be finite and nonnegative.")
        if not np.isfinite(feas_tol) or feas_tol < 0.0:
            raise SolveError("feas_tol must be finite and nonnegative.")
        if iter_limit is not None and iter_limit <= 0:
            raise SolveError("iter_limit must be positive.")

        # Store only validated public settings before constructing the native
        # worker; a failed build therefore never leaves a partially built Solver.
        self._sdpr = sdpr
        self._backend = backend
        self._gap_tol = gap_tol
        self._feas_tol = feas_tol
        self._iter_limit = iter_limit
        self._config = config
        self._worker = self._build_worker()
        self._solved = False
        self._closed = False
        self._solve_report = None
        self._clique_functionals_cache = None
        self._refresh_status_stats()

    @property
    def backend(self):
        """Return the fixed backend name. The native worker is built during Solver
        construction, so this setting is read-only for the worker's lifetime."""
        return self._backend

    @property
    def gap_tol(self):
        """Return the fixed relative-gap tolerance supplied to the backend. The
        value remains synchronized with both the native worker and solve reports."""
        return self._gap_tol

    @property
    def feas_tol(self):
        """Return the fixed feasibility tolerance supplied to the backend.

        Moment extraction uses the same accuracy scale as an absolute floor for
        deciding whether a clique functional can be normalized safely.
        """
        return self._feas_tol

    @property
    def iter_limit(self):
        """Return the fixed optional backend iteration limit. ``None`` leaves the
        native solver's iteration policy in effect."""
        return self._iter_limit

    @property
    def config(self):
        """Return a copy of the fixed native backend configuration. Returning a copy
        prevents caller mutation from making the public settings disagree with the
        worker that was already built."""
        return dict(self._config)

    @property
    def built(self):
        """Return whether the backend worker remains usable."""
        return self._worker is not None and self._worker.built

    @property
    def solved(self):
        """Return whether a solve report is available."""
        return self._solved

    @property
    def status(self):
        """Return a copy of qualitative Solver status fields."""
        return to_python_report(self._status)

    @property
    def stats(self):
        """Return a copy of numerical Solver statistics."""
        stats = dict(self._stats)
        for name in ('native', 'factorization', 'memory', 'settings'):
            if name in stats:
                stats[name] = dict(stats[name])
        return to_python_report(stats)

    ################################################################################
    # Solve and extraction
    ################################################################################

    def solve(self, *, verbose=False):
        """Solve the selected native backend task once.

        Inputs:
            verbose:
                Boolean. Whether to stream backend progress.

        Outputs:
            report:
                Report. Normalized solution information.
        """
        verbose = bool(verbose)

        if self._closed:
            raise PipelineError("A closed Solver cannot be solved.")
        if self._solved:
            report = dict(self._solve_report)
            report['changed'] = False
            return to_python_report(report)

        # Commit solved state only after the worker returns a complete normalized
        # report. Backend exceptions leave the Solver retryable and unsolved.
        report = self._worker.solve(verbose=verbose)
        self._solved = True
        self._solve_report = dict(report)
        self._refresh_status_stats()
        return to_python_report(report)

    def extract_means(self, *, variables=None):
        """Extract stitched normalized first moments from clique functionals.

        Inputs:
            variables:
                Sequence or None. Variable names to extract.

        Outputs:
            means:
                dict. Variable names mapped to stitched means.
        """
        variables = (
            None if variables is None else cast_variable_names(variables)
        )

        self._require_solved()
        return extract_means(
            self._sdpr,
            self._clique_functionals(),
            variables=variables,
        )

    def extract_moments(self, moments):
        """Extract stitched normalized moments from clique functionals.

        Inputs:
            moments:
                MomentSpec. Named variable-exponent mappings.

        Outputs:
            moments_extracted:
                dict. Requested names mapped to stitched moments.
        """
        moments = cast_moment_spec(moments)

        self._require_solved()
        return extract_moments(
            self._sdpr,
            self._clique_functionals(),
            moments,
        )

    def extract_clique_moments(self, moments, *, cliques=None):
        """Extract normalized moments separately for containing cliques.

        Inputs:
            moments:
                MomentSpec. Named variable-exponent mappings.
            cliques:
                Sequence or None. Clique IDs to include.

        Outputs:
            moments_by_clique:
                dict. Clique IDs mapped to requested clique moments.
        """
        moments = cast_moment_spec(moments)
        cliques = (
            None if cliques is None else cast_clique_ids(cliques)
        )

        self._require_solved()
        return extract_clique_moments(
            self._sdpr,
            self._clique_functionals(),
            moments,
            cliques=cliques,
        )

    def extract_moment_diagnostics(self, *, variables=None):
        """Report clique-functional conditioning and shared-moment agreement.

        Inputs:
            variables:
                Sequence or None. Variables to compare.

        Outputs:
            report:
                Report. Mass, cancellation, and disagreement diagnostics.
        """
        variables = (
            None if variables is None else cast_variable_names(variables)
        )

        self._require_solved()
        return extract_moment_diagnostics(
            self._sdpr,
            self._clique_functionals(),
            variables=variables,
        )

    ################################################################################
    # Resource lifecycle and inspection
    ################################################################################

    def close(self):
        """Close the backend worker and discard cached clique functionals.

        Outputs:
            changed:
                Boolean. True when the worker changed to its closed state.
        """
        if self._closed:
            return False
        changed = self._worker.close()
        self._clique_functionals_cache = None
        self._closed = True
        self._refresh_status_stats()
        return changed

    def __enter__(self):
        """Return the Solver for context-manager use. Context-manager use pairs
        construction with guaranteed native resource release even when solving
        or extraction raises."""
        return self

    def __exit__(self, exception_type, exception, traceback):
        """Close the Solver without suppressing exceptions. Returning false
        preserves the original exception after native resources have been
        released."""
        self.close()
        return False

    def __str__(self):
        """Return a readable Solver summary. The summary is meant for interactive
        inspection and therefore uses the same status and statistics fields
        exposed programmatically."""
        return format_summary('Rhiza Solver', self._status, self._stats)

    def __repr__(self):
        """Return a concise developer representation. The representation
        highlights identity and lifecycle information without expanding large
        sampled arrays."""
        return (
            f"Solver(backend='{self.backend}', built={self.built}, "
            f"solved={self._solved}, closed={self._closed})"
        )

    ################################################################################
    # Internal lifecycle helpers
    ################################################################################

    def _build_worker(self):
        """Build the native worker for the selected backend.

        Outputs:
            worker:
                BackendWorker. Built native backend adapter.
        """
        # Backend modules expose the same worker contract, keeping the public
        # Solver lifecycle independent of native task and cone representations.
        builders = {
            'MOSEK': build_mosek,
            'SCS': build_scs,
        }
        return builders[self.backend](
            self._sdpr,
            gap_tol=self.gap_tol,
            feas_tol=self.feas_tol,
            iter_limit=self.iter_limit,
            config=self._config,
        )

    def _require_solved(self):
        """Require a usable solution before moment extraction."""
        if self._closed:
            raise PipelineError("A closed Solver cannot extract moments.")
        if not self._solved:
            raise PipelineError("Solve the SDPR before extracting moments.")
        if not self._solve_report.get('success', False):
            raise SolveError("The backend did not return a usable solution.")

    def _clique_functionals(self):
        """Reconstruct and cache compact clique functionals for extraction.

        The backend dual vector is copied only on the first extraction call.
        Reversing row scaling, validating masses, and computing conditioning
        diagnostics then happen once. Later moment queries reuse only retained
        sample indexes and weights rather than dense vectors over every sample.

        Outputs:
            functionals:
                tuple. Compact clique functionals in clique-ID order.
        """
        if self._clique_functionals_cache is None:
            self._clique_functionals_cache = reconstruct_clique_functionals(
                self._sdpr,
                self._worker.get_system_duals(),
                mass_abs_tol=self._mass_abs_tol(),
            )
        return self._clique_functionals_cache

    def _mass_abs_tol(self):
        """Derive the absolute floor for normalizing clique functionals.

        The backend feasibility target controls how closely the unit-mass
        stationarity equation is solved. Cancellation roundoff is assessed
        separately from this absolute scale after each signed functional is
        reconstructed.

        Outputs:
            mass_abs_tol:
                Float. Absolute near-zero mass tolerance.
        """
        return max(100.0*np.finfo(float).eps, 10.0*self.feas_tol)

    def _refresh_status_stats(self):
        """Refresh cheap Solver status and statistics. Recomputing the derived
        fields in one place keeps summaries and programmatic diagnostics
        synchronized after state changes."""
        self._status = {
            'backend': self.backend,
            'built': self.built,
            'solved': self._solved,
            'closed': self._closed,
        }
        if self._solve_report is not None:
            self._status['solution_status'] = self._solve_report.get(
                'solution_status'
            )
            self._status['success'] = self._solve_report.get('success')
        # Re-read worker statistics after builds, solves, and closes so the
        # interactive summary always reflects the same lifecycle snapshot.
        self._stats = (
            dict(self._worker.stats)
            if self._worker is not None else {}
        )
