"""Backend worker interface shared by numerical solver adapters.

The contract exposes only the operations Solver uses: solve, retrieve submitted
clique-system duals, inspect statistics, and release resources.
"""

from math import isfinite

from rhiza.diagnostics import SolveError


################################################################################
# Shared numerical conversion
################################################################################


def restore_objective_value(value, *, scale, offset, backend):
    """Undo SDPR's exact affine objective normalization.

    Backends solve the centered, scaled certificate problem because this keeps
    solver tolerances independent of benign objective offsets and units. Public
    reports use the original canonical objective convention, so both primal and
    dual values are mapped back through the same positive scale and offset.

    Inputs:
        value:
            Float or None. Normalized backend objective.
        scale:
            Float. Positive objective scale.
        offset:
            Float. Original objective offset.
        backend:
            String. Backend name used in an error message.

    Outputs:
        restored:
            Float or None. Objective in the original canonical units.
    """
    if value is None:
        return None
    restored = float(scale)*float(value) + float(offset)
    if not isfinite(restored):
        raise SolveError(
            f"{backend} returned an objective that cannot be restored to "
            "the original scale."
        )
    return float(restored)


################################################################################
# Backend worker
################################################################################


class BackendWorker:
    """Define the runtime interface owned by a Solver.

    Concrete workers keep native tasks, settings, and status conventions inside
    their adapters while exposing the one dual vector used for moment extraction.
    A worker may release its native workspace after solving while remaining usable
    through copied Rhiza-owned report and dual data.
    """

    @property
    def built(self):
        """Return whether the backend worker remains usable."""
        raise NotImplementedError

    @property
    def closed(self):
        """Return whether the worker was explicitly closed."""
        raise NotImplementedError

    @property
    def stats(self):
        """Return backend build and solve statistics."""
        raise NotImplementedError

    def solve(self, *, verbose=False):
        """Solve the native backend task.

        Inputs:
            verbose:
                Boolean. Whether to stream backend progress.

        Outputs:
            report:
                Report. Normalized solution information.
        """
        raise NotImplementedError

    def get_system_duals(self):
        """Return submitted clique-system duals after a successful solve.

        Outputs:
            system_duals:
                ndarray. Concatenated submitted equation duals.
        """
        raise NotImplementedError

    def close(self):
        """Close the worker and discard retained solution data.

        Outputs:
            changed:
                Boolean. True when the worker changed to its closed state.
        """
        raise NotImplementedError
