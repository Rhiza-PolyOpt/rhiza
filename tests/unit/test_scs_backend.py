"""Unit tests for direct SCS assembly and solution normalization."""

from enum import Enum
import json

import numpy as np
import pytest

import rhiza as rz
import rhiza.backends.scs as scs_backend
from rhiza.backends.scs import (
    _assemble_scs_data,
    _normalize_solution_status,
    _resolve_settings,
    _svec_indices,
    build_scs,
)
from rhiza.diagnostics import SolutionStatus, SolveError
from tests.fixtures.reference_problem import make_reference_pop


################################################################################
# Test configuration
################################################################################


pytestmark = pytest.mark.scs


################################################################################
# Recording SCS module
################################################################################


class _RecordingSolver:
    """Record constructor data and return a deterministic SCS solution."""

    def __init__(self, module, data, cone, settings):
        """Store native inputs for inspection.

        Inputs:
            module:
                _RecordingSCS. Owning module-like object.
            data:
                Mapping. SCS problem data.
            cone:
                Mapping. SCS cone data.
            settings:
                Mapping. Native settings.
        """
        self.module = module
        self.data = data
        self.cone = cone
        self.settings = settings

    def solve(self):
        """Return deterministic primal and dual vectors."""
        num_variables = self.data['A'].shape[1]
        num_cone_rows = self.data['A'].shape[0]
        x = np.linspace(1.0, 2.0, num_variables)
        y = np.zeros(num_cone_rows)
        num_scalar = self.cone['z']
        y[:num_scalar] = -np.arange(1, num_scalar + 1)

        row_offset = num_scalar
        for size in self.cone['s']:
            matrix = np.eye(size)
            if size >= 2:
                matrix[1,0] = 0.25
                matrix[0,1] = 0.25
            values = []
            for column in range(size):
                for row in range(column, size):
                    value = matrix[row,column]
                    if row != column:
                        value *= np.sqrt(2.0)
                    values.append(value)
            stop = row_offset + len(values)
            y[row_offset:stop] = values
            row_offset = stop

        return {
            'x': x,
            'y': y,
            's': np.zeros(num_cone_rows),
            'info': {
                'status': 'solved',
                'status_val': self.module.SCS_SOLVED,
                'iter': 17,
                'pobj': 2.0,
                'dobj': 1.5,
                'res_pri': 1e-7,
                'res_dual': 2e-7,
                'gap': 0.5,
                'setup_time': 12.0,
                'solve_time': 34.0,
                'lin_sys_solver': 'sparse-direct-amd-qdldl',
                'scale': 0.2,
                'scale_updates': 3,
                'accepted_accel_steps': 4,
                'rejected_accel_steps': 1,
            },
        }


class _RecordingSCS:
    """Provide SCS constants and a recording workspace constructor."""

    SCS_SOLVED = 1
    SCS_SOLVED_INACCURATE = 2
    SCS_UNBOUNDED = -1
    SCS_INFEASIBLE = -2
    SCS_INDETERMINATE = -3
    SCS_FAILED = -4
    SCS_SIGINT = -5
    SCS_UNBOUNDED_INACCURATE = -6
    SCS_INFEASIBLE_INACCURATE = -7
    SCS_UNFINISHED = 0

    def __init__(self):
        """Create a module-like recording object."""
        self.last_solver = None

    def SCS(self, data, cone, **settings):
        """Create and retain one recording workspace."""
        self.last_solver = _RecordingSolver(self, data, cone, settings)
        return self.last_solver


class _EnumRecordingSCS(_RecordingSCS):
    """Accept the SCS 3.3 selector API and reject obsolete native keywords."""

    class LinearSolver(Enum):
        AUTO = 'auto'
        QDLDL = 'qdldl'
        CPU_INDIRECT = 'cpu_indirect'
        MKL = 'mkl'
        ACCELERATE = 'accelerate'
        CPU_DENSE = 'cpu_dense'
        GPU_INDIRECT = 'gpu_indirect'
        CUDSS = 'cudss'

    def SCS(self, data, cone, **settings):
        """Validate solver selection before recording native inputs."""
        for key in ('use_indirect', 'gpu', 'mkl', 'cudss'):
            if key in settings:
                raise TypeError("Obsolete SCS selector: " + key)
        self.LinearSolver(settings.get('linear_solver', 'auto'))
        return super().SCS(data, cone, **settings)


################################################################################
# Fixture construction
################################################################################


def _make_reference_sdpr():
    """Build one compressed reference relaxation.

    Outputs:
        sdpr:
            SDPR. Finalized reference relaxation.
    """
    pop = make_reference_pop(num_samples=80)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=4)
    sdpr.compress(method='PQR')
    sdpr.finalize()
    return sdpr


@pytest.fixture(scope='module')
def selector_sdpr():
    """Reuse an immutable relaxation for native setting and reporting checks."""
    return _make_reference_sdpr()


################################################################################
# Exact cone assembly
################################################################################


def test_scs_data_matches_sdpr_coefficient_maps():
    """Match scalar signs, PSD adjoints, row order, and objectives."""
    sdpr = _make_reference_sdpr()
    data, cone, assembly = _assemble_scs_data(sdpr)
    matrix = data['A'].toarray()

    assert data['A'].format == 'csc'
    assert data['A'].has_sorted_indices
    assert data['A'].shape == (
        assembly['num_cone_rows'],
        assembly['num_system_rows'],
    )
    assert cone['z'] == sdpr.stats['num_scalar_coeffs']
    assert cone['s'] == [
        gramian['gramian_size'] for gramian in sdpr._gramians
    ]
    np.testing.assert_allclose(
        data['b'][:cone['z']], sdpr._scalar_objective()
    )
    np.testing.assert_allclose(data['b'][cone['z']:], 0.0)

    scalar_expected = np.zeros((
        sdpr.stats['num_scalar_coeffs'],
        sdpr.stats['num_rows_retained'],
    ))
    psd_expected = np.zeros((
        assembly['num_psd_cone_rows'],
        sdpr.stats['num_rows_retained'],
    ))
    gramian_row_offsets = []
    row_offset = cone['z']
    for size in cone['s']:
        gramian_row_offsets.append(row_offset)
        row_offset += size*(size + 1)//2
    objective_expected = np.empty(sdpr.stats['num_rows_retained'])

    system_row_offset = 0
    for clique_id, system in enumerate(sdpr._clique_systems):
        num_rows = len(system['row_indices'])
        objective_expected[
            system_row_offset:system_row_offset + num_rows
        ] = sdpr._clique_objective_evals(clique_id)

        for chunk in sdpr._iter_scalar_coeff_chunks(
            clique_id, max_num_coeffs=10_000_000
        ):
            rows = chunk.scalar_indices
            columns = slice(
                system_row_offset, system_row_offset + num_rows
            )
            scalar_expected[rows,columns] = chunk.coeff_evals.T

        for chunk in sdpr._iter_gramian_coeff_chunks(
            clique_id, max_num_coeffs=10_000_000
        ):
            gramian = sdpr._gramians[chunk.gramian_id]
            size = gramian['gramian_size']
            offset = (
                gramian_row_offsets[chunk.gramian_id]
                - cone['z']
            )
            svec = _svec_indices(
                size, chunk.gramian_rows, chunk.gramian_columns
            )
            values = -chunk.coeff_evals.T.copy()
            off_diagonal = chunk.gramian_rows != chunk.gramian_columns
            values[off_diagonal,:] /= np.sqrt(2.0)
            columns = slice(
                system_row_offset, system_row_offset + num_rows
            )
            psd_expected[offset + svec,columns] = values

        system_row_offset += num_rows

    np.testing.assert_allclose(data['c'], objective_expected)
    np.testing.assert_allclose(matrix[:cone['z'],:], scalar_expected)
    np.testing.assert_allclose(matrix[cone['z']:,:], psd_expected)
    assert assembly['num_scalar_entries_emitted'] >= data['A'][:cone['z'],:].nnz
    assert assembly['num_gramian_entries_emitted'] >= data['A'][cone['z']:,:].nnz


def test_scs_assembly_never_materializes_a_complete_clique_system(monkeypatch):
    """Build SCS data only through bounded coefficient iterators."""
    sdpr = _make_reference_sdpr()

    def fail_materialization(clique_id):
        """Reject the forbidden dense backend path."""
        raise AssertionError("backend requested a full clique system")

    monkeypatch.setattr(
        sdpr,
        '_materialize_augmented_clique_system',
        fail_materialization,
    )
    data, _, _ = _assemble_scs_data(sdpr)
    assert data['A'].nnz > 0


################################################################################
# Worker lifecycle and solution orientation
################################################################################


def test_scs_worker_maps_dual_orientation_and_system_weights(monkeypatch):
    """Recover Rhiza objectives and the system duals used for extraction."""
    sdpr = _make_reference_sdpr()
    fake_scs = _RecordingSCS()
    monkeypatch.setattr(scs_backend, '_import_scs', lambda: fake_scs)
    worker = build_scs(
        sdpr,
        gap_tol=1e-5,
        feas_tol=2e-5,
        iter_limit=321,
        config={
            'normalize': False,
            'eps_abs': 9.0,
            'custom_setting': 7,
        },
    )

    report = worker.solve(verbose=True)
    assert report['success']
    assert report['solution_status'] == 'OPTIMAL'
    assert 'problem_status' not in report
    assert 'termination_status' not in report
    objective_scale = sdpr.stats['objective_scale']
    objective_offset = sdpr.stats['objective_offset']
    assert report['primal_objective'] == (
        objective_scale*1.5 + objective_offset
    )
    assert report['dual_objective'] == (
        objective_scale*2.0 + objective_offset
    )
    assert report['relative_gap'] == 0.25
    assert report['num_iterations'] == 17

    settings = fake_scs.last_solver.settings
    assert settings['use_indirect'] is False
    assert settings['normalize'] is False
    assert settings['adaptive_scale'] is True
    assert settings['eps_abs'] == 2e-5
    assert settings['eps_rel'] == 1e-5
    assert settings['max_iters'] == 321
    assert settings['verbose'] is True
    assert settings['custom_setting'] == 7

    expected_duals = np.linspace(
        1.0, 2.0, sdpr.stats['num_rows_retained']
    )
    np.testing.assert_allclose(
        worker.get_system_duals(), expected_duals
    )
    # Native y and s vectors are validated during solve but are not retained;
    # the backend contract exposes only the weights consumed by Solver.
    assert not hasattr(worker, '_solution')

    stats = worker.stats
    assert stats['native']['termination_status'] == 'solved'
    assert stats['native']['setup_time'] == 0.012
    assert stats['native']['solve_time'] == 0.034
    assert stats['native']['linear_system_mode'] == 'SPARSE_DIRECT'
    assert stats['native']['native_linear_system_solver'] == (
        'sparse-direct-amd-qdldl'
    )
    assert stats['num_matrix_nonzeros'] > 0
    assert worker.close()
    assert not worker.close()


def test_scs_solve_failure_leaves_assembled_data_available(monkeypatch):
    """Permit a clean retry when native optimization raises before commit."""
    sdpr = _make_reference_sdpr()
    fake_scs = _RecordingSCS()
    original_solve = _RecordingSolver.solve
    num_calls = {'solve': 0}

    def fail_once(self):
        """Raise on the first native solve and succeed on the second."""
        num_calls['solve'] += 1
        if num_calls['solve'] == 1:
            raise RuntimeError("temporary native failure")
        return original_solve(self)

    monkeypatch.setattr(scs_backend, '_import_scs', lambda: fake_scs)
    monkeypatch.setattr(_RecordingSolver, 'solve', fail_once)
    worker = build_scs(
        sdpr,
        gap_tol=1e-5,
        feas_tol=1e-5,
        iter_limit=None,
        config={},
    )

    with pytest.raises(SolveError, match="SCS optimization failed"):
        worker.solve()
    assert worker._data is None
    assert worker._cone is None
    assert worker._native_solver is not None
    assert not worker._solved

    report = worker.solve()
    assert report['success']
    assert worker._data is None
    assert worker._cone is None
    assert worker._native_solver is None
    assert worker.solve()['changed'] is False


def test_scs_solution_statuses_are_normalized_from_exit_flags():
    """Normalize documented SCS integer exit flags."""
    scs = _RecordingSCS()
    cases = {
        1: SolutionStatus.OPTIMAL,
        2: SolutionStatus.OPTIMAL_INACCURATE,
        -1: SolutionStatus.INFEASIBLE,
        -6: SolutionStatus.INFEASIBLE,
        -2: SolutionStatus.UNBOUNDED,
        -7: SolutionStatus.UNBOUNDED,
        0: SolutionStatus.UNKNOWN,
        -3: SolutionStatus.UNKNOWN,
        -4: SolutionStatus.SOLVER_ERROR,
        -5: SolutionStatus.SOLVER_ERROR,
    }
    for value, expected in cases.items():
        assert _normalize_solution_status(scs, value) is expected


################################################################################
# SCS vectorization and settings
################################################################################


def test_svec_indexing_follows_scs_column_major_order():
    """Follow SCS lower-triangle column-major PSD conventions."""
    rows = np.array([0, 1, 1, 2, 2, 2])
    columns = np.array([0, 0, 1, 0, 1, 2])
    indexes = _svec_indices(3, rows, columns)
    np.testing.assert_array_equal(indexes, [0, 1, 3, 2, 4, 5])


def test_public_scs_settings_override_native_config():
    """Give public tolerances, limits, and verbosity final precedence."""
    settings = _resolve_settings(
        {
            'eps_abs': 1.0,
            'eps_rel': 2.0,
            'max_iters': 3,
            'verbose': False,
            'use_indirect': True,
        },
        gap_tol=1e-5,
        feas_tol=2e-5,
        iter_limit=400,
        verbose=True,
    )
    assert settings['eps_abs'] == 2e-5
    assert settings['eps_rel'] == 1e-5
    assert settings['max_iters'] == 400
    assert settings['verbose'] is True
    assert settings['use_indirect'] is True


@pytest.mark.parametrize(
    'module_type', (_RecordingSCS, _EnumRecordingSCS), ids=('scs32', 'scs33')
)
@pytest.mark.parametrize('selectors, selection, mode', (
    ({}, 'qdldl', 'SPARSE_DIRECT'),
    ({'use_indirect': False}, 'qdldl', 'SPARSE_DIRECT'),
    ({'use_indirect': True}, 'cpu_indirect', 'SPARSE_INDIRECT'),
    ({'gpu': True, 'use_indirect': False}, 'cudss', 'GPU_DIRECT'),
    ({'gpu': True, 'use_indirect': True}, 'gpu_indirect', 'GPU_INDIRECT'),
    ({'mkl': True}, 'mkl', 'MKL_DIRECT'),
))
def test_scs_selector_compatibility_and_setting_precedence(
    monkeypatch, selector_sdpr, module_type, selectors, selection, mode
):
    """Preserve caller intent and public settings across native selector APIs."""
    if module_type is _EnumRecordingSCS and not selectors:
        selection, mode = 'auto', 'AUTO'
    fake_scs = module_type()
    monkeypatch.setattr(scs_backend, '_import_scs', lambda: fake_scs)
    config = dict(selectors)
    config.update({
        'normalize': False,
        'eps_abs': 9.0,
        'eps_rel': 8.0,
        'max_iters': 7,
        'verbose': False,
    })
    original_config = dict(config)
    worker = build_scs(
        selector_sdpr,
        gap_tol=1e-5,
        feas_tol=2e-5,
        iter_limit=321,
        config=config,
    )

    assert worker.solve(verbose=True)['success']
    settings = fake_scs.last_solver.settings
    if module_type is _EnumRecordingSCS:
        assert settings['linear_solver'] == selection
        assert not {'use_indirect', 'gpu', 'mkl', 'cudss'} & settings.keys()
    else:
        assert 'linear_solver' not in settings
        assert settings['use_indirect'] is selectors.get('use_indirect', False)
        for key, value in selectors.items():
            assert settings[key] == value
    assert settings['normalize'] is False
    assert settings['eps_abs'] == 2e-5
    assert settings['eps_rel'] == 1e-5
    assert settings['max_iters'] == 321
    assert settings['verbose'] is True
    assert config == original_config
    assert worker.stats['native']['linear_system_mode'] == mode
    assert json.loads(json.dumps(worker.stats))['settings'] == settings
    assert worker.close()


@pytest.mark.parametrize('as_enum', (False, True), ids=('string', 'enum'))
@pytest.mark.parametrize('selection, mode', (
    ('auto', 'AUTO'),
    ('qdldl', 'SPARSE_DIRECT'),
    ('cpu_indirect', 'SPARSE_INDIRECT'),
    ('mkl', 'MKL_DIRECT'),
    ('accelerate', 'ACCELERATE_DIRECT'),
    ('cpu_dense', 'DENSE_DIRECT'),
    ('gpu_indirect', 'GPU_INDIRECT'),
    ('cudss', 'GPU_DIRECT'),
))
def test_scs_explicit_solver_selection_is_reported_as_a_string(
    monkeypatch, selector_sdpr, as_enum, selection, mode
):
    """Accept native strings or enums without leaking enums into public reports."""
    fake_scs = _EnumRecordingSCS()
    monkeypatch.setattr(scs_backend, '_import_scs', lambda: fake_scs)
    value = fake_scs.LinearSolver(selection) if as_enum else selection
    config = {'linear_solver': value}
    worker = build_scs(
        selector_sdpr,
        gap_tol=1e-5,
        feas_tol=1e-5,
        iter_limit=None,
        config=config,
    )

    assert worker.solve()['success']
    assert fake_scs.last_solver.settings['linear_solver'] == selection
    assert config['linear_solver'] == value
    stats = json.loads(json.dumps(worker.stats))
    assert stats['settings']['linear_solver'] == selection
    assert stats['native']['linear_system_mode'] == mode
    assert worker.close()


@pytest.mark.parametrize('config, message', (
    ({'linear_solver': 'qdldl', 'use_indirect': False}, 'not both'),
    ({'linear_solver': 'qdldl', 'gpu': False}, 'not both'),
    ({'linear_solver': 'qdldl', 'mkl': False}, 'not both'),
    ({'linear_solver': 'qdldl', 'cudss': False}, 'not both'),
    ({'linear_solver': 'unknown'}, 'valid'),
    ({'mkl': True, 'use_indirect': True}, 'MKL indirect'),
    ({'gpu': True, 'mkl': True}, 'only one'),
    ({'cudss': True}, 'gpu=True'),
))
def test_scs_invalid_selectors_fail_before_workspace_construction(
    monkeypatch, selector_sdpr, config, message
):
    """Reject ambiguous or unsupported selection without committing a solve."""
    fake_scs = _EnumRecordingSCS()
    monkeypatch.setattr(scs_backend, '_import_scs', lambda: fake_scs)
    original_config = dict(config)
    worker = build_scs(
        selector_sdpr,
        gap_tol=1e-5,
        feas_tol=1e-5,
        iter_limit=None,
        config=config,
    )

    with pytest.raises(SolveError, match=message):
        worker.solve()
    assert fake_scs.last_solver is None
    assert worker._data is not None
    assert worker._cone is not None
    assert not worker._solved
    assert config == original_config
    assert worker.close()
