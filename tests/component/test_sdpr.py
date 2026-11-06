"""Component tests for correlatively decomposed SDPR construction."""

import numpy as np
import pytest

import rhiza as rz
from rhiza.diagnostics import DegreeError, PipelineError, SDPRError
from rhiza.tools.linalg import normalize_rows
from tests.fixtures.reference_problem import make_reference_pop


################################################################################
# Construction and degrees
################################################################################


def test_requires_finalized_pop():
    """Reject SDPR construction from a mutable POP."""
    pop = make_reference_pop(num_samples=40)
    with pytest.raises(PipelineError):
        rz.SDPR(pop, sos_degree=4)


def test_requires_even_sos_degree(reference_pop):
    """Reject odd base SOS degrees."""
    with pytest.raises(DegreeError):
        rz.SDPR(reference_pop, sos_degree=3)


def test_sparse_reference_recipe(reference_pop):
    """Construct the expected sparse reference relaxation structure."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    assert sdpr.stats['num_cliques'] == 3
    assert sdpr.stats['num_separator_edges'] == 2
    assert sum(
        gramian['kind'] == 'BASE' for gramian in sdpr._gramians
    ) == 3
    assert sum(
        gramian['kind'] == 'INEQUALITY' for gramian in sdpr._gramians
    ) == 2


def test_dense_reference_has_one_clique():
    """Represent the dense special case with one clique and no separators."""
    pop = make_reference_pop(num_samples=80, use_sparse_scopes=False)
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=4)
    assert sdpr.stats['num_cliques'] == 1
    assert sdpr.stats['num_separator_edges'] == 0


def test_sampled_identities_have_no_multipliers(reference_pop):
    """Omit free multipliers for equality evals already zero on samples."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    assert not any(
        block['kind'] == 'EQUALITY_MULTIPLIER'
        for block in sdpr._scalar_blocks
    )


def test_explicit_equalities_have_multipliers():
    """Construct one free polynomial multiplier per nonzero equality."""
    pop = make_reference_pop(
        num_samples=100,
        use_sampled_equalities=False,
    )
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=4)
    equality_blocks = [
        block for block in sdpr._scalar_blocks
        if block['kind'] == 'EQUALITY_MULTIPLIER'
    ]
    assert len(equality_blocks) == 2
    assert all(block['mul_degree'] == 2 for block in equality_blocks)


def test_explicit_multiplier_degree_policy():
    """Resolve multiplier degrees under an explicit override."""
    pop = make_reference_pop(
        num_samples=100,
        use_sampled_equalities=False,
    )
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=4, mul_degree=3)
    inequality_degrees = {
        gramian['mul_degree']
        for gramian in sdpr._gramians
        if gramian['kind'] == 'INEQUALITY'
    }
    equality_degrees = {
        block['mul_degree']
        for block in sdpr._scalar_blocks
        if block['kind'] == 'EQUALITY_MULTIPLIER'
    }
    assert inequality_degrees == {2}
    assert equality_degrees == {3}


def test_default_separator_degree(reference_pop):
    """Resolve a missing separator degree to sos_degree."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    assert sdpr.sep_degree == 4


def test_sdpr_configuration_is_read_only(reference_pop):
    """Keep eager recipe data synchronized with its public configuration."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    settings = {
        'sos_degree': 2,
        'mul_degree': 2,
        'sep_degree': 2,
        'rank_tol': 1e-8,
        'basis_rank_limit': 1,
        'enable_component_splitting': False,
    }
    for name, value in settings.items():
        with pytest.raises(AttributeError):
            setattr(sdpr, name, value)
    assert sdpr.stats['sos_degree'] == 4
    assert sdpr.stats['sep_degree'] == 4


def test_sdpr_integral_settings_are_not_truncated(reference_pop):
    """Accept integer-valued reals and reject fractional degree settings."""
    sdpr = rz.SDPR(
        reference_pop,
        sos_degree=4.0,
        mul_degree=2.0,
        sep_degree=4.0,
        basis_rank_limit=3.0,
    )
    assert sdpr.sos_degree == 4
    assert sdpr.mul_degree == 2
    assert sdpr.sep_degree == 4
    assert sdpr.basis_rank_limit == 3

    with pytest.raises(DegreeError, match='integer'):
        rz.SDPR(reference_pop, sos_degree=3.5)
    with pytest.raises(DegreeError, match='integer'):
        rz.SDPR(reference_pop, sos_degree=4, mul_degree=1.5)
    with pytest.raises(SDPRError, match='integer'):
        rz.SDPR(reference_pop, sos_degree=4, basis_rank_limit=1.5)


def test_separator_degree_sufficiency_uses_actual_residual_degree():
    """Report multiplier products that exceed the chosen communication degree."""
    x_evals = np.linspace(-1.0, 1.0, 24)
    pop = rz.POP()
    pop.add_variable("x", x_evals)
    pop.add_objective(x_evals**2, ('x',), 2)
    pop.add_constraint(1.0 - x_evals**2, ('x',), 2, form='>=0')
    pop.finalize()

    insufficient = rz.SDPR(
        pop,
        sos_degree=2,
        mul_degree=3,
        sep_degree=2,
    )
    assert insufficient.stats['required_sep_degree'] == 4
    assert insufficient.status['sep_degree_insufficient']

    sufficient = rz.SDPR(
        pop,
        sos_degree=2,
        mul_degree=3,
        sep_degree=4,
    )
    assert sufficient.stats['required_sep_degree'] == 4
    assert not sufficient.status['sep_degree_insufficient']


################################################################################
# Algebraic assembly
################################################################################


def test_max_objective_sign_equivalence():
    """Canonicalize MIN f and MAX minus-f to the same clique objectives."""
    pop_min = make_reference_pop(num_samples=80, objective_form='MIN')
    pop_max = make_reference_pop(num_samples=80, objective_form='MAX')
    pop_min.finalize()
    pop_max.finalize()
    sdpr_min = rz.SDPR(pop_min, sos_degree=4)
    sdpr_max = rz.SDPR(pop_max, sos_degree=4)
    for first, second in zip(
        sdpr_min._clique_systems, sdpr_max._clique_systems
    ):
        np.testing.assert_allclose(
            first['objective_evals'], second['objective_evals']
        )


def _make_affine_objective_sdpr(scale, offset):
    """Build one quadratic objective after a positive affine transform."""
    x_evals = np.linspace(-2.0, 2.0, 80)
    base_evals = (x_evals - 0.7)**2
    objective_evals = scale*base_evals + offset
    pop = rz.POP()
    pop.add_variable('x', x_evals)
    pop.add_objective(objective_evals, ('x',), 2)
    pop.finalize()
    return rz.SDPR(pop, sos_degree=2), objective_evals


def test_objective_affine_normalization_preserves_the_declared_objective():
    """Center offsets and remove units without changing sampled values."""
    reference, _ = _make_affine_objective_sdpr(1.0, 0.0)
    transformed, objective_evals = _make_affine_objective_sdpr(1e4, 1e6)

    np.testing.assert_allclose(
        transformed._clique_systems[0]['objective_evals'],
        reference._clique_systems[0]['objective_evals'],
        rtol=2e-11,
        atol=2e-11,
    )
    reconstructed = (
        transformed.stats['objective_scale']
        * transformed._clique_systems[0]['objective_evals']
        + transformed.stats['objective_offset']
    )
    np.testing.assert_allclose(
        reconstructed, objective_evals, rtol=0.0, atol=2e-10
    )
    np.testing.assert_allclose(
        transformed.stats['objective_scale'],
        1e4*reference.stats['objective_scale'],
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        transformed.stats['objective_offset'],
        1e4*reference.stats['objective_offset'] + 1e6,
        rtol=2e-15,
    )


def test_constant_objective_uses_unit_scale_and_exact_offset():
    """Represent a constant objective by its offset and a zero normalized row."""
    x_evals = np.linspace(-1.0, 1.0, 12)
    pop = rz.POP()
    pop.add_variable('x', x_evals)
    pop.add_objective(np.full(x_evals.shape, 3.5), ('x',), 0)
    pop.finalize()

    sdpr = rz.SDPR(pop, sos_degree=2)
    assert sdpr.stats['objective_scale'] == 1.0
    assert sdpr.stats['objective_offset'] == 3.5
    np.testing.assert_array_equal(
        sdpr._clique_systems[0]['objective_evals'],
        np.zeros(x_evals.shape),
    )


def test_inequality_sign_equivalence():
    """Canonicalize LE g and GE minus-g to the same Gramian coefficients."""
    pop_le = make_reference_pop(num_samples=80, inequality_form='<=0')
    pop_ge = make_reference_pop(num_samples=80, inequality_form='>=0')
    pop_le.finalize()
    pop_ge.finalize()
    sdpr_le = rz.SDPR(pop_le, sos_degree=4)
    sdpr_ge = rz.SDPR(pop_ge, sos_degree=4)
    for clique_id in range(3):
        np.testing.assert_allclose(
            sdpr_le._materialize_clique_system_evals(clique_id),
            sdpr_ge._materialize_clique_system_evals(clique_id),
        )


def test_separator_coefficients_cancel(reference_pop):
    """Cancel shared separator polynomials across endpoint systems."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    rng = np.random.default_rng(10)
    coeffs = np.zeros(sdpr._num_scalar_coeffs())
    for block in sdpr._scalar_blocks:
        if block['kind'] == 'SEPARATOR':
            start = block['start']
            stop = block['stop']
            coeffs[start:stop] = rng.normal(size=block['size'])

    total = np.zeros(reference_pop.num_samples)
    for system in sdpr._clique_systems:
        for contribution in system['scalar_contributions']:
            block = sdpr._scalar_blocks[contribution['block_id']]
            if block['kind'] != 'SEPARATOR':
                continue
            evals = sdpr._materialize_scalar_contribution(contribution)
            total += evals@coeffs[block['start']:block['stop']]
    np.testing.assert_allclose(total, 0.0, atol=1e-13)


def test_streamed_coefficients_match_dense_reference(reference_pop):
    """Reproduce each clique coefficient matrix from streamed chunks."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    sdpr.finalize()
    for clique_id in range(3):
        scalar = [
            chunk.coeff_evals
            for chunk in sdpr._iter_scalar_coeff_chunks(
                clique_id, max_num_coeffs=10000000
            )
        ]
        gramian = [
            chunk.coeff_evals
            for chunk in sdpr._iter_gramian_coeff_chunks(
                clique_id, max_num_coeffs=10000000
            )
        ]
        actual = np.hstack(scalar + gramian)
        system = sdpr._clique_systems[clique_id]
        expected = (
            system['row_scales'][:,None]
            * sdpr._materialize_clique_system_evals(clique_id)[
                system['row_indices'],:
            ]
        )
        np.testing.assert_allclose(actual, expected)


################################################################################
# Analysis and lifecycle
################################################################################


def test_analyze_does_not_compress(reference_pop):
    """Analyze every clique rank without selecting sampled equations."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    report = sdpr.analyze()
    assert report['num_cliques'] == 3
    assert report['sample_margin_check'] == 'POSITIVE_MARGIN'
    assert report['num_cliques_checked'] == 3
    assert not sdpr.compressed


def test_compress_is_transactional_and_idempotent(reference_pop):
    """Commit all clique selections once and return the stored report later."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    first = sdpr.compress(method='PQR')
    assert first['clique_size_threshold'] is None
    assert first['num_cliques_selected'] == 3
    assert first['num_cliques_skipped'] == 0
    assert all(item['compression_applied'] for item in first['cliques'])
    rows = [system['row_indices'].copy() for system in sdpr._clique_systems]
    second = sdpr.compress(method='PQR')
    assert first['changed']
    assert not second['changed']
    for before, system in zip(rows, sdpr._clique_systems):
        np.testing.assert_array_equal(before, system['row_indices'])


def test_compression_clique_size_threshold_selects_small_cliques(
    reference_pop,
    monkeypatch,
):
    """Compress only cliques at or below the declared size threshold."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    original = sdpr._materialize_augmented_clique_system
    materialized = []

    def track_selected_cliques(clique_id):
        """Record selected cliques and reject accidental large materialization."""
        materialized.append(clique_id)
        assert len(sdpr.cliques[clique_id]) <= 2
        return original(clique_id)

    monkeypatch.setattr(
        sdpr,
        '_materialize_augmented_clique_system',
        track_selected_cliques,
    )
    report = sdpr.compress(
        method='PQR',
        clique_size_threshold=2,
    )

    assert materialized == [0, 1]
    assert report['clique_size_threshold'] == 2
    assert report['num_cliques_selected'] == 2
    assert report['num_cliques_skipped'] == 1
    assert report['num_rows_original'] == 3*reference_pop.num_samples
    assert report['num_rows_retained'] < report['num_rows_original']
    assert tuple(
        item['compression_applied'] for item in report['cliques']
    ) == (True, True, False)
    assert tuple(
        item['clique_size'] for item in report['cliques']
    ) == (2, 2, 3)
    assert report['cliques'][2]['rank'] is None
    assert report['cliques'][2]['num_rows_retained'] == (
        reference_pop.num_samples
    )
    assert sdpr.status['sample_margin_check'] == 'PARTIAL'
    assert sdpr._clique_systems[2]['row_indices'] is None
    assert sdpr._clique_systems[2]['rank_analysis'] is None

    repeated = sdpr.compress(clique_size_threshold=2)
    assert not repeated['changed']
    assert repeated['clique_size_threshold'] == 2
    with pytest.raises(PipelineError, match='settings cannot change'):
        sdpr.compress(clique_size_threshold=3)

    assert sdpr.finalize()
    np.testing.assert_array_equal(
        sdpr._clique_systems[2]['row_indices'],
        np.arange(reference_pop.num_samples),
    )


def test_compression_clique_size_threshold_validation(reference_pop):
    """Accept positive integer thresholds and reject invalid selectors."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    report = sdpr.compress(clique_size_threshold=2.0)
    assert report['clique_size_threshold'] == 2

    for value in (0, -1, 1.5, True):
        other = rz.SDPR(reference_pop, sos_degree=4)
        with pytest.raises(SDPRError):
            other.compress(clique_size_threshold=value)


def test_compression_changed_is_false_when_no_rows_are_removed():
    """Use the same change semantics at report and clique levels."""
    x_evals = np.array([-1.0, 0.0, 1.0])
    pop = rz.POP()
    pop.add_variable('x', x_evals)
    pop.add_objective(x_evals**2, ('x',), 2)
    pop.finalize()

    report = rz.SDPR(pop, sos_degree=2).compress(method='PQR')
    assert not report['changed']
    assert report['num_rows_original'] == report['num_rows_retained']
    assert all(not item['changed'] for item in report['cliques'])


def test_compression_report_isolated_from_caller_mutation(reference_pop):
    """Keep stored compression diagnostics independent of returned mappings."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    first = sdpr.compress(method='PQR')
    original_rows = first['cliques'][0]['num_rows_retained']
    first['cliques'][0]['num_rows_retained'] = -1

    repeated = sdpr.compress(method='PQR')
    assert repeated['cliques'][0]['num_rows_retained'] == original_rows


def test_finalize_locks_compression(reference_pop):
    """Finalize row maps and reject later structural mutation."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    assert sdpr.finalize()
    assert not sdpr.finalize()
    with pytest.raises(PipelineError):
        sdpr.compress()


def test_basis_rank_limit_applies_to_separator_bases(reference_pop):
    """Apply one quotient-basis rank limit to every basis category."""
    sdpr = rz.SDPR(
        reference_pop,
        sos_degree=4,
        basis_rank_limit=1,
    )
    separator_bases = [
        basis for basis in sdpr._bases if basis['is_separator']
    ]
    assert separator_bases
    assert all(basis['basis_rank'] == 1 for basis in separator_bases)
    assert all(basis['basis_rank_limited'] for basis in separator_bases)
    assert sdpr.status['basis_rank_limited']


def test_analysis_reports_rank_evidence_without_row_selection(reference_pop):
    """Return rank evidence without compression-specific diagnostics."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    report = sdpr.analyze()
    assert report['sample_margin_check'] == 'POSITIVE_MARGIN'
    assert all(item['sample_margin'] > 0 for item in report['cliques'])
    for item in report['cliques']:
        assert 'changed' not in item
        assert 'num_rows_retained' not in item
        assert 'row_span_residual' not in item


def test_compression_failure_does_not_commit_partial_rows(reference_pop, monkeypatch):
    """Leave every clique uncompressed when a later selection fails."""
    import rhiza.sdpr as sdpr_module

    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    original = sdpr_module.compress_clique_system
    calls = {'count': 0}

    def fail_second(
        system_evals, *, method, rank_tol, rank_analysis=None
    ):
        """Fail the second compression while preserving the first result."""
        calls['count'] += 1
        if calls['count'] == 2:
            raise RuntimeError("deliberate compression failure")
        return original(
            system_evals,
            method=method,
            rank_tol=rank_tol,
            rank_analysis=rank_analysis,
        )

    monkeypatch.setattr(sdpr_module, 'compress_clique_system', fail_second)
    with pytest.raises(RuntimeError, match="deliberate"):
        sdpr.compress(method='PQR')
    assert not sdpr.compressed
    assert all(
        system['row_indices'] is None for system in sdpr._clique_systems
    )


def test_uncompressed_finalize_does_not_materialize_full_system(
    reference_pop,
    monkeypatch,
):
    """Compute uncompressed row scales through bounded coefficient chunks."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)

    def fail_materialization(clique_id):
        """Raise when the forbidden test path is reached."""
        raise AssertionError("full clique materialization was used")

    monkeypatch.setattr(
        sdpr,
        '_materialize_augmented_clique_system',
        fail_materialization,
    )
    assert sdpr.finalize()
    assert all(
        system['row_indices'].size == reference_pop.num_samples
        for system in sdpr._clique_systems
    )



def test_streamed_row_norms_match_materialized_reference(reference_pop):
    """Match bounded row-norm accumulation to direct normalization."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    for clique_id in range(sdpr.stats['num_cliques']):
        augmented = sdpr._materialize_augmented_clique_system(clique_id)
        _, expected = normalize_rows(augmented)
        actual = sdpr._compute_uncompressed_row_scales(clique_id)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_sample_margin_probe_accumulates_clique_coverage(reference_pop):
    """Move from unchecked through partial to complete rank evidence."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    assert sdpr.status == {
        'sample_margin_check': 'UNCHECKED',
        'basis_rank_limited': False,
        'sep_degree_insufficient': False,
    }
    assert tuple(enumerate(sdpr.cliques)) == (
        (0, ('x0', 'x1')),
        (1, ('x1', 'x2')),
        (2, ('x2', 'x3', 'x4')),
    )

    first = sdpr.analyze(num_cliques=1, seed=4)
    assert first['sample_margin_check'] == 'PARTIAL'
    assert first['num_cliques_checked'] == 1
    assert first['num_cliques_checked_this_call'] == 1
    assert len(first['cliques']) == 1
    assert first['cliques'][0]['scope'] == sdpr.cliques[
        first['cliques'][0]['clique_id']
    ]

    second = sdpr.analyze()
    assert second['sample_margin_check'] == 'POSITIVE_MARGIN'
    assert second['num_cliques_checked'] == 3
    assert second['num_cliques_checked_this_call'] == 2
    assert len(second['cliques']) == 2

    cached = sdpr.analyze()
    assert cached['sample_margin_check'] == 'POSITIVE_MARGIN'
    assert cached['num_cliques_checked_this_call'] == 0
    assert len(cached['cliques']) == 3


def test_sample_margin_probe_exact_ids_and_validation(reference_pop):
    """Analyze requested clique IDs and reject ambiguous selections."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    report = sdpr.analyze(clique_ids=(2, 0))
    assert tuple(item['clique_id'] for item in report['cliques']) == (2, 0)
    assert report['sample_margin_check'] == 'PARTIAL'

    with pytest.raises(SDPRError, match='not both'):
        sdpr.analyze(clique_ids=(1,), num_cliques=1)
    with pytest.raises(SDPRError, match='duplicates'):
        sdpr.analyze(clique_ids=(1, 1))
    with pytest.raises(SDPRError, match='out of range'):
        sdpr.analyze(clique_ids=(9,))
    with pytest.raises(SDPRError, match='positive'):
        sdpr.analyze(num_cliques=0)


def test_random_sample_margin_probe_is_deterministic(reference_pop):
    """Select the same unchecked clique subset for a fixed seed."""
    first = rz.SDPR(reference_pop, sos_degree=4)
    second = rz.SDPR(reference_pop, sos_degree=4)
    first_ids = tuple(
        item['clique_id']
        for item in first.analyze(num_cliques=2, seed=17)['cliques']
    )
    second_ids = tuple(
        item['clique_id']
        for item in second.analyze(num_cliques=2, seed=17)['cliques']
    )
    assert first_ids == second_ids


def test_analysis_is_rank_only(reference_pop, monkeypatch):
    """Avoid PQR row selection during a sample-margin probe."""
    import rhiza.sdpr as sdpr_module

    sdpr = rz.SDPR(reference_pop, sos_degree=4)

    def fail_compression(*args, **kwargs):
        """Raise if analysis enters the compression path."""
        raise AssertionError("compression was used during analysis")

    monkeypatch.setattr(sdpr_module, 'compress_clique_system', fail_compression)
    report = sdpr.analyze(num_cliques=1, seed=0)
    assert report['num_cliques_checked_this_call'] == 1


def test_compression_reuses_cached_rank_analysis(reference_pop, monkeypatch):
    """Skip repeated SVD rank analysis after a complete probe."""
    import rhiza.tools.compression as compression_module

    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    sdpr.analyze()

    def fail_rank_analysis(*args, **kwargs):
        """Raise if compression recomputes a cached rank."""
        raise AssertionError("cached rank analysis was not reused")

    monkeypatch.setattr(
        compression_module,
        'analyze_balanced_system_rank',
        fail_rank_analysis,
    )
    report = sdpr.compress(method='PQR')
    assert report['num_rows_retained'] < report['num_rows_original']
    assert sdpr.status['sample_margin_check'] == 'POSITIVE_MARGIN'


def test_analysis_after_compression_uses_cached_records(reference_pop, monkeypatch):
    """Report compressed clique ranks without rematerializing systems."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    sdpr.compress(method='PQR')

    def fail_materialization(*args, **kwargs):
        """Raise if cached compression diagnostics are ignored."""
        raise AssertionError("clique system was rematerialized")

    monkeypatch.setattr(
        sdpr,
        '_materialize_augmented_clique_system',
        fail_materialization,
    )
    report = sdpr.analyze()
    assert report['num_cliques_checked_this_call'] == 0
    assert report['sample_margin_check'] == 'POSITIVE_MARGIN'


def test_sample_margin_check_detects_basis_and_system_zero_margin():
    """Distinguish early basis evidence from complete clique-system evidence."""
    two_points = np.array([-1.0, 1.0])
    basis_pop = rz.POP()
    basis_pop.add_variable('x', two_points)
    basis_pop.add_objective((two_points - 0.2)**2, ('x',), 2)
    basis_pop.finalize()
    basis_sdpr = rz.SDPR(basis_pop, sos_degree=2)
    assert basis_sdpr.status['sample_margin_check'] == 'NO_MARGIN'
    assert basis_sdpr.analyze()['minimum_sample_margin'] == 0

    three_points = np.linspace(-1.0, 1.0, 3)
    system_pop = rz.POP()
    system_pop.add_variable('x', three_points)
    system_pop.add_objective((three_points - 0.2)**2, ('x',), 2)
    system_pop.finalize()
    system_sdpr = rz.SDPR(system_pop, sos_degree=2)
    assert system_sdpr.status['sample_margin_check'] == 'UNCHECKED'
    report = system_sdpr.analyze()
    assert report['sample_margin_check'] == 'NO_MARGIN'
    assert report['cliques'][0]['sample_margin'] == 0


def _make_scaled_constraint_sdpr(scale, *, form):
    """Construct one explicit normalized constraint recipe."""
    x_evals = np.linspace(-2.0, 2.0, 64)
    pop = rz.POP()
    pop.add_variable('x', x_evals)
    pop.add_objective((x_evals - 0.3)**2, ('x',), 2)
    pop.add_constraint(scale*x_evals, ('x',), 1, form=form)
    pop.finalize()
    return rz.SDPR(pop, sos_degree=2)


def test_constraint_normalization_preserves_coefficient_recipe():
    """Make legal equality and inequality rescaling numerically invariant."""
    for form, scales in (
        ('==0', (1e-12, 1e-8, 1.0, 1e8, 1e12)),
        ('>=0', (1e-12, 1e-8, 1.0, 1e8, 1e12)),
    ):
        reference = _make_scaled_constraint_sdpr(scales[0], form=form)
        reference_matrix = reference._materialize_clique_system_evals(0)
        for scale in scales[1:]:
            sdpr = _make_scaled_constraint_sdpr(scale, form=form)
            np.testing.assert_allclose(
                sdpr._materialize_clique_system_evals(0),
                reference_matrix,
                atol=1e-13,
                rtol=1e-13,
            )
            expected = 2.0*abs(scale)
            np.testing.assert_allclose(
                sdpr.stats['constraint_scale_range'],
                (expected, expected),
            )


def test_subnormal_inequality_normalization_stays_finite():
    """Normalize tiny inequality evals before forming Gramian products."""
    x_evals = np.array([-1.0, 0.0, 1.0])

    def build(scale):
        """Build the same sampled inequality at one positive scale."""
        pop = rz.POP()
        pop.add_variable('x', x_evals)
        pop.add_objective((x_evals - 0.3)**2, ('x',), 2)
        pop.add_constraint(scale*x_evals, ('x',), 1, form='>=0')
        pop.finalize()
        return rz.SDPR(pop, sos_degree=2)

    reference = build(1.0)._materialize_clique_system_evals(0)
    smallest = np.nextafter(0.0, 1.0)
    tiny = build(smallest)._materialize_clique_system_evals(0)
    assert np.all(np.isfinite(tiny))
    np.testing.assert_allclose(tiny, reference, atol=0.0, rtol=0.0)


def test_scalar_streaming_avoids_complete_contribution_materialization(
    monkeypatch,
):
    """Use bounded scalar slices in row norms and backend emission."""
    sdpr = _make_scaled_constraint_sdpr(1.0, form='==0')

    def fail_materialization(*args, **kwargs):
        """Raise if the complete scalar block is constructed."""
        raise AssertionError("complete scalar entry was materialized")

    monkeypatch.setattr(sdpr, '_materialize_scalar_contribution', fail_materialization)
    sdpr.finalize()
    chunks = list(sdpr._iter_scalar_coeff_chunks(0, max_num_coeffs=32))
    assert chunks
    assert max(chunk.coeff_evals.size for chunk in chunks) <= 64


def test_scalar_chunk_respects_element_budget():
    """Bound the first equality-multiplier temporary by the chunk request."""
    rng = np.random.default_rng(45)
    num_samples = 500
    xyz = rng.normal(size=(num_samples, 3))
    pop = rz.POP()
    for column, name in enumerate(('x', 'y', 'z')):
        pop.add_variable(name, xyz[:,column])
    pop.add_objective(np.sum(xyz**2, axis=1), ('x', 'y', 'z'), 2)
    pop.add_constraint(xyz[:,0], ('x', 'y', 'z'), 1, form='==0')
    pop.finalize()
    sdpr = rz.SDPR(pop, sos_degree=4)
    equality_contribution = next(
        contribution
        for contribution in sdpr._clique_systems[0]['scalar_contributions']
        if contribution['kind'] == 'EQUALITY_MULTIPLIER'
    )
    _, _, chunk = next(sdpr._iter_scalar_contribution_eval_chunks(
        equality_contribution,
        max_num_coeffs=num_samples,
    ))
    assert chunk.size <= num_samples


def test_compressed_scalar_chunk_slices_columns_before_rows():
    """Apply compressed row selection only after bounding basis columns."""
    sdpr = _make_scaled_constraint_sdpr(1.0, form='==0')
    equality_contribution = next(
        contribution
        for contribution in sdpr._clique_systems[0]['scalar_contributions']
        if contribution['kind'] == 'EQUALITY_MULTIPLIER'
    )
    basis = sdpr._bases[equality_contribution['basis_id']]
    indexes = []

    class TrackingArray(np.ndarray):
        """Record indexing order while retaining ndarray behavior."""

        def __new__(cls, values):
            """Create a tracking view over one basis-evaluation array."""
            obj = np.asarray(values).view(cls)
            obj.indexes = indexes
            return obj

        def __array_finalize__(self, source):
            """Propagate the shared index log to derived NumPy views."""
            self.indexes = getattr(source, 'indexes', indexes)

        def __getitem__(self, key):
            """Record one indexing operation before ordinary array access."""
            self.indexes.append(key)
            return super().__getitem__(key)

    basis['basis_evals'] = TrackingArray(basis['basis_evals'])
    row_indices = np.array([0, 3, 7, 11], dtype=np.int32)
    next(sdpr._iter_scalar_contribution_eval_chunks(
        equality_contribution,
        row_indices=row_indices,
        max_num_coeffs=len(row_indices),
    ))

    first_rows, first_columns = indexes[0]
    assert first_rows == slice(None)
    assert isinstance(first_columns, slice)
    assert first_columns.stop - first_columns.start == 1
    np.testing.assert_array_equal(indexes[1][0], row_indices)


def test_sdpr_public_reports_are_json_serializable(reference_pop):
    """Expose only ordinary Python scalar leaves in public diagnostics."""
    import json

    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    analysis = sdpr.analyze(num_cliques=1, seed=0)
    compression = sdpr.compress(method='PQR')
    for report in (sdpr.status, sdpr.stats, analysis, compression):
        json.dumps(report)


def test_sdpr_diagnostic_schema_is_compact(reference_pop):
    """Keep status and statistics free of redundant object metadata."""
    sdpr = rz.SDPR(reference_pop, sos_degree=4)
    assert set(sdpr.status) == {
        'sample_margin_check',
        'basis_rank_limited',
        'sep_degree_insufficient',
    }
    assert set(sdpr.stats) == {
        'sos_degree',
        'sep_degree',
        'required_sep_degree',
        'num_samples',
        'num_bases',
        'num_cliques',
        'num_gramians',
        'enable_component_splitting',
        'num_component_split_gramians',
        'num_scalar_coeffs',
        'num_separator_coeffs',
        'num_psd_coeffs',
        'num_psd_coeffs_without_component_splitting',
        'num_rows_original',
        'num_rows_retained',
        'objective_offset',
        'objective_scale',
        'constraint_scale_range',
        'num_separator_edges',
        'num_separator_components',
        'num_separator_cycles',
    }
