"""Lazy SHAP, partial dependence, ICE, and accumulated local effects."""

from itertools import product
from inspect import signature
from collections.abc import Mapping

import numpy as np
import polars as pl

from ins_gbm.preprocessing.encoder import _NUMERIC_FILL
from ins_gbm.progress import PipelineCancelled, ProgressEvent

from .adapter import PredictionAdapter
from .results import EffectResult, ShapResult, _validate_xlim


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _valid_numeric(values):
    values = np.asarray(values, dtype=float)
    # ModelData uses float32; compare in that dtype to recognize the rounded sentinel.
    return np.isfinite(values) & (values.astype(np.float32) != np.float32(_NUMERIC_FILL))


class ExplanationReport:
    """Explain fitted predictions without fitting or consulting outcome values.

    Parameters
    ----------
    fitted : FittedPipeline or EnsembleResult
        Fitted prediction object.
    data : ModelData
        Original population to explain, before fitted encoding/preprocessing.
        Exposure/offset remain fixed for each row; targets are not consulted.
    background : ModelData, optional
        Explicit SHAP reference population, sampled uniformly, independently
        of population averaging weights. Required only for SHAP; may equal data.
    feature_space : {'auto', 'raw', 'processed'}, default='auto'
        Auto uses raw inputs for ensembles and processed components for single
        models. Raw interventions replay each member's fitted transforms.
    prediction_type : {'response', 'rate', 'link'}, default='response'
        Response includes exposure where applicable; Poisson rate removes it.
        Link is log final response, not log rate. Gamma does not support rate.
    seed : int, default=42
        Random seed for reproducible row/background sampling, permutation SHAP,
        and selection of displayed ICE curves.
    progress_callback : callable, optional
        Called with a ``ProgressEvent`` at method and prediction-batch boundaries.
    should_stop : callable, optional
        Zero-argument function checked at progress boundaries. Returning true
        raises ``PipelineCancelled``.
    batch_size : int, default=1024
        Maximum rows per adapter prediction call. Larger batches reduce call
        overhead but use more memory; this does not limit explanation rows.
    shap_kwargs : mapping, optional
        Options for ``shap()``. If supplied (even an empty mapping), compute SHAP
        during construction and retain these options as defaults for later
        calls. None leaves SHAP lazy. Requires ``background``.

    Notes
    -----
    Only the most recent SHAP result is cached, in memory. Keep data, background,
    and report configuration unchanged when reusing a report. PDP, ICE, and ALE
    are calculated separately and do not use SHAP contributions.
    """

    def __init__(self, fitted, data, *, background=None, prediction_type="response", seed=42,
                 progress_callback=None, should_stop=None, batch_size=1024, feature_space="auto",
                 shap_kwargs=None):
        if shap_kwargs is not None and not isinstance(shap_kwargs, Mapping):
            raise TypeError("shap_kwargs must be a mapping of SHAP options or None")
        data.validate_for_prediction()
        self.adapter = PredictionAdapter(fitted, data, prediction_type, feature_space)
        self.data = data
        self.frame = self.adapter.frame
        self.background = background
        self.seed = seed
        self.progress_callback = progress_callback
        self.should_stop = should_stop
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self._shap_defaults = dict(shap_kwargs or {})
        self._shap_cache = None
        if shap_kwargs is not None:
            self.shap()

    @property
    def feature_names(self):
        """Names accepted by diagnostic methods, including processed components."""
        return list(self.adapter.names)

    @property
    def feature_metadata(self):
        """Coordinate types, transform lineage, and fitted PCA descriptions."""
        return self.adapter.metadata

    def _event(self, method, current=None, total=None):
        if self.should_stop and self.should_stop():
            raise PipelineCancelled("Explanation cancelled")
        if self.progress_callback:
            self.progress_callback(ProgressEvent("explain", method, current, total))

    def _predict(self, frame, indices):
        output = []
        for start in range(0, frame.height, self.batch_size):
            self._event("prediction", start, frame.height)
            output.append(self.adapter.predict(frame.slice(start, self.batch_size), indices[start:start + self.batch_size]))
        return np.concatenate(output)

    def _weights(self, weights, indices):
        values = self.data.weight if weights is None else weights
        if isinstance(values, str):
            if values != "uniform":
                raise ValueError("weights must be an aligned array or 'uniform'")
            values = None
        if values is None:
            return np.ones(len(indices))
        values = np.asarray(values, dtype=float)
        if values.shape != (self.frame.height,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("weights must be finite, nonnegative, and align with data")
        selected = values[indices]
        if selected.sum() <= 0:
            raise ValueError("Selected rows must have positive total averaging weight")
        return selected

    def _rows(self, max_samples):
        if max_samples is None:
            return np.arange(self.frame.height)
        cap = _positive_integer(max_samples, "max_samples")
        return np.sort(np.random.default_rng(self.seed).choice(self.frame.height, min(cap, self.frame.height), replace=False))

    def _settings(self, method, indices, weights, **kwargs):
        objective = self.adapter.objective
        scale = self.adapter.prediction_type
        units = "severity" if objective == "gamma" else (
            "expected count" if self.data.exposure is not None and scale != "rate" else "frequency rate")
        if scale == "link":
            units = f"log {units}"
        return {"method": method, "prediction_type": scale, "objective": objective,
                "batch_size": self.batch_size, "feature_space": self.adapter.feature_space,
                "units": units, "seed": self.seed, "row_indices": indices.tolist(),
                "averaging_weights": weights.tolist(), "feature_metadata": self.adapter.metadata,
                "context": "Exposure and offsets fixed per explained row; not attributed",
                "interpretation": "Model behavior, not causal effects. PDP/ICE can extrapolate under feature dependence; SHAP depends on the reference population. " + (
                    "Raw feature effects include all paths through fitted preprocessing." if self.adapter.feature_space == "raw" else
                    "Component effects vary processed scores, not source variables."),
                **kwargs}

    def _features(self, features, limit=2):
        names = [features] if isinstance(features, str) else list(features)
        if not 1 <= len(names) <= limit or len(set(names)) != len(names):
            raise ValueError(f"Specify between one and {limit} distinct features")
        unknown = set(names) - set(self.frame.columns)
        if unknown:
            raise ValueError(f"Unknown diagnostic features: {sorted(unknown)}")
        return names

    def _grid(self, frame, name, grid, grid_size, percentiles):
        metadata = self.adapter.metadata[name]
        if grid is not None:
            values = list(grid)
        elif metadata["kind"] == "categorical":
            values = frame[name].unique(maintain_order=True).to_list()
            if len(values) > 30:
                raise ValueError(f"{name}: more than 30 levels; supply an explicit grid")
        else:
            values = frame[name].cast(pl.Float64).to_numpy()
            values = values[_valid_numeric(values)]
            if not len(values):
                raise ValueError(f"{name}: no finite nonmissing numeric values")
            if not 0 <= percentiles[0] <= percentiles[1] <= 1:
                raise ValueError("percentiles must satisfy 0 <= low <= high <= 1")
            values = np.unique(np.quantile(values, np.linspace(*percentiles, _positive_integer(grid_size, "grid_size")))).tolist()
        if not values:
            raise ValueError("Grid cannot be empty")
        if metadata["kind"] == "numeric" and not _valid_numeric(values).all():
            raise ValueError("Numeric grids must exclude missing values/sentinels")
        return list(dict.fromkeys(values))

    def _intervene(self, frame, changes):
        expressions = []
        for name, value in changes.items():
            dtype = frame.schema[name]
            # Continuous interventions on integer inputs must not be truncated.
            if self.adapter.metadata[name]["kind"] == "numeric":
                dtype = pl.Float64
            expressions.append(pl.lit(value).cast(dtype).alias(name))
        return frame.with_columns(expressions)

    def _pdp_ice(self, features, *, method, grids=None, grid_size=20, percentiles=(.05, .95),
                 weights=None, max_samples=None, centered=False, reference=None, display_rows=100):
        names = self._features(features, 1 if method == "ice" else 2)
        indices = self._rows(max_samples)
        frame = self.frame[indices.tolist()]
        w = self._weights(weights, indices)
        grids = grids or {}
        if set(grids) - set(names):
            raise ValueError("Grid keys must name the requested diagnostic features")
        axes = [self._grid(frame, n, grids.get(n), grid_size, percentiles) for n in names]
        records, supports = [], []
        predictions = []
        # Nearest grid cell provides a consistent empirical support overlay.
        assignments = []
        for name, axis in zip(names, axes):
            if self.adapter.metadata[name]["kind"] == "numeric":
                values = frame[name].cast(pl.Float64).to_numpy()
                codes = np.argmin(np.abs(values[:, None] - np.asarray(axis, dtype=float)[None, :]), axis=1)
                codes[~_valid_numeric(values)] = -1
            else:
                codes = np.asarray([axis.index(v) if v in axis else -1 for v in frame[name].to_list()])
            assignments.append(codes)
        combinations = list(product(*(range(len(axis)) for axis in axes)))
        for current, cell in enumerate(combinations):
            self._event(method, current, len(combinations))
            values = [axis[i] for axis, i in zip(axes, cell)]
            changed = self._intervene(frame, dict(zip(names, values)))
            pred = self._predict(changed, indices)
            predictions.append(pred)
            coordinates = dict(zip(("value", "value_2"), values))
            mask = np.ones(len(indices), dtype=bool)
            for codes, i in zip(assignments, cell):
                mask &= codes == i
            supports.append({**coordinates, "count": int(mask.sum()), "weight": float(w[mask].sum()),
                             "unsupported": not bool(mask.any())})
            if method == "pdp":
                records.append({**coordinates, "prediction": float(np.average(pred, weights=w))})
        if method == "ice":
            axis = axes[0]
            ref_index = (len(axis) // 2 if self.adapter.metadata[names[0]]["kind"] == "numeric" else 0) if reference is None else axis.index(reference)
            baseline = predictions[ref_index] if centered else np.zeros(len(indices))
            for value, pred in zip(axis, predictions):
                records.extend({"row": row, "source_row": int(indices[row]), "value": value,
                                "prediction": float(v), "weight": float(w[row])}
                               for row, v in enumerate(pred - baseline))
        settings = self._settings(method, indices, w, features=names, grids=dict(zip(names, axes)),
                                  centered=centered, reference=axes[0][ref_index] if method == "ice" and centered else None,
                                  display_rows=_positive_integer(display_rows, "display_rows"),
                                  support_definition="Rows assigned to nearest numeric grid point or exact categorical level")
        return EffectResult(pl.DataFrame(records), pl.DataFrame(supports), settings)

    def pdp(self, features, **kwargs):
        """Calculate a partial dependence curve or two-feature surface.

        Parameters
        ----------
        features : str or sequence of str
            One or two distinct names from ``feature_names``. Each grid cell
            replaces these features for every selected row before averaging.
        grids : mapping, optional
            Feature name to explicit grid values. Unspecified numeric axes use
            quantiles; categorical axes use all observed levels (at most 30
            unless an explicit grid is supplied). Numeric grids exclude missing
            values and the library missing-value sentinel.
        grid_size : int, default=20
            Number of quantiles per automatic numeric axis before deduplication.
            Two automatic axes can require grid_size squared population passes.
        percentiles : tuple of float, default=(0.05, 0.95)
            Lower/upper quantile bounds for automatic numeric grids, in [0, 1].
        weights : array-like or 'uniform', optional
            Nonnegative averaging weights aligned to the full input population,
            before sampling. None uses ``data.weight``, or equal weights when
            absent. 'uniform' explicitly selects equal weights.
        max_samples : int, optional
            Uniform sample cap for averaging rows, without replacement, using
            the report seed. None uses all rows. Grids use the selected rows.
        centered : bool, default=False
            Shared ICE option; has no effect on PDP predictions.
        reference : scalar, optional
            Shared ICE option; has no effect on PDP predictions.
        display_rows : int, default=100
            Shared ICE plotting option; has no effect on the PDP calculation.

        Returns
        -------
        EffectResult
            Weighted mean predictions in ``table``, observed grid-cell counts
            and weights in ``support``, and calculation details in ``settings``.
            Exposure and offsets remain fixed per row. No result is cached.
        """
        return self._pdp_ice(features, method="pdp", **kwargs)

    def ice(self, feature, *, grid=None, **kwargs):
        """Calculate individual conditional expectation curves for selected rows.

        Parameters
        ----------
        feature : str
            Name from ``feature_names`` to replace along each row's curve.
        grid : sequence, optional
            Explicit values to evaluate. None uses numeric quantiles or all
            observed categories (at most 30 without an explicit grid).
        grid_size : int, default=20
            Quantile count for an automatic numeric grid, before deduplication.
        percentiles : tuple of float, default=(0.05, 0.95)
            Lower/upper quantile bounds in [0, 1] for an automatic numeric grid.
        weights : array-like or 'uniform', optional
            Nonnegative weights aligned to all input rows before sampling.
            None uses ``data.weight`` or equal weights if absent; 'uniform'
            forces equal weights. Stored with curves; does not alter predictions.
        max_samples : int, optional
            Uniform sample cap, without replacement, using the report seed.
            None computes curves for every row. Automatic grids use this sample.
        centered : bool, default=False
            Subtract each row's prediction at the reference grid value.
        reference : scalar, optional
            Grid value used for centering. None chooses the middle numeric grid
            entry or first categorical entry. Explicit values must be in grid.
        display_rows : int, default=100
            Maximum curves shown by ``plot()``, sampled using the report seed.
            All calculated curves remain in the result table.

        Returns
        -------
        EffectResult
            Per-row predictions (or centered differences), support, and settings.
            ``row`` indexes the selected sample; ``source_row`` indexes the full
            input population. Exposure and offsets stay fixed. No result is cached.
        """
        return self._pdp_ice(feature, method="ice", grids={feature: grid} if grid is not None else None, **kwargs)

    def ale(self, features, *, bins=None, edges=None, weights=None, max_samples=None, xlim=None):
        """Calculate numeric accumulated local effects within observed bins.

        Parameters
        ----------
        features : str or sequence of str
            One or two distinct numeric names from ``feature_names``. Two names
            produce second-order effects adjusted for first-order main effects.
        bins : int, optional
            Quantile bin count per automatic axis, before duplicate edges are
            removed. Defaults to 20 for one feature or 10 for two features.
            Reduce this if cells are empty or have zero total weight.
        edges : mapping, optional
            Feature name to explicit finite, strictly increasing bin boundaries
            covering all selected nonmissing values. Unspecified axes use bins.
        weights : array-like or 'uniform', optional
            Nonnegative weights aligned to the full input population before
            sampling. Used for local averaging and centering. None uses
            ``data.weight`` or equal weights if absent; 'uniform' forces equality.
        max_samples : int, optional
            Uniform row cap sampled without replacement using the report seed.
            None starts with all rows. Missing/nonfinite values and the numeric
            missing sentinel in requested features are excluded after sampling.
        xlim : pair of float or None, optional
            Displayed variable-axis bounds for one-feature ALE, e.g. (0, 2) or
            (None, 2) to cap only the upper end. Each bound must be finite or
            None (automatic). None uses automatic limits. Bounds are retained
            for ``plot()`` and exported figures; they do not change bins, rows,
            effects, centering, or support. Two-feature ALE does not support xlim.
            This zoom cannot add detail absent from the calculated bins; supply
            explicit edges if more resolution near zero is needed.

        Returns
        -------
        EffectResult
            Centered effects in ``table``, bin counts/weights in ``support``, and
            settings including excluded rows. Empty or zero-weight cells raise
            ValueError. Exposure and offsets stay fixed. No result is cached.
        """
        names = self._features(features)
        xlim = _validate_xlim(xlim)
        if xlim is not None and len(names) != 1:
            raise ValueError("xlim requires a one-feature numeric plot")
        if set(edges or {}) - set(names):
            raise ValueError("Edge keys must name the requested diagnostic features")
        if any(self.adapter.metadata[n]["kind"] != "numeric" for n in names):
            raise ValueError("ALE supports numeric coordinates; use PDP/ICE for categories")
        indices = self._rows(max_samples)
        frame = self.frame[indices.tolist()]
        w = self._weights(weights, indices)
        valid = np.ones(len(indices), dtype=bool)
        for name in names:
            valid &= _valid_numeric(frame[name].cast(pl.Float64).to_numpy())
        excluded = int((~valid).sum())
        indices, w = indices[valid], w[valid]
        frame = self.frame[indices.tolist()]
        if not len(indices) or w.sum() <= 0:
            raise ValueError("ALE needs nonmissing rows with positive total weight")
        bins = _positive_integer(bins if bins is not None else (20 if len(names) == 1 else 10), "bins")
        axes, codes, constants = [], [], []
        for name in names:
            values = frame[name].cast(pl.Float64).to_numpy()
            explicit = (edges or {}).get(name)
            axis = np.asarray(explicit, dtype=float) if explicit is not None else np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
            if axis.ndim != 1 or not len(axis) or not _valid_numeric(axis).all() or np.any(np.diff(axis) <= 0):
                raise ValueError("ALE edges must be strictly increasing finite nonmissing values")
            constant = len(np.unique(values)) == 1
            if axis[0] > values.min() or axis[-1] < values.max():
                raise ValueError("ALE edges must cover all nonmissing observations")
            if len(axis) == 1:
                axis = np.repeat(axis, 2)
            axes.append(axis)
            constants.append(constant)
            codes.append(np.clip(np.searchsorted(axis, values, side="right") - 1, 0, len(axis) - 2))
        shape = tuple(len(a) - 1 for a in axes)
        delta, mass, counts = np.zeros(shape), np.zeros(shape), np.zeros(shape, dtype=int)
        for current, cell in enumerate(np.ndindex(shape)):
            self._event("ale", current, int(np.prod(shape)))
            mask = np.ones(len(indices), dtype=bool)
            for code, i in zip(codes, cell):
                mask &= code == i
            counts[cell] = mask.sum()
            mass[cell] = w[mask].sum()
            if mass[cell] <= 0:
                raise ValueError("Empty/zero-weight ALE cell; reduce bins or supply suitable edges")
            local = frame.filter(pl.Series(mask))
            changes = np.zeros(mask.sum())
            for corner in product((0, 1), repeat=len(names)):
                assignments = {n: a[i + c] for n, a, i, c in zip(names, axes, cell, corner)}
                sign = (-1) ** (len(names) - sum(corner))
                changes += sign * self._predict(self._intervene(local, assignments), indices[mask])
            delta[cell] = np.average(changes, weights=w[mask])
        if len(names) == 1:
            nodes = np.r_[0, np.cumsum(delta)]
            effect = (nodes[:-1] + nodes[1:]) / 2
        else:
            nodes = np.pad(delta.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
            # Conditional local first differences remove lower-order ALE effects.
            dx = np.diff(nodes, axis=0)
            dx = np.sum((dx[:, :-1] + dx[:, 1:]) / 2 * mass, axis=1) / mass.sum(axis=1)
            dy = np.diff(nodes, axis=1)
            dy = np.sum((dy[:-1, :] + dy[1:, :]) / 2 * mass, axis=0) / mass.sum(axis=0)
            nodes -= np.r_[0, np.cumsum(dx)][:, None] + np.r_[0, np.cumsum(dy)][None, :]
            effect = (nodes[:-1, :-1] + nodes[1:, :-1] + nodes[:-1, 1:] + nodes[1:, 1:]) / 4
        effect -= np.average(effect, weights=mass)
        if any(constants):
            effect[:] = 0
        records, support = [], []
        for cell in np.ndindex(shape):
            coordinates = {key: float((axis[i] + axis[i + 1]) / 2)
                           for key, axis, i in zip(("value", "value_2"), axes, cell)}
            records.append({**coordinates, "effect": float(effect[cell])})
            support.append({**coordinates, "count": int(counts[cell]), "weight": float(mass[cell]), "unsupported": False})
        settings = self._settings("ale", indices, w, features=names, edges=dict(zip(names, [a.tolist() for a in axes])),
                                  excluded_missing_rows=excluded, centered=True,
                                  centering="Weighted mean of bin-center effects is zero",
                                  order=len(names), xlim=xlim)
        return EffectResult(pl.DataFrame(records), pl.DataFrame(support), settings)

    def shap(self, **kwargs):
        """Calculate SHAP once and reuse the most recent matching result.

        Options supplied through ``explain(shap_kwargs=...)`` are defaults for
        subsequent calls. Explicit options override them. The report's data,
        background, and configuration must remain unchanged while it is reused.

        Parameters
        ----------
        engine : {'auto', 'tree', 'permutation'}, default='auto'
            'auto' tries native tree SHAP and falls back to permutation SHAP
            when ineligible. 'tree' requires a supported single model, processed
            numeric coordinates, and a compatible prediction scale; otherwise
            raises ValueError. 'permutation' explains the complete prediction
            function and supports raw features and ensembles.
        max_background : int, default=100
            Maximum uniformly sampled background rows, without replacement.
            Background sampling ignores averaging weights. A representative
            original ``ModelData`` must be supplied to ``explain(background=...)``.
        max_samples : int, optional
            Uniform cap on explained rows, sampled without replacement using the
            report seed. None explains all rows, which can be expensive. Each
            selected row receives contributions for every report feature.
        n_permutations : int, default=20
            Permutation cycles per explained row and repeat. More cycles improve
            approximation at greater prediction cost. Ignored by tree SHAP.
        n_repeats : int, default=1
            Independent seeded permutation runs, retained separately for stability
            summaries. Tree SHAP repeats reuse the same deterministic values.
        weights : array-like or 'uniform', optional
            Nonnegative summary weights aligned to the full input population,
            before sampling. None uses ``data.weight`` or equal weights if absent;
            'uniform' forces equal weights. Does not change individual
            contributions or the background reference distribution.

        Returns
        -------
        ShapResult
            Signed contributions for all features, checked predictions/baselines,
            explained samples, settings, and summary/plot/export helpers. Reuses
            the most recent result when effective options match; changed options
            recalculate it. Results use the report's ``prediction_type`` scale.
        """
        options = {**self._shap_defaults, **kwargs}
        bound = signature(self._compute_shap).bind(**options)
        bound.apply_defaults()
        key_options = dict(bound.arguments)
        weights = key_options.pop("weights")
        if weights is not None and not isinstance(weights, str):
            values = np.asarray(weights, dtype=float)
            weights = (values.shape, values.tobytes())
        key = (tuple(key_options.items()), weights)
        if self._shap_cache is not None and self._shap_cache[0] == key:
            return self._shap_cache[1]
        result = self._compute_shap(**options)
        self._shap_cache = (key, result)
        return result

    def _compute_shap(self, *, engine="auto", max_background=100, max_samples=None, n_permutations=20,
             n_repeats=1, weights=None):
        """SHAP on the selected final prediction scale, with a deliberate background."""
        if self.background is None:
            raise ValueError("Supply background=ModelData to explain() for SHAP")
        if engine not in {"auto", "tree", "permutation"}:
            raise ValueError("engine must be auto, tree, or permutation")
        try:
            import shap
        except ImportError as exc:
            raise ImportError("SHAP explanations require ins_gbm[explain] (pip install 'ins_gbm[explain]')") from exc
        self._event("shap")
        indices = self._rows(max_samples)
        frame = self.frame[indices.tolist()]
        w = self._weights(weights, indices)
        self.background.validate_for_prediction()
        background = self.adapter.transform(self.background)
        cap = _positive_integer(max_background, "max_background")
        background_indices = np.sort(np.random.default_rng(self.seed).choice(background.height, min(cap, background.height), replace=False))
        background = background[background_indices.tolist()]
        cycles = _positive_integer(n_permutations, "n_permutations")
        repeats = _positive_integer(n_repeats, "n_repeats")
        reason = None
        tree = None
        if engine != "permutation":
            try:
                tree = self._tree(shap, frame, background, indices)
            except Exception as exc:
                if isinstance(exc, PipelineCancelled):
                    raise
                reason = str(exc)
                if engine == "tree":
                    raise ValueError(f"Native tree SHAP is ineligible: {reason}") from exc
        selected_engine = "tree" if tree is not None else "permutation"
        actual = self._predict(frame, indices)
        records, predictions = [], []
        if tree is None:
            encoded, reference, decode = self._encode(frame, background)
        for repeat in range(repeats):
            if tree is not None:
                contributions, baselines = tree
            else:
                contributions = np.zeros((frame.height, len(self.adapter.names)))
                baselines = np.zeros(frame.height)
                masker = shap.maskers.Independent(reference, max_samples=len(reference))
                for row in range(frame.height):
                    self._event("shap", repeat * frame.height + row, repeats * frame.height)
                    context_index = int(indices[row])

                    def predict(values):
                        return self._predict(decode(values), np.full(len(values), context_index, dtype=int))

                    explainer = shap.PermutationExplainer(predict, masker,
                                                        seed=self.seed + repeat * frame.height + row,
                                                        feature_names=self.adapter.names)
                    explanation = explainer(encoded[row:row + 1], max_evals=cycles * (2 * encoded.shape[1] + 1), silent=True)
                    contributions[row] = explanation.values[0]
                    baselines[row] = np.asarray(explanation.base_values).reshape(-1)[0]
            residual = actual - baselines - contributions.sum(axis=1)
            if not np.allclose(residual, 0, atol=1e-6, rtol=0) and not np.allclose(actual, baselines + contributions.sum(axis=1), atol=1e-6, rtol=2e-5):
                raise ValueError("SHAP reconstruction failed on the selected prediction scale")
            for row in range(frame.height):
                predictions.append({"repeat": repeat, "row": row, "source_row": int(indices[row]),
                                    "base_value": float(baselines[row]), "prediction": float(actual[row]),
                                    "residual": float(residual[row])})
                records.extend({"repeat": repeat, "row": row, "feature": name,
                                "contribution": float(contributions[row, j])}
                               for j, name in enumerate(self.adapter.names))
        settings = self._settings("shap", indices, w, engine=selected_engine, requested_engine=engine,
                                  fallback_reason=reason, background_indices=background_indices.tolist(),
                                  background_distribution="Uniform empirical rows; averaging weights do not alter reference probabilities",
                                  n_permutations=cycles, n_repeats=repeats, shap_version=shap.__version__,
                                  perturbation="interventional", features=self.adapter.names)
        return ShapResult(pl.DataFrame(records), pl.DataFrame(predictions), frame, settings, w)

    def _encode(self, frame, background):
        """Represent categories as reversible codes, keeping each variable one player."""
        arrays, references, categories = [], [], {}
        for name in self.adapter.names:
            if self.adapter.metadata[name]["kind"] == "categorical":
                levels = list(dict.fromkeys(frame[name].to_list() + background[name].to_list()))
                categories[name] = levels
                arrays.append([levels.index(v) for v in frame[name].to_list()])
                references.append([levels.index(v) for v in background[name].to_list()])
            else:
                arrays.append(frame[name].cast(pl.Float64).to_numpy())
                references.append(background[name].cast(pl.Float64).to_numpy())

        def decode(values):
            columns = []
            for j, name in enumerate(self.adapter.names):
                if name in categories:
                    raw = [categories[name][int(v)] for v in values[:, j]]
                    columns.append(pl.Series(name, raw, dtype=frame.schema[name]))
                else:
                    columns.append(pl.Series(name, values[:, j]).cast(frame.schema[name]))
            return pl.DataFrame(columns)

        return np.column_stack(arrays).astype(float), np.column_stack(references).astype(float), decode

    def _tree(self, shap, frame, background, indices):
        if self.adapter.ensemble is not None:
            raise ValueError("Final ensemble predictions require the general explainer")
        if self.adapter.feature_space == "raw":
            raise ValueError("Raw-feature explanations require the general explainer")
        member = self.adapter.members[0]
        fitted = member.pipeline.fitted_model
        if any(meta["kind"] != "numeric" for meta in self.adapter.metadata.values()):
            raise ValueError("Categorical/grouped one-hot coordinates require the general explainer")
        if list(member.columns) != self.adapter.names:
            raise ValueError("Model columns and diagnostic players differ")
        combined = np.vstack([frame.to_numpy(), background.to_numpy()]).astype(float)
        if not _valid_numeric(combined).all():
            raise ValueError("Missing values require the general explainer")
        scale = self.adapter.prediction_type
        rf = fitted.framework == "random_forest"
        if rf:
            if scale == "link":
                raise ValueError("Log Random Forest predictions require the general explainer")
            minimum = min(float(est.tree_.value.min()) for est in fitted.model.estimators_)
            factor = np.ones(len(indices))
            if fitted.objective == "poisson" and scale == "response" and self.data.exposure is not None:
                factor = self.data.exposure[indices.tolist()].to_numpy().astype(float)
            if minimum * min(1, factor.min()) <= 1e-10:
                raise ValueError("Random Forest clipping can affect native SHAP scale")
        elif fitted.framework in {"lightgbm", "xgboost", "catboost"}:
            if scale != "link":
                raise ValueError("Exponentiated tree predictions require the general explainer on response/rate scales")
            factor = np.ones(len(indices))
        else:
            raise ValueError(f"Unsupported native framework {fitted.framework}")
        explainer = shap.TreeExplainer(fitted.model, data=background.to_numpy().astype(np.float32),
                                       feature_perturbation="interventional", model_output="raw")
        explanation = explainer(combined.astype(np.float32), check_additivity=True)
        all_values = np.asarray(explanation.values)
        if all_values.shape != combined.shape:
            raise ValueError("Unsupported native SHAP output shape")
        native_output = np.asarray(explanation.base_values).reshape(-1) + all_values.sum(axis=1)
        # Supported wrappers have a known affine relationship to raw tree
        # output. Validate it without an O(number_of_rows ** 2) prediction pass.
        actual = self._predict(frame, indices)
        adjustment = actual - native_output[:len(indices)] * factor
        context = np.zeros(len(indices))
        if not rf:
            if self.data.offset is not None:
                context += self.data.offset[indices.tolist()].to_numpy()
            if fitted.objective == "poisson" and self.data.exposure is not None:
                context += np.log(self.data.exposure[indices.tolist()].to_numpy())
        intercept = adjustment - context
        if not np.allclose(intercept, intercept[0], atol=1e-6, rtol=2e-5):
            raise ValueError("Native output is not affine-equivalent to wrapper predictions")
        reference_actual = self._predict(background, np.full(background.height, indices[0], dtype=int))
        reference_expected = native_output[len(indices):] * factor[0] + adjustment[0]
        if not np.allclose(reference_actual, reference_expected, atol=1e-6, rtol=2e-5):
            raise ValueError("Native reference predictions disagree with wrapper predictions")
        baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0]) * factor + adjustment
        return all_values[:len(indices)] * factor[:, None], baseline
