"""Reusable explanation tables and presentation figures."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import polars as pl


def _save(fig, output_path):
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def _label(value):
    if value is None:
        return "missing"
    return f"{value:.5g}" if isinstance(value, (int, float, np.number)) else str(value)


def _metadata(directory, settings):
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "settings.json").write_text(
        json.dumps(settings, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
    )


@dataclass
class EffectResult:
    """PDP/ICE predictions or centered ALE effects, plus empirical support.

    Parameters
    ----------
    table : polars.DataFrame
        Grid coordinates and predictions/effects; ICE also includes sample-row
        and original-population row indices.
    support : polars.DataFrame
        Observed grid/bin row counts, weights, and unsupported-cell indicators.
    settings : dict
        Calculation method, feature metadata, prediction units, sampling,
        averaging weights, and plotting options produced by the report.
    """

    table: pl.DataFrame
    support: pl.DataFrame
    settings: dict

    def plot(self, output_path=None):
        """Plot stored effects with observed support, without recalculating them.

        Parameters
        ----------
        output_path : str or pathlib.Path, optional
            File to save the figure to; its suffix selects the image format.
            None returns the figure without saving. Parent directory must exist.

        Returns
        -------
        matplotlib.figure.Figure
            Curve, ICE profiles, or two-feature heatmap with support counts.
            ICE displays at most settings['display_rows'] seeded sample curves;
            this does not remove calculated rows from ``table``.
        """
        import matplotlib.pyplot as plt

        features = self.settings["features"]
        kind = self.settings["method"]
        value_column = "effect" if kind == "ale" else "prediction"
        fig, (ax, density) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[4, 1])
        if len(features) == 2:
            xs = self.table["value"].unique(maintain_order=True).to_list()
            ys = self.table["value_2"].unique(maintain_order=True).to_list()
            z = np.full((len(ys), len(xs)), np.nan)
            for row in self.table.iter_rows(named=True):
                z[ys.index(row["value_2"]), xs.index(row["value"])] = row[value_column]
            im = ax.imshow(z, origin="lower", aspect="auto")
            ax.set_xticks(range(len(xs)), [_label(v) for v in xs], rotation=45, ha="right")
            ax.set_yticks(range(len(ys)), [_label(v) for v in ys])
            ax.set_ylabel(features[1])
            fig.colorbar(im, ax=ax, label=self.settings["units"] + (" (centered effect)" if kind == "ale" else ""))
            for cell in self.support.filter(pl.col("unsupported")).iter_rows(named=True):
                ax.scatter(xs.index(cell["value"]), ys.index(cell["value_2"]), color="white", edgecolors="black", marker="X", s=55)
        else:
            values = self.table["value"].unique(maintain_order=True).to_list()
            numeric = self.settings["feature_metadata"][features[0]]["kind"] == "numeric"
            x = np.asarray(values, dtype=float) if numeric else np.arange(len(values))
            if kind == "ice":
                ids = self.table["row"].unique(maintain_order=True).to_list()
                cap = self.settings.get("display_rows", 100)
                ids = np.random.default_rng(self.settings["seed"]).choice(ids, min(cap, len(ids)), replace=False)
                for row_id in ids:
                    y = self.table.filter(pl.col("row") == int(row_id))[value_column]
                    if numeric:
                        ax.plot(x, y, alpha=.25)
                    else:
                        ax.scatter(x, y, alpha=.25, s=12)
            else:
                if numeric:
                    ax.plot(x, self.table[value_column], marker="o")
                else:
                    ax.scatter(x, self.table[value_column])
            if not numeric:
                ax.set_xticks(x, [_label(v) for v in values], rotation=45, ha="right")
            ax.set_ylabel(self.settings["units"] + (" (centered difference)" if kind == "ale" or self.settings.get("centered") else ""))
        ax.set_xlabel(features[0])
        ax.set_title(f"{kind.upper()} — {self.settings['prediction_type']}")
        if "count" in self.support.columns:
            colors = ["tomato" if unsupported else "gray" for unsupported in self.support["unsupported"]]
            density.bar(np.arange(self.support.height), self.support["count"], color=colors)
            absent = np.flatnonzero(self.support["unsupported"].to_numpy())
            if len(absent):
                density.scatter(absent, np.zeros(len(absent)), color="tomato", marker="x", label="No observed support")
                density.legend(fontsize="small")
            density.set_ylabel("Rows")
            density.set_xlabel("Grid/bin support (table order)")
        else:
            density.set_axis_off()
        return _save(fig, output_path)

    def export(self, directory):
        """Write effect tables, settings, and figures from this stored result.

        Parameters
        ----------
        directory : str or pathlib.Path
            Destination directory, created with parents if needed. Writes
            effects.csv, support.csv, settings.json, effect.png, and effect.svg.
            Existing files with these names are overwritten.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        _metadata(directory, self.settings)
        self.table.write_csv(Path(directory) / "effects.csv")
        self.support.write_csv(Path(directory) / "support.csv")
        for suffix in ("png", "svg"):
            plt.close(self.plot(Path(directory) / f"effect.{suffix}"))


@dataclass
class ShapResult:
    """Long-form signed contributions and independently checked predictions.

    Parameters
    ----------
    contributions : polars.DataFrame
        Signed contribution for each repeat, sample-row index, and feature.
    predictions : polars.DataFrame
        Per-repeat/per-row baseline, prediction, reconstruction residual, and
        source_row index in the original explanation population.
    samples : polars.DataFrame
        Explained feature values in sampled-row order, without repeat duplication.
    settings : dict
        Prediction scale/units, feature metadata, engine, and sampling settings.
    weights : numpy.ndarray
        Nonnegative averaging weights aligned to samples, used for summaries.

    Notes
    -----
    Summary, plotting, and export methods reuse these tables without recalculating
    SHAP. Keep a cached result unchanged when sharing it through a report.
    """

    contributions: pl.DataFrame
    predictions: pl.DataFrame
    samples: pl.DataFrame
    settings: dict
    weights: np.ndarray = field(repr=False)

    def _grouped(self, groups=None):
        frame = self.contributions
        if not groups:
            return frame
        mapping = {}
        features = set(frame["feature"].unique())
        for group, members in groups.items():
            for name in members:
                if name not in features or name in mapping:
                    raise ValueError(f"Unknown or overlapping display-group feature: {name}")
                mapping[name] = group
        remaining = features - mapping.keys()
        if set(groups) & remaining:
            raise ValueError("Display-group names collide with ungrouped features")
        return frame.with_columns(pl.col("feature").replace(mapping)).group_by(
            "repeat", "row", "feature", maintain_order=True
        ).agg(pl.col("contribution").sum())

    def summary(self, groups=None):
        """Summarize weighted feature importance across rows and repeats.

        Parameters
        ----------
        groups : mapping of str to sequence of str, optional
            Display-group name to feature names. Contributions are summed with
            their signs within each row/repeat before taking absolute values.
            Features must exist and belong to at most one group; group names must
            not collide with ungrouped features. Other features remain separate.
            None summarizes individual features. Does not recalculate SHAP.

        Returns
        -------
        polars.DataFrame
            Sorted by descending mean_abs_contribution. Includes weighted signed
            mean_contribution and repeat_std, the population standard deviation
            of each repeat's weighted mean absolute contribution.
        """
        frame = self._grouped(groups)
        records = []
        for name in frame["feature"].unique(maintain_order=True):
            selected = frame.filter(pl.col("feature") == name)
            row_values = selected.group_by("row").agg(pl.col("contribution").mean()).sort("row")
            run_scores = []
            for repeat in selected["repeat"].unique():
                values = selected.filter(pl.col("repeat") == repeat).sort("row")["contribution"].to_numpy()
                run_scores.append(np.average(np.abs(values), weights=self.weights))
            records.append({"feature": name, "mean_abs_contribution": float(np.mean(run_scores)),
                            "mean_contribution": float(np.average(row_values["contribution"], weights=self.weights)),
                            "repeat_std": float(np.std(run_scores))})
        return pl.DataFrame(records).sort("mean_abs_contribution", descending=True)

    def segment_summary(self, segments):
        """Summarize stored SHAP contributions within population segments.

        Parameters
        ----------
        segments : sequence
            One label per explained sample, in ``samples`` order, rather than
            one per original input row or per repeat. Labels are cast to strings
            and reused across repeats. Use predictions['source_row'] from a
            single repeat, ordered by row, to map original-population labels
            to the selected samples.

        Returns
        -------
        polars.DataFrame
            Per-segment/per-feature weighted mean absolute contribution, weighted
            signed mean contribution, and unique sample count. Averages include
            all repeats; means are null for segments with zero total weight.
        """
        if len(segments) != self.samples.height:
            raise ValueError("Segments must align with explained samples")
        labels = pl.Series("segment", segments).cast(pl.String).to_list()
        frame = self.contributions.with_columns(
            pl.Series("segment", [labels[r] for r in self.contributions["row"]]),
            pl.Series("weight", [self.weights[r] for r in self.contributions["row"]]),
        )
        return frame.group_by("segment", "feature").agg(
            pl.when(pl.col("weight").sum() > 0).then(
                (pl.col("contribution").abs() * pl.col("weight")).sum() / pl.col("weight").sum()
            ).otherwise(None).alias("mean_abs_contribution"),
            pl.when(pl.col("weight").sum() > 0).then(
                (pl.col("contribution") * pl.col("weight")).sum() / pl.col("weight").sum()
            ).otherwise(None).alias("mean_contribution"),
            pl.col("row").n_unique().alias("n_rows"),
        )

    def plot_importance(self, top_n=20, groups=None, output_path=None):
        """Plot a horizontal bar chart of weighted mean absolute SHAP values.

        Parameters
        ----------
        top_n : int, default=20
            Number of highest-importance features or display groups to show.
        groups : mapping of str to sequence of str, optional
            Display-group name to nonoverlapping existing feature names. Signed
            contributions are summed per row/repeat before computing importance;
            ungrouped features remain separate. See ``summary()``.
        output_path : str or pathlib.Path, optional
            Image file to save; suffix selects format. Parent must exist.
            None returns the figure without saving.

        Returns
        -------
        matplotlib.figure.Figure
            Importance chart built from stored contributions and summary weights.
        """
        import matplotlib.pyplot as plt

        table = self.summary(groups).head(top_n).reverse()
        fig, ax = plt.subplots(figsize=(8, max(4, table.height * .3)))
        ax.barh(table["feature"], table["mean_abs_contribution"])
        ax.set_xlabel(f"Mean absolute SHAP contribution ({self.settings['units']})")
        return _save(fig, output_path)

    def plot_beeswarm(self, top_n=20, output_path=None):
        """Plot per-row SHAP contributions colored by feature values.

        Parameters
        ----------
        top_n : int, default=20
            Number of features to show, ranked by weighted mean absolute SHAP.
            Each dot uses a row's contribution averaged across repeats; points
            are not sampled or sized by averaging weights.
        output_path : str or pathlib.Path, optional
            Image file to save; suffix selects format. Parent must exist.
            None returns the figure without saving.

        Returns
        -------
        matplotlib.figure.Figure
            Beeswarm of stored contributions. Numeric values determine color;
            categories use display codes rather than a numerical effect scale.
        """
        import matplotlib.pyplot as plt

        names = self.summary().head(top_n)["feature"].to_list()[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * .35)))
        for index, name in enumerate(names):
            values = self.contributions.filter(pl.col("feature") == name).group_by("row").agg(
                pl.col("contribution").mean()).sort("row")["contribution"].to_numpy()
            color = self.samples[name]
            if color.dtype.is_numeric() and self.settings["feature_metadata"][name]["kind"] == "numeric":
                colors = color.cast(pl.Float64).to_numpy().copy()
                colors[~np.isfinite(colors)] = np.nan
            else:
                colors = color.cast(pl.String).fill_null("missing").rank("dense").to_numpy()
            # Pack points vertically within narrow contribution bins.
            width = np.ptp(values)
            bins = np.floor((values - values.min()) / (width if width else 1) * 40).astype(int)
            offsets = np.zeros(len(values))
            for cell in np.unique(bins):
                positions = np.flatnonzero(bins == cell)
                steps = np.arange(len(positions))
                offsets[positions] = np.where(steps % 2, -(steps + 1) // 2, steps // 2)
            offsets *= .3 / max(1, np.abs(offsets).max())
            ax.scatter(values, index + offsets, c=colors, cmap="coolwarm", s=12, alpha=.65, plotnonfinite=True)
        ax.set_yticks(range(len(names)), names)
        ax.axvline(0, color="gray", linewidth=.7)
        ax.set_xlabel(f"SHAP contribution ({self.settings['units']}); color = feature value/level")
        return _save(fig, output_path)

    def plot_waterfall(self, row=0, repeat=0, top_n=20, output_path=None):
        """Show how signed contributions connect one baseline to its prediction.

        Parameters
        ----------
        row : int, default=0
            Zero-based index within ``samples``, not the original input row
            number. The corresponding prediction records its ``source_row``.
        repeat : int, default=0
            Zero-based SHAP repeat to display. Unknown row/repeat pairs raise
            ValueError; contributions are not averaged across repeats here.
        top_n : int or None, default=20
            Number of largest absolute contributions shown individually. Any
            remaining features form an additional signed-sum 'Other' bar.
            None shows all features; a truncation limit must be positive.
        output_path : str or pathlib.Path, optional
            Image file to save; suffix selects format. Parent must exist.
            None returns the figure without saving.

        Returns
        -------
        matplotlib.figure.Figure
            Waterfall with baseline and final prediction on the stored scale.
        """
        import matplotlib.pyplot as plt

        table = self.contributions.filter((pl.col("row") == row) & (pl.col("repeat") == repeat))
        pred = self.predictions.filter((pl.col("row") == row) & (pl.col("repeat") == repeat))
        if pred.height != 1:
            raise ValueError("Unknown row/repeat")
        base = pred["base_value"][0]
        table = table.sort(pl.col("contribution").abs(), descending=True)
        values = table["contribution"].to_numpy()
        labels = table["feature"].to_list()
        if top_n is not None and len(values) > top_n:
            if top_n < 1:
                raise ValueError("top_n must be positive")
            labels = labels[:top_n] + [f"Other {len(values) - top_n} features"]
            values = np.r_[values[:top_n], values[top_n:].sum()]
        starts = base + np.r_[0, np.cumsum(values)[:-1]]
        fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * .3)))
        ax.barh(labels, values, left=starts, color=np.where(values >= 0, "tomato", "steelblue"))
        ax.axvline(base, color="gray", linestyle="--", label=f"Reference {base:.5g}")
        final = pred["prediction"][0]
        ax.axvline(final, color="black", label=f"Prediction {final:.5g}")
        ax.set_xlabel(self.settings["units"])
        ax.legend()
        return _save(fig, output_path)

    def plot_dependence(self, feature, color_by=None, output_path=None):
        """Plot a feature's observed values against its stored SHAP contributions.

        Parameters
        ----------
        feature : str
            Explained feature name present in ``samples`` and ``contributions``.
            Each sample's contribution is averaged across repeats. This plots
            SHAP values; it does not calculate a partial dependence curve.
        color_by : str, optional
            Feature in ``samples`` used to color points. Numeric values are used
            directly; categorical values use display codes. None uses one color.
        output_path : str or pathlib.Path, optional
            Image file to save; suffix selects format. Parent must exist.
            None returns the figure without saving.

        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot on the stored prediction scale, with categorical axis
            labels and a color bar where applicable.
        """
        import matplotlib.pyplot as plt

        values = self.contributions.filter(pl.col("feature") == feature).group_by("row").agg(
            pl.col("contribution").mean()).sort("row")
        if not values.height:
            raise ValueError(f"Unknown feature {feature}")
        x = self.samples[feature]
        numeric = self.settings["feature_metadata"][feature]["kind"] == "numeric"
        if numeric:
            x_values = x.to_numpy()
            labels = None
        else:
            labels = x.cast(pl.String).fill_null("missing").unique(maintain_order=True).to_list()
            x_values = [labels.index(v) for v in x.cast(pl.String).fill_null("missing")]
        color = None
        if color_by:
            series = self.samples[color_by]
            color = series.to_numpy() if series.dtype.is_numeric() else series.cast(pl.String).fill_null("missing").rank("dense").to_numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        dots = ax.scatter(x_values, values["contribution"], c=color, cmap="viridis" if color_by else None)
        if color_by:
            fig.colorbar(dots, ax=ax, label=color_by + " (category codes where applicable)")
        if labels is not None:
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_xlabel(feature)
        ax.set_ylabel(f"SHAP contribution ({self.settings['units']})")
        return _save(fig, output_path)

    def export(self, directory):
        """Write stored SHAP tables, settings, and summary figures to disk.

        Parameters
        ----------
        directory : str or pathlib.Path
            Destination directory, created with parents if needed. Writes
            contributions.csv, predictions.csv, samples.csv, importance.csv,
            settings.json, and importance/beeswarm figures in PNG and SVG.
            Existing files with these names are overwritten. Does not recalculate
            SHAP; figures and importance use the stored contributions.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        _metadata(directory, self.settings)
        for name in ("contributions", "predictions", "samples"):
            getattr(self, name).write_csv(Path(directory) / f"{name}.csv")
        self.summary().write_csv(Path(directory) / "importance.csv")
        for suffix in ("png", "svg"):
            plt.close(self.plot_importance(output_path=Path(directory) / f"importance.{suffix}"))
            plt.close(self.plot_beeswarm(output_path=Path(directory) / f"beeswarm.{suffix}"))
