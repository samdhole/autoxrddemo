import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from autoxrd.loader import XRDData
from autoxrd.fitter import XRDFitter, FitResult
from autoxrd.analyzer import XRDAnalyzer
from autoxrd.reporter import HTMLReporter


def make_mock_fit_result(
    name="TestSample",
    center=28.44,
    fwhm=0.20,
    r_squared=0.98,
    wavelength=1.54056,
) -> FitResult:
    all_peaks = [
        {"center": center, "fwhm": fwhm, "amplitude": 80.0, "eta": 0.5,
         "d_spacing": wavelength / (2.0 * np.sin(np.deg2rad(center / 2.0))),
         "relative_intensity": 100.0},
        {"center": center + 3.0, "fwhm": fwhm, "amplitude": 40.0, "eta": 0.5,
         "d_spacing": wavelength / (2.0 * np.sin(np.deg2rad((center + 3.0) / 2.0))),
         "relative_intensity": 50.0},
    ]
    return FitResult(
        name=name,
        dominant_peak={"center": center, "fwhm": fwhm, "amplitude": 80.0, "eta": 0.5},
        all_peaks=all_peaks,
        r_squared=r_squared,
        aic=-100.0,
        bic=-90.0,
        n_peaks=2,
        wavelength=wavelength,
    )


class TestScherrerAndDSpacing:
    def test_d_spacing_known_value(self):
        # Quartz 26.65°: d = 1.54056 / (2 * sin(13.325°)) ≈ 3.343 Å
        d = XRDAnalyzer._d_spacing(26.65, 1.54056)
        assert abs(d - 3.343) < 0.005

    def test_scherrer_known_value(self):
        # center=28.44°, fwhm=0.20°, λ=1.54056 Å
        # β = deg2rad(0.20) ≈ 0.003491 rad
        # θ = 14.22° = 0.24824 rad, cos(θ) ≈ 0.97059
        # D = 0.9 * 1.54056 / (0.003491 * 0.97059) Å ≈ 409.7 Å ≈ 41.0 nm
        D = XRDAnalyzer._scherrer(0.20, 28.44, 1.54056)
        assert abs(D - 41.0) < 2.0

    def test_scherrer_capped_at_max(self):
        # Very narrow fwhm → huge crystallite → capped at CRYSTALLITE_SIZE_CAP_NM
        D = XRDAnalyzer._scherrer(0.001, 28.44, 1.54056)
        assert D == XRDAnalyzer.CRYSTALLITE_SIZE_CAP_NM

    def test_scherrer_tiny_nonzero_fwhm_hits_instrument_cap(self):
        D = XRDAnalyzer._scherrer(1e-10, 28.44, 1.54056)
        assert D == XRDAnalyzer.CRYSTALLITE_SIZE_CAP_NM

    def test_scherrer_cap_boundary(self):
        # Derive the FWHM that yields exactly CRYSTALLITE_SIZE_CAP_NM, then
        # verify the cap triggers just above and not just below.
        # D = (K * lam) / (beta_rad * cos(theta))  →  beta_rad = (K * lam) / (D_A * cos(theta))
        cap = XRDAnalyzer.CRYSTALLITE_SIZE_CAP_NM  # 200 nm
        lam, center, K = 1.54056, 28.44, XRDAnalyzer.K_SCHERRER
        theta = np.deg2rad(center / 2)
        beta_rad = (K * lam) / (cap * 10 * np.cos(theta))  # cap*10: nm → Å
        fwhm_boundary = float(np.rad2deg(beta_rad))
        # Slightly wider FWHM → smaller crystallite, below cap
        assert XRDAnalyzer._scherrer(fwhm_boundary * 1.01, center, lam) < cap
        # Slightly narrower FWHM → larger crystallite, hits cap
        assert XRDAnalyzer._scherrer(fwhm_boundary * 0.99, center, lam) == cap

    def test_scherrer_returns_nan_for_zero_fwhm(self):
        D = XRDAnalyzer._scherrer(0.0, 28.44, 1.54056)
        assert np.isnan(D)

    def test_d_spacing_returns_nan_for_zero_angle(self):
        d = XRDAnalyzer._d_spacing(0.0, 1.54056)
        assert np.isnan(d)


class TestBuildSummaryTable:
    def test_returns_dataframe(self):
        fr = make_mock_fit_result()
        table = XRDAnalyzer.build_summary_table({"TestSample": fr})
        assert isinstance(table, pd.DataFrame)

    def test_column_names(self):
        fr = make_mock_fit_result()
        table = XRDAnalyzer.build_summary_table({"TestSample": fr})
        expected = ["Sample", "Phase", "2θ (°)", "d-spacing (Å)", "FWHM (°)",
                    "Crystallite Size (nm)", "R²", "AIC", "BIC", "N_peaks", "Flag"]
        assert list(table.columns) == expected

    def test_phase_extracted_from_name(self):
        fr = make_mock_fit_result(name="Quartz__R040031")
        table = XRDAnalyzer.build_summary_table({"Quartz__R040031": fr})
        assert table.iloc[0]["Phase"] == "Quartz"

    def test_d_spacing_correct(self):
        fr = make_mock_fit_result(center=26.65)
        table = XRDAnalyzer.build_summary_table({"s": fr})
        assert abs(table.iloc[0]["d-spacing (Å)"] - 3.343) < 0.01

    def test_r_squared_present(self):
        fr = make_mock_fit_result(r_squared=0.987)
        table = XRDAnalyzer.build_summary_table({"s": fr})
        assert abs(table.iloc[0]["R²"] - 0.987) < 0.0001

    def test_empty_fit_results_returns_schema(self):
        table = XRDAnalyzer.build_summary_table({})
        expected = ["Sample", "Phase", "2θ (°)", "d-spacing (Å)", "FWHM (°)",
                    "Crystallite Size (nm)", "R²", "AIC", "BIC", "N_peaks", "Flag"]
        assert list(table.columns) == expected
        assert table.empty

    def test_zero_peak_fit_result_builds_nan_summary_row(self):
        fr = FitResult(
            name="no_peaks",
            dominant_peak={"center": np.nan, "fwhm": np.nan, "amplitude": np.nan, "eta": np.nan},
            all_peaks=[],
            r_squared=0.0,
            aic=0.0,
            bic=0.0,
            n_peaks=0,
            wavelength=1.54056,
        )

        table = XRDAnalyzer.build_summary_table({"no_peaks": fr})

        assert len(table) == 1
        assert table.iloc[0]["N_peaks"] == 0
        assert np.isnan(table.iloc[0]["2θ (°)"])
        assert np.isnan(table.iloc[0]["FWHM (°)"])


class TestFlagOutliers:
    def _make_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Sample": ["A", "B", "C", "D", "E"],
            "Phase": ["Q", "C", "F", "M", "K"],
            "2θ (°)": [26.0, 29.0, 28.0, 35.0, 12.0],
            "d-spacing (Å)": [3.4, 3.1, 3.2, 2.6, 7.2],
            "FWHM (°)": [0.15, 0.16, 0.14, 0.15, 1.50],  # K is outlier
            "Crystallite Size (nm)": [60.0, 58.0, 62.0, 60.0, 6.0],  # K is outlier
            "R²": [0.98, 0.97, 0.99, 0.96, 0.55],  # K below POOR_FIT_R2_THRESHOLD (0.70)
            "AIC": [-100.0, -95.0, -110.0, -98.0, -80.0],
            "BIC": [-90.0, -85.0, -100.0, -88.0, -70.0],
            "N_peaks": [3, 4, 3, 5, 2],
            "Flag": ["", "", "", "", ""],
        })

    def test_poor_fit_flagged(self):
        table = self._make_table()
        result = XRDAnalyzer.flag_outliers(table)
        assert "Poor fit" in result.iloc[4]["Flag"]

    def test_flag_column_present(self):
        table = self._make_table()
        result = XRDAnalyzer.flag_outliers(table)
        assert "Flag" in result.columns
        assert "Flag (IQR supplement)" not in result.columns

    def test_no_false_positives_on_uniform_data(self):
        # All values identical — no outliers should be flagged
        df = pd.DataFrame({
            "Sample": ["A", "B", "C"],
            "Phase": ["Q", "C", "F"],
            "2θ (°)": [26.0, 26.0, 26.0],
            "d-spacing (Å)": [3.4, 3.4, 3.4],
            "FWHM (°)": [0.15, 0.15, 0.15],
            "Crystallite Size (nm)": [60.0, 60.0, 60.0],
            "R²": [0.98, 0.98, 0.98],
            "AIC": [-100.0, -100.0, -100.0],
            "BIC": [-90.0, -90.0, -90.0],
            "N_peaks": [3, 3, 3],
            "Flag": ["", "", ""],
        })
        result = XRDAnalyzer.flag_outliers(df)
        assert all(result["Flag"] == "")

    def test_instrument_limited_size_flagged(self):
        df = self._make_table()
        df.loc[0, "Crystallite Size (nm)"] = XRDAnalyzer.CRYSTALLITE_SIZE_CAP_NM
        result = XRDAnalyzer.flag_outliers(df)
        assert "Instrument-limited" in result.loc[0, "Flag"]

    def test_single_row_iqr_does_not_crash_or_flag_iqr(self):
        df = self._make_table().iloc[[0]].copy()
        result = XRDAnalyzer.flag_outliers(df)
        assert len(result) == 1
        assert "IQR" not in result.iloc[0]["Flag"]

    def test_single_row_nan_fwhm_and_size_do_not_flag_iqr(self):
        df = self._make_table().iloc[[0]].copy()
        df.loc[df.index[0], "FWHM (°)"] = np.nan
        df.loc[df.index[0], "Crystallite Size (nm)"] = np.nan

        result = XRDAnalyzer.flag_outliers(df)

        assert len(result) == 1
        assert result.iloc[0]["Flag"] == ""


class TestHTMLReporter:
    def test_style_table_accepts_empty_r2_dataframe(self):
        html = HTMLReporter(Path("."))._style_table(pd.DataFrame(columns=["R²"]))

        assert "xrd-summary-table" in html

    def test_render_includes_buyer_facing_process_signal(self):
        results = {
            "Sample_01": make_mock_fit_result(name="Sample_01", center=26.65, fwhm=0.08, r_squared=0.92),
            "Sample_02": make_mock_fit_result(name="Sample_02", center=26.65, fwhm=0.14, r_squared=0.93),
            "Sample_03": make_mock_fit_result(name="Sample_03", center=26.65, fwhm=0.22, r_squared=0.91),
        }
        summary = XRDAnalyzer.flag_outliers(XRDAnalyzer.build_summary_table(results, parse_phase=lambda n: n))
        reporter = HTMLReporter(Path(__file__).resolve().parent.parent / "templates")

        html = reporter.render(
            summary_table=summary,
            fit_results=results,
            xrd_data={},
            metadata={"title": "Memo", "analyst": "A", "sample_count": 3},
        )

        assert "Process Signal" in html
        assert "Decision Implication" in html
        assert "Validation Notes" in html
        assert "FWHM broadens from 0.0800° to 0.2200°" in html
        assert "dominant reflection remains position-stable" in html
        assert "scientist time" in html

    def test_render_escapes_untrusted_metadata_and_sample_text(self):
        malicious_name = '<script>alert("sample")</script>'
        fr = make_mock_fit_result(name=malicious_name)
        summary = XRDAnalyzer.build_summary_table({malicious_name: fr})
        summary = XRDAnalyzer.flag_outliers(summary)
        reporter = HTMLReporter(Path(__file__).resolve().parent.parent / "templates")

        html = reporter.render(
            summary_table=summary,
            fit_results={malicious_name: fr},
            xrd_data={},
            metadata={
                "title": '<script>alert("title")</script>',
                "analyst": '<img src=x onerror=alert("analyst")>',
                "sample_count": 1,
            },
        )

        assert "<script>" not in html
        assert "<img src=x" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;img src=x" in html

    def test_render_uses_only_available_figures_for_missing_xrd_data(self):
        fr = make_mock_fit_result(name="missing_from_xrd")
        summary = XRDAnalyzer.build_summary_table({"missing_from_xrd": fr})
        reporter = HTMLReporter(Path(__file__).resolve().parent.parent / "templates")

        html = reporter.render(
            summary_table=summary,
            fit_results={"missing_from_xrd": fr},
            xrd_data={},
            metadata={"title": "Memo", "analyst": "A", "sample_count": 1},
        )

        assert "missing_from_xrd" in html
        assert "data:image/png;base64" not in html


class TestBuildPeakTable:
    def test_returns_dataframe(self):
        fr = make_mock_fit_result()
        table = XRDAnalyzer.build_peak_table({"TestSample": fr})
        assert isinstance(table, pd.DataFrame)

    def test_has_correct_columns(self):
        fr = make_mock_fit_result()
        table = XRDAnalyzer.build_peak_table({"TestSample": fr})
        expected = {"Sample", "Phase", "Peak #", "2θ (°)", "d-spacing (Å)",
                    "FWHM (°)", "Crystallite Size (nm)", "Rel. Intensity (%)", "η"}
        assert expected <= set(table.columns)

    def test_row_count_matches_total_peaks(self):
        fr1 = make_mock_fit_result(name="A")
        fr2 = make_mock_fit_result(name="B")
        table = XRDAnalyzer.build_peak_table({"A": fr1, "B": fr2})
        assert len(table) == fr1.n_peaks + fr2.n_peaks

    def test_empty_all_peaks_returns_empty_peak_schema(self):
        fr = FitResult(
            name="empty",
            dominant_peak={"center": np.nan, "fwhm": np.nan, "amplitude": np.nan, "eta": np.nan},
            all_peaks=[],
            r_squared=0.0,
            aic=0.0,
            bic=0.0,
            n_peaks=0,
            wavelength=1.54056,
        )
        table = XRDAnalyzer.build_peak_table({"empty": fr})
        expected = ["Sample", "Phase", "Peak #", "2θ (°)", "d-spacing (Å)",
                    "FWHM (°)", "Crystallite Size (nm)", "Rel. Intensity (%)", "η"]
        assert table.empty
        assert list(table.columns) == expected


class TestBuildTrendModel:
    def _make_peak_table(self) -> pd.DataFrame:
        rows = []
        for i, sample in enumerate(["S1", "S2", "S3", "S4", "S5"]):
            rows.append({
                "Sample": sample, "Peak #": 1,
                "2θ (°)": 26.65, "FWHM (°)": 0.08 + i * 0.03,
            })
            rows.append({
                "Sample": sample, "Peak #": 2,
                "2θ (°)": 36.50, "FWHM (°)": 0.10 + i * 0.02,
            })
        return pd.DataFrame(rows)

    def test_schema(self):
        peak_table = self._make_peak_table()
        models = XRDAnalyzer.build_trend_model(
            peak_table, ["S1", "S2", "S3", "S4", "S5"]
        )
        required = {
            "Peak #", "Center (°)", "N_obs",
            "Position slope (°/sample)", "Position R²",
            "FWHM slope (°/sample)", "FWHM R²",
        }
        assert required <= set(models.columns)

    def test_row_count(self):
        peak_table = self._make_peak_table()
        models = XRDAnalyzer.build_trend_model(
            peak_table, ["S1", "S2", "S3", "S4", "S5"]
        )
        assert len(models) == 2  # two peak families

    def test_fwhm_slope_positive_for_increasing_fwhm(self):
        peak_table = self._make_peak_table()
        models = XRDAnalyzer.build_trend_model(
            peak_table, ["S1", "S2", "S3", "S4", "S5"]
        )
        row = models[models["Peak #"] == 1].iloc[0]
        assert abs(row["FWHM slope (°/sample)"] - 0.03) < 1e-4

    def test_min_obs_produces_nan(self):
        # Only 2 samples for Peak #1 — below threshold of 3
        rows = [
            {"Sample": "S1", "Peak #": 1, "2θ (°)": 26.65, "FWHM (°)": 0.08},
            {"Sample": "S2", "Peak #": 1, "2θ (°)": 26.65, "FWHM (°)": 0.10},
        ]
        models = XRDAnalyzer.build_trend_model(
            pd.DataFrame(rows), ["S1", "S2"]
        )
        assert np.isnan(models.iloc[0]["FWHM slope (°/sample)"])
        assert models.iloc[0]["N_obs"] == 2

    def test_empty_peak_table_returns_empty_with_schema(self):
        models = XRDAnalyzer.build_trend_model(pd.DataFrame(), ["S1", "S2"])
        assert len(models) == 0
        assert "FWHM slope (°/sample)" in models.columns

    def test_constant_fwhm_gives_nan_r2(self):
        rows = [
            {"Sample": f"S{i}", "Peak #": 1, "2θ (°)": 26.65 + i * 0.01, "FWHM (°)": 0.10}
            for i in range(1, 6)
        ]
        models = XRDAnalyzer.build_trend_model(
            pd.DataFrame(rows), [f"S{i}" for i in range(1, 6)]
        )
        assert np.isnan(models.iloc[0]["FWHM R²"])

    def test_extra_samples_filtered_out(self):
        peak_table = self._make_peak_table()  # S1–S5
        models = XRDAnalyzer.build_trend_model(peak_table, ["S1", "S2", "S3"])
        assert all(models["N_obs"] == 3)

    def test_nan_fwhm_does_not_poison_position_fit(self):
        # 3 of 5 samples have NaN FWHM → only 2 valid FWHM values, below the ≥3 threshold,
        # so FWHM slope is NaN. All 5 positions are valid → position fit succeeds.
        rows = [
            {"Sample": f"S{i}", "Peak #": 1, "2θ (°)": 26.65 + i * 0.01,
             "FWHM (°)": float("nan") if i in (2, 3, 4) else 0.08 + i * 0.01}
            for i in range(1, 6)
        ]
        models = XRDAnalyzer.build_trend_model(
            pd.DataFrame(rows), [f"S{i}" for i in range(1, 6)]
        )
        assert not np.isnan(models.iloc[0]["Position slope (°/sample)"])
        assert np.isnan(models.iloc[0]["FWHM slope (°/sample)"])

    def test_r_squared_clamped_to_zero_for_pathological_underfit(self, monkeypatch):
        rows = [
            {"Sample": "S1", "Peak #": 1, "2θ (°)": 0.0, "FWHM (°)": 0.10},
            {"Sample": "S2", "Peak #": 1, "2θ (°)": 100.0, "FWHM (°)": 0.11},
            {"Sample": "S3", "Peak #": 1, "2θ (°)": 0.0, "FWHM (°)": 0.12},
        ]

        monkeypatch.setattr(np, "polyfit", lambda *_args, **_kwargs: np.array([0.0, 0.0]))

        models = XRDAnalyzer.build_trend_model(pd.DataFrame(rows), ["S1", "S2", "S3"])

        assert models.iloc[0]["Position R²"] == 0.0


class TestParsePhase:
    def test_default_splits_on_double_underscore(self):
        fr = make_mock_fit_result(name="Quartz__R040031")
        table = XRDAnalyzer.build_summary_table({"Quartz__R040031": fr})
        assert table.iloc[0]["Phase"] == "Quartz"

    def test_custom_parse_phase_summary_table(self):
        fr = make_mock_fit_result(name="Sample_A_300C")
        table = XRDAnalyzer.build_summary_table(
            {"Sample_A_300C": fr},
            parse_phase=lambda n: n,
        )
        assert table.iloc[0]["Phase"] == "Sample_A_300C"

    def test_custom_parse_phase_peak_table(self):
        fr = make_mock_fit_result(name="Sample_B_400C")
        table = XRDAnalyzer.build_peak_table(
            {"Sample_B_400C": fr},
            parse_phase=lambda n: n.split("_")[1],
        )
        assert table.iloc[0]["Phase"] == "B"

    def test_custom_parse_phase_trend_figure_positional(self):
        rows = [
            {"Sample": "Sample_A_300C", "Peak #": 1, "2θ (°)": 26.0, "FWHM (°)": 0.10},
            {"Sample": "Sample_B_400C", "Peak #": 1, "2θ (°)": 26.1, "FWHM (°)": 0.11},
            {"Sample": "Sample_C_500C", "Peak #": 1, "2θ (°)": 26.2, "FWHM (°)": 0.12},
        ]
        figure = HTMLReporter.build_trend_figure(
            pd.DataFrame(rows),
            ["Sample_A_300C", "Sample_B_400C", "Sample_C_500C"],
            lambda n: n.split("_")[1],
        )
        assert isinstance(figure, str)
        assert len(figure) > 0

    def test_build_trend_figure_rejects_non_callable_parse_phase(self):
        rows = [
            {"Sample": "S1", "Peak #": 1, "2θ (°)": 26.0, "FWHM (°)": 0.10},
        ]
        with pytest.raises(TypeError, match="parse_phase must be callable"):
            HTMLReporter.build_trend_figure(
                pd.DataFrame(rows),
                ["S1"],
                parse_phase="not-callable",
            )

    def test_build_trend_figure_draws_trend_model_overlays(self, monkeypatch):
        from matplotlib.axes import Axes

        rows = [
            {"Sample": "S1", "Peak #": 1, "2θ (°)": 26.0, "FWHM (°)": 0.10},
            {"Sample": "S2", "Peak #": 1, "2θ (°)": 26.1, "FWHM (°)": 0.12},
            {"Sample": "S3", "Peak #": 1, "2θ (°)": 26.2, "FWHM (°)": 0.14},
        ]
        trend_models = pd.DataFrame([{
            "Peak #": 1,
            "Center (°)": 26.1,
            "N_obs": 3,
            "Position slope (°/sample)": 0.1,
            "Position R²": 1.0,
            "FWHM slope (°/sample)": 0.02,
            "FWHM R²": 1.0,
        }])
        dashed_lines = []
        original_plot = Axes.plot

        def spy_plot(self, *args, **kwargs):
            if kwargs.get("ls") == "--":
                dashed_lines.append(args)
            return original_plot(self, *args, **kwargs)

        monkeypatch.setattr(Axes, "plot", spy_plot)
        figure = HTMLReporter.build_trend_figure(
            pd.DataFrame(rows),
            ["S1", "S2", "S3"],
            trend_models=trend_models,
        )

        assert isinstance(figure, str)
        assert len(dashed_lines) == 2

    def test_build_trend_figure_accepts_single_sample_batch(self):
        rows = [
            {"Sample": "S1", "Peak #": 1, "2θ (°)": 26.0, "FWHM (°)": 0.10},
        ]

        figure = HTMLReporter.build_trend_figure(pd.DataFrame(rows), ["S1"])

        assert isinstance(figure, str)
        assert len(figure) > 0

    def test_build_trend_figure_filters_samples_before_index_mapping(self, monkeypatch):
        from matplotlib.axes import Axes

        rows = [
            {"Sample": "S1", "Peak #": 1, "2θ (°)": 26.0, "FWHM (°)": 0.10},
            {"Sample": "EXTRA", "Peak #": 1, "2θ (°)": 27.0, "FWHM (°)": 0.20},
        ]
        plotted_x = []
        original_plot = Axes.plot

        def spy_plot(self, *args, **kwargs):
            if len(args) >= 2 and getattr(args[0], "name", None) == "_idx":
                plotted_x.extend(list(args[0]))
            return original_plot(self, *args, **kwargs)

        monkeypatch.setattr(Axes, "plot", spy_plot)
        figure = HTMLReporter.build_trend_figure(pd.DataFrame(rows), ["S1"])

        assert isinstance(figure, str)
        assert plotted_x == [1, 1]
        assert not any(pd.isna(x) for x in plotted_x)


class TestSampleFigure:
    def test_empty_fit_arrays_use_dataframe_fallback(self):
        df = pd.DataFrame({"two_theta": [], "intensity": []})
        fr = FitResult(
            name="empty",
            dominant_peak={"center": np.nan, "fwhm": np.nan, "amplitude": np.nan, "eta": np.nan},
            all_peaks=[],
            r_squared=0.0,
            aic=0.0,
            bic=0.0,
            n_peaks=0,
            wavelength=1.54056,
        )

        figure = HTMLReporter(Path("."))._build_sample_figure("empty", df, fr)

        assert isinstance(figure, str)
        assert len(figure) > 0
