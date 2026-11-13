import csv

import numpy as np
import pytest

from singingmaterials import peak_fitting

from conftest import THZ, make_dos_dict


def gaussian_fwhm(f, centre, fwhm, height):
    return height * np.exp(-4.0 * np.log(2.0) * ((f - centre) / fwhm) ** 2)


@pytest.fixture
def two_peak_dos():
    """Two well separated Gaussians, so the fit should recover them exactly."""
    f = np.linspace(0.1 * THZ, 30 * THZ, 3000)
    peaks = [(8 * THZ, 2 * THZ, 1.0), (20 * THZ, 1 * THZ, 2.0)]  # centre, fwhm, height
    dos = sum(gaussian_fwhm(f, *p) for p in peaks)
    return f, dos, peaks


def test_fit_recovers_separated_gaussians(two_peak_dos):
    f, dos, peaks = two_peak_dos
    fit = peak_fitting.fit_peaks(f, dos, n_peaks=2, smoothing=0.1)

    spacing = f[1] - f[0]
    for (centre, fwhm, height), c, w, h in zip(peaks, fit["centres"], fit["fwhms"], fit["heights"]):
        assert c == pytest.approx(centre, abs=spacing)
        assert w == pytest.approx(fwhm, rel=0.02)
        assert h == pytest.approx(height, rel=0.01)
    assert np.max(np.abs(fit["residuals"])) < 0.02
    assert fit["fitted_modes"] == pytest.approx(fit["dos_modes"], rel=0.01)


def test_peaks_are_in_frequency_order_and_capped_by_candidates(two_peak_dos):
    f, dos, _ = two_peak_dos
    fit = peak_fitting.fit_peaks(f, dos, n_peaks=10, smoothing=0.1)
    assert len(fit["centres"]) == fit["n_candidates"] <= 10
    assert np.all(np.diff(fit["centres"]) > 0)


def test_empty_gap_is_not_a_candidate():
    """Between two sharp peaks the DOS is zero; the curvature dips on both sides, but the gap is not a peak."""
    f = np.linspace(0.1 * THZ, 30 * THZ, 3000)
    dos = gaussian_fwhm(f, 10 * THZ, 0.3 * THZ, 5.0) + gaussian_fwhm(f, 16 * THZ, 0.3 * THZ, 5.0)
    candidates, _ = peak_fitting.find_candidate_peaks(f / THZ, dos, smoothing=0.1)
    assert all(dos[c] > 0.1 for c in candidates)


def test_projections():
    dos_dict = make_dos_dict()
    assert list(peak_fitting.fit_dos_peaks(dos_dict=dos_dict, n_peaks=1, projection="total")) == ["total"]
    assert sorted(peak_fitting.fit_dos_peaks(dos_dict=dos_dict, n_peaks=1, projection="site")) == ["A_1", "B_1"]
    assert sorted(peak_fitting.fit_dos_peaks(dos_dict=dos_dict, n_peaks=1, projection="element")) == ["A", "B"]
    with pytest.raises(ValueError):
        peak_fitting.fit_dos_peaks(dos_dict=dos_dict, projection="orbital")


def read_csv(path):
    with open(path) as handle:
        rows = [row for row in csv.reader(handle) if not row[0].startswith("#")]
    return rows[0], rows[1:]


def test_save_total_csv(tmp_path):
    fits = peak_fitting.fit_dos_peaks(dos_dict=make_dos_dict(), n_peaks=2, smoothing=0.1)
    path = peak_fitting.save_peak_fit(fits, tmp_path / "total.csv")
    header, rows = read_csv(path)

    assert header == ["Peak Index", "Peak Height", "Frequency", "FWHM", "DOS Ratio", "Normalized DOS Ratio",
                      "Frequency Ratio"]
    assert len(rows) == 2
    assert sum(float(r[5]) for r in rows) == pytest.approx(1.0)
    assert max(float(r[4]) for r in rows) == pytest.approx(1.0)


def test_save_projected_csv_has_species(tmp_path):
    fits = peak_fitting.fit_dos_peaks(dos_dict=make_dos_dict(), n_peaks=1, smoothing=0.1, projection="element")
    path = peak_fitting.save_peak_fit(fits, tmp_path / "element.csv")
    header, rows = read_csv(path)

    assert header[0] == "Species"
    assert sorted(r[0] for r in rows) == ["A", "B"]
    # ratios are relative to the tallest peak across all species (B is twice as tall as A in the synthetic DOS)
    ratio = {r[0]: float(r[5]) for r in rows}
    assert ratio["B"] == pytest.approx(1.0)
    assert ratio["A"] == pytest.approx(0.5, rel=0.01)
    assert sum(float(r[6]) for r in rows) == pytest.approx(1.0)


def test_settings_can_vary_by_species():
    fits = peak_fitting.fit_dos_peaks(dos_dict=make_dos_dict(), projection="element",
                                      n_peaks={"A": 1}, smoothing={"B": 0.2})
    assert len(fits["A"]["centres"]) == 1
    assert fits["A"]["smoothing"] == peak_fitting.DEFAULT_SMOOTHING
    assert fits["B"]["smoothing"] == 0.2
    with pytest.raises(ValueError, match="not element projections"):
        peak_fitting.fit_dos_peaks(dos_dict=make_dos_dict(), projection="element", n_peaks={"Ca": 2})


def test_plots(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from singingmaterials import visualisation

    fits = peak_fitting.fit_dos_peaks(dos_dict=make_dos_dict(), n_peaks=2, projection="element")
    assert len(visualisation.plot_peak_fits(fits)) == 2
    assert len(visualisation.plot_peak_fit(fits["A"]).figure.axes) == 1


def test_temperature_weights_the_dos_and_labels_the_csv(tmp_path):
    from singingmaterials.dos_stats import scale_by_occupation

    dos_dict = make_dos_dict()
    fits = peak_fitting.fit_dos_peaks(dos_dict=dos_dict, n_peaks=2, temperature=300)
    total = dos_dict["projection"]["total"]
    expected = scale_by_occupation(total["densities"], total["frequencies"], 300)
    np.testing.assert_allclose(fits["total"]["densities"], expected)
    assert fits["total"]["temperature"] == 300

    path = peak_fitting.save_peak_fit(fits, tmp_path / "thermal.csv")
    with open(path) as handle:
        assert "temperature: 300K" in handle.read()
    with pytest.raises(ValueError):
        peak_fitting.fit_dos_peaks(dos_dict=dos_dict, temperature=0)


def test_min_frequency_leaves_out_low_frequencies(two_peak_dos):
    f, dos, _ = two_peak_dos
    fit = peak_fitting.fit_peaks(f, dos, n_peaks=2, smoothing=0.1, min_frequency=5.0)
    assert fit["frequencies"][0] >= 5.0 * THZ
    np.testing.assert_allclose(f[fit["peak_indices"]], fit["centres"])  # indices still refer to the full grid
