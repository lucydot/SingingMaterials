"""Fit a phonon density of states with a set of Gaussian peaks.

Each peak is a symmetric Gaussian. Peak positions come from the curvature of the smoothed DOS (so shoulders are
found as well as maxima). Each peak's centre and height are fixed to the DOS, so that only the widths are fitted.
Peaks are added one at a time, each where the current fit leaves the most DOS unexplained.

The fit can be done for the total DOS, or separately for each element or site projected DOS (PDOS).
"""

import csv

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks

_THZ = 1e12
DEFAULT_N_PEAKS = 9
DEFAULT_SMOOTHING = 0.3  # THz


def gaussian(f, height, centre, fwhm):
    """Symmetric Gaussian peak with the given height at `centre` and full width at half maximum `fwhm`."""
    return height * np.exp(-4.0 * np.log(2.0) * ((f - centre) / fwhm) ** 2)


def peak_model(f, heights, centres, fwhms):
    """Sum of Gaussian peaks."""
    return sum(
        (gaussian(f, h, c, w) for h, c, w in zip(heights, centres, fwhms)),
        np.zeros_like(f, dtype=float),
    )


def find_candidate_peaks(frequencies_thz, densities, smoothing=0.3):
    """Return (indices of candidate peak positions, smoothed DOS).

    Candidate peaks are wherethe smoothed DOS bends downwards most strongly (maxima of -d2g/df2).
    `smoothing` is the width (THz) of the Gaussian smoothing applied first, so small wiggles are not picked.
    The height condition requires the DOS to bend downwards so that zero values in the DOS aren't identified as peaks."""
    smoothed = gaussian_filter1d(
        densities, smoothing / (frequencies_thz[1] - frequencies_thz[0])
    )
    neg_curvature = -np.gradient(
        np.gradient(smoothed, frequencies_thz), frequencies_thz
    )
    threshold = 0.01 * np.max(neg_curvature)
    candidates, _ = find_peaks(neg_curvature, prominence=threshold, height=threshold)
    return candidates, smoothed


def _fit_widths(frequencies_thz, densities, peaks):
    """Fit the widths (THz) of Gaussians fixed at the given peak indices, with heights fixed to the DOS there."""
    heights, centres = densities[peaks], frequencies_thz[peaks]
    spacing = frequencies_thz[1] - frequencies_thz[0]
    fwhm_guess = np.full(len(peaks), (frequencies_thz[-1] - frequencies_thz[0]) / 25.0)
    return minimize(
        lambda fwhms: np.sum(
            (densities - peak_model(frequencies_thz, heights, centres, fwhms)) ** 2
        ),
        fwhm_guess,
        method="L-BFGS-B",
        bounds=[(2 * spacing, None)]
        * len(peaks),  # narrower than ~2 grid points falls between the data points
        options={"maxiter": 2000, "maxfun": 50000, "ftol": 1e-10},
    )


def fit_peaks(
    frequencies,
    densities,
    n_peaks=DEFAULT_N_PEAKS,
    smoothing=DEFAULT_SMOOTHING,
    label="total",
    min_frequency=0.0,
):
    """Fit a single DOS (frequencies in Hz) with up to n_peaks Gaussians. `smoothing` is in THz.

    Frequencies below min_frequency (THz) are left out of the fit; the returned curves start there.
    Returns a dict with the peak indices into the full `frequencies` array, centres and FWHMs (Hz), heights,
    the modes in each peak, the fitted and smoothed curves and the residuals."""
    frequencies = np.asarray(frequencies, dtype=float)
    densities = np.asarray(densities, dtype=float)
    start = int(
        np.searchsorted(frequencies, min_frequency * _THZ)
    )  # first grid point at or above min_frequency
    frequencies, densities = frequencies[start:], densities[start:]
    f_thz = frequencies / _THZ

    candidates, smoothed = find_candidate_peaks(f_thz, densities, smoothing)
    smooth_px = smoothing / (f_thz[1] - f_thz[0])

    # add peaks one at a time, each where the current fit leaves the most DOS unexplained; the residual is
    # smoothed, so a broad unexplained region outranks a single sharp spike next to an existing peak
    peaks, residual, res = [], densities.copy(), None
    for _ in range(min(n_peaks, len(candidates))):
        residual_smooth = gaussian_filter1d(residual, smooth_px)
        peaks.append(
            max(
                (p for p in candidates if p not in peaks),
                key=lambda p: residual_smooth[p],
            )
        )
        res = _fit_widths(f_thz, densities, peaks)
        residual = densities - peak_model(f_thz, densities[peaks], f_thz[peaks], res.x)

    order = np.argsort(f_thz[peaks])
    peaks = np.array(peaks, dtype=int)[order]
    fwhms_thz = res.x[order] if res is not None else np.array([])
    heights = densities[peaks]
    fitted = peak_model(f_thz, heights, f_thz[peaks], fwhms_thz)
    modes = np.array(
        [
            np.trapezoid(gaussian(f_thz, h, c, w), f_thz)
            for h, c, w in zip(heights, f_thz[peaks], fwhms_thz)
        ]
    )

    return {
        "label": label,
        "frequencies": frequencies,
        "densities": densities,
        "smoothed": smoothed,
        "smoothing": smoothing,
        "min_frequency": min_frequency,
        "peak_indices": peaks + start,
        "centres": frequencies[peaks],
        "heights": heights,
        "fwhms": fwhms_thz * _THZ,
        "modes": modes,
        "fitted": fitted,
        "residuals": densities - fitted,
        "fitted_modes": np.trapezoid(fitted, f_thz),
        "dos_modes": np.trapezoid(densities, f_thz),
        "converged": bool(res.success) if res is not None else True,
        "n_candidates": len(candidates),
    }


def _projections(dos_dict, projection):
    """Return {label: (frequencies, densities)} for 'total', each 'element' or each 'site'."""
    if projection == "total":
        total = dos_dict["projection"]["total"]
        return {"total": (total["frequencies"], total["densities"])}
    if projection == "site":
        source = {k: v for k, v in dos_dict["projection"].items() if k != "total"}
    elif projection == "element":
        if "element" in dos_dict:
            source = dos_dict["element"]
        else:
            from singingmaterials.dos_stats import element_projections

            source = element_projections(dos_dict["projection"])
    else:
        raise ValueError(
            f"projection must be 'total', 'element' or 'site', not {projection!r}"
        )
    return {label: (v["frequencies"], v["densities"]) for label, v in source.items()}


def _setting_for(setting, label, default):
    """A setting is either one value for every projection, or a dict {label: value}; labels missing from the
    dict use the default."""
    if isinstance(setting, dict):
        return setting.get(label, default)
    return setting


def fit_dos_peaks(
    mp_id=None,
    dos_dict=None,
    n_peaks=DEFAULT_N_PEAKS,
    smoothing=DEFAULT_SMOOTHING,
    projection="total",
    temperature=None,
    min_frequency=0.0,
):
    """Fit the phonon DOS of a material with Gaussian peaks.

    Give either an mp_id (fetched from the Materials Project and cached) or a dos_dict as returned by
    mp_interface.get_dos_raw_mp or dos_stats.dos_stats_analysis.

    `projection` is 'total' for the total DOS, or 'element' / 'site' to fit each projected DOS separately.

    n_peaks and smoothing (THz) are either one value for every projection, or a dict by element / site label,
    e.g. n_peaks={"Ca": 6, "C": 4}; labels missing from a dict use the defaults (9 peaks, 0.3 THz).

    With temperature=None the athermal DOS is fitted. With a temperature (K), each DOS is first weighted by the
    Bose-Einstein occupation at that temperature (dos_stats.scale_by_occupation).

    min_frequency (THz) leaves the lowest frequencies out of the fit when unphysical spikes are found after thermal occupation"""
    if dos_dict is None:
        if mp_id is None:
            raise ValueError("You must give an mp_id or a dos_dict")
        from singingmaterials import mp_interface

        dos_dict = mp_interface.get_dos_raw_mp(mp_id)

    projections = _projections(dos_dict, projection)
    for name, setting in (("n_peaks", n_peaks), ("smoothing", smoothing)):
        if isinstance(setting, dict) and set(setting) - set(projections):
            raise ValueError(
                f"{name} has labels {sorted(set(setting) - set(projections))} that are not "
                f"{projection} projections of this material: {sorted(projections)}"
            )

    if temperature is not None and temperature <= 0:
        raise ValueError(
            "temperature must be above 0 K (use temperature=None for the athermal DOS)"
        )

    fits = {}
    for label, (frequencies, densities) in projections.items():
        if temperature is not None:
            from singingmaterials.dos_stats import scale_by_occupation

            densities = scale_by_occupation(
                np.asarray(densities, dtype=float), frequencies, temperature
            )
        fit = fit_peaks(
            frequencies,
            densities,
            label=label,
            min_frequency=min_frequency,
            n_peaks=_setting_for(n_peaks, label, DEFAULT_N_PEAKS),
            smoothing=_setting_for(smoothing, label, DEFAULT_SMOOTHING),
        )
        fit["mp_id"] = mp_id or dos_dict.get("metadata", {}).get("mp_id")
        fit["projection"] = projection
        fit["temperature"] = temperature
        fits[label] = fit
    return fits


def save_peak_fit(fits, filename=None):
    """Write the fitted peaks to a CSV file, one row per peak, and return the file name.

    Columns should be self explanatory, except forDOS Ratio and Frequency Ratio (relative to the tallest peak) and
    Normalized DOS Ratio (DOS ratios summing to 1 over the file). For a projected (element or site) fit the ratios are relative to the tallest peak across
    all species."""
    fits = list(fits.values())
    mp_id = fits[0].get("mp_id")
    projection = fits[0].get("projection", "total")
    temperature = fits[0].get("temperature")
    temperature_label = "athermal" if temperature is None else f"{temperature:g}K"
    if filename is None:
        filename = (
            f"peak_fit_{mp_id}_{projection}"
            + ("" if temperature is None else f"_{temperature_label}")
            + ".csv"
        )
    projected = projection != "total"

    labels = [fit["label"] for fit in fits for _ in fit["peak_indices"]]
    indices = np.concatenate([fit["peak_indices"] for fit in fits])
    heights = np.concatenate([fit["heights"] for fit in fits])
    centres = np.concatenate([fit["centres"] for fit in fits])
    fwhms = np.concatenate([fit["fwhms"] for fit in fits])
    smoothings = sorted({fit["smoothing"] for fit in fits})

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"# mp_id: {mp_id}"])
        writer.writerow(
            [
                f"# projection: {projection}, temperature: {temperature_label}, "
                f"smoothing: {', '.join(map(str, smoothings))} THz, frequencies in Hz"
            ]
        )
        header = [
            "Peak Index",
            "Peak Height",
            "Frequency",
            "FWHM",
            "DOS Ratio",
            "Normalized DOS Ratio",
            "Frequency Ratio",
        ]
        writer.writerow((["Species"] if projected else []) + header)
        if len(heights) == 0:
            return filename
        ref = np.argmax(heights)
        dos_ratios = heights / heights[ref]
        norm_dos_ratios = dos_ratios / np.sum(dos_ratios)
        freq_ratios = centres / centres[ref]
        for i in range(len(heights)):
            row = [
                indices[i],
                heights[i],
                centres[i],
                fwhms[i],
                dos_ratios[i],
                norm_dos_ratios[i],
                freq_ratios[i],
            ]
            writer.writerow(([labels[i]] if projected else []) + row)
    return filename


def print_peak_fit(fit):
    """Print a summary of one fit (frequencies shown in THz)."""
    temperature = fit.get("temperature")
    print(
        f"{fit['label']}"
        + ("" if temperature is None else f" at {temperature:g} K")
        + f": {len(fit['peak_indices'])} Gaussian peaks "
        f"(from {fit['n_candidates']} candidates, smoothing {fit['smoothing']} THz). "
        f"DOS integral {fit['dos_modes']:.3f}, fitted {fit['fitted_modes']:.3f}"
    )
    if not fit["converged"]:
        print("  Warning: the width fit did not fully converge")
    for i, (c, h, w, m) in enumerate(
        zip(fit["centres"], fit["heights"], fit["fwhms"], fit["modes"])
    ):
        print(
            f"  Peak {i + 1}: Center = {c / _THZ:.3f} THz, Amplitude = {h:.4f}, FWHM = {w / _THZ:.3f} THz, "
            f"Integrated Modes = {m:.2f}"
        )
