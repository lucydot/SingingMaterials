import matplotlib.pyplot as plt
import numpy as np
from singingmaterials.dos_stats import scale_by_occupation


def animate_dos_vs_temperature(dos_dict, site_order=None, interval=500):
    from matplotlib.animation import FuncAnimation

    """
    Create matplotlib animation showing DOS projections and band centres vs temperature.
    
    interval : ms between frames
    """

    projections = dos_dict["projection"]
    mp_id = dos_dict["metadata"]["mp_id"]

    # Extract temperatures (as sorted floats)
    temps = list(next(iter(projections.values()))["stats"]["thermal"].keys())
    temps = sorted(int(t) for t in temps)

    # Decide plotting order
    if site_order is None:
        site_order = list(projections.keys())

    # Consistent colours
    cmap = plt.get_cmap("tab10")
    colours = {site: cmap(i % 10) for i, site in enumerate(site_order)}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(f"Phonon DOS vs Temperature ({mp_id})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("DOS")

    ax.set_autoscale_on(False)

    lines = {}
    centre_lines = {}
    q25_lines = {}
    q75_lines = {}

    # Initialise empty lines
    for site in site_order:
        f = projections[site]["frequencies"]

        (line,) = ax.plot(f, np.zeros_like(f), color=colours[site], label=site)
        lines[site] = line

        centre = ax.axvline(0, color=colours[site], linestyle="--", alpha=0.7)
        centre_lines[site] = centre

        q25 = ax.axvline(
            0, color=colours[site], linestyle=":", alpha=0.3, linewidth=1.0
        )
        q75 = ax.axvline(
            0, color=colours[site], linestyle=":", alpha=0.3, linewidth=1.0
        )

        q25_lines[site] = q25
        q75_lines[site] = q75

    ax.legend()

    # Fix axis limits for stability
    all_f = projections[site_order[0]]["frequencies"]
    ax.set_xlim(all_f.min(), all_f.max())

    ax.set_ylim(
        0,
        1.1
        * max(
            (projections[s]["densities"] / projections[s]["densities"].sum()).max()
            for s in site_order
        ),
    )

    def update(frame):
        T = temps[frame]

        ax.set_title(f"Phonon DOS at T = {int(T)} K ({mp_id})")

        for site in site_order:
            f = projections[site]["frequencies"]

            # reconstruct weighted DOS visually
            weighted = scale_by_occupation(projections[site]["densities"], f, T)
            weighted = weighted / weighted.sum()

            lines[site].set_ydata(weighted)

            centre = projections[site]["stats"]["thermal"][str(T)]["band_centre"]
            centre_lines[site].set_xdata([centre, centre])

            q25 = projections[site]["stats"]["thermal"][str(T)]["quantile_25"]
            q75 = projections[site]["stats"]["thermal"][str(T)]["quantile_75"]

            q25_lines[site].set_xdata([q25, q25])
            q75_lines[site].set_xdata([q75, q75])

        return (
            list(lines.values())
            + list(centre_lines.values())
            + list(q25_lines.values())
            + list(q75_lines.values())
        )

    anim = FuncAnimation(fig, update, frames=len(temps), interval=interval, blit=True)

    return anim


def plot_dos_at_temperature(dos_dict, T=None, site_order=None):
    """
    Plot projected DOS and band statistics at a single temperature.

    T=None → athermal DOS
    """
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams.update({"font.size": 10})
    projections = dos_dict["projection"]
    mp_id = dos_dict["metadata"]["mp_id"]

    if site_order is None:
        site_order = list(projections.keys())

    cmap = plt.get_cmap("tab10")
    colours = {site: cmap(i % 10) for i, site in enumerate(site_order)}

    ONE_MM = 1 / 25.4  # Convert mm to inches
    fig, ax = plt.subplots(figsize=(170 * ONE_MM, 140 * ONE_MM))  # single column
    # ax.set_title(
    #    f"Phonon DOS {'(athermal)' if T is None else f'at T = {int(T)} K'} — {mp_id}"
    # )
    ax.set_xlabel("Frequency (THz)", fontsize=10)
    ax.set_ylabel("Density of States", fontsize=10)

    ax.set_autoscale_on(False)

    lines = {}

    for site in site_order:
        if site == "total":
            pass
        else:
            f = projections[site]["frequencies"]

            if T is None:
                weighted = projections[site]["densities"]
            else:
                weighted = scale_by_occupation(projections[site]["densities"], f, T)

            weighted = weighted / weighted.sum()

            # Convert to LaTeX-style subscript
            if site == "total":
                formatted_label = "Total"
            else:
                element, number = site.split("_")
                formatted_label = f"{element}"

            ax.plot(f / 1e12, weighted, color=colours[site], label=formatted_label)

            # Band centre
            if T is None:
                centre = projections[site]["stats"]["athermal"]["band_centre"] / 1e12
                q25 = projections[site]["stats"]["athermal"]["quantile_25"] / 1e12
                q75 = projections[site]["stats"]["athermal"]["quantile_75"] / 1e12
            else:
                centre = projections[site]["stats"]["thermal"][str(T)]["band_centre"]
                q25 = projections[site]["stats"]["thermal"][str(T)]["quantile_25"]
                q75 = projections[site]["stats"]["thermal"][str(T)]["quantile_75"]

            ax.axvline(
                centre, color=colours[site], linestyle="--", linewidth=1, alpha=0.6
            )

            ax.axvline(q25, color=colours[site], linestyle=":", linewidth=1, alpha=0.6)
            ax.axvline(q75, color=colours[site], linestyle=":", linewidth=1, alpha=0.6)

    # Fixed y-scale
    ymax = max(
        (projections[s]["densities"] / projections[s]["densities"].sum()).max()
        for s in site_order
    )
    ax.set_ylim(0, ymax)

    ax.set_xlim(
        projections[site_order[0]]["frequencies"].min() / 1e12,
        projections[site_order[0]]["frequencies"].max() / 1e12,
    )

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig("BAs.png", dpi=320)

    plt.show()

    return plt


def plot_bandstructure(
    mp_id=None, bandstructure=None, ax=None, color="black", ylim=None
):
    """
    Plot a phonon band structure along its high-symmetry path, with frequency in THz.

    Give either an mp_id (fetched from the Materials Project and cached as {mp_id}_bs.json) or a pymatgen
    PhononBandStructureSymmLine as `bandstructure`. Pass `ax` to draw into an existing axes.
    """
    if bandstructure is None:
        if mp_id is None:
            raise ValueError("You must give an mp_id or a bandstructure")
        from singingmaterials import mp_interface

        bandstructure = mp_interface.bandstructure_from_mp_id(mp_id)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    distance = np.asarray(bandstructure.distance)
    bands = np.asarray(bandstructure.bands)  # shape (n_bands, n_qpoints), THz

    # draw each branch of the path separately, so jumps in the path (e.g. "X|K") are not joined up
    for branch in bandstructure.branches:
        segment = slice(branch["start_index"], branch["end_index"] + 1)
        for band in bands:
            ax.plot(distance[segment], band[segment], color=color, linewidth=1)

    # high-symmetry labels: a label at the end of one branch and the start of the next share a tick
    ticks = {}
    for i, qpoint in enumerate(bandstructure.qpoints):
        if qpoint.label is None:
            continue
        label = (
            r"$\Gamma$" if qpoint.label in ("GAMMA", "\\Gamma", "Γ") else qpoint.label
        )
        x = round(float(distance[i]), 8)
        if x in ticks and ticks[x] != label:
            ticks[x] = f"{ticks[x]}|{label}"
        else:
            ticks[x] = label
    for x in ticks:
        ax.axvline(x, color="grey", linewidth=0.5)
    ax.set_xticks(list(ticks.keys()), list(ticks.values()))

    ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
    ax.set_xlim(distance[0], distance[-1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel("Frequency (THz)")
    if mp_id is not None:
        ax.set_title(f"Phonon band structure ({mp_id})")

    return ax


def plot_dos(
    mp_id=None, dos_dict=None, ax=None, color="black", xlim=None, vertical=False
):
    """
    Plot the total phonon density of states, with frequency in THz.

    Give either an mp_id (fetched from the Materials Project and cached as {mp_id}_dos.json) or a dos_dict as
    returned by mp_interface.get_dos_raw_mp or dos_stats.dos_stats_analysis. Pass `ax` to draw into an existing
    axes. vertical=True puts frequency on the y-axis.
    """
    if dos_dict is None:
        if mp_id is None:
            raise ValueError("You must give an mp_id or a dos_dict")
        from singingmaterials import mp_interface

        dos_dict = mp_interface.get_dos_raw_mp(mp_id)
    elif mp_id is None:
        mp_id = dos_dict.get("metadata", {}).get("mp_id")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    total = dos_dict["projection"]["total"]
    frequencies = np.asarray(total["frequencies"]) / 1e12  # Hz -> THz
    densities = np.asarray(total["densities"])  # states/THz

    if vertical:
        ax.plot(densities, frequencies, color=color, linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xlim(left=0)
        if xlim is not None:
            ax.set_ylim(xlim)
        ax.set_xlabel("DOS (states/THz)")
        ax.set_ylabel("Frequency (THz)")
    else:
        ax.plot(frequencies, densities, color=color, linewidth=1)
        ax.set_ylim(bottom=0)
        ax.set_xlim(xlim if xlim is not None else (frequencies[0], frequencies[-1]))
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("DOS (states/THz)")
    if mp_id is not None:
        ax.set_title(f"Phonon DOS ({mp_id})")

    return ax


def plot_peak_fit(fit, show_smoothed=True):
    """
    Plot a Gaussian peak fit from peak_fitting.fit_peaks / fit_dos_peaks: the DOS, the fit and each peak.
    Frequency in THz.
    """
    f = np.asarray(fit["frequencies"]) / 1e12  # Hz -> THz

    fig, ax1 = plt.subplots(figsize=(9, 4.5), layout="constrained")

    dos_label = (
        "Calculated DOS"
        if fit.get("temperature") is None
        else "Bose-Einstein weighted DOS"
    )
    ax1.plot(f, fit["densities"], color="black", linewidth=2, label=dos_label)
    if show_smoothed:
        ax1.plot(
            f, fit["smoothed"], color="darkorange", linewidth=1.2, label="Smoothed DOS"
        )
    ax1.plot(
        f, fit["fitted"], color="crimson", linestyle="--", linewidth=2, label="Fit"
    )

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(fit["centres"]), 1)))
    for color, h, c, w in zip(
        colors, fit["heights"], fit["centres"] / 1e12, fit["fwhms"] / 1e12
    ):
        peak = h * np.exp(-4.0 * np.log(2.0) * ((f - c) / w) ** 2)
        ax1.plot(f, peak, color=color, linewidth=1.3, alpha=0.9)
        ax1.fill_between(f, 0, peak, color=color, alpha=0.15)

    title = (
        "Phonon DOS fit"
        if fit.get("temperature") is None
        else f"Phonon DOS fit at {fit['temperature']:g} K"
    )
    if fit.get("mp_id"):
        title += f" ({fit['mp_id']}"
        title += ")" if fit["label"] == "total" else f", {fit['label']})"
    ax1.set_title(title)
    ax1.set_ylabel("Phonon DOS (states / THz)")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(f[0], f[-1])
    ax1.grid(alpha=0.25)
    ax1.set_xlabel("Frequency (THz)")

    return ax1


def plot_peak_fits(fits, show_smoothed=False):
    """
    Plot several Gaussian peak fits from peak_fitting.fit_dos_peaks (e.g. one per element) as stacked panels, one
    per projection.
    """
    fits = list(fits.values()) if isinstance(fits, dict) else list(fits)
    fig, axes = plt.subplots(
        len(fits),
        1,
        figsize=(9, 2.2 * len(fits) + 0.8),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    for ax, fit in zip(axes[:, 0], fits):
        f = np.asarray(fit["frequencies"]) / 1e12  # Hz -> THz
        ax.plot(f, fit["densities"], color="black", linewidth=1.5, label="DOS")
        if show_smoothed:
            ax.plot(
                f, fit["smoothed"], color="darkorange", linewidth=1, label="Smoothed"
            )
        ax.plot(
            f,
            fit["fitted"],
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label="Fit",
        )
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(fit["centres"]), 1)))
        for color, h, c, w in zip(
            colors, fit["heights"], fit["centres"] / 1e12, fit["fwhms"] / 1e12
        ):
            peak = h * np.exp(-4.0 * np.log(2.0) * ((f - c) / w) ** 2)
            ax.fill_between(f, 0, peak, color=color, alpha=0.25)
        ax.text(
            0.01,
            0.92,
            f"{fit['label']}  ({len(fit['centres'])} peaks)",
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
        )
        ax.set_ylim(bottom=0)
        ax.set_ylabel("DOS (states/THz)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("Frequency (THz)")
    f0 = np.asarray(fits[0]["frequencies"]) / 1e12
    axes[-1, 0].set_xlim(f0[0], f0[-1])
    mp_id = fits[0].get("mp_id")
    if mp_id:
        temperature = fits[0].get("temperature")
        at = "" if temperature is None else f" at {temperature:g} K"
        fig.suptitle(
            f"Phonon DOS fit by {fits[0].get('projection', 'projection')}{at} ({mp_id})"
        )
    return axes[:, 0]
