from dotenv import find_dotenv, load_dotenv
import numpy as np
import os
from singingmaterials import utilities
from pathlib import Path
import json
import pickle
import warnings

# use python-dotenv library for storing secrets in a .env file in the working directory or one of its parents.
# usecwd is needed as the default search starts from this file, which is in site-packages once installed.
load_dotenv(find_dotenv(usecwd=True))


def _mp_rester():
    """return a client for the Materials Project. mp-api is an optional dependency, so only import when needed."""
    try:
        from mp_api.client import MPRester
    except ImportError as e:
        raise ImportError(
            "Reading from the Materials Project needs mp-api, which is an optional dependency. "
            'Install it with: pip install "SingingMaterials[mp]"'
        ) from e
    return MPRester(os.getenv("MP_API_KEY"))


def gamma_frequencies_from_mp_id(mp_id):
    """return phonon frequencies (in Hz) at gamma point from for a material hosted on the Materials Project.
    Material is identified using unique ID number. Note that to use this feature you need a Materials
    Project API key (https://materialsproject.org/api)."""

    with _mp_rester() as mpr:
        try:
            bs = mpr.get_phonon_bandstructure_by_material_id(mp_id)
        except Exception as e:
            raise ValueError(
                f"Materials Project entry {mp_id} does not appear to have phonon data"
            ) from e
    print("extracting frequencies for qpoint {}".format(bs.qpoints[0].cart_coords))

    phonon_frequencies = list(bs.to_pmg.bands[:, 0] * 1e12)  # convert from THz to Hz
    phonon_frequencies = utilities.process_imaginary(phonon_frequencies)
    print("phonon frequencies are (Hz):", phonon_frequencies)

    return phonon_frequencies


def dos_data_from_mp_id(mp_id):
    """return dos data obect. This is for a material hosted on the Materials Project.
    Material is identified using unique ID number. Note that to use this feature you need a Materials
    Project API key (https://materialsproject.org/api)."""

    with _mp_rester() as mpr:
        try:
            dos = mpr.get_phonon_dos_by_material_id(mp_id)
        except Exception as e:
            raise ValueError(
                f"Materials Project entry {mp_id} does not appear to have phonon data"
            ) from e

    return dos


def bandstructure_from_mp_id(mp_id):
    """return the phonon band structure (a pymatgen PhononBandStructureSymmLine, frequencies in THz) for a
    material hosted on the Materials Project. The result is cached in the working directory as
    {mp_id}_bs.json; delete this file to re-fetch. Needs a Materials Project API key."""
    from pymatgen.phonon.bandstructure import (
        PhononBandStructureSymmLine,
    )  # installed with mp-api

    filepath = Path(f"{mp_id}_bs.json")
    if filepath.is_file():
        print("Fetching band structure from existing file...")
        with open(filepath) as handle:
            return PhononBandStructureSymmLine.from_dict(json.load(handle))

    print("Fetching band structure from Materials Project servers...")
    with _mp_rester() as mpr:
        try:
            bs = mpr.get_phonon_bandstructure_by_material_id(mp_id).to_pmg
        except Exception as e:
            raise ValueError(
                f"Materials Project entry {mp_id} does not appear to have phonon data"
            ) from e

    with open(filepath, "w") as handle:
        json.dump(bs.as_dict(), handle)
    return bs


def get_dos_raw_mp(mp_id):
    """get the full and projected densities. Return as a nested dictionary. Arg is the materials project ID.
    The result is cached in the working directory as {mp_id}_dos.json; delete this file to re-fetch."""

    filepath = Path(f"{mp_id}_dos.json")
    legacy_filepath = Path(f"{mp_id}_dos.pickle")
    if filepath.is_file():
        print("Fetching from existing file...")
        dos_dict = load_dos_json(filepath)
    elif legacy_filepath.is_file():
        print("Fetching from existing file...")
        warnings.warn(
            f"{legacy_filepath} was cached by an older version of singingmaterials, which could "
            "misalign densities and frequencies when imaginary modes were removed. Delete it to "
            "re-fetch from the Materials Project."
        )
        with open(legacy_filepath, "rb") as handle:
            dos_dict = pickle.load(handle)
    else:
        print("Fetching from Materials Project servers...")
        dos = dos_data_from_mp_id(mp_id)

        dos_dict = {}
        dos_dict["metadata"] = {"mp_id": mp_id}
        dos_dict["projection"] = {}

        raw_frequencies = np.array(dos.frequencies) * 1e12  # convert from THz to Hz
        frequencies = utilities.process_imaginary(raw_frequencies)
        dos_dict["metadata"]["bin_width"] = frequencies[1] - frequencies[0]
        print(f"bin width is {dos_dict['metadata']['bin_width'] / 1e12} THz")

        # filter densities against the unfiltered frequencies so the two stay aligned
        densities = utilities.process_imaginary_dos(dos.densities, raw_frequencies)
        dos_dict["projection"]["total"] = {
            "densities": densities,
            "frequencies": frequencies,
        }

        for i, site in enumerate(dos.structure.relabel_sites().sites):
            densities = utilities.process_imaginary_dos(
                dos.projected_densities[i], raw_frequencies
            )
            dos_dict["projection"][site.label] = {
                "densities": densities,
                "frequencies": frequencies,
            }

        save_dos_json(dos_dict, filepath)

    return dos_dict


def save_dos_json(dos_dict, filepath):
    """save a raw dos dict (metadata and projections, no stats) to a json file."""
    serialisable = {
        "metadata": {
            key: (float(value) if isinstance(value, np.floating) else value)
            for key, value in dos_dict["metadata"].items()
        },
        "projection": {
            site: {key: np.asarray(values).tolist() for key, values in data.items()}
            for site, data in dos_dict["projection"].items()
        },
    }
    with open(filepath, "w") as handle:
        json.dump(serialisable, handle)


def load_dos_json(filepath):
    """load a raw dos dict saved by save_dos_json, restoring the numpy arrays."""
    with open(filepath) as handle:
        dos_dict = json.load(handle)
    for data in dos_dict["projection"].values():
        for key in data:
            data[key] = np.array(data[key])
    return dos_dict
