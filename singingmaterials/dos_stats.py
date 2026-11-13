import numpy as np
import pandas as pd
import math
import re
import scipy
from scipy import constants
from singingmaterials import mp_interface

def dos_stats_analysis(mp_id=None,phonopy_filename=None,temp=None):
    """for each entry in a dos_dict, calculate the integrated dos, the phonon band centre, the quantiles and the IQR and of the dos distribution in Hz (discounting any negative frequencies) and add these to the nested dicts. The dos is optionally weighted by Bose Einstein occupation at a specified temp. DOS can be gotten from the materials project (mp_id) or from a phonopy summary file from the phonondb database (phonopy_filename).

    Stats are calculated per site (dos_dict['projection'], which also holds the 'total' dos) and per element
    (dos_dict['element'], where the densities of all sites of the same element are summed). Both have the same structure."""
    
    if temp is not None:
        if not hasattr(temp, '__iter__'):
            temp = [temp]    # if single temp provided as scalar, convert it to iterable list
        if 0 in temp:
            print ("cannot calculate stats for 0K")
        temp = [t for t in temp if t != 0]    # copy, so the caller's list is not modified
    if mp_id:    
        dos_dict = mp_interface.get_dos_raw_mp(mp_id)
    elif phonopy_filename:
        from singingmaterials import phonopy_interface    # optional dependency, so only import when needed
        dos_dict = phonopy_interface.get_dos_raw_phonopy(phonopy_filename)
    else:
        raise ValueError('You must specify a dos data source (mp_id or phonopy_filename)')

    dos_dict['element'] = element_projections(dos_dict['projection'])

    for group in ('projection', 'element'):
        for entry in dos_dict[group].values():
            add_stats(entry, temp)

    return dos_dict

def element_from_site_label(site_label):
    """return the element symbol from a site label, e.g. 'O_1' -> 'O', 'Mg' -> 'Mg', 'Fe2+_3' -> 'Fe'."""
    match = re.match(r"[A-Z][a-z]*", site_label)
    if match is None:
        raise ValueError(f"cannot identify an element in site label '{site_label}'")
    return match.group(0)

def element_projections(projection):
    """sum the site-projected densities over all sites of the same element. Returns a dict keyed by element
    symbol, with the same 'densities' and 'frequencies' structure as the site projections. 'total' is excluded."""
    elements = {}
    for site, site_dict in projection.items():
        if site == 'total':
            continue
        element = element_from_site_label(site)
        if element in elements:
            elements[element]['densities'] = elements[element]['densities'] + site_dict['densities']
        else:
            elements[element] = {'densities': np.array(site_dict['densities'], dtype=float),
                                 'frequencies': site_dict['frequencies']}
    return elements

def calculate_stats(f, densities):
    """return a dict of the distribution stats for a single dos."""
    return {
        'band_centre': phonon_band_centre(f, densities),
        'integrated_dos': integrated_dos(f, densities),
        'quantile_25': weighted_quantile(f, densities, 0.25),
        'quantile_75': weighted_quantile(f, densities, 0.75),
        'IQR': phonon_dos_IQR(f, densities),
        'shannon_entropy': phonon_shannon_entropy(f, densities),
        'densities': densities,
    }

def add_stats(entry, temp=None):
    """add athermal (and, if temp is given, thermal) stats to a dict holding 'densities' and 'frequencies'."""
    densities = entry['densities']
    f = entry['frequencies']
    entry['stats'] = {'athermal': calculate_stats(f, densities)}
    if temp:
        entry['stats']['thermal'] = {str(t): calculate_stats(f, scale_by_occupation(densities, f, t))
                                     for t in temp}

def dos_dict_to_dataframe(dos_dict, by="site"):
    """convert the nested dictionary to a pandas dataframe. by="site" (default) gives one row per site
    (including 'total'), by="element" gives one row per element."""
    if by == "site":
        group, label_column = "projection", "site"
    elif by == "element":
        group, label_column = "element", "element"
    else:
        raise ValueError(f'by must be "site" or "element", not "{by}"')

    rows = []

    mp_id = dos_dict["metadata"]["mp_id"]

    for label, label_data in dos_dict[group].items():
        stats = label_data["stats"]

        # athermal row
        row_base = {
            "mp_id": mp_id,
            label_column: label,
            "temperature": None
        }

        for key, value in stats["athermal"].items():
            row_base[key] = value

        rows.append(row_base)

        # thermal rows
        if "thermal" in stats:
            for T, values in stats["thermal"].items():
                row = {
                    "mp_id": mp_id,
                    label_column: label,
                    "temperature": float(T)
                }
                row.update(values)
                rows.append(row)

    return pd.DataFrame(rows)

def phonon_shannon_entropy(f,dos):
    p = dos / np.sum(dos)  # normalise to give a probability distribution
    p_nonzero = p[p > 0]  # mask out zero probs. this is legit because they are zero in the summation anyhow
    return -np.sum(p_nonzero*np.log(p_nonzero))
    
def phonon_band_centre(f,dos):
    """for each particular chemical species, get the phonon band centre in Hz (discounting any negative frequencies). Return as a dictionary with species string as key."""
    return scipy.integrate.simpson(dos*f,f)/scipy.integrate.simpson(dos,f)
    
def integrated_dos(f,dos):
    """for each particular chemical species, get the integrated dos (discounting any negative frequencies). Return as a dictionary with species string as key."""
    return scipy.integrate.simpson(dos,f)

def phonon_dos_IQR(f,dos):
    """return the inter-quartile range of the dos distribution"""    
    return weighted_quantile(f, dos, 0.75) - weighted_quantile(f, dos, 0.25)
    
def bose_einstien_distribution(energy,temperature):
    if energy <= 0:
        return 0.0    # occupation diverges at zero energy; non-positive modes are excluded from the stats
    return 1 / math.expm1(energy/(constants.Boltzmann*temperature))

def frequency_to_energy(frequency):
    """convert frequency in Hz to energy in joules"""
    energy = constants.h*frequency
    return energy

def scale_by_occupation(densities, phonon_frequencies, temperature):
    """Re-scale the density by the Bose Einstein occupation factor for a state with energy h*frequency at specified temperature."""
    BE_occupations = np.array([bose_einstien_distribution(frequency_to_energy(frequency),temperature) 
                           for frequency in phonon_frequencies])

    # scale by occupation expressed as a fraction of the total occupation
    scaled_densities = densities*BE_occupations

    return scaled_densities

def weighted_quantile(values, weights, quantile):
    """
    Compute a weighted quantile of discrete data.
    
    values   : array-like data points
    weights  : array-like weights (same length as values)
    quantile : float in [0,1] (e.g. 0.25, 0.75)
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    # sort by values
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    # cumulative distribution
    cum_weights = np.cumsum(weights)
    cum_weights /= cum_weights[-1]

    return np.interp(quantile, cum_weights, values)

