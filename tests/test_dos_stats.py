import copy
import math

import numpy as np
import pytest
from scipy import constants

from singingmaterials import dos_stats

from conftest import SITE_PARAMS, THZ


# --- statistics on a known distribution ---

@pytest.fixture
def gaussian_dos():
    f = np.linspace(0.1 * THZ, 20 * THZ, 4000)
    centre, sigma = 8 * THZ, 1 * THZ
    dos = np.exp(-0.5 * ((f - centre) / sigma) ** 2)
    return f, dos, centre, sigma


def test_band_centre_is_mean(gaussian_dos):
    f, dos, centre, _ = gaussian_dos
    assert dos_stats.phonon_band_centre(f, dos) == pytest.approx(centre, rel=1e-6)


def test_integrated_dos(gaussian_dos):
    f, dos, _, sigma = gaussian_dos
    assert dos_stats.integrated_dos(f, dos) == pytest.approx(sigma * math.sqrt(2 * math.pi), rel=1e-6)


def test_quantiles_and_iqr(gaussian_dos):
    f, dos, centre, sigma = gaussian_dos
    q25 = dos_stats.weighted_quantile(f, dos, 0.25)
    q75 = dos_stats.weighted_quantile(f, dos, 0.75)
    assert q25 < centre < q75
    assert dos_stats.weighted_quantile(f, dos, 0.5) == pytest.approx(centre, rel=1e-3)
    # IQR of a normal distribution is ~1.349 sigma
    assert dos_stats.phonon_dos_IQR(f, dos) == pytest.approx(1.349 * sigma, rel=1e-2)


def test_shannon_entropy_is_positive_and_maximal_for_uniform():
    f = np.arange(100.0)
    uniform = np.ones(100)
    peaked = np.exp(-0.5 * ((f - 50) / 5) ** 2)
    assert dos_stats.phonon_shannon_entropy(f, uniform) == pytest.approx(math.log(100))
    assert 0 < dos_stats.phonon_shannon_entropy(f, peaked) < math.log(100)


def test_shannon_entropy_ignores_zero_densities():
    f = np.arange(4.0)
    assert dos_stats.phonon_shannon_entropy(f, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(math.log(2))


# --- thermal occupation ---

def test_bose_einstein_matches_formula():
    energy = dos_stats.frequency_to_energy(5 * THZ)
    T = 300
    expected = 1 / (math.exp(energy / (constants.Boltzmann * T)) - 1)
    assert dos_stats.bose_einstien_distribution(energy, T) == pytest.approx(expected)


def test_bose_einstein_zero_energy_does_not_divide_by_zero():
    assert dos_stats.bose_einstien_distribution(0.0, 300) == 0.0


def test_scale_by_occupation_weights_low_frequencies_more():
    f = np.array([1, 10]) * THZ
    scaled = dos_stats.scale_by_occupation(np.ones(2), f, 300)
    assert scaled[0] > scaled[1] > 0


# --- full analysis pipeline ---

def test_requires_a_data_source():
    with pytest.raises(ValueError):
        dos_stats.dos_stats_analysis()


def test_athermal_stats(patch_mp):
    d = dos_stats.dos_stats_analysis(mp_id="mp-test")
    for site, p in SITE_PARAMS.items():
        stats = d["projection"][site]["stats"]["athermal"]
        assert stats["band_centre"] == pytest.approx(p["centre"], rel=1e-3)
        assert stats["shannon_entropy"] > 0
        assert "thermal" not in d["projection"][site]["stats"]


def test_thermal_stats_use_scaled_densities(patch_mp):
    d = dos_stats.dos_stats_analysis(mp_id="mp-test", temp=[300])
    stats = d["projection"]["A_1"]["stats"]
    athermal, thermal = stats["athermal"], stats["thermal"]["300"]
    # occupation favours low frequencies, so everything shifts down
    assert thermal["band_centre"] < athermal["band_centre"]
    assert thermal["shannon_entropy"] != pytest.approx(athermal["shannon_entropy"])
    expected = dos_stats.phonon_shannon_entropy(None, thermal["densities"])
    assert thermal["shannon_entropy"] == pytest.approx(expected)


def test_scalar_temperature_accepted(patch_mp):
    d = dos_stats.dos_stats_analysis(mp_id="mp-test", temp=300)
    assert list(d["projection"]["A_1"]["stats"]["thermal"]) == ["300"]


def test_zero_kelvin_dropped_without_mutating_caller(patch_mp):
    temps = [0, 300, 600]
    d = dos_stats.dos_stats_analysis(mp_id="mp-test", temp=temps)
    assert temps == [0, 300, 600]
    assert sorted(d["projection"]["A_1"]["stats"]["thermal"]) == ["300", "600"]


def test_dos_dict_to_dataframe(patch_mp):
    d = dos_stats.dos_stats_analysis(mp_id="mp-test", temp=[300, 600])
    df = dos_stats.dos_dict_to_dataframe(d)
    n_sites = len(d["projection"])
    assert len(df) == n_sites * 3  # athermal + two temperatures
    assert set(df["temperature"].dropna()) == {300.0, 600.0}
    assert (df["mp_id"] == "mp-test").all()


# --- by-element stats ---

@pytest.mark.parametrize("label, element", [("O_1", "O"), ("Mg", "Mg"), ("Fe2+_3", "Fe"), ("A_1", "A")])
def test_element_from_site_label(label, element):
    assert dos_stats.element_from_site_label(label) == element


def test_element_projections_sum_sites_of_same_element():
    f = np.arange(3.0)
    projection = {
        "total": {"densities": np.array([6.0, 6.0, 6.0]), "frequencies": f},
        "O_1": {"densities": np.array([1.0, 2.0, 3.0]), "frequencies": f},
        "O_2": {"densities": np.array([3.0, 2.0, 1.0]), "frequencies": f},
        "Mg": {"densities": np.array([2.0, 2.0, 2.0]), "frequencies": f},
    }
    elements = dos_stats.element_projections(projection)
    assert set(elements) == {"O", "Mg"}
    np.testing.assert_array_equal(elements["O"]["densities"], [4.0, 4.0, 4.0])
    np.testing.assert_array_equal(elements["Mg"]["densities"], [2.0, 2.0, 2.0])
    # the site densities are not modified
    np.testing.assert_array_equal(projection["O_1"]["densities"], [1.0, 2.0, 3.0])


def test_element_stats(patch_mp):
    d = dos_stats.dos_stats_analysis(mp_id="mp-test", temp=[300])
    assert "total" not in d["element"]
    assert set(d["element"]) == {"A", "B"}
    for site, p in SITE_PARAMS.items():
        element = dos_stats.element_from_site_label(site)
        stats = d["element"][element]["stats"]
        assert stats["athermal"]["band_centre"] == pytest.approx(p["centre"], rel=1e-3)
        site_thermal = d["projection"][site]["stats"]["thermal"]["300"]
        assert stats["thermal"]["300"]["band_centre"] == pytest.approx(site_thermal["band_centre"])


def test_element_band_centre_is_weighted_mean_of_sites(patch_mp, monkeypatch):
    # make B_1 a second site of element A, so A has two sites
    raw = copy.deepcopy(patch_mp)
    raw["projection"]["A_2"] = raw["projection"].pop("B_1")
    from singingmaterials import mp_interface
    monkeypatch.setattr(mp_interface, "get_dos_raw_mp", lambda mp_id: copy.deepcopy(raw))

    d = dos_stats.dos_stats_analysis(mp_id="mp-test")
    sites = [d["projection"][s]["stats"]["athermal"] for s in ("A_1", "A_2")]
    expected = (sum(s["band_centre"] * s["integrated_dos"] for s in sites)
                / sum(s["integrated_dos"] for s in sites))
    assert list(d["element"]) == ["A"]
    assert d["element"]["A"]["stats"]["athermal"]["band_centre"] == pytest.approx(expected)


def test_dos_dict_to_dataframe_by_element(patch_mp):
    d = dos_stats.dos_stats_analysis(mp_id="mp-test", temp=[300, 600])
    df = dos_stats.dos_dict_to_dataframe(d, by="element")
    assert len(df) == len(d["element"]) * 3
    assert set(df["element"]) == {"A", "B"}
    with pytest.raises(ValueError):
        dos_stats.dos_dict_to_dataframe(d, by="atom")
