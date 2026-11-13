# Singing Materials

`SingingMaterials` is a Python package which reads and analyses atomistic simulation data, and uses [Strauss](https://www.audiouniverse.org/research/strauss) to turn it into sound; *to hear how materials sing*.

A description of this package has been published in the following [pre-print](https://arxiv.org/abs/2603.29037), and is accepted to the [ICAD proceedings](https://interactive-sonification.org/icad2026/). A `.bib` file is in the repo, and the plain-text citation is: Whalley, L., Shepherd, R., Boehringer, J., Knotts, S., Vickers, P., Caselton, G., Harrison, C., Hogg, B., Ratliff, D., Davenport, C., & Portas, A. (2026). *Singing Materials: Initial experiments in applying sonification to phonon spectra.* arXiv:2603.29037 [cond-mat.mtrl-sci]

This repository holds the package only. For examples of how it has been used, including scripts, input files and sound examples, see the [SingingMaterials_Dissemination](https://github.com/lucydot/SingingMaterials_Dissemination) repository.

## What is this?

- The package currently focuses on the [sonification](#what-is-sonification) of phonon density of states (DOS) data, which is read in from the [Materials Project](https://next-gen.materialsproject.org/) or [phonondb](https://github.com/atztogo/phonondb/blob/main/mdr/phonondb/README.md) databases.
- It calculates statistics for each site in a material, such as the phonon band centre and the Shannon entropy, optionally weighted by temperature. This part of the package can also be used without producing any sound.
- Three sonification types are supported: spectral, synth and choral. These can be mixed together in different ways (superimpose, concatenate or a mixture of the two). Mixing is implemented through `ffmpeg`.
- If you want to mix sonifications you must use `.yml` files rather than the command line interface. The command line interface does not allow mixing.
- The package is designed to be extendable, so that new types of phonon data, new data interfaces, and new sonifications can be implemented in the future.
- This package also powers the `AtomicStrudel` web app in-development [here](https://github.com/lucydot/AtomicStrudel).

This is alpha-release software: it has been tested, but there are likely still bugs - please use with caution! If you spot any issues please raise this on the [Issues page](https://github.com/lucydot/SingingMaterials/issues).

## What is Sonification?

Sonification is the use of non-speech audio to convey information. A well-known scientific example is the geiger counter which produces an audible click when it detects an ionisation event. Sonification is being increasingly used within the physical sciences, in particular for [astrophysical research](https://www.scientificamerican.com/video/listen-to-the-astonishing-chirp-of-two-black-holes-merging). 
Sonification has also been used as a tool for music composition by a [number of contemporary classical composers](https://www.straebel.com/files/Straebel%202010_Sonification%20Metaphor.pdf). 
The continuum from data representation to data abstraction is briefly discussed in the paper [`Making data sing`](https://www.researchgate.net/profile/Atau-Tanaka/publication/312740596_Making_Data_Sing_Embodied_Approaches_to_Sonification/links/5fc6b5f2299bf188d4e8d59e/Making-Data-Sing-Embodied-Approaches-to-Sonification.pdf) (Parkinson and Tanaka, 2013).

## Installation

```bash
pip install "SingingMaterials[all]"
```

SingingMaterials supports Python 3.11 to 3.14. On newer versions of Python, dependencies such as `h5py` may not have pre-built wheels yet, and the install can fail.

This installs everything. The package is then imported in Python as `singingmaterials`. To leave out dependencies you don't need, choose from these options instead:

| Option | Needed to | Installs |
|---|---|---|
| `sound` | create sonifications | [Strauss](https://www.audiouniverse.org/research/strauss) |
| `mp` | read data from the Materials Project | [mp-api](https://docs.materialsproject.org/downloading-data/using-the-api/getting-started) |
| `phonopy` | read phonondb files | [phonopy](https://phonopy.github.io/phonopy/) |

For example, `pip install "SingingMaterials[mp]"` is enough to read Materials Project data and calculate statistics, without producing any sound. You need at least one of `mp` or `phonopy` to read in new data: with `pip install SingingMaterials` alone, only DOS data already cached in your working directory can be read. 

To read data from the Materials Project, [generate an API key](https://next-gen.materialsproject.org/api) and store it as `MP_API_KEY` in a `.env` file in your working directory (or one of its parents), or set it as an environment variable.

Mixing sonifications also needs [`ffmpeg`](https://ffmpeg.org/) on your path.

The choral sonification uses the Solitary Choir samples by [James Peacock](https://www.pianobook.co.uk/profile/jamespeacock42), which are not distributed with this package. Set `PHONON_CHORAL_SAMPLES` to the folder holding the sample `.wav` files, as an environment variable or in your `.env` file, or pass `sample_path` to `sonify_site_choral`.

To work on the package itself, clone this repository and run `pip install -e ".[all,test]"` from its root. The test suite is then run with `pytest`.

## Usage

### Creating sonifications

From the command line:

```bash
phonon-sonify mp-3953 --info --temp 300          # print the sites available for a material
phonon-sonify mp-3953 --temp 300 --all-sites     # sonify all sites (spectral mode by default)
phonon-sonify mp-3953 --site O_6 --mode synth --lfo
phonon-sonify-yaml my_spec.yml                   # create and mix sonifications from a spec file
```

See [`spec_example.yml`](https://github.com/lucydot/SingingMaterials/blob/main/singingmaterials/spec_example.yml) for the spec file format.

From Python:

```python
from singingmaterials import PhononDOSSonifier

sonifier = PhononDOSSonifier(mp_id="mp-3953", temperatures=[300], duration=10.0)
sonifier.print_available_sites(temperature=300)
sonifier.sonify_all_sites(temperature=300, output_path="mp-3953.wav")
```

### Data and statistics only

```python
from singingmaterials import visualisation
from singingmaterials.dos_stats import dos_dict_to_dataframe, dos_stats_analysis

dos_dict = dos_stats_analysis(mp_id="mp-3953", temp=300)
print(
    dos_dict["projection"]["total"]["stats"]["thermal"]["300"]["band_centre"]
)  # in Hz
print(dos_dict_to_dataframe(dos_dict))
visualisation.plot_dos_at_temperature(dos_dict)
```

Stats are given per site in `dos_dict["projection"]` (which also holds the `"total"` DOS), and per element in `dos_dict["element"]`, where the densities of all sites of the same element are summed:

```python
for element, data in dos_dict["element"].items():
    print(element, data["stats"]["athermal"]["band_centre"])  # in Hz

print(dos_dict_to_dataframe(dos_dict, by="element"))
```

### Fitting the DOS with Gaussian peaks

The DOS can be summarised as a set of Gaussian peaks, for the total DOS or for each element's projected DOS, and the peaks saved to a CSV file:

```python
from singingmaterials import peak_fitting, visualisation

fits = peak_fitting.fit_dos_peaks("mp-3953", projection="element",
                                  n_peaks={"Ca": 6, "C": 4, "O": 6}, smoothing={"Ca": 0.1})
visualisation.plot_peak_fits(fits)
peak_fitting.save_peak_fit(fits)  # -> peak_fit_mp-3953_element.csv
```

Pass `temperature` (K) to fit the DOS weighted by the Bose-Einstein occupation at that temperature. The [peak fitting tutorial](tutorials/peak_fitting.ipynb) explains the method and how to choose the settings.

## Package structure

`mp_interface.py` : interface to the Materials Project database holding density of states data.

`phonopy_interface.py` : reads in data from a phonopy yml file, and generates density of states data for a given mesh. The phonopy files are distributed as part of [phonondb](https://github.com/atztogo/phonondb/blob/main/mdr/phonondb/README.md). This module requires phonopy to be installed.

`dos_stats.py` : this reads in density of states data via one of the interfaces and post-processes it ready for analysis or sonification.

`peak_fitting.py` : fits the density of states (total, or projected by element or site) with a set of Gaussian peaks, and saves the peaks to a CSV file.

`visualisation.py` : functions for plotting the density of states data, the band structure and the peak fits. because seeing things is still useful.

`utilities.py` : small utility functions.

`frequency_mapping.py` : functions for mapping from phonon frequencies to audible frequencies.

`phonon_dos_sonifier.py`: takes the post-processed density of states data and sonifies it using Strauss. Three sonification strategies are implemented: spectral (using the Strauss spectraliser), synth (using the Strauss synth) and choral (using the strauss sampler and choral samples). This module requires Strauss to be installed.

`phonon_mixer.py` : takes the `.wav` files created by `phonon_dos_sonifier.py` and mixes them together with `ffmpeg`.

`cli.py` : command-line-interface for the `phonon_dos_sonifier` module. Allows the user to create `.wav` files for the spectral, synth or choral sonifications, but does not allow user to mix these together.

`run_from_yaml.py` : reads in a `.yml` spec file (see `spec_example.yml`) and uses it to create and mix sonifications.

`spec_example.yml` : example yml file which specifies how to create the sonifications and mix them together.

`mods.py` : dictionaries specifying mods to the Strauss synthesiser parameters.

## Citation

If you use this package please cite the following [pre-print](https://arxiv.org/abs/2603.29037), which will later be published as part of the [ICAD proceedings](https://interactive-sonification.org/icad2026/):

Whalley, L., Shepherd, R., Boehringer, J., Knotts, S., Vickers, P., Caselton, G., Harrison, C., Hogg, B., Ratliff, D., Davenport, C., & Portas, A. (2026). *Singing Materials: Initial experiments in applying sonification to phonon spectra.* arXiv:2603.29037 [cond-mat.mtrl-sci]

## AI usage statement

The main functionality of SingingMaterials was implemented without AI. From October 2026, AI was used to make changes to this repository, including creating the test suite. Commit `b11c2ef` ("Create LICENSE.md", 15 September 2026) is the last commit made before using AI. Commits after this, starting with `6d034a0` ("Fix crashes and incorrect results in phonon_sonification"), may have been written using [Claude Code](https://claude.com/claude-code) with Opus 5.5. These commits carry a `Co-Authored-By: Claude` line in their commit message, so they can be listed with `git log --grep="Co-Authored-By: Claude"`.

