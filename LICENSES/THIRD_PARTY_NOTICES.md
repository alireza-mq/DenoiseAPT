# Third-party notices

The runnable demonstration depends on NumPy and PyTorch. Their packages and
licenses are distributed by their respective projects and are not bundled in
this repository.

The optional TSB-AD-U data is not redistributed. The download and curation code
is provided by the TSB-AD project, while individual constituent datasets retain
their own terms. Review the TSB-AD license table and each original source before
downloading, redistributing, or using those records commercially.

`denoiseapt/publication_metrics.py` contains a minimal NumPy port of the VUS-PR
algorithm from TSB-AD 1.5, commit
`e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48`. The relevant upstream files are
`TSB_AD/evaluation/basic_metrics.py` and `TSB_AD/evaluation/metrics.py` at
<https://github.com/TheDatumOrg/TSB-AD>. The upstream code is Apache-2.0; the
full license text is included in
[`APACHE-2.0-THIRD-PARTY.txt`](APACHE-2.0-THIRD-PARTY.txt).

The external-denoiser benchmark additionally used PyWavelets 1.8.0 and
Noisereduce 3.0.3 under their MIT licenses. Those packages are not vendored.

The RINS-T comparison used an official-architecture/repository-recipe
adaptation at commit `95d1d9b44b44ba771b2400f5fb68fe42447c5fa2`. No software
license was present in the pinned checkout, so no RINS-T source is redistributed
here. Obtain it directly from its authors and review its current terms before
use.

No third-party JavaScript, fonts, or images are fetched by the browser UI.
