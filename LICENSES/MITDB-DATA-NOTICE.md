# MIT-BIH Arrhythmia Database replay data notice

`data/prepared/tsb_ad_mitdb_anomaly_preservation_replay.npz` contains a derived
512-time-point ECG window from the MIT-BIH Arrhythmia Database, together with
controlled corruptions and frozen benchmark outputs generated for the
DenoiseAPT demonstration.

The MIT-BIH Arrhythmia Database was created by George B. Moody and Roger G.
Mark and is distributed by PhysioNet under the Open Data Commons Attribution
License 1.0:

- Database record: https://physionet.org/content/mitdb/1.0.0/
- Dataset DOI: https://doi.org/10.13026/C2F305
- License: https://physionet.org/content/mitdb/view-license/1.0.0/

Suggested source citation: G. B. Moody and R. G. Mark, "The impact of the
MIT-BIH Arrhythmia Database," *IEEE Engineering in Medicine and Biology
Magazine*, vol. 20, no. 3, pp. 45-50, 2001.

The replay is provided with attribution under those terms. It is an
illustrative, post-hoc selected held-out replay rather than a new benchmark, and
it is not covered by the repository's MIT license. It is intended only to
demonstrate signal-processing behavior and is not a medical diagnosis or
clinical validation.
