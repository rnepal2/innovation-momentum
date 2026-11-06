# Retrospective empirical spectral sensitivities

Recorded after acquiring the two fixed NGSIM blocks and inspecting their quality profiles and uncentered spectra. US-101 has 212,427 rows and I-80 has 226,638. There are no duplicate selected keys or missing selected fields; positive per-vehicle time increments are 0.1 s. Extreme position/speed discrepancies remain, with maximum discrepancies about 8.24 and 40.86 m/s. No raw observations are changed.

The primary 50 m grid has a fully observed 13-bin strip for US-101. I-80 has only seven contiguous complete bins and fails the fixed eight-bin minimum. Its registered 25 m sensitivity has 13 complete bins, covering a shorter physical interval. Retain the failed primary grid. Create figures for every eligible resolution and identify I-80's 25 m figure as a sensitivity.

For every eligible fixed field, add constant temporal detrending alongside the existing undetrended periodogram. Keep the same spatial strip and sampling. This tests whether the large-scale integrated power is dominated by a persistent spatial speed profile. Report absolute integrated power, first-two-mode fraction, and fraction of original power remaining. Temporal centering changes the estimand and can remove slow physical variation as well as stationary profile differences; no causal or stationary-covariance conclusion follows.

Count position/speed mismatch above 5 m/s and inspect the largest examples. This is an audit diagnostic, with no automatic exclusion or outcome-selected filtering. Binning, smoothing, and spatial support can alter spectra. Different roads, dates, physical strip lengths, and temporal resolutions prevent direct inference about site-specific instability from the mode fractions.
