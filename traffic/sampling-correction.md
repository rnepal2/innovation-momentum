# Sampling inverse correction

Independent sensor-coordinate covariance validation detected inconsistent numerical rank handling in the conditional spatial reconstruction. The uniform-12 dictionary has two exact nuisance-mode aliases. The reported rank and nullspace calculation excluded those directions, while NumPy's default pseudoinverse retained some floating-point remnants. This contaminated otherwise identifiable q=3 coefficients.

The reconstruction now uses relative SVD cutoff 10^-10 consistently for the inverse and nullspace. An exact dictionary fixture verifies that recovered q=3 coefficients annihilate every declared nuisance mode on the uniform-12 mask. Independent least-squares recovery and full sensor-coordinate covariance propagation check the implementation.

The correction affects analysis of existing physical arrays; no physical parameters or observations change. Complete-mask conclusions are unchanged. Previous diagnostic and rejection-summary tables are preserved before reanalysis. Separate analysis-source archives record the corrected code. The simulator-clock correction remains documented separately in timing-correction.md.
