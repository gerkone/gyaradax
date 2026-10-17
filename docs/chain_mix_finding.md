# The 6 discrepant <k_perp^2> pairs: chain-mix test

Written 2026-10-17. Kinetic's session ended before this could be sent; it
answers their last question.

## Background

On 25 paired runs (GKW `_Lin_gkw` vs gyaradax `_Lin_gy`, same physics), 19
agree on `k_perp_eff_squared` to <=1e-5 and 6 disagree by 0.3-0.5% median,
up to 18% at the worst ky: extra_iteration_3, 6, 11, 17, 19, 22.

Ruled out already: run length. The gyaradax side ran to t=320, the same as
GKW, converged with late drift <=6e-6 (only iteration_22 at 3e-2).

## Kinetic's mechanism

Both codes renormalise EACH kx chain (ballooning theta0) separately, so
subdominant chains are not exp(-dgamma*t) suppressed. Between
renormalisations (GKW: naverage 40 x dtim 0.01 = 0.4) each chain grows at
its own gamma, so the relative chain amplitudes at a dump are off by
exp(dgamma * tau), tau being the time since the last renormalisation.

Prediction: discrepancy scales with dgamma BETWEEN chains, not with how
dominant the leading chain is.

## Test, GKW side, all 25 pairs

Per unstable ky: chain labels from `mode_label`, their growth rates from
`growth_rates_all_modes` at the final time, dgamma = max - min across
chains. Median over unstable ky.

| group | n | dgamma median | predicted mix error exp(dgamma*0.4)-1 |
|---|--:|--:|--:|
| agreeing | 19 | 0.4545 | 19.9% |
| discrepant | 6 | 0.4914 | 21.7% |

Mann-Whitney U, one-sided: **p = 0.018**.

## Reading

The ordering is real and in the predicted direction. The magnitude is not:
the predicted mix error is ~20% for BOTH groups, yet 19 runs agree to 1e-5.
The mechanism as stated predicts error everywhere and we see it in six.

Reconciliation, which is kinetic's own parenthetical: the operative term is
exp(dgamma * tau) with tau the time since the last renormalisation AT THE
DUMP INSTANT, not the full 0.4. If both codes dump at the same phase of the
cycle, tau ~ 0 and the chains agree whatever dgamma is. The six would be the
phase-mismatched ones, with dgamma setting only how badly it hurts. That
fits the p = 0.018 ordering, the 19 near-exact agreements, and the earlier
observation that all 8 runs with a symmetry-locked +/-kx pair
(2nd/1st = exactly 1.000, so dgamma = 0) agree.

Testing the phase part needs each code's dump time relative to its own
renormalisation cycle, which was not available here.

Caveats: n = 6 against 19, the hypothesis was formed after seeing which runs
disagreed, and the effect is 0.491 against 0.455. Weak, post-hoc evidence,
consistent with the mechanism but not proof of it.

## The consequence, which does not depend on the test

"Reproducible between codes" and "physically defined" differ. Both codes
share the per-chain normalisation, which is why the all-chain weights agree
to 0.1%; a shared arbitrary convention is still arbitrary. The physics
argument for using chain0 (theta0 = 0) for the weights and <k_perp^2>
stands on its own.

**But before adopting chain0 as primary:** the earlier cross-code check had
all-kx agreeing to 0.1% and chain0 disagreeing by 11%. If chain0 were the
physically defined quantity it should agree BETTER, not worse. That points
at the chain0 SELECTION differing between codes -- GKW via kx_connect.dat,
gyaradax via mode_label/ixzero -- rather than at chain0 being wrong. On the
8 symmetry-locked runs chain0 is unambiguous, so if those also disagree by
11% the selection is the problem. Resolve that first, or we swap a
well-defined-but-conventional quantity for a physical one computed
inconsistently.
