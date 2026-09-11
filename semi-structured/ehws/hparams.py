"""Phase 2 learning rate / penalty (lambda) schedule, per model and sparsity.

Values for OPT-125M and OPT-1.3B are taken directly from ELSA's own
Table 5 (`ELSA.pdf`, page 16) -- since Phase 2 is architecturally the
same ADMM structure ELSA itself uses (global joint optimization under
the true loss), reusing ELSA's own tuned hyperparameters for the models
it shares with us is far more defensible than guessing. Ellipsis rows
in that table's OPT-125M column stop at 80%; 90% reuses 80%'s value
(the table's own apparent convention -- LLaMA-2-13B similarly shares one
cell across 50-90%). 30% isn't in ELSA's table at all (their sweep
starts at 50%); we reuse 50%'s values, extrapolating the table's own
trend of *smaller* lambda / *larger* lr at higher sparsity (i.e. 30%
should if anything want an even gentler setting than 50%, so reusing
50%'s is a conservative choice, not an aggressive one).

`fla-hub/hgrn-1.3B-100B` has no published *table* reference (it isn't one
of ELSA's paper models), but real ELSA's own code was actually run
against it: `ELSA-official/results/hgrn-1.3b-0.8/run.log` is a complete,
un-truncated log of that exact run (`admm_lr=2e-4, admm_lmda=0.01,
admm_lmda_schedule_mode=constant, admm_beta2=0.95`, 4096 steps, interval
32, batch_size=1 x grad_accum=8, 32768 *unique* C4 training docs, no
resampling) and its final line is `[('wikitext2', 54.083...), ('c4',
36.917...)]` -- the exact number this whole project's gap-closing effort
targets. **This is measured ground truth, not an approximation** -- found
2026-09-11 while auditing EHWS against ELSA-official/ for this session's
gap-closing pass, sitting unnoticed in the sibling repo the entire time
this project used OPT-1.3B's *paper-table* row (lr=1e-3, lambda=5e-5,
cosine schedule) as a "closest analogue" guess instead. The two are wildly
different: real ELSA's lambda (0.01, constant) is 200x EHWS's previous
5e-5 peak (itself only reached at the end of a 0->max cosine ramp, so the
effective average pull during training was even further below that) --
*every* HGRN data point collected in this project before this session
(the original lr/n_calib tuning, the damping sweep, Z-step ablation, KD
temperature, alpha_kd) ran at that 200x-too-small lambda throughout, so
none of them cleanly isolated what the ADMM penalty term itself was
contributing. See `_HGRN_1_3B` below: sparsity=0.80 now uses these
measured values; every other sparsity has no ground truth and still
falls back to the OPT-1.3B paper-table approximation.

**One point has been retuned, not just copied.** ELSA's own values were
tuned for their pure cross-entropy objective; this method's X-step also
mixes in a knowledge-distillation term (`alpha_kd=0.5`), which dilutes
the effective gradient magnitude enough that ELSA's own learning rate is
too conservative here. `Extreme_Layer_Global_Pruning_Unstructured`
measured this directly at OPT-125M/50%: scaling `lr` to 3x ELSA's
published value (lambda and schedule unchanged) took that method's
WikiText2 perplexity from 50.99 to 35.05. Applying the same 3x scaling
here -- this package's own Phase 1/Phase 2 redesign (see `ehws/admm.py`'s
module docstring) hasn't been independently re-swept against it yet.
"""

from __future__ import annotations

# sparsity -> (lr, lambda) for a model column reusing ELSA's own values.
_OPT_125M = {
    0.30: (1e-5, 1e-2),
    0.50: (3e-5, 1e-2),  # retuned: 3x ELSA's own lr (1e-5) for the CE+KD objective -- see module docstring
    0.60: (5e-5, 5e-3),
    0.70: (3e-4, 2e-3),  # retuned: 3x ELSA's own lr (1e-4), same fix as 0.50, previously never applied here
    0.80: (6e-4, 1e-3),  # retuned: 3x ELSA's own lr (2e-4) -- see module docstring
    0.90: (6e-4, 1e-3),  # retuned: reuses 0.80's retuned lr, matching ELSA's own table convention of 90% reusing 80%
}
_OPT_125M_SCHEDULE = "constant"

_OPT_1_3B = {
    0.30: (1e-1, 5e-5),
    0.50: (1e-1, 5e-5),
    0.60: (1e-2, 5e-5),
    0.70: (5e-3, 5e-5),
    0.80: (1e-3, 5e-5),
    0.90: (1e-3, 5e-5),
}
_OPT_1_3B_SCHEDULE = "cosine"

# sparsity -> (lr, lambda, schedule). Same OPT-1.3B-approximation values/
# schedule as _OPT_1_3B at every sparsity except 0.80, which is real
# measured ground truth from ELSA-official/results/hgrn-1.3b-0.8/run.log
# (admm_lr=2e-4, admm_lmda=0.01 constant -- see module docstring). Only
# 0.80 has ever actually been run through real ELSA's own code against
# this model; the rest remain the same unverified analogue as before.
_HGRN_1_3B = {
    0.30: (1e-1, 5e-5, "cosine"),
    0.50: (1e-1, 5e-5, "cosine"),
    0.60: (1e-2, 5e-5, "cosine"),
    0.70: (5e-3, 5e-5, "cosine"),
    0.80: (2e-4, 1e-2, "constant"),  # measured (run.log), not approximated -- see module docstring
    0.90: (1e-3, 5e-5, "cosine"),
}


def _nearest_table_sparsity(table: dict, sparsity: float) -> float:
    return min(table.keys(), key=lambda s: abs(s - sparsity))


def get_phase2_hparams(model_name: str, sparsity: float) -> tuple[float, float, str]:
    """Return (lr, lambda_max, schedule) for Phase 2 at this sparsity.

    Falls back to OPT-1.3B's schedule/values (documented approximation,
    see module docstring) for any model not explicitly tabulated here.
    """
    name = model_name.lower()
    if "opt-125m" in name:
        table, schedule = _OPT_125M, _OPT_125M_SCHEDULE
    elif "opt-1.3b" in name or "opt1.3b" in name:
        table, schedule = _OPT_1_3B, _OPT_1_3B_SCHEDULE
    elif "hgrn" in name:
        key = _nearest_table_sparsity(_HGRN_1_3B, sparsity)
        return _HGRN_1_3B[key]
    else:
        table, schedule = _OPT_1_3B, _OPT_1_3B_SCHEDULE  # anything else not tabulated: closest analogue

    key = _nearest_table_sparsity(table, sparsity)
    lr, lam = table[key]
    return lr, lam, schedule
