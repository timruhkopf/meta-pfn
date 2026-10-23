"""All metric names are constants: `<split>/<name>` series, `best/...`, `final/...` summaries."""

TRAIN_MSE = "train/mse"
FINAL_TRAIN_MSE = "final/train_mse"
FINAL_EVAL_MSE = "final/eval_mse"
FINAL_W_ERROR = "final/w_error"

# headroom (E0)
PAIRS_DONE = "progress/pairs_done"
GAIN_BAYES = "final/gain_bayes"  # + "/n<n_A>": mean over pairs of NLL_lower - NLL_bayes (nats per query)
GAIN_ORACLE = "final/gain_oracle_T"  # + "/n<n_A>": NLL_lower - NLL_oracle_T
TIME_PREFIX = "time/"

# sampler_check (E0 follow-up)
GAIN_KIND_SMC = "final/gain_kind_smc"  # + "/n<n_A>": NLL_lower - NLL of the SMC learner told the true kind
GAIN_KIND_IS = "final/gain_kind_is"  # + "/n<n_A>": NLL_lower - NLL of the importance sampler told the true kind

# outmap (E0b)
GAIN_ORACLE_G = "final/gain_oracle_g"  # + "/n<n_A>": NLL_lower_bayes - NLL_oracle_g (GAIN_BAYES is vs lower_bayes)

# regpfn (phase 1)
REGPFN_NLL_N5 = "final/nll_n5"  # mean NLL on A's queries at n_A = 5 (no B for aonly; unperturbed seed for back)
METAPFN_GAIN_N5 = "final/gain_n5"  # metapfn: NLL(A-only PFN) - NLL(model) at n_A = 5, mean over the 4 related cells
