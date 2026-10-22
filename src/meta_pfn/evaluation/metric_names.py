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
