# References

Annotated literature for this project. BibTeX for every entry lives in `references.bib` under the same key; `just refs-check` verifies that both files agree.

**Status values:** `to-read` · `skimmed` · `read` · `cited` · `to-verify` (id or BibTeX not yet checked against the source)

**Entry template**

```markdown
### [key] Authors (Year). *Title*. Venue. DOI or arXiv id
- **Status:**
- **Claim:** what the paper shows, in one sentence, in our own words.
- **Relevance:** supports / contradicts / baseline / method source / dataset / evaluation — and why.
- **Used in:** configs, code modules, research log entries that depend on it.
- **Notes:** setup details we must match (splits, metric definitions, seeds, compute).
```

---

## Problem & motivation

### [huang2023landscapes] Huang, Li (2023). *On the Hyperparameter Loss Landscapes of Machine Learning Models: An Exploratory Study*. arXiv:2311.14014
- **Status:** skimmed (to-verify: the exact statistics we calibrate against)
- **Claim:** Publishes landscape statistics across datasets for HPO of the same model, and finds the landscapes similar across datasets (correlated rankings, overlapping good regions).
- **Relevance:** motivation and calibration target: how related real B and A are. Our ρ grid (relatedness; σ before 2026-10-22) (Spearman 0.9 / 0.7) is chosen against it.
- **Used in:** `pairs.py`; E0 log §2

### [snoek2014warping] Snoek, Swersky, Zemel, Adams (2014). *Input Warping for Bayesian Optimization of Non-stationary Functions*. ICML 2014. arXiv:1402.0929
- **Status:** read
- **Claim:** Learning a per-axis Kumaraswamy (Beta-CDF) warp of the inputs makes GP-based BO handle non-stationary landscapes.
- **Relevance:** method source for the `warp` family: per-axis Kumaraswamy CDFs model how HPO axes stretch.
- **Used in:** `pairs.WarpFamily`

## Closest related work

### [feurer2018rgpe] Feurer, Letham, Hutter, Bakshy (2018). *Practical Transfer Learning for Bayesian Optimization*. arXiv:1802.02219
- **Status:** read
- **Claim:** RGPE: an ensemble of per-task GPs weighted by how well each ranks the target's observations.
- **Relevance:** the standard transfer-BO baseline; it weights whole tasks and doesn't register them, which is exactly the gap a Meta-PFN would fill.
- **Used in:** README; E1 baselines (planned)

### [wistuba2021fsbo] Wistuba, Grabocka (2021). *Few-Shot Bayesian Optimization with Deep Kernel Surrogates*. ICLR 2021. arXiv:2101.07667
- **Status:** skimmed
- **Claim:** A deep kernel meta-learned across tasks gives strong few-shot BO on HPO benchmarks.
- **Relevance:** a meta-learned surrogate baseline (not in context: it fine-tunes per task).
- **Used in:** README

### [perrone2018ablr] Perrone, Jenatton, Seeger, Archambeau (2018). *Multiple Adaptive Bayesian Linear Regression for Scalable Bayesian Optimization with Warm Start*. NeurIPS 2018. arXiv:1712.02902
- **Status:** skimmed
- **Claim:** A shared neural feature map with per-task Bayesian linear heads warm-starts BO from related tasks.
- **Relevance:** transfer with a shared representation but no explicit registration.
- **Used in:** README

### [rakotoarison2024ifbo] Rakotoarison, Adriaensen, Mallik, Garibov, Bergman, Hutter (2024). *In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization*. ICML 2024. arXiv:2404.16795
- **Status:** read
- **Claim:** A PFN trained on a learning-curve prior does freeze-thaw BO in context.
- **Relevance:** the predecessor's code base (ifBO); a PFN-for-HPO precedent.
- **Used in:** decision 0002

## Method building blocks

### [mueller2022pfn] Müller, Hollmann, Pineda Arango, Grabocka, Hutter (2022). *Transformers Can Do Bayesian Inference*. ICLR 2022. arXiv:2112.10510
- **Status:** read
- **Claim:** A transformer trained on data sampled from a prior approximates that prior's posterior predictive in one forward pass.
- **Relevance:** why the frozen PFN can serve as the Bayes oracle's predictive and likelihood in E0, and why a Meta-PFN trained on a pair prior learns transfer.
- **Used in:** `oracle.py`

### [mueller2023pfns4bo] Müller, Feurer, Hollmann, Hutter (2023). *PFNs4BO: In-Context Learning for Bayesian Optimization*. ICML 2023. arXiv:2305.17535
- **Status:** read
- **Claim:** PFNs trained on a BNN prior are strong BO surrogates.
- **Relevance:** source of the BNN prior and PFN design that pfn-repro reproduces.
- **Used in:** pfn-repro

### [vapnik2009lupi] Vapnik, Vashist (2009). *A new learning paradigm: Learning using privileged information*. Neural Networks 22, 544–557. doi:10.1016/j.neunet.2009.06.042
- **Status:** read
- **Claim:** Information available only at training time (privileged) can speed up learning of the test-time predictor.
- **Relevance:** the principle behind supervising a Meta-PFN's registration with the known warp.
- **Used in:** README; E1 (planned)

## Baselines

E0's predictors are its baselines: A alone (`lower`), pooling without registration (`naive`), the known warp (`oracle_T`).

## Datasets & benchmarks

Synthetic pairs on pfn-repro's BNN prior (`pairs.py`); no external datasets in Phase 0.

## Evaluation & statistics

### [bouthillier2021variance] Bouthillier et al. (2021). *Accounting for Variance in Machine Learning Benchmarks*. MLSys 2021. arXiv:2103.03098
- **Status:** read
- **Claim:** Ignoring sources of variation leads to wrong conclusions; randomize them and compare paired.
- **Relevance:** paired comparisons over pairs, the Monte-Carlo error of the oracle as its own source of variance.
- **Used in:** `analysis/headroom.py`

## Tools & software

PyTorch, Hydra, MLflow, uv (cite as each project recommends).

---

## Reading queue

- Swersky, Snoek, Adams (2013). *Multi-Task Bayesian Optimization*. NeurIPS 2013. Multi-task GPs for transfer between related HPO tasks (the coinciding-landscapes assumption). BibTeX to fetch; to-verify.
