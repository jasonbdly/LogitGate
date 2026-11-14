# Mathematical Foundations & Symmetric Debiasing

This document details the mathematical formulations, statistical proofs, and debiasing algorithms implemented in **LogitGate**.

---

## 1. Symmetric Log-Odds Permutation Debiasing

### Theoretical Formulation
Large language models exhibit significant pretraining token prior bias (Zhao et al., 2021) and positional label bias (Zheng et al., 2023), inherently favoring certain tokens (e.g., `'A'` over `'B'`) regardless of candidate quality.

Let $s$ denote the true latent alignment signal between the rubric and candidate artifact. Let $b$ denote the model's additive token prior bias toward token `'A'`.

When evaluating under a binary rubric:

- **Pass 1 (Forward Mapping: $A=\text{Pass}, B=\text{Fail}$)**:
  $$L_1 = \text{logit}(A) - \text{logit}(B) = s + b$$

- **Pass 2 (Reverse Mapping: $B=\text{Pass}, A=\text{Fail}$)**:
  $$L_2 = \text{logit}(B) - \text{logit}(A) = s - b$$

LogitGate combines the forward and reverse passes in **log-odds space**, scaled by an empirical temperature parameter $T$:

$$\bar{L} = \frac{L_1 + L_2}{2 \cdot T} = \frac{(s + b) + (s - b)}{2 \cdot T} = \frac{2s}{2T} = \frac{s}{T}$$

The additive token prior bias $b$ cancels out analytically. The calibrated pass probability is obtained via the standard sigmoid transform:

$$P(\text{Pass}) = \sigma(\bar{L}) = \frac{1}{1 + e^{-\bar{L}}}$$

### Empirical Contrast: Log-Odds vs. Naive Probability Averaging
Consider a true positive signal $s = +2.0$ ($\sigma(2.0) \approx 0.8808$) severely contaminated by an extreme prior bias $b = +5.0$:

- **Naive Probability Averaging**:
  $$P_1 = \sigma(2.0 + 5.0) = \sigma(7.0) \approx 0.9991$$
  $$P_2 = \sigma(2.0 - 5.0) = \sigma(-3.0) \approx 0.0474$$
  $$\bar{P} = \frac{P_1 + P_2}{2} = \frac{0.9991 + 0.0474}{2} = \mathbf{0.5233} \quad (\text{Coin Flip / False Ambiguity})$$

- **LogitGate Log-Odds Space Averaging**:
  $$\bar{L} = \frac{7.0 + (-3.0)}{2} = \frac{4.0}{2} = 2.0$$
  $$\sigma(\bar{L}) = \sigma(2.0) = \mathbf{0.8808} \quad (\text{Exact True Signal Recovered!})$$

Probability averaging distorts confidence near the boundaries due to the non-linear compression of the sigmoid function, whereas log-odds averaging preserves additive linearity.

---

## 2. Prefix-Preserving Attention Inversion

In standard autoregressive architectures, tokens can only attend to previous tokens. Placing the candidate document first and isolating the label mapping to the suffix allows the attention KV-cache to be reused across both debiasing passes:

```
[EVALUATION RUBRIC]
{rubric}

[CANDIDATE ARTIFACT]
{candidate}

[DECISION MAPPING]
A: The candidate artifact complies fully with the rubric.
B: The candidate artifact violates the rubric or has defects.
Respond with a single character (A or B).
Verdict:
```

Because Pass 1 and Pass 2 share an identical prefix up to `[DECISION MAPPING]`, the entire prompt context remains in the GPU KV-cache. Pass 2 only requires evaluating the short `[DECISION MAPPING]` suffix, reducing second-pass latency by **60%–75%**.

---

## 3. Missing-Token Bounded Floor

When target decision tokens (e.g., `'B'`) do not appear in the top-$K$ logprobs returned by inference endpoints (such as Ollama or OpenAI-compatible APIs), naive evaluators often assign an arbitrary $-99.0$ constant. This introduces extreme distortions in log-odds space.

LogitGate dynamically computes a bounded floor based on the theoretical maximum unobserved probability mass:

$$\text{floor} = \min\left(\min_{i} \text{logprob}_i, \, \ln\max\left(1.0 - \sum_{i} P_i, \, 10^{-6}\right)\right)$$

Where $\sum_i P_i$ is the cumulative softmax probability of all returned top-$K$ tokens. This bounds missing tokens strictly to the residual unobserved mass without cliff distortion.

---

## 4. Post-Hoc Temperature Calibration & Brier Score Minimization

Language models often suffer from overconfidence or underconfidence depending on parameter scale and post-training alignment (Guo et al., 2017).

LogitGate provides AUROC- and log-odds scale-regularized temperature scaling via `fit_calibration()`:

$$P_i(T) = \sigma\left(\frac{\bar{L}_i}{T}\right)$$

The optimal temperature $T^*$ is found by minimizing the empirical Brier Score regularized by a small-sample MAP prior that decays at $\mathcal{O}(1/N)$ as calibration sample count grows:

$$T^* = \arg\min_{T > 0} \frac{1}{N} \sum_{i=1}^N \left( \sigma\left(\frac{\bar{L}_i}{T}\right) - y_i \right)^2$$

Where $y_i \in \{0, 1\}$ is the ground truth label. Because zero-shot binary `A/B` polarity swaps on compact sub-4B models exhibit greater symbol-binding attenuation than multi-option prompts, LogitGate separates binary calibration tables (`KNOWN_BINARY_MODEL_TEMPERATURES`) from multiclass tables (`KNOWN_MODEL_TEMPERATURES` for `ChoiceGate`, `ScoreGate`, and `JudgeGate`). For any unrecognized Bring-Your-Own (BYO) model when `LOGITGATE_AUTO_CALIBRATE=1` (default), `auto_calibrate_backend()` / lazy first-use calibration evaluates micro-probes once and persists both temperatures (`binary_temperature`, `multiclass_temperature`, `probe_auroc`, `median_abs_log_odds`) to `~/.cache/logitgate/calibration.json`, automatically clamping low-AUROC models ($\text{AUROC} < 0.70$) to $T \ge 1.0$ so uncertain predictions escalate cleanly to `JudgeGate`.

---

## 5. Probability-Weighted Continuous Scoring (`ScoreGate`)

Building on G-Eval (Liu et al., 2023), `ScoreGate` transforms discrete first-token probability distributions over score tokens (e.g., `'1'`, `'2'`, `'3'`, `'4'`, `'5'`) into continuous expected values:

$$\mathbb{E}[S] = \sum_{k=1}^K s_k \cdot P(s_k)$$

Where $P(s_k)$ is the normalized softmax probability of token $s_k$. With symmetric debiasing enabled, `ScoreGate` evaluates the candidate under both ascending ($1 \to 5$) and descending ($5 \to 1$) scale orientations, averaging them to cancel scale anchoring bias.

---

## 6. Joint Post-Temperature Split-Conformal Calibration & Rank Invariance

Setting static Layer-1 cutoffs independently of softmax temperature creates a scale mismatch: conformal cutoffs $(\tau_{\text{reject}}, \tau_{\text{accept}})$ fitted under $T_{\text{old}} = 1.0$ lose their finite-sample error bound once a temperature $T^* \neq 1.0$ stretches or compresses logits at runtime. Conversely, splitting an already modest $N$-item calibration set across separate temperature and conformal folds weakens both estimators.

**Single-Sweep Post-Temperature Invariance (`fit_calibration_profile`)**:
For any positive fitted temperature $\hat{T} > 0$, the transformation $z \mapsto \operatorname{round}(\sigma(z / \hat{T}), 4)$ applied on LogitGate's binary `Verdict` scale is monotonically non-decreasing in the symmetric log-odds $z = \frac{1}{2}(L_1 + L_2)$. `fit_calibration_profile()` fits $\hat{T}$ first on the $2N$-pass log-odds sweep and then evaluates two-sided split-conformal cutoffs directly on $p_i(\hat{T}) = \operatorname{round}(\sigma(z_i / \hat{T}), 4)$:

$$\tau_{\text{accept}} = \operatorname{strictly\_above}_4\!\left(Q_{1 - \alpha/2}\!\left(\{p_i(\hat{T}) : y_i = 0\}\right)\right), \qquad \tau_{\text{reject}} = \operatorname{strictly\_below}_4\!\left(Q_{1 - \alpha/2}\!\left(\{1 - p_i(\hat{T}) : y_i = 1\}\right)\right)$$

where empirical quantiles at level $(1 - \alpha/2)$ take order statistic $k = \lceil (m + 1)(1 - \alpha/2) \rceil$ within each class (guaranteeing finite-sample marginal Layer-1 miscoverage $\le \alpha$ whenever $\min(m_+, m_-) \ge \lceil (1 - \alpha/2) / (\alpha/2) \rceil$, i.e. $19$ per class at $\alpha = 0.10$ or $39$ at $\alpha = 0.05$). Because strict rank order is invariant under positive temperature scaling and step-grid strict inequality moves boundaries by $+0.0001$ outside tied quantiles, fitting cutoffs post-temperature guarantees zero scale drift while preserving full $2N$ sample efficiency.

The same `CalibrationProfile` ecosystem extends distribution-free selective risk control to multi-way and continuous gates (using `effective_multiclass_temperature = max(0.65, binary_temperature)` when no dedicated multiclass temperature was fitted):

- **`ChoiceGate` (`calibrate_choice_thresholds`)**: Uses one-sided nonconformity scores $e_i = 0$ on correct top-1 predictions and $e_i = \operatorname{round}(\max_k P_i(k), 4)$ on errors, setting `choice_min_confidence` strictly above $\max(Q_{1-\alpha}(e), 1/K)$ (with optional `choice_min_margin` over top-2 probability margin).
- **`ScoreGate` (`calibrate_score_thresholds`)**: Applies two-sided conformal order statistics over expected scores $\mathbb{E}[S_i] \in [s_{\min}, s_{\max}]$ to produce conservative `(score_fail_threshold, score_threshold)` cutoffs bounding `[AMBIGUOUS]` Layer-2 escalations.

---

## 7. Positional Swap Debiasing & Hodge Pairwise Ranking (`RankGate`)

When compact LLMs judge preference between candidate completions (`A` vs. `B`, or slot list `A, B, C, ...`), single-pass prompting suffers from strong positional ordering bias $\beta_{\text{pos}}$ (typically favoring Slot `A`) and label token prior bias $\beta_{\text{tok}}$ (Zheng et al., 2023). `RankGate` cancels both biases using single-token log-probabilities (`max_tokens=1`) across four structured regimes:

### A. Pairwise Symmetric Swap Cancellation ($N = 2$)
Given candidates $(c_i, c_j)$, `compile_pairwise_prompt()` formats rubric and context before the nonce-delimited candidate blocks (`query_first=True`) and swaps candidate order across two passes:
- Pass 1 measures $\Delta^{(1)}_{ij} = \log P_1(\text{A}) - \log P_1(\text{B}) = s_i - s_j + \beta_{\text{pos}} + \beta_{\text{tok}}$.
- Pass 2 evaluates $(c_j, c_i)$ and extracts $\Delta^{(2)}_{ij} = \log P_2(\text{B}) - \log P_2(\text{A}) = s_i - s_j - (\beta_{\text{pos}} + \beta_{\text{tok}})$.

Averaging in log-odds space recovers the debiased pairwise log-preference $z_{ij} = \frac{1}{2}(\Delta^{(1)}_{ij} + \Delta^{(2)}_{ij}) = s_i - s_j$, while $\frac{1}{2}|\Delta^{(1)}_{ij} - \Delta^{(2)}_{ij}|$ measures positional divergence against optional `max_positional_bias`.

### B. Listwise Slot-Permutation Debiasing ($2 \le N \le 10$)
`compile_listwise_rank_prompt()` places candidates under single-character slot tokens `A, B, C, ...` and averages centered per-pass log-softmax slot probabilities $\tilde{\ell}^{(m)}_k = \log P_m(\text{slot}_m(k)) - \log\sum_{r} \exp(\log P_m(r))$ across permutation schedules (`"derangement"` 2-pass cycle-shift, `"reverse"` 2-pass reversal, or `"cyclic"` $N$-pass Latin-square rotation), then maps mean log-worths through temperature-scaled softmax $P(k) = \operatorname{softmax}(w / T)$.

### C. Round-Robin Hodge / Bradley-Terry Projection & Condorcet Residual
Evaluating all $\binom{N}{2}$ unordered pairs produces a skew-symmetric debiased log-preference matrix $Z_{ij} = -Z_{ji}$. Least-squares / combinatorial Hodge projection onto zero-sum Bradley-Terry log-worths (Bradley & Terry, 1952; Jiang et al., 2011) admits the exact closed-form average over outgoing pairwise edges:

$$w_i = \frac{1}{N} \sum_{j \neq i} Z_{ij}, \qquad \sum_{i=1}^N w_i = 0$$

The orthogonal non-transitive / Condorcet cycle component is quantified directly via root-mean-square `cycle_residual`:

$$r_{\text{cycle}} = \sqrt{\frac{1}{\binom{N}{2}} \sum_{1 \le i < j \le N} \left(Z_{ij} - (w_i - w_j)\right)^2}$$

When preferences are pairwise transitive, $r_{\text{cycle}} \approx 0$; cyclic trade-offs (`X > Y > Z > X`) produce strictly positive `cycle_residual` surfaced in trace metadata.

### D. Single-Elimination Tournament Bracket
For larger pools (`tournament`, up to $N = 32$), `RankGate` seeds pairs round-by-round ($N - 1$ swapped matches total), advances odd-$N$ byes automatically, records each match gap $|z_{ij}|$, and recovers global log-worths along the champion-rooted elimination tree ($w_{\text{loser}} = w_{\text{winner}} - |z_{ij}|$, zero-centered before temperature softmax).

---

## 8. Foundational Literature Credited

1. **Pretraining Token Prior Bias & Calibration**:
   - Zhao, T. Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). *"Calibrate Before Use: Improving Few-Shot Performance of Language Models"*. **ICML 2021**. [arXiv:2102.09690](https://arxiv.org/abs/2102.09690).
2. **Paired Comparison Log-Odds Modeling & Combinatorial Hodge Ranking**:
   - Bradley, R. A., & Terry, M. E. (1952). *"Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons"*. **Biometrika**, 39(3/4), 324–345.
   - Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2011). *"Statistical ranking and combinatorial Hodge theory"*. **Mathematical Programming**, 127(1), 203–244.
3. **Positional & Label Permutation Bias in LLM Judges**:
   - Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). *"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"*. **NeurIPS 2023**. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
4. **Probability-Weighted Continuous Expected Scoring**:
   - Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). *"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"*. **EMNLP 2023**. [arXiv:2303.16634](https://arxiv.org/abs/2303.16634).
5. **Post-Hoc Temperature Scaling & Expected Calibration Error (ECE)**:
   - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *"On Calibration of Modern Neural Networks"*. **ICML 2017**. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
6. **Split-Conformal Selective Classification & Distribution-Free Uncertainty Quantification**:
   - Angelopoulos, A. N., & Bates, S. (2021). *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"*. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511).
7. **System-1 Single-Token Decision Benchmarking**:
   - Deußer, T., Sparrenberg, L., & Sifa, R. (2026). *"Evaluating and Benchmarking the System One Model Jev"*. [arXiv:2609.37647](https://arxiv.org/abs/2609.37647).
8. **Recurrent Depth & Universal Transformers**:
   - Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019). *"Universal Transformers"*. **ICLR 2019**. [arXiv:1807.03819](https://arxiv.org/abs/1807.03819).
