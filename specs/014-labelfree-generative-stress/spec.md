# Feature Specification: labelfree-generative-stress

**Feature Branch**: `014-labelfree-generative-stress`

**Created**: 2026-09-14

**Status**: Draft — clarified

**Input**: User description: "Feature: labelfree-generative-stress. This is research/label-free-study brief 06 … Implement research/labelfree/stress.py: (1) a stationary block-bootstrap path generator, (2) adapters that sample coherent sequences from each existing GAN type (report which cannot), (3) the Tail-GAN calibration of each generator against the incumbent NNNC_MLX's pooled-WF P&L distribution, (4) a stress report for a candidate policy. Sample offline into a bank; never sample inside a training loop." Plus user scope constraints of 2026-09-14: multi-task GAN variants are out of scope; TabDDPM/WGAN/CTAB-GAN+ were recently rebuilt and their current implementations are authoritative.

## Context

This brief is **infrastructure, not an alpha source**. The literature is unambiguous that a
generator cannot add information the ~1.5 M real bars lack: agents trained inside learned
market simulators learn the simulator's artefacts (Coletta 2023), the most faithful crypto
generator had negative CAGR (CTBench 2025), recursive self-training collapses tails (Shumailov
2024), and return-conditioned diffusion planners copy luck (Paster 2022). So **no policy is
trained on synthetic data here.**

The question is the one real data cannot answer: *how does a policy behave on plausible paths it
has never seen?* — the robustness veto every other brief assumes exists.

### The measurement taken before this spec was written

The brief flagged one pivotal uncertainty: whether the repo's generators emit coherent
*sequences* or independent rows. It is settled, and it reshapes the feature.

| generator | `generate()` returns | verdict |
|---|---|---|
| TabDDPM | `(n, 1, num_features)` — *"the trailing seq axis exists so `balance_single_task`'s `_SQUEEZE_SEQ_DIM_TYPES` path can squeeze it"* | rows |
| WGAN | `(n, 1, F)` — "3D with seq_len=1" | rows |
| CTAB-GAN+ | `pd.DataFrame` | rows |
| **CGAN** | `(n, seq_len, F)` | **sequences** |

**None of the three rebuilt generators can emit a coherent sequence.** They are class-conditional
row samplers built for class balancing, which is what they were needed for and is not a defect.
But a row sampler cannot produce a plausible price *path*, and the policy under test is
path-dependent — stops, grace windows, holding limits and take-profit all depend on the order
bars arrive in. Shuffled rows would exercise none of it.

The brief anticipated this exact contingency: *"if they do not generate coherent sequences, use a
block bootstrap of real sequences or a GARCH-t path model as the generator, which is the required
baseline anyway."*

### What the study has already established

| brief | verdict |
|---|---|
| 01 direct policy optimisation | NOT RESOLVED |
| 02 label warm-start | NOT RESOLVED — 92% of the advantage was turnover |
| 03 meta-labelling | KILL — nested AUC 0.4978 |
| 04 RL in a pooled env | KILL — pooled reward IQM −0.1301 |
| 05 gradient-free policy search | KILL — PBO 0.7128 over 48,015 evaluations |

**No brief produced an adoptable candidate**, which bears directly on what this harness has to
stress — see Q2.

Separately, `docs/GAN_SCORECARD.md` records that GAN augmentation does not help P&L at any
fidelity (8-arm rebuild, no GAN beat baseline). That is a settled finding about *training*
augmentation and is not re-litigated here; this feature asks only whether a generator can
reproduce a P&L distribution well enough to be a **veto instrument**.

## Clarifications

### Session 2026-09-14

- Q: The starter prompt inherits protocol §5's gates, but §5 gate 1 is "pooled-WF annualised
  return ≥ control + 3.35pp/yr" — a POLICY-performance gate, and this feature delivers a tool
  with no return. How does §5 apply? → A: **§5 governs the policies the harness stresses, not the
  harness.** The harness's own acceptance is the brief's Tail-GAN calibration (SC-001), exactly
  as the brief states ("its 'acceptance' is that at least one generator passes the Tail-GAN
  calibration on the incumbent"). §5 is recorded as inherited-but-not-applicable-to-the-tool,
  and applies unchanged to any policy this harness later vetoes.
- Q: The brief depends on "a candidate policy from brief 1/2/3 to stress", but no brief produced
  an adoptable candidate — 01 and 02 NOT RESOLVED, 03/04/05 KILL. What gets stressed? → A: **The
  incumbent `NNNC_MLX`.** It is the only deployed policy, it is already the calibration
  reference, and a robustness verdict on production is genuinely informative. Stressing a killed
  candidate would be a robustness verdict on noise.
- Q: None of the three rebuilt generators can emit coherent sequences, and CGAN is the only repo
  type that can. Build CGAN adapters? → A: **No — the block bootstrap is the ONLY generator.**
  CGAN has been left behind by the others and is not maintained, so probing it would measure a
  stale implementation. **No GAN adapter is built.** The sequence-capability finding is still
  reported (US3) because it is a durable fact about the repo's tooling and is the reason for the
  pivot, but no learned generator is carried.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A generator whose paths are worth trusting (Priority: P1) 🎯 MVP

A researcher needs at least one path generator whose synthetic paths reproduce the **incumbent
policy's own P&L distribution** — because a generator that cannot reproduce the P&L of a policy
we already understand tells us nothing about a policy we do not.

**Why this priority**: every later story samples from it. This is the Tail-GAN criterion, and it
is the feature's only real acceptance condition.

**Independent Test**: sample paths, run the incumbent through them, and compare the resulting
P&L distribution, tail risk and drawdown against **the same policy scored by the same evaluator
on the real panels** — the evaluator held constant so only the paths vary.

**Acceptance Scenarios**:

1. **Given** a generator, **When** the incumbent is run on its sampled paths, **Then** the
   distance between synthetic and real P&L distributions is reported as a number, not asserted.
2. **Given** a generator that fails calibration, **When** results are reported, **Then** it is
   dropped from the harness and the failure is recorded as a finding about that generator.
3. **Given** any generator, **When** it is sampled, **Then** sampling happens **offline into a
   fixed bank of paths**, never inside a training or evaluation loop.

---

### User Story 2 - A path generator that does not need a GAN (Priority: P2)

A researcher has a stationary block bootstrap over real feature/return sequences, which
preserves the local autocorrelation and volatility clustering a path-dependent policy reacts to.

**Why this priority**: per clarification Q3 this is the **only** generator. The three rebuilt
GANs cannot emit sequences, and CGAN — the one type that can — is unmaintained, so probing it
would measure a stale implementation rather than the repo's live tooling.

**Independent Test**: bootstrap paths reproduce the real series' autocorrelation and volatility
clustering within a stated tolerance, and the incumbent's P&L on them passes calibration.

**Acceptance Scenarios**:

1. **Given** real sequences, **When** paths are sampled with a block length of 1–3 days,
   **Then** each path preserves within-block ordering and the block length is recorded.
2. **Given** the bootstrap passes calibration, **When** the harness is assembled, **Then** it is
   the sole generator and no learned generator is carried.

---

### User Story 3 - An honest account of what the repo's generators can do (Priority: P3)

A researcher can state, per generator type, whether it can produce coherent sequences at all,
verified against the running code rather than its documentation.

**Why this priority**: this is a durable finding about the repo's own tooling. It also prevents
the next reader from re-deriving it, and it is the reason the design pivots to the bootstrap.

**Acceptance Scenarios**:

1. **Given** each in-scope generator, **When** its output shape is probed, **Then** the result
   records sequence capability and the evidence for it.
2. **Given** a generator that emits rows only, **When** it is considered for stress use, **Then**
   it is excluded with the reason stated, never silently adapted by shuffling rows.

---

### User Story 4 - A stress verdict on a policy (Priority: P4)

A researcher runs a policy across ≥1,000 sampled paths and receives a distribution, not a point
estimate: P&L quantiles, drawdown distribution, and the fraction of paths that lose money — with
the incumbent's figures beside them for reference.

**Acceptance Scenarios**:

1. **Given** a policy, **When** it is stressed, **Then** the report carries quantiles and the
   loss-path fraction for both the policy and the incumbent.
2. **Given** a stress result, **When** it is reported, **Then** it is labelled a **veto, not a
   gate** — no policy may be promoted on stress results alone.

### Edge Cases

- **A generator produces paths the policy never trades.** Zero trades is not a stress result; it
  is an inadmissible path bank, and must be reported as such rather than scored as flat.
- **The bootstrap is indistinguishable from real data.** That is a pass for the harness and
  simultaneously evidence that the learned generators are unnecessary.
- **A generator reproduces the body but not the tail.** Tail behaviour is the entire point of a
  stress harness; body-only agreement fails calibration.
- **Sampling is slower than the stress run it serves.** Diffusion sampling is the known
  bottleneck; the bank must be reusable across candidates or the harness cannot be run routinely.
- **A path bank is reused after the feature set changes.** The bank must record what it was
  generated from, or a stale bank silently stresses the wrong thing.

## Requirements *(mandatory)*

### Functional Requirements

**Generation**

- **FR-001**: Paths MUST be sampled **offline into a persisted bank**, never inside a training or
  evaluation loop.
- **FR-002**: A path bank MUST record its generator, parameters, source data range and feature
  set, so a stale bank is detectable.
- **FR-003**: The block bootstrap MUST preserve within-block ordering, record its block length,
  and apply **one shared block index across all pairs** so cross-pair co-movement survives
  resampling.
- **FR-003a**: The block length MUST be **swept**, and calibration reported per value.
- **FR-004**: Each in-scope generator's ability to emit coherent sequences MUST be determined
  from its **implementation's return contract in the running code, cited by `file:line`**, and
  recorded. No GAN model is instantiated or loaded, so the census cannot be broken by a missing
  or untrained checkpoint.
- **FR-005**: A generator that cannot emit coherent sequences MUST be excluded with its reason
  stated, and MUST NOT be adapted by shuffling or concatenating independent rows.

**Calibration**

- **FR-006**: The generator MUST be calibrated by running the **incumbent** policy on its paths
  and comparing the P&L distribution against **the same policy scored by the same evaluator on
  the real panels**. The evaluator is held constant so the comparison varies only the paths; the
  archived freqtrade trades are a **secondary sanity reference**, reported but never gating.
- **FR-007**: Calibration MUST report distributional distance, a tail-risk measure and maximum
  drawdown as numbers.
- **FR-008**: A generator failing calibration MUST be dropped from the harness, and the failure
  recorded as a finding about that generator.
- **FR-009**: No learned generator is built, adapted or carried (clarification Q3). The
  sequence-capability finding is reported as a fact about the repo's tooling, not as a
  shortlist.

**Stress**

- **FR-010**: A stress run MUST use ≥1,000 sampled paths.
- **FR-011**: Stress output MUST be a distribution — P&L quantiles, drawdown distribution and the
  fraction of paths with negative net return — never a point estimate.
- **FR-012**: The policy stressed is the **incumbent `NNNC_MLX`** (clarification Q2). Where a
  future candidate is stressed, the incumbent's figures MUST appear beside it.
- **FR-013**: A path bank on which the policy takes no trades MUST be reported inadmissible, and
  the check MUST route through the study's shared `admissibility` gate rather than a local
  reimplementation — that module exists because the check kept being rewritten per site and each
  rewrite guarded only one side.
- **FR-014**: Stress results MUST be labelled a **veto, not a gate**; no policy may be promoted
  on them alone.

**Scope**

- **FR-015**: The multi-task GAN variants (MT_DDPM, MT_WGAN, MT_CTAB_GAN) are **out of scope**.
- **FR-016**: The rebuilt TabDDPM, WGAN and CTAB-GAN+ implementations are authoritative; any
  figure carried over from `docs/GAN_SCORECARD.md` MUST be re-verified against them or cited as
  historical.
- **FR-017**: No policy may be trained on synthetic data by this feature.

**Process**

- **FR-018**: All runs MUST execute sequentially.
- **FR-019**: Only files this feature creates or modifies MAY be staged.
- **FR-020**: A result is admissible only with a specification, plan, task list and a clean
  analyze report produced before implementation and again after.
- **FR-021**: The result report MUST be written even when every generator fails.

### Key Entities

- **Path bank**: a persisted set of sampled paths with the provenance needed to detect staleness.
- **Generator**: anything that produces paths — the block bootstrap or a learned model.
- **Calibration**: the comparison of a generator's incumbent-P&L distribution against real.
- **Stress report**: the distributional verdict on one policy across one path bank.
- **Incumbent**: the production `NNNC_MLX` policy, both the calibration reference and the
  comparison baseline.

## Success Criteria *(mandatory)*

### Harness acceptance (the brief's own gate)

- **SC-001**: The block bootstrap reproduces the incumbent's pooled-WF P&L distribution within a
  **pre-registered** tolerance on distributional distance, tail risk and maximum drawdown. It is
  the only generator (Q3), so this gate carries the feature alone.
- **SC-002**: The calibration tolerance is fixed in the design document **before** the
  calibration is run, not chosen after seeing the distance.
- **SC-003**: Sequence capability is reported for every in-scope generator, each claim carrying a
  `file:line` citation into the running code.
- **SC-004**: Every path bank is reproducible from its recorded provenance.
- **SC-005**: Sampling happens offline; no sampling occurs inside a training or evaluation loop.
- **SC-006**: A stress run covers **≥1,000 paths, where one path is a complete resampled panel
  across all pairs** — not 1,000 per pair — and reports a distribution: P&L quantiles, drawdown
  distribution, loss-path fraction.
- **SC-006a**: Calibration is reported **per block length** across the swept range, never for a
  single chosen value.

### Kill gates

- **SC-007**: If the block bootstrap fails calibration, the feature is **killed** and no harness
  is built on an uncalibrated generator. With no learned generator in scope there is no fallback,
  which is why SC-001 is the feature's single load-bearing gate.
- **SC-008**: The inability of TabDDPM, WGAN and CTAB-GAN+ to emit coherent sequences is reported
  as a finding about the repo's tooling, whatever the calibration outcome.
- **SC-009**: Stress results are never sufficient to promote a policy — they may only veto.

### Process gates

- **SC-010**: The feature directory holds a specification, plan, task list and a clean analyze
  report produced before implementation and again after.
- **SC-011**: The result report is written whatever the outcome, per protocol §6.
- **SC-012**: Multi-task variants appear nowhere in the results except as an explicit
  out-of-scope note.
- **SC-013**: Protocol §5's gates are recorded as applying to policies this harness vetoes, not
  to the harness itself (clarification Q1).

## Assumptions

- The incumbent `NNNC_MLX` is the calibration reference because it is the only policy actually
  deployed, and its real pooled-WF P&L distribution (378 trades, 15 folds) already exists.
- Brief 05's exit surrogate (`research/labelfree/es/exits.py`, calibrated at Spearman +0.9766
  against real backtests) is the natural policy evaluator for stress runs, since it is fast
  enough for ≥1,000 paths and its fidelity is already measured.
- "Plausible paths" means paths preserving the local autocorrelation and volatility clustering of
  real 15-minute crypto bars; it does not mean paths drawn from the true data-generating process,
  which is unknown.

## Non-Goals

- Training any policy on synthetic data.
- Re-litigating whether GAN augmentation helps P&L — settled by `docs/GAN_SCORECARD.md`.
- The multi-task GAN variants.
- Building or probing any learned generator, CGAN included (clarification Q3).
- The adjacent approaches listed in `briefs/09-excluded-and-why.md`.
- Improving any generator's fidelity. This feature measures generators; it does not fix them.
