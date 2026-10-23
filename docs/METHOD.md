# Method and implementation contract

This document maps the software to **“Predicting Before Propagating:
Role-Conditioned Operator Predictive Control for Deep Graph Neural Networks”**
and its supplementary material. Equation, section, algorithm and table numbers
below refer to those two supplied documents. The code implements the ROPC
method; running it does not establish the numerical benchmark results reported
in the paper.

## 1. What is specified by the paper

The primary configuration in Supplementary Appendix A uses representation width
128, nonlinear width 128, 16 fine roles with support two, four coarse roles,
four transport operators, source bounds `[0.05, 0.40]`, self-loop mass one and
representation radius three. The sketch has width 16 and the tie-inclusive
sensor budget is 1,024 nodes. There are 25 observation channels and 20 tangent
action coordinates/latent state coordinates. The bilinear interaction rank is
four.

Planning uses a three-step horizon, three majorant updates and eight primal-dual
iterations per update. The action trust box has half-width 0.025, warm-start
damping is 0.5, and the smoothing, base-anchor, value and support weights are
0.02, 0.05, 0.3 and 0.5. The health weight is four. The target head smoothness
bound is four, the planning blend ramp is `1e-4`, and the execution blend ramp
is `1e-3`. The anytime value uses hazard `1/32`, discount `31/32`, and a
320-step continuation cap. The primary policy iteration has three rounds,
GRU width 48 and four DAgger rounds.

The full-size configurations retain these values. Smoke/quick presets reduce
budgets to verify the execution path. Their outputs are development checks,
not paper-budget benchmark results.

## 2. Component map

| Paper component | Implementation | Contract |
| --- | --- | --- |
| Feasible action polytope, Eq. (8), Appendix D Eq. (18) | `model.ActionGeometry`, `project_block` | Nonnegative role action, unit sum, source bounds, transport floor, optional trust/erosion box |
| Role likelihood and hierarchy, Section IV-A, Appendix E | `model.RoleEncoder` | Simplex fine memberships, row-stochastic hierarchy, bounded fine compatibility, within-graph structural likelihood |
| Fused sparse and global role operators, Eqs. (5), (7) | `model.UnionSupport`, `FusedSpMM`, `kernel_values`, `global_role_route`, `PropagationBlock` | Row-normalized sparse operators on one union support; global route remains separate and factorized |
| Representation invariance, Lemma 1 | `model.project_ball_rows`, `SpectralLinear`, `PropagationBlock` | Radius-limited encoder and numerically enforced spectral bounds on the tied nonlinear block |
| Reference schedule | `model.SplineSchedule` | Interpolation of feasible actions; two knots give a coordinate-wise monotone reference |
| Observation, Lemma 3, Appendix B | `model.equivariant_hash`, `build_sensed_set`, `Sensor`, `Normalizer` | Label-free tie-inclusive sensing, original baseline edge weights and frozen training normalization |
| Observer and transition, Eq. (9), Proposition 5 | `control.TransitionModel` | Corrected/prior states are distinct; explicit depth-zero null columns; projected model constraints |
| Anytime value and stage heads, Section V-C | `control.ConvexHead`; fitting in `training.py` | Convexity in observation at fixed context, norm-derived smoothness, detached true continuation targets |
| Interventions, Lemma 12, Eq. (20), excitation Eq. (21) | `control.RotatingDesign`, `slack_radii`, `eroded_bounds`, `pooled_response`; collection in `training.py` | Completed antithetic blocks, common coordinate radii, exact inequality erosion, source/residual structural mask only at depth zero |
| Health and support penalties | `control.HealthRegion`, `SupportDistance` | Frozen depth-conditioned covariance/center and action support metric |
| Learned-objective planner, Eqs. (11)–(13), Lemmas 5 and 10 | `control.Planner` | Fixed iteration budgets, a tangent convex majorant, bounded rollout remainder, continuous certificate blend |
| Reliability and execution, Eq. (14), Appendix B | `control.ReliabilityScorer`, `Envelope`, `margin_scaled_execution` | Frozen rank calibration, monotone finite envelope ramps and margin-scaled execution; explicit fallback for unsupported bins |
| Causal deployment, Algorithm 1 | `control.Controller` | Sense, correct, propose, plan, gate, then commit the executed action and that decision's recurrent memory |
| Planner/amortizer iteration, Algorithm 3 | `training.py`, staged `runner.py` | Detached branches, head/model fitting, DAgger, final freeze/refit, tuning, audit, evaluation |
| Task definitions and metrics | `registry.py`, `datasets.py`, `tasks.py` | Dataset-specific label shapes, graph/node/link readouts, missing-label masks and split-level metrics |

## 3. Actions, operators and state bounds

Each coarse role has action `(source, residual, transport_1, …, transport_M)`.
The tangent chart is the Helmert basis of the zero-sum subspace. Its first
coordinate is `(source − residual)/sqrt(2)`; there is one such coordinate per
coarse role. Because the initial current representation equals the source
representation, those columns are exactly zero in the depth-zero response.
Weak but nonzero responses at later depths receive excitation-dependent ridge
regularization and are not structurally deleted. This is the distinction made
in Supplementary Appendix E and Table XIX.

The projection retains the uploaded finite KKT enumeration. The small linear
systems are solved in float64, then converted to the caller's dtype. A bounded
dual bisection handles degenerate active sets. Input feasibility and final
constraint residuals are checked; an empty erosion or a failed projection
raises an error instead of executing an infeasible action. This numerical
fallback is a disclosed implementation choice; it is not the paper's idealized
pure active-set operation count.

Sparse row sums use local CSR segment reductions. Subtracting global float32
prefix sums is avoided because it loses small row masses on large graphs.
Duplicate input arcs are accumulated for operator construction. The structural
Bernoulli likelihood instead uses unique ordered distinct pairs, samples true
nonedges only, excludes pairs between different graphs in a batch, and weights
the two sample populations to estimate the complete within-graph objective.
The choice to exclude self-pairs from that likelihood is explicit; self-loop
mass in the transport operators remains one by default.

The supplied documents do not give complete separate equations for every
coarse/reverse candidate. This implementation uses the supplied code's
conventions: `Pr0` is the reversed baseline and `Px` is the reversed fine-gated
operator. For `Pc`, `Pi.T @ B @ Pi` is divided by the outer product of the
column masses of `Pi`. This makes it a weighted average of fine compatibility
entries and preserves their stated bounds. `Pg` is exactly the factorized
global route in Eq. (7), independently for every graph in a batch.

The nonlinear block's spectral normalization uses an SVD at optimizer-step
boundaries. Between updates, the bound
`||W_reference||_2 + ||W − W_reference||_F` remains valid, including after a
parameter perturbation. Forward passes do not mutate the normalization state.
A few power iterations alone are a lower estimate of the spectral norm and
would not establish the stated invariant-ball constraint.

## 4. Observation and task semantics

The five global channels are participation rank, source alignment, baseline
edge energy, log variance ratio and predictive uncertainty. Each coarse role
has log dispersion ratio, centroid separation, a weighted RV coefficient,
uncertainty and edge energy. The sensor uses only edges with **both endpoints
in the sensed set**, and retains their original `P0` row weights. It does not
renormalize an induced subgraph. This follows Supplementary Appendix B.

The label-free hash refines float32 feature bit patterns and in/out degrees
through two integer neighborhood-sum rounds. Integer reductions avoid
floating-point summation-order changes in the selected set. All ties at the
cutoff are included. Thus 1,024 is a nominal budget, not a hard cap, and the
actual selected count/overshoot must be reported.

Classification uncertainty is normalized categorical entropy. Binary and
multilabel tasks use Bernoulli entropy. Regression has no predictive variance
head specified in the paper: the task adapter uses the label-free log of one
plus the node readout's root-mean-square deviation from its graph mean. Link
uncertainty averages Bernoulli entropy of existing-edge scores at incident
source nodes. These last two definitions are implementation conventions, not
measurements or uniquely specified formulas from the paper. See `DATASETS.md`
for feature encodings, readouts, positional encodings and evaluator details.

**LRIM preprocessing boundary.** The public LRIM data contain binary spin
features on a periodic four-neighbor lattice and node energy-change targets.
The supplement describes continuous features and no sensor-cut ties for the
larger LRIM graphs. The supplied files do not identify the transformation that
would produce those continuous features. This repository preserves official
spin features and its documented positional encoding; it does not introduce
node identifiers or artificial noise to force an exact sensor count. Ties and
the resulting sensing cost can therefore differ from the paper's account.

The shared task adapter selects categorical cross-entropy, masked Bernoulli
loss, MAE or squared loss as appropriate. Graph targets use a graph readout;
node targets retain node indexing; PCQM-Contact uses pair scores. Regression
and multilabel targets retain their original shapes and floating dtype, and
missing entries are masked rather than converted to class identifiers.

## 5. Observer, value contexts and causal memory

The observer uses
`x = x_prior + K_o (z − C x_prior)` with the gain computed by a linear solve.
The constraints on the endpoint transition matrices, bilinear factors,
observation matrix and diagonal observation noise are enforced after updates.
The action bound is expanded across all role blocks before constraining the
bilinear action matrix. The model exposes the actual trained matrix norms and
observer contraction diagnostic.

The value context contains the depth feature, prior latent state, recurrent
base state and previous executed action. Value fitting uses the memory that
generated the recorded continuation. During optimization, the terminal
context follows the base's nominal rollout, and the one-step lookahead uses
the nominal memory after the base proposal. Both contexts stay fixed while
candidate actions vary, as prescribed in Section V-C and Appendix E. Their
mismatch with the memory produced by another executed action belongs to the
contrast error calibrated by the envelope.

Every `Controller.step` record contains its own next recurrent states and,
for ROPC, its full planned sequence. `commit` consumes that record. It does not
read a mutable pending tuple that may have been overwritten by an offline
branch or continuation. The prior for the next real observation is predicted
from the **executed** action. The base recurrence itself advances once at the
current observation, before that action is executed.

Masked roles are fixed to base actions in the planner's trust polytope and in
student gates. Replay accepts a complete per-graph action trace; actions from
the first graph are not broadcast to other graphs. Replay equality and backend
determinism are tests on the actual platform, not implied by selecting a CSR
operator.

## 6. What the planner certificate covers

The objective is Eq. (11): intermediate stage heads, the terminal value,
health penalties and action regularization. The health majorant is evaluated
in displacement coordinates as
`K_j (V − V_anchor) + Sigma_j^(-1/2) (Xi z_anchor_j − mu_j)`.
The anchor is subtracted exactly once. Consequently the health term and the
whole majorant agree with the objective at the incumbent.

Softplus curvature and the actual head weights bound the head Hessian and
gradient. The planner uses at least the actual head smoothness bound, even if
it is larger than the requested target. It does not silently assume that a
configured number is a certificate.

Rollout curvature is bounded over every feasible action sequence, hence over
the entire trust region. Starting from the actual corrected state, the code
propagates a state-norm bound using the affine context/bias terms and the
simplex action-norm bound. The first- and second-derivative bounds obey
`L_next <= a L + b` and `K_next <= a K + 2 c L`, where the coefficients come
from the current trained matrix norms. The observation curvature is
`beta_j = ||C||_2 K_j`, with `beta_1 = 0`. Empirical training maxima and a
multiplier on a nominal trajectory are not used as domain certificates.

The remainder coefficient follows Eq. (12). PDHG has the fixed configured
budget and is not assumed to converge. The candidate is blended with the
incumbent according to its decrease in the convex majorant, Eq. (13).
Regression tests check tangency, sampled majorization, feasible actions and
decrease of the learned objective with an active health hinge. These numerical
tests support the implementation; they do not replace the paper's assumptions
or prove task-loss improvement on the true graph network.

## 7. Calibration, audit scope and unsupported bins

Reliability uses pooled tied empirical-CDF knots with continuous interpolation
and continuous constant extension at the first/last knot. A single training
record does not produce an undefined standard deviation. Finite calibration
slopes give the piecewise-linear envelope in Appendix B, followed by the
margin-scaled execution rule in Eq. (14).

Insufficient support is represented by an infinite slope. That bin and any
part of its incoming ramp with positive weight force the base action. The
implementation avoids `0 * infinity` and `infinity − infinity`, so the
executed action, blend weight and recorded certificate remain finite. These
explicit disabled regions do not have a finite global Lipschitz envelope; a
finite-envelope continuity claim must not be made for such a policy. Saved
envelopes encode unsupported slopes as `null` plus an explicit Boolean mask,
so strict JSON files do not contain nonstandard `Infinity` tokens.

The paper's graph-level Clopper–Pearson statement requires independent audit
graphs, a policy frozen before the audit, and no policy selection using those
audit outcomes. Fresh reserve graphs are needed for the pre-registered
fallback policy. Multiple layers, branch actions, sensing seeds, perturbations
or subgraphs from the same transductive graph are not independent graph draws.
Single-graph results must therefore be labeled empirical trajectory/decision
diagnostics; they do not establish the paper's independent-graph guarantee.
All-bin, simultaneous and whole-trajectory events must retain their actual
units and denominators. A small run or an empty bin cannot substantiate the
reported coverage levels.

## 8. Dataset and experiment coverage

`registry.py` contains every primary row of Supplementary Table IX and supplies
the corresponding configuration and launcher. The table below records its
scientific settings; full task/feature provenance is in `DATASETS.md`.

| Dataset | Deploy / train ceiling | Transport bank | Floor | Batch |
| --- | --- | --- | ---: | --- |
| ogbn-arxiv | 32 / 64 | P0, Pr0, Pg, Pf | 2 | Full graph |
| ogbn-products | 16 / 64 | P0, Pg, Pf, Pc | 2 | 50k-node training subgraphs |
| ogbn-proteins | 16 / 64 | P0, Pg, Pf, Px | 4 | 50k-node training subgraphs |
| Roman-empire | 32 / 64 | P0, Pr0, Pg, Px | 2 | Full graph |
| Amazon-ratings | 16 / 64 | P0, Pg, Px, Pc | 2 | Full graph |
| Minesweeper | 16 / 64 | P0, Pg, Pf, Pc | 4 | Full graph |
| PascalVOC-SP | 64 / 64 | P0, Pg, Pf, Pc | 4 | 32 |
| COCO-SP | 64 / 64 | P0, Pg, Pf, Pc | 4 | 32 |
| Peptides-func | 32 / 64 | P0, Pg, Pf, Px | 8 | 128 |
| Peptides-struct | 16 / 64 | P0, Pg, Pf, Px | 8 | 128 |
| PATTERN | 16 / 64 | P0, Pg, Pc, Px | 2 | 128 |
| CLUSTER | 16 / 64 | P0, Pg, Pc, Px | 2 | 128 |
| LRIM-16 | 32 / 32 | P0, Pg, Px, Pc | 8 | 32 |
| LRIM-32 | 64 / 64 | P0, Pg, Px, Pc | 8 | 8 |
| LRIM-64 | 96 / 128 | P0, Pg, Px, Pc | 8 | 1 |
| LRIM-128 | 128 / 128 | P0, Pg, Px, Pc | 8 | 1 |
| CIFAR10 | 32 / 64 | P0, Pg, Pf, Pc | 4 | 128 |
| ZINC | 16 / 64 | P0, Pg, Pf, Px | 4 | 128 |
| MNIST | 32 / 64 | P0, Pg, Pf, Pc | 4 | 128 |
| MalNet-Tiny | 32 / 64 | P0, Pr0, Pg, Pc | 4 | 16 |
| ogbg-molhiv | 16 / 64 | P0, Pg, Pf, Px | 4 | 128 |
| PCQM-Contact | 32 / 64 | P0, Pg, Pf, Pc | 4 | 128 |
| ogbg-molpcba | 16 / 64 | P0, Pg, Pf, Px | 4 | 256 |
| ogbg-ppa | 32 / 64 | P0, Pg, Pf, Pc | 4 | 64 |

LRIM-256 is additionally a **transfer target** in protocol P5, not a 25th
independently trained Table IX configuration. P5 trains on LRIM-16 with random
depth up to 64 and deploys at twice the lattice side, including depth 512 for
side 256. It must not be confused with the primary LRIM-16 ceiling of 32.

The supplied paper also specifies matched comparator searches, ten paired
runs, prospective family rules, depth exposure, true-model planning arms,
student-capacity/budget sweeps, timing measurements and statistical reductions.
Dataset support is distinct from completing that entire experiment matrix.
Reported paper numbers, the paper's large random-majorant test counts, GPU
latencies and independent-graph audit outcomes are not inserted into generated
results. Only results produced by the selected commands and checkpoint are
reported as measurements.

## 9. Verification boundary

`tests/test_model_control.py` checks projection feasibility and its variational
inequality across all paper floors, eroded-box vertices, spectral bounds,
large-prefix row-sum cancellation, structural likelihood populations,
node/edge-order hash behavior, sensor uncertainty, reference monotonicity,
depth-zero Jacobians, observer contraction, convex-head Hessians, planner
majorants and learned-objective descent, strict-JSON envelope round trips,
reliability continuity, causal branch commits and per-graph replay.

Dataset tests and end-to-end synthetic runs exercise task adapters, split
handling and stage transitions. A successful smoke run is not a completed
training run on the downloaded benchmarks. Full training, server execution,
GPU determinism/performance and scientific reproduction require the actual
environment, data, budgets and saved seed-level records.

### OGB cache compatibility

`ogb_compat.py` preserves OGB's metadata, raw readers, graph processor and
official splits. It supplies an explicit `weights_only=False` only for named
official OGB processed/split caches, whose NumPy/PyG objects are not compatible
with the newer restricted default. It does not replace `torch.load` globally
or change the installed PyTorch build. These paths are for trusted official
dataset caches.

Explicit prepare/run commands download noninteractively into a staging
directory. A missing release marker on an existing dataset is reported as a
stale/incomplete cache; existing data are not deleted. Offline tests exercise
node and PyG cache loading, saved/CSV splits, legacy `_pyg` directories and
the stale-release failure. This compatibility path has been tested with the
available CPU runtime, not by claiming an unperformed run on the target
server's PyTorch version.
