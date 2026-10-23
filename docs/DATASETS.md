# Datasets and task contracts

Supplementary Table IX is the source of the primary banks, floors, training
ceilings, deployment depths, and batch sizes. Every row is an independent
experiment directory. `python -m ropc list --json` prints the machine-readable
registry used by the loaders and launchers.

## Primary configurations

All rows use four coarse roles, 20 tangent action coordinates, horizon three,
three planner–amortizer rounds and a nominal sensing budget of 1,024 nodes under
the full preset. The cut is tie-inclusive, so it is not a hard sample-size cap.

| Dataset | Task / metric | L* / Lmax | Bank | rmin | Training batch |
|---|---|---|---|---|---|
| ogbn-arxiv | Node class / accuracy | 32 / 64 | P0, Pr0, Pg, Pf | 2 | Full graph |
| ogbn-products | Node class / accuracy | 16 / 64 | P0, Pg, Pf, Pc | 2 | 50k-node induced blocks |
| ogbn-proteins | Node multilabel / ROC-AUC | 16 / 64 | P0, Pg, Pf, Px | 4 | 50k-node induced blocks |
| Roman-empire | Node class / accuracy | 32 / 64 | P0, Pr0, Pg, Px | 2 | Full graph |
| Amazon-ratings | Node class / accuracy | 16 / 64 | P0, Pg, Px, Pc | 2 | Full graph |
| Minesweeper | Binary node / ROC-AUC | 16 / 64 | P0, Pg, Pf, Pc | 4 | Full graph |
| PascalVOC-SP | Node class / macro-F1 | 64 / 64 | P0, Pg, Pf, Pc | 4 | 32 graphs |
| COCO-SP | Node class / macro-F1 | 64 / 64 | P0, Pg, Pf, Pc | 4 | 32 graphs |
| Peptides-func | Graph multilabel / AP | 32 / 64 | P0, Pg, Pf, Px | 8 | 128 graphs |
| Peptides-struct | Graph regression / MAE | 16 / 64 | P0, Pg, Pf, Px | 8 | 128 graphs |
| PATTERN | Node class / balanced accuracy | 16 / 64 | P0, Pg, Pc, Px | 2 | 128 graphs |
| CLUSTER | Node class / balanced accuracy | 16 / 64 | P0, Pg, Pc, Px | 2 | 128 graphs |
| LRIM-16 | Node regression / log10-MSE | 32 / 32 | P0, Pg, Px, Pc | 8 | 32 graphs |
| LRIM-32 | Node regression / log10-MSE | 64 / 64 | P0, Pg, Px, Pc | 8 | 8 graphs |
| LRIM-64 | Node regression / log10-MSE | 96 / 128 | P0, Pg, Px, Pc | 8 | 1 graph |
| LRIM-128 | Node regression / log10-MSE | 128 / 128 | P0, Pg, Px, Pc | 8 | 1 graph |
| CIFAR10 | Graph class / accuracy | 32 / 64 | P0, Pg, Pf, Pc | 4 | 128 graphs |
| ZINC | Graph regression / MAE | 16 / 64 | P0, Pg, Pf, Px | 4 | 128 graphs |
| MNIST | Graph class / accuracy | 32 / 64 | P0, Pg, Pf, Pc | 4 | 128 graphs |
| MalNet-Tiny | Graph class / accuracy | 32 / 64 | P0, Pr0, Pg, Pc | 4 | 16 graphs |
| ogbg-molhiv | Binary graph / ROC-AUC | 16 / 64 | P0, Pg, Pf, Px | 4 | 128 graphs |
| PCQM-Contact | Link prediction / filtered MRR | 32 / 64 | P0, Pg, Pf, Pc | 4 | 128 graphs |
| ogbg-molpcba | Graph multilabel / AP | 16 / 64 | P0, Pg, Pf, Px | 4 | 256 graphs |
| ogbg-ppa | Graph class / accuracy | 32 / 64 | P0, Pg, Pf, Pc | 4 | 64 graphs |

PATTERN/CLUSTER use the benchmark's class-averaged recall, often called weighted
or balanced accuracy. The paper labels these scores as accuracy. The evaluator
aggregates the whole split before calculating this metric. Scores are stored as
fractions rather than percentages. MAE and log10-MSE are minimized; all other
metrics in the table are maximized.

## Official providers

| Datasets | Provider / API |
|---|---|
| OGB node datasets | [Open Graph Benchmark node tasks](https://ogb.stanford.edu/docs/nodeprop/) |
| OGB graph datasets | [Open Graph Benchmark graph tasks](https://ogb.stanford.edu/docs/graphprop/) |
| Heterophily datasets | [Official data](https://github.com/yandex-research/heterophilous-graphs), through PyG `HeterophilousGraphDataset` |
| Pascal, COCO, Peptides, PCQM-Contact | [Long Range Graph Benchmark](https://github.com/vijaydwivedi75/lrgb), through PyG `LRGBDataset` |
| PATTERN, CLUSTER, MNIST, CIFAR10 | [Benchmarking GNNs](https://github.com/graphdeeplearning/benchmarking-gnns), through PyG `GNNBenchmarkDataset` |
| ZINC | PyG `ZINC(subset=True)`, the standard 12k graph benchmark |
| MalNet-Tiny | PyG `MalNetTiny`, using its released train/validation/test split |
| LRIM | [Official LRIM repository](https://github.com/iJorl/lrim_graph_benchmark) and [released data](https://huggingface.co/datasets/jmathys/lrim_graph_benchmark) |

The LRIM filenames are `lrim_{side}_0.6_10k.pt`: the hard setting, 10,000 released
graphs per lattice size. The official sequential 80/10/10 split is retained.
Targets are node energy changes. The reported metric is
`log10(mean((prediction-target)**2))`; “absolute log10-MSE” means the actual score,
not its absolute value or a difference against a baseline.

LRIM-256 is registered only as a transfer target. It is not a 25th Table IX
training configuration. The paper's P5 additionally changes LRIM-16's training
ceiling to 64 and deploys on sides 32–256 at twice the side length. An ordinary
LRIM-16 run at its Table IX ceiling does not reproduce P5.
The [P5 protocol guide](../protocols/lrim-size-transfer/README.md) supplies the
separate source configuration and `ropc transfer` command.

## Input preprocessing

Node inputs remain floating point. Categorical and multilabel targets retain
their appropriate shape and dtype. Missing labels remain missing and are masked
in both loss and metric calculations.

The static encoder receives these label-free features:

- OGB node datasets retain their released features. Proteins derives node
  features by averaging incident edge features; species labels do not become
  input features.
- Molecules use fixed one-hot encodings of the nine OGB atom categories and
  three bond categories. ZINC uses its 28 atom and four bond categories.
- PATTERN/CLUSTER use fixed categorical vocabularies. MNIST/CIFAR10 append their
  released superpixel coordinates to pixel features.
- MalNet-Tiny uses a local degree profile: degree and the minimum, maximum, mean
  and population standard deviation of neighboring degrees.
- Available edge attributes are retained in `GraphData.edge_attr` and their
  encoded incoming means are appended to node inputs. The propagation bank is
  the paper's structural/role bank, not a learned dynamic edge-message block.
- Graph families append exact random-walk return probabilities for steps
  1 through 16. This explicit RWSE choice is cached as preprocessing. The
  supplement requests RWSE but does not supply every feature-encoder setting.
  Verified periodic LRIM lattices use translation symmetry to calculate the
  same return probabilities from one evolving walk. Other topologies use
  sparse matrix powers.

These feature adapters make all task types executable without silently discarding
edge information. They are explicit implementation conventions where the source
files did not include a family encoder. They do not establish numerical parity
with every baseline's released input encoder or the paper's approximate 500k
parameter comparison budget. Actual parameter counts and preprocessing cost must
accompany a comparison run.

The official LRIM input has one binary spin feature per node. It is not a source
of continuous features that guarantees a unique sensing hash. The implementation
preserves ties instead of adding node IDs or random noise to force the sample to
1,024 nodes. Record the realized sensing-set size when comparing this data
pipeline with the supplement's reported sampling counts.

## Readouts, losses and evaluation

Node tasks read out each node representation. Graph tasks mean-pool node
representations within each graph before the shared linear readout. Disjoint
batches never pool across different graphs. PCQM-Contact scores endpoint dot
products and trains on the release's labeled positive/negative edge candidates.

Classification uses cross-entropy; binary/multilabel tasks use masked binary
cross-entropy with logits. PATTERN, CLUSTER and superpixel segmentation use
class-frequency weighting. Regression uses MAE for MAE benchmarks and MSE for
LRIM. The controller's stage cost clips and rescales this task loss using a
training-fixed loss scale.

ROC-AUC and AP are calculated after collecting predictions over the complete
evaluation split. Columns without both label classes do not contribute, matching
the OGB evaluator convention; a completely undefined metric is `NaN`, not a
fabricated zero. Macro-F1 includes the classes present in the truth/predictions.

PCQM-Contact uses the released LRGB `mrr_filtered` convention: remove other known
positive destinations for the same source from the ranking candidates, preserve
self candidates, use the upstream argsort tie convention, compute MRR within each
graph, then average across graphs. `mrr_filtered_noself` is a distinct metric and
is not substituted silently. See the
[official edge head](https://github.com/vijaydwivedi75/lrgb/blob/main/graphgps/head/inductive_edge.py).

Predictive entropy is the sensor's uncertainty channel for class/binary tasks.
For regression, the label-free channel is the log of one plus the dispersion of
node predictions around their graph mean. This is a documented sensor convention,
not a calibrated predictive-variance model. Link tasks use mean edge-prediction
entropy at each source node.

## Supervision and audit partitions

The official validation and test splits are retained. They are not used to fit
task parameters, the observer, value heads, or execution envelopes.

For graph families, a deterministic seed-dependent permutation of **official
training graph IDs** assigns 15% to value/identification, 10% to tuning, 10%
(at most 2,000) to audit, and 8% (at most 1,600) to reserve audit. Remaining graphs
train the task route. These are disjoint whole-graph partitions. The full preset
uses the resulting full audit partitions; development presets may truncate them
and report that reduced budget. Calibration never creates new “independent”
samples by perturbing a shared family backbone.

For single graphs, 15% of official training labels are held out for value and
identification fitting. This split is fixed before subgraph sampling. Tuning,
audit and reserve perturbation seeds are disjoint, but all refer to one graph;
the corresponding audit is empirical. A heterophily split is selected by one
mask column, never by flattening all ten columns.

Products and proteins train on induced blocks containing at most 50,000 nodes.
The blocks follow a reverse-Cuthill–McKee ordering, with block order shuffled each
epoch. Local label indices are mapped from the fixed global task/value split.
The uploaded source does not specify the partitioner algorithm; the
reverse-Cuthill–McKee choice is an explicit implementation convention.
Validation and final test metrics use the original full graph.

## Data identity and caches

Each dataset has its own directory under `--data-root`. Official family files
are hashed with SHA-256 and combined with the split manifest and preprocessing
version. File digests are cached against file size and modification time and
recomputed when either changes. Single-graph identities cover canonical tensors
and supervision splits. The manifest records the identity scope.

Prepared family graphs carry their exact feature specification. Missing or
mismatched specifications are rejected with an instruction to select a fresh
data root, rather than silently reuse obsolete features. A per-dataset lock
serializes initial downloads and preprocessing when multiple Slurm seeds start
together.

Caches, checkpoints, logs and run outputs are excluded by `.gitignore`. The Git
repository contains code and configuration, not redistributed benchmark data.
