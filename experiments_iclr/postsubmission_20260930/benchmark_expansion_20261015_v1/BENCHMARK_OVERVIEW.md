# What the proposed benchmark will test

We want several useful predictions from one shared GNN. Each route has a small
private component that can steer the shared model. The benchmark asks which
training rules and private components improve prediction quality, and whether
they can outperform four separately trained copies of the same native model.

The current proposal contains **6,460 training runs**. It is being implemented
and checked; the full campaign waits for your review and GO. The completed
24-run molecular pilot, completed three-run BN follow-up and running nine-fit
normalization/ingredient study are separate development studies. They do not replace any of the proposed benchmark runs.

## The comparisons made in every setting

A setting fixes the dataset, reference backbone and one documented source
configuration. For each of its ten trials we train:

1. **Reference reproduction:** one native model with the source paper/release
   training horizon and selection protocol. Its selected checkpoint must also
   be scored with the source test protocol to check the reported baseline.
   Final-scoring source now covers all task families. Its native numerical
   checks and job/report integration remain pending; validation results alone
   do not establish reproduction. Source-configuration ambiguities are disclosed.
2. **Matched native single and independent ensemble:** four separate native
   models, each with its own weights, optimizer and selected checkpoint. Model0
   is also the single-model comparison. Their averaged predictions form I4.
   We use the longer training controller applied to the shared methods.
3. **Original GNNM:** one shared native model, four private input/output factor
   routes, the original training loss and average prediction.
4. **Candidate shared models:** the explicit recipes below. Every candidate
   retains one shared native backbone and the same native inputs/operators.

The independent ensemble consumes four training runs. Averaging its saved
predictions consumes no additional training run. The original GNNM is already
included in the shared-method counts below.

## Datasets and backbones

| Task | Datasets | Source architectures/configurations | Settings | Shared recipes per setting | Fits |
| --- | --- | --- | ---: | ---: | ---: |
| Node classification | CiteSeer, PubMed, Coauthor-CS, Coauthor-Physics | PolyFormer, Mono and Cheb source configurations | 8 | 19 | 1,920 |
| Molecular graph classification | ogbg-molbace, ogbg-molhiv | GCN, GCN with virtual node, GIN, GIN with virtual node; dropout0 and0.5 | 16 | 19 | 3,840 |
| Link prediction | ogbl-collab | NCN, NCNC | 2 | 18 | 460 |
| Link prediction | ogbl-collab | BUDDY | 1 | 19 | 240 |

For example, a node setting has ten trials of one source reproduction, four
native bodies and19 shared recipes: **10 × (1 + 4 + 19) = 240 fits**. Eight
node settings give1,920. The same counting explains every row. There are seven
datasets, eight architecture families and27 source settings in total.

The source papers name both molecular dropout choices; the exact winning
choice for each published score is unclear. Both are fixed, disclosed settings.
We will not choose a winner after testing and call it exact paper reproduction.
There is no LR, weight-decay or other hyperparameter search in this proposal.

## What changes in the shared methods

| Idea | Concept | Required comparison |
| --- | --- | --- |
| Original GNNM | Four factor routes use one backbone and average their predictions | Native single and independent ensemble |
| Ensemble-aware training | Let private factors, or both private and shared weights, learn partly from the final averaged prediction | Original own-route loss; private-only versus all-parameter credit |
| Class geometry | Align embeddings of labelled examples in the same class; separately test a variance floor that preserves differences between distinct examples | No auxiliary; alignment alone; alignment with variance |
| Link contrast | Align the same query across routes and compare sampled alternatives, including alternatives sharing an endpoint | Same-query alignment; global alternatives; endpoint alternatives |
| Persistent responsibility | Give routes different positive training weights based on graph structure or molecular size | Original equal weighting; a matched randomized assignment |
| Factual context | Teach routes similarity targets based on different observed aspects of the graph | All routes learn the same target mixture versus routes assigned different targets |
| Private nonlinear correction | Add a small zero-start nonlinear correction for each route around the shared input representation | Original diagonal factors; factors plus correction; correction without factors |
| Learned aggregation | Learn global or prediction-dependent weights for the four predictions | Uniform average under the identical training loss; selected learned bank also scored with its fixed average |
| Private BN running state | Give each route its own running statistics while sharing all learned native affine weights | Original shared BN buffers; applies only where the native architecture already has BN |

The original node within-route supervised contrastive rule and fixed positive
bootstrap weighting are retained where they were explicitly defined. Molecular
and link objectives have their own meaningful task targets and reductions;
we do not silently substitute a node classification loss for link ranking.

Four combinations test whether useful components reinforce each other:
private correction plus geometry; structural responsibility plus ensemble-aware
private training; private correction plus uniformly pooled training; and private
correction plus prediction-dependent pooling. Their separate ingredient controls
are retained. This is a limited set of motivated combinations, not every possible
mixture of coefficients and methods.

There are25 exact recipe definitions across the tasks, including original GNNM
and controls. These are not25 new inventions. Each concrete definition, native
hyperparameter and applicability rule is in [the source roster](expanded_study_design_v1/PROPOSED_ROSTER.json)
and [native settings](expanded_study_design_v1/NATIVE_SETTINGS.json).

## Training, reliability and compute

The reference reproductions retain their source schedule. The matched models
use generous caps and moderate validation patience: PolyFormer minimum2,000,
cap10,000 and patience100; molecular GNNs and NCN/NCNC minimum100, cap1,000 and
patience50. BUDDY's exact source-compatible controller is separately recorded.
Patience starts after required source phases. Each model selects its own first
best native validation score. A cap reached while still improving is reported
as unresolved training, not a converged result.

We retain individual seed scores, paired differences, variation, convergence
curves, failure costs and unsuccessful methods. We also inspect route quality
and what averaging repairs or harms. A favorable seed or separated standard
deviations does not establish a reliable advantage. The broad matrix is
exploratory; the separate confirmation proposal requires a frozen candidate,
new paired trials and its own reviewed budget.

Once all jobs are implemented and resource checks pass, the final package will
provide one command per GPU, with up to two compatible jobs running on each.
The current authorization allows only the dedicated allocation. Deployment to
18.77 comes later. Measured full-epoch costs are required before promising a
completion time or dividing6,460 runs across three GPUs. Fit counts alone do
not measure GPU work.
