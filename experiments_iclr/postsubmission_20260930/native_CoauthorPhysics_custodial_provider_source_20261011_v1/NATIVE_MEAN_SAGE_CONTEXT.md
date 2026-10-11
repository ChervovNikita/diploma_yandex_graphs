# Published mean GraphSAGE context on Physics

Primary paper: *Pitfalls of Graph Neural Network Evaluation*, arXiv1811.05868, Table1: Physics GS-mean93.0±0.8% test accuracy, compared with GCN92.8±1.0. These are literature values averaged over100 splits and20 initializations per split. No project outcome was read here.

Pinned author commit `1e72912a0810cdf27ae54fd589a3b43358a2b161`. Exact model/config/base/train/metrics/early-stopping/preprocess files are preserved in `pinned_sources/`. Optimized meanSAGE config: depth2, hidden32, dropout.4, Adam lr.001, no schedule. `config/train.conf.yaml` sets raw feature normalization false, max100000 epochs, KDD criterion, tolerance50. `normalize_features=false` is retained; the optional helper actually uses L1 row normalization and is inactive.

For each layer, drop its input, then form `z = mean_(A+I)(h W_agg) + h W_skip + b`; divide each row by `max(L2norm(z),1)`; apply ReLU only after layer1. Layer2 logits are also row-normalized. The same dropped input feeds aggregate and skip projections. The author explicitly sets one diagonal adjacency entry per node. This is distinct from PyG provider loop removal: provider495924 no-loop entries; common model530417 including34493 loops, as root separately froze.

Own-label mean cross entropy plus TFslim L2 is minimized. Only first-layer aggregate and skip matrices are regularized: `.1*tf.nn.l2_loss = .05*sum(weights²)`. Biases and second-layer weights have no L2. Weights use Glorot, biases start at zero. Optimizer is Adam, not AdamW.

The paper's prose calls its early stopping strict. Actual `KDDCriterion.should_stop` compares `val_loss <= val_loss_min`, resetting patience and replacing the checkpoint on ties. It selects total SELECT CE+L2, uses50 consecutive worse epochs, and restores trainable variables before final TEST evaluation. The native driver evaluates TEST only after training. Primary metric `metrics.accuracy` is sklearn `accuracy_score`; macro F1 is secondary.

Root's ten masks preserve20TRAIN and30selector perclass and add30CAL perclass. They are custom prospective masks on one graph, not official canonical masks or ten independent graph populations. Native paper restTEST size therefore differs by150 reservedCAL nodes. Current core ownCE/h128/LN/.2/AdamW.001/earliest strict accuracy is a separately frozen adapted method recipe, not this exact published native recipe.
