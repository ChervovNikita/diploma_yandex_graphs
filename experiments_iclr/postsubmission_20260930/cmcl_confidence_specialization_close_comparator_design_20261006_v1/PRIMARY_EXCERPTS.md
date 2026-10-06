# Bounded primary details added to the saved CMCL method

Main PDF: https://proceedings.mlr.press/v70/lee17b/lee17b.pdf (same SHA5dcc65 as saved method).
Supplement: https://proceedings.mlr.press/v70/lee17b/lee17b-supp.pdf (SHA6fdf83).
Author code: paper page6 footnote authenticates https://github.com/chhwang/cmcl; inspected immutable commit57f41f4b166b8544c56580f9d32ee98b997e3f59 (2018-08-04). Exact blob pins are in SOURCE_BINDINGS.json. Current code is not certified as the original2017 reported experiment revision.

## Published missing details, transcribed

Page6 section5.1: top-1 is calculated by averaging output probabilities from all models and choosing the largest class; oracle error means none of the members predicts correctly. Classification uses global contrast normalization/ZCA whitening and no data augmentation. K-overlap changes sum_m v_i,m=1 toK. Page7 large-model setup lists beta{.5,.75,1,1.25,1.5} andK{2,3,4}; exact per-model winning values are not supplied in the scoped paragraphs.

Supplement appendixA: smallCNN LR.01/momentum.9/Nesterov/weightdecay.0005/dampening0/batch64/200epochs, LRtimes.2 at60/120/160. ResNet20 batch128/weightdecay.0001/momentum.9/LR.1, times.1 at82/123,164epochs. VGG17/GoogLeNet18 batch128/weightdecay.0005/Nesterov.9/LR.1 times.2 at25/50/75,100epochs. These image schedules are provenance, not graph defaults.

## Author-code selected excerpts

### README.md

```text
# Confident Multiple Choice Learning

This code is for the paper "Confident Multiple Choice Learning".

## Preliminaries

It is tested under Ubuntu Linux 16.04.1 and Python 2.7 environment, and requries following Python packages to be installed:

* [TensorFlow](https://github.com/tensorflow/tensorflow): version 1.0.0 or above. Only GPU version is available.
* [Torchfile](https://github.com/bshillingford/python-torchfile): version 0.0.2 or above.

*Simple torchfile installation:*

    pip install torchfile

## Dataset 

We provide the following datasets in torch format:

* CIFAR-10 whitened: [pre-processed data (1.37GB)](https://www.ndsl.kaist.edu/~changho/cmcl/dataset/cifar10_whitened.t7)
* SVHN (excluding the extra dataset): [pre-processed data (2.27GB)](https://www.ndsl.kaist.edu/~changho/cmcl/dataset/svhn_preprocessed.t7)

## Example scripts

* [`run_CMCL.sh`](run_CMCL.sh): train the models using "Confident multiple choice learning".
* [`run_MCL.sh`](run_MCL.sh): train the models using "Multiple choice learning".
* [`run_IE.sh`](run_IE.sh): train the models using "Independent ensemble".

## All training options:

    python src/ensemble.py \
    --dataset=cifar \
    --model_type=resnet \
    --batch_size=128 \
    --num_model=5 \
    --loss_type=cmcl_v0 \
    --k=4 \
    --beta=0.75 \
    --feature_sharing=True \
    --test=False

* `dataset`         : supports `cifar` and `svhn`.
* `model_type`      : supports `vggnet`, `googlenet`, and `resnet`.
* `batch_size`      : we use batch size 128.
* `num_model`       : number of models to ensemble.
* `loss_type`       : supports `independent`, `mcl`, `cmcl_v0`, and `cmcl_v1`.
* `k`               : overlap parameter.
* `beta`            : penalty parameter.
* `feature_sharing` : use feature sharing if `True`.
* `test`            : if `True`, test the result of previous training, otherwise run a new training.
```

### run_CMCL.sh

```text
###### Example script to run the code ######
#!/bin/sh

# dataset:    Supports cifar, svhn.
# model_type: Supports vggnet, googlenet, resnet.
# batch_size: We use batch size 128.
# num_model:  # of models to ensemble.
# loss_type:  Supports independent, mcl, cmcl_v0, cmcl_v1.
# k:          Overlap parameter.
# beta:       Penalty parameter.
# feature_sharing: Use feature sharing if True.
COMMAND="python src/ensemble.py \
--dataset=cifar \
--model_type=resnet \
--batch_size=128 \
--num_model=5 \
--loss_type=cmcl_v0 \
--k=4 \
--gpu=$1 \
--beta=$2 \
--feature_sharing=True"
COMMAND_TRAIN="$COMMAND --test=False"
COMMAND_TEST="$COMMAND --test=True"

MODEL="CMCL_run-gpu-$1-beta-$2"

# run train
echo $COMMAND_TRAIN
$COMMAND_TRAIN > "log_$MODEL"

# run test
echo $COMMAND_TEST
$COMMAND_TEST >> "log_$MODEL"
```

### src/ensemble.py

```text
12: tf.app.flags.DEFINE_string('data_dir', './dataset', 'Directoty to store input dataset')
13: tf.app.flags.DEFINE_string('dataset', 'cifar', 'Supported: cifar, svhn')
14: tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of images to process in a batch.')
15: tf.app.flags.DEFINE_string('model_type', 'resnet', 'Supported: vggnet, googlenet, resnet')
16: tf.app.flags.DEFINE_integer('num_model', 5, 'How many models to ensemble.')
17: tf.app.flags.DEFINE_string('loss_type', 'cmcl_v1', 'Supported: independent, mcl, cmcl_v0, cmcl_v1')
18: tf.app.flags.DEFINE_integer('k', 4, 'Overlap parameter')
19: tf.app.flags.DEFINE_integer('gpu', 0, 'GPU to use')
20: tf.app.flags.DEFINE_float('beta', 0.75, '')
21: tf.app.flags.DEFINE_boolean('feature_sharing', True, 'Use feature sharing if True.')
22: tf.app.flags.DEFINE_boolean('test', True, 'Run test if True else run train')
```

### src/model.py

```text
113:         elif FLAGS.loss_type == 'cmcl_v0':
114:             # CMCL version 0: confident oracle loss with exact gradient
115:             a = FLAGS.beta
116:             softmax_list = [tf.clip_by_value(tf.nn.softmax(logits),1e-10, 1.0) for logits in logits_list]
117:             entropy_list = [-tf.log(num_class+0.)-tf.reduce_mean(tf.log(softmax),1) for softmax in softmax_list]
118:             loss_list = []
119:             for m in range(FLAGS.num_model):
120:                 loss_list.append(closs_list[m] + a*tf.add_n(entropy_list[:m]+entropy_list[m+1:]))
121:             with tf.device('/cpu:0'):
122:                 temp, min_index = tf.nn.top_k(-tf.transpose(loss_list), FLAGS.k)
123:             min_index = tf.transpose(min_index)
124: 
125:             new_loss = 0
126:             for m in range(FLAGS.num_model):
127:                 for topk in range(FLAGS.k):
128:                     condition = tf.equal(min_index[topk], m)
129:                     new_loss += tf.reduce_sum(tf.where(condition, closs_list[m] - a*entropy_list[m], tf.zeros(closs_list[0].get_shape())))
130:             new_loss += tf.reduce_sum(a*tf.add_n(entropy_list))
131:             total_loss += new_loss/FLAGS.batch_size
```
```text
184:         # backward pass
185:         var_grads = model.OPTIMIZER(lr).compute_gradients(total_loss)
186:         apply_grads = model.OPTIMIZER(lr).apply_gradients(var_grads, global_step=global_step)
```
```text
188:         # softmax results
189:         softmax_list = [tf.nn.softmax(logits) for logits in logits_list]
190: 
191:         # prediction results
192:         pred_list = [tf.cast(tf.argmax(softmax, 1), tf.int32) for softmax in softmax_list]
193: 
194:         # comparison is 1 if prediction equals to label, else 0.
195:         comp_list = [tf.cast(tf.equal(labels, pred), tf.float32) for pred in pred_list]
196: 
197:         # error rate results of models
198:         err_list = [100.*(1.-tf.reduce_mean(comp)) for comp in comp_list]
199: 
200:         # ensemble top-1 error rate
201:         pred_top1 = tf.cast(tf.argmax(tf.add_n(softmax_list), 1), tf.int32)
202:         comp_top1 = tf.cast(tf.equal(labels, pred_top1), tf.float32)
203:         top1_err = 100.*(1.-tf.reduce_mean(comp_top1))
204: 
205:         # oracle error rate
206:         comp_oracle = tf.minimum(tf.add_n(comp_list), 1.)
207:         oracle_err = 100.*(1.-tf.reduce_mean(comp_oracle))
```

### src/feature_sharing.py

```text
6: def feature_sharing(features):
7:     """Feature sharing operation.
8:     Args:
9:       features: List of hidden features from models.
10:     """
11:     nmodel = len(features)
12:     with tf.variable_scope('feature_sharing'):
13:         shape = features[0].get_shape()
14:         output = [0.]*nmodel
15:         for from_idx in range(nmodel):
16:             for to_idx in range(nmodel):
17:                 if from_idx == to_idx:
18:                     # don't drop hidden features within a model.
19:                     mask = 1.
20:                 else:
21:                     # randomly drop features to share with another model.
22:                     mask = tf.floor(0.7 + tf.random_uniform(shape))
23:                 output[to_idx] += mask * features[from_idx]
24:         return output
```

### src/resnet.py

```text
10: if FLAGS.dataset == 'cifar':
11:     MAX_STEPS = 64000
12:     VAR_LIST = [0.1, 0.01, 0.001]
13:     PIVOT_LIST = [0, 32000, 48000]
14:     WD_FACTOR = 0.0001
15: elif FLAGS.dataset == 'svhn':
16:     MAX_STEPS = 93120
17:     VAR_LIST = [0.1, 0.01, 0.001]
18:     PIVOT_LIST = [0, 46560, 69840]
19:     WD_FACTOR = 0.0001
20: else:
21:     raise ValueError('Not supported dataset: %s' % FLAGS.dataset)
22: 
23: def OPTIMIZER(lr):
24:     return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=False)
```
```text
66:         for m in range(FLAGS.num_model):
67:             l = images
68:             with tf.variable_scope('model_%d' % m):
69:                 l = layers.conv('conv_init', l, 16, stride=1)
70:                 l = residual('res_1_1', l, 16, 16, 1)
71:                 l = residual('res_1_2', l, 16, 16, 1)
72:                 l = residual('res_1_3', l, 16, 16, 1)
73:                 features.append(l)
74: 
75:         # stochastically share hidden features right before the first pooling
76:         if FLAGS.feature_sharing:
77:             features = feature_sharing(features)
```
