# cae_ensemble inspected method scope

Primary URL: https://arxiv.org/html/2111.11108v1

Zero full-paper or author-source credit.

## Block 35

3. The CAE-Ensemble Framework

## Block 36

This section details the proposed CAE-Ensemble . In Section 3.1 , we first cover basic CAE models using convolutional sequence-to-sequence autoencoders that are able to capture temporal dependencies in time series data while ensuring efficiency. Next, in Section 3.2 , we describe a diversity-driven ensemble CAE-Ensemble that consists of the basic CAE models while taking into account diversity during training, along with an efficient training method using parameter transferring for the proposed ensemble. Finally, in Section 3.3 , we propose an strategy for tuning hyperparameters in a fully unsupervised manner. The complete process is summarized in Algorithm 1 and a framework overview is shown in Figure 2 .

## Block 37

Algorithm 1 CAE-Ensemble Input : Raw time series $\mathcal{T}$ , Number of Basic Models $M$ Output : Outlier scores 1 $\mathit{outlierScores}[]\leftarrow\emptyset,\,\mathit{savedParam}\leftarrow\emptyset$ ; 2 $\mathcal{T}\leftarrow\textnormal{{$\mathit{ReScale}$}}(\mathcal{T})$ ; 3 $\beta,\lambda,w\leftarrow\textnormal{{$\mathit{HyperparameterSelection}$}}(\mathcal{T})$ ; /* Sec 3.3 */ 4 $\mathcal{T}_{\mathit{windows}}\leftarrow\textnormal{{$\mathit{SplitIntoWindows}$}}(\mathcal{T},w)$ ; 5 $\mathbf{X}\leftarrow\textnormal{{$\mathit{Embedding}$}}(\mathcal{T}_{\mathit{windows}})$ ; 6 for $i\leftarrow 1$ to $M$ do 7 if $i$ = 1 then /* Build the first basic model CAE $f_{1}$ . */ 8 $\theta_{f_{1}}\leftarrow$ $\mathit{OptimizeNormal}$ ( $f_{1}$ ); 9 $\mathbf{\hat{X}^{(1)}}\leftarrow f_{1}(\mathbf{X})$ ; 10 else /* Build the $i$ -th basic model CAE $f_{i}$ . */ 11 $\theta_{f_{i}}\leftarrow\textnormal{{$\mathit{OptimizeDiverse}$}}(f_{i},\lambda,\mathit{savedParam})$ ; /* $\mathit{OptimizeDiverse}$ uses Eq. 13 . */ 12 $\mathbf{\hat{X}^{(i)}}\leftarrow f_{i}(\mathbf{X})$ ; 13 $\mathit{outlierScores}[i]\leftarrow\langle\|\mathbf{x}_{1}-\hat{\mathbf{x}}_{1}^{(i)}\|_{2}^{2},\ldots,\|\mathbf{x}_{w}-\hat{\mathbf{x}}_{w}^{(i)}\|_{2}^{2}\rangle$ ; 14 $\mathit{savedParam}\leftarrow$ Randomly select the fraction $\beta$ of the parameters $\theta_{f_{i}}$ from basic model CAE $f_{i}$ ; 15 return $\mathit{median(outlierScores)}$

## Block 38

Before introducing the model, we cover a pre-processing step: a raw time series is pre-processed into time series windows that are then used for training and testing ( 29 ) . The pre-processing first re-scales an observation $x$ in the time series to $\displaystyle z=\frac{x-\mu}{\sigma}$ , where $\mu$ is the mean and $\sigma$ is the standard deviation of the observations in the training time series. This is to prevent that magnitude differences among different dimensions in a time series affect the reconstruction errors differently, which is a common pre-processing technique ( 30 ) .

## Block 39

We then create sliding windows of size $w$ that slide one observation at a time. For example, for $\mathcal{T}=\langle\mathbf{s}_{1},\mathbf{s}_{2},\dots,\mathbf{s}_{C}\rangle$ , the first window is $\langle\mathbf{s}_{1},\mathbf{s}_{2},\dots,\mathbf{s}_{w}\rangle$ and the second is $\langle\mathbf{s}_{2},\mathbf{s}_{3},\dots,\mathbf{s}_{w+1}\rangle$ , etc.

## Block 72

3.2. Diversity-Driven Ensembles

## Block 73

We use the proposed CAE s as basic models in the ensemble called CAE-Ensemble . Ensembles are generally able to improve overall accuracy by combining the outputs from individual basic models ( 17 ) . A naive approach is to first train multiple basic models. Then, each basic model is used to reconstruct embedded time series $\mathbf{X}$ . The average over the reconstructed time series from all basic models is used as the final reconstructed time series for the ensemble, as defined in Equation 8 .

## Block 74

(8) $\displaystyle F(\mathbf{X})=\frac{1}{M}\sum_{m=1}^{M}f_{m}(\mathbf{X}),$

## Block 75

where $M$ is the total number of basic models, $f_{m}(\cdot)$ refers to the $m$ -th basic model, and $F(\cdot)$ is the output of the ensemble.

## Block 76

This naive approach has two limitations: (1) If the basic models are similar, the accuracy of the ensemble is similar to those of the basic models, meaning that the ensemble is not substantially better than each basic model. (2) The computational cost of the ensemble is often $M$ times that of a single basic model. When $M$ is large, the ensemble can be very expensive to train.

## Block 77

The first limitation has been addressed partially by using basic models with different structures, e.g., by randomly modifying the connections among computational units ( 17 , 19 ) . However, basic models with different structures may not produce diverse outputs.

## Block 78

We introduce a diversity-driven ensemble to address the above two limitations, as depicted in Figure 8 . Rather than training different basic models independently ( 17 , 19 ) , we generate the basic models one by one. When training a basic model, we design an objective function that considers not only the accuracy of the model but also its difference compared to the previous basic models, which ensures diversity among basic models, thus addressing the first limitation. In addition, when training a basic model, we transfer a portion of the parameters from the previous basic model to the model instead of training the model from scratch. This helps significantly reduce the training time, thus addressing the second limitation.

## Block 79

3.2.1. Basic Model Generation

## Block 80

Inspired by Born-again Neural Networks ( 25 ) ( BANN ) and AdaBoost Negative Correlation ( 35 ) , we iteratively generate basic models in the model training epochs. For example, we may generate a basic model per $n$ training epochs. If $n=10$ , when training an ensemble using 200 epochs, we can then generate 20 basic models.

## Block 81

In the first epoch, we create the first basic model $f_{1}(\cdot)$ , and we start training this basic model using embedded time series $\mathbf{X}$ . We then add the first basic model to the ensemble. After $n$ epochs, the first basic model is partially trained. We continue to generate the second basic model $f_{2}(\cdot)$ by transferring a portion of the parameters learned in the first basic model to $f_{2}(\cdot)$ . Then, we train the second basic model. The training and basic model generation processes continue until the last training epoch.

## Block 82

More specifically, given a basic model $f_{m-1}(\cdot)$ with $\theta_{f_{m-1}}$ trained parameters obtained in the previous epochs, the newly generated basic model $f_{m}(\cdot)$ receives a randomly selected fraction $\beta$ of its parameters from $f_{m-1}(\cdot)$ . Then, the remaining fraction $1-\beta$ of the parameters must be trained in subsequent epochs. This approach enables us to train a large number of ensemble components with low training time. Figure 9 shows the process of generating new basic models with parameter transfer.

## Block 83

Note that the proposed ensemble differs from a Snapshot Ensemble ( 36 ) , where the trained parameters are transferred completely among the basic models.

## Block 84

3.2.2. Diversity Metric

## Block 85

Based on existing studies ( 22 ) , an ensemble benefits from consisting of diverse basic models. The more diverse the basic models in an ensemble are, the more accurate the ensemble often achieves. To quantify the diversity explicitly, we define a diversity metric $\mathit{DIV}_{f_{m},f_{n}}(\cdot)$ to measure the dissimilarity between two basic models $f_{m}(\cdot)$ and $f_{n}(\cdot)$ .

## Block 86

(9) $\displaystyle\mathit{DIV}_{f_{m},f_{n}}(\mathbf{X})={||f_{m}(\mathbf{X})-f_{n}(\mathbf{X})||_{2}},$

## Block 87

Here, a larger $\mathit{DIV}_{f_{m},f_{n}}(\mathbf{X})$ value indicates a larger difference between the outputs of basic models $f_{m}(\cdot)$ and $f_{n}(\cdot)$ .

## Block 88

The design of diversity metric $\mathit{DIV}_{f_{m},f_{n}}(\cdot)$ is inspired by the supervised diversity metric ( 37 ) . However, the supervised diversity metric computes the $\mathrm{softmax}$ outputs of two basic models and compares the outputs with ground-truth labels to quantify the diversity. In our unsupervised setting, we do not have ground truth labels. Instead, we aim for different basic models with different reconstructions, i.e., basic models with diverse outputs.

## Block 89

Next, we extend the diversity metric $\mathit{DIV}_{F}(\cdot)$ to measure the diversity of an ensemble model $F(\cdot)$ as follows.

## Block 90

(10) $\displaystyle\mathit{DIV}_{F}(\mathbf{X})=\frac{2}{M(M-1)}\sum^{M}_{m=1}\sum^{M}_{n=m+1}\mathit{DIV}_{f_{m},f_{n}}(\mathbf{X}),$

## Block 91

where a large $\mathit{DIV}_{F}(\cdot)$ value indicates that ensemble model $F(\cdot)$ is more diverse.

## Block 92

3.2.3. Diversity-Driven Objective Function

## Block 93

Based on the proposed diversity metric, we define the objective function for training each basic model in the ensemble in two parts.

## Block 94

First, a basic model $f_{m}(\cdot)$ should reconstruct the time series accurately, which is the same as for all autoencoder models. Thus, the objective function $\mathcal{J}_{f_{m}}$ measures the difference between the input embedded time series vectors and the reconstructed vectors.

## Block 95

(11) $\displaystyle\mathcal{J}_{f_{m}}=||\mathbf{X}-\mathbf{\hat{X}}||^{2}_{2}=||\mathbf{X}-f_{m}(\mathbf{X})||^{2}_{2}$

## Block 96

Second, the basic model should increase the diversity of the current ensemble. Thus, the objective function $\mathcal{K}_{f_{m}}$ utilizing Equation 9 , measures the diversity between the basic model $f_{m}(\mathbf{X})$ and the current ensemble $F(\mathbf{X})$ .

## Block 97

(12) $\displaystyle\mathcal{K}_{f_{m}}=||f_{m}(\mathbf{X})-F(\mathbf{X})||^{2}_{2}$

## Block 98

Finally, the objective function $\mathcal{O}_{f_{m}}$ for training the basic model $f_{m}(\cdot)$ is defined as a combination of $\mathcal{J}_{f_{m}}$ and $\mathcal{K}_{f_{m}}$ .

## Block 99

(13) $\displaystyle\arg\min_{\theta_{f_{m}}}\mathcal{L}_{f_{m}}=\arg\min_{\theta_{f_{m}}}\mathcal{J}_{f_{m}}-\lambda\mathcal{K}_{f_{m}},$

## Block 100

where $\lambda$ is a factor that controls the importance of the diversity, and $\theta_{f_{m}}$ represents the learnable parameters of model $f_{m}(\cdot)$ .

## Block 101

3.2.4. Ensemble Outlier Score

## Block 102

We aggregate the outlier scores from all basic models to obtain the final outlier score. Given $M$ basic models that reconstruct the embedded time series $\mathbf{X}=\langle\mathbf{x}_{1},\mathbf{x}_{2},\dots,\mathbf{x}_{w}\rangle$ , we obtain $M$ reconstructions $\hat{\mathbf{X}}^{(m)}=\langle\hat{\mathbf{x}}^{(m)}_{1},\hat{\mathbf{x}}^{(m)}_{2},\dots,\hat{\mathbf{x}}^{(m)}_{w}\rangle$ , where $1\leq m\leq M$ . For each vector $\mathbf{x}_{t}$ in the embedded time series $\mathbf{X}$ , we obtain $M$ reconstruction errors as follows.

## Block 103

(14) $\displaystyle\langle\|\mathbf{x}_{t}-\hat{\mathbf{x}}_{t}^{(1)}\|_{2}^{2},\|\mathbf{x}_{t}-\hat{\mathbf{x}}_{t}^{(2)}\|_{2}^{2},\ldots,\|\mathbf{x}_{t}-\hat{\mathbf{x}}_{t}^{(M)}\|_{2}^{2}\rangle$

## Block 104

We use the $\mathrm{median}$ of the $M$ errors as the final outlier score of vector $\mathbf{x}_{t}$ as follows.

## Block 105

(15) $OS(\mathbf{x}_{t})=\mathrm{median}\left(\|\mathbf{x}_{t}-\hat{\mathbf{x}}_{t}^{(1)}\|_{2}^{2},\ldots,\|\mathbf{x}_{t}-\hat{\mathbf{x}}_{t}^{(M)}\|_{2}^{2}\right)$

## Block 106

We use $\mathrm{median}(\cdot)$ instead of $\mathrm{mean}(\cdot)$ because $\mathrm{median}(\cdot)$ reduces the influence of the reconstruction errors from the basic models that overfit to the original time series ( 19 ) .
