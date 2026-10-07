# dibs inspected method scope

Primary URL: https://arxiv.org/html/2003.04514v3

Zero full-paper or author-source credit.

## Block 25

Notation

## Block 26

In this paper, we consider a network with a shared encoder $f(\cdot)$ and multiple stochastic task-specific decoders $g_{i}(\cdot)$ where both $f(\cdot)$ and $g_{i}(\cdot)$ are parameterized using neural networks. Each encoder encodes an image, $X$ , to a shared latent space, $Z$ , which is then used by each task specific decoder to obtain a prediction $Y_{i}$ , where $Y_{i}$ is the $i$ -th prediction from the model. Fig 1 describes the architecture visually.

## Block 27

DIBS : Diverse Information Bottleneck in Ensembles

## Block 28

We propose a method for ensemble learning ( Hansen and Salamon 1990 ) by promoting diversity among pairwise latent ensemble variables and by enforcing an information bottleneck ( Alemi et al. 2016 ) between each latent $\tilde{Z}_{i}$ and the input $X$ . Formally, we consider a set of $K$ decoders $\{\theta_{1},...\theta_{K}\}\sim p(\Theta)$ sampled from some given initial distribution $p(\Theta)$ . Given input data $X$ , we want to learn a shared encoding $Z$ , and $K$ decoders that map the latent state $Z$ to $\tilde{Z}_{i}$ ’s and $\tilde{Z}_{i}$ ’s to the $K$ output predictions $\{Y_{1},...,Y_{K}\}$ .

## Block 29

We posit that for effective learning through ensembles, there must be some diversity among the members of the ensemble, since each ensemble member is by assumption a weak learner, and individual performance is not as important as collective performance ( Melville and Mooney 2004 ) . However, promoting diversity randomly among the members is likely to result in uninformative/irrelevant aspects of data being captured by them. Hence, in addition to task-specific standard likelihood maximization, we introduce the need for a diversity enforcing constraint, and a bottleneck constraint. To accomplish the latter, we build upon the Variational Information Bottleneck (VIB) formulation ( Alemi et al. 2016 ) by constraining the information flow from input $X$ to the outputs $Y_{i}$ , we introduce the information bottleneck term $-\mathcal{I}(\tilde{Z}_{i},X;\theta)$ . For diversity maximization between ensembles, we design an anti-clustering and diversity-inducing generative adversarial loss, described in the next section.

## Block 30

Adversarial Model for Diversity Maximization

## Block 31

We adopt an adversarial learning approach based on the intuition of diversity maximization among the $K$ models. Our method is inspired by Adversarial Autoencoders ( Makhzani et al. 2015 ) , which proposes a natural scheme for combining adversarial training with variational inference. Here, our aim is to maximize separation in distribution between ensemble pairs $q(\tilde{z}_{i}|x),q(\tilde{z}_{j}|x)$ , such that samples $(\hat{z}_{1},\hat{z}_{2})$ are indistinguishable to a discriminator if $\hat{z}_{1}\sim q(\tilde{z}_{i}|x)$ , $\hat{z}_{2}\sim q(\tilde{z}_{j}|x)$ with $i=j$ and they are distinguishable if $i\neq j$ . To this end, we frame the adversarial loss, such that the $K$ generators $q(\tilde{z}_{i}|x)\;\forall i\in[1,..,K]$ trick the discriminator into thinking that samples from $q(\tilde{z}_{i}|x)$ and $q(\tilde{z}_{j}|x)$ are samples from different distributions.

## Block 32

We start with $r(\tilde{z})$ , a prior distribution on $\tilde{z}$ . In our case, this is normal $\mathcal{N}(0,\mathbb{I})$ , but more complex priors are also supported, in the form of implicit models. We want to make each encoder $q(\tilde{z}_{i}|x)$ to be close in distribution to this prior, but sufficiently far from other encoders, so that overlap is minimized. Unlike typical GANs ( Goodfellow et al. 2014 ) , the discriminator of our diversity inducing loss takes in a pair of samples ( $\hat{z}_{1},\hat{z}_{2}$ ) instead of just one sample. Hence, we have the following possibilities for the different sources of a pair of latents: 1) $\hat{z}_{1}\sim r(\tilde{z})$ and $\hat{z}_{2}\sim r(\tilde{z})$ , 2) $\hat{z}_{1}\sim q(\tilde{z}_{i}|x)$ and $\hat{z}_{2}\sim r(\tilde{z})$ , 3) $\hat{z}_{1}\sim q(\tilde{z}_{i}|x)$ and $\hat{z}_{2}\sim q(\tilde{z}_{i}|x)$ , and 4) $\hat{z}_{1}\sim q(\tilde{z}_{i}|x)$ and $\hat{z}_{2}\sim q(\tilde{z}_{j}|x)$ , with $i\neq j$

## Block 33

Let $D(\cdot)$ denote the discriminator, which is a feed-forward neural network that takes in a pair $(\hat{z}_{1},\hat{z}_{2})$ as input and outputs a $0$ (fake) or a $1$ (real). There are $K$ generators corresponding to each $q(\tilde{z}_{j}|z)$ , and the deterministic encoder $z=f(x)$ . We denote the parameters of all these generators, as well as the deterministic encoder as $G$ , to simplify notation. These generators are trained by minimizing the following loss over $G$ :

## Block 34

$\displaystyle L_{G}$ $\displaystyle=\mathbb{E}_{\hat{z}_{1}\sim q(\tilde{z}_{i}|x),\;\hat{z}_{2}\sim q(\tilde{z}_{j}|x)}[\log D(\hat{z}_{1},\hat{z}_{2})]$ $\displaystyle+\mathbb{E}_{\hat{z}_{1}\sim r(\tilde{z}),\;\hat{z}_{2}\sim q(\tilde{z}_{i}|x)}[\log(1-D(\hat{z}_{1},\hat{z}_{2}))]$ $\displaystyle+\mathbb{E}_{\hat{z}_{1}\sim q(\tilde{z}_{i}|x),\;\hat{z}_{2}\sim q(\tilde{z}_{i}|x)}[\log(1-D(\hat{z}_{1},\hat{z}_{2}))]$ (3)

## Block 35

Given a fixed discriminator $D$ , the first term encourages pairs of different encoder heads to be distinguishable. The second term encourages each encoder to overlap with the prior. The third term encourages samples from the same encoder to be indistinguishable.

## Block 36

On the other hand, given a fixed generator $G$ , the discriminator is trained by maximizing the following objective function with respect to $D$ :

## Block 37

$\displaystyle L_{D}$ $\displaystyle=\mathbb{E}_{\hat{z}_{1}\sim r(\tilde{z}),\;\hat{z}_{2}\sim r(\tilde{z})}[\log D(\hat{z}_{1},\hat{z}_{2})]$ $\displaystyle+\mathbb{E}_{\hat{z}_{1}\sim q(\tilde{z}_{i}|x),\;\hat{z}_{2}\sim q(\tilde{z}_{j}|x)}[\log D(\hat{z}_{1},\hat{z}_{2})]$ $\displaystyle+\mathbb{E}_{\hat{z}_{1}\sim r(\tilde{z}),\;\hat{z}_{2}\sim q(\tilde{z}_{i}|x)}[\log(1-D(\hat{z}_{1},\hat{z}_{2}))]$ (4)

## Block 38

The first term encourages the discriminator to not distinguish between samples from the prior. The second term aims to maximize overlap between different encoders, as an adversarial objective to what the generator is aiming to do in Eqn 3 . The third term minimizes overlap between the prior and each encoder.

## Block 39

It is important to note that the generators do not explicitly appear in the loss function because they are implicitly represented through the samples $\hat{z}_{1}\sim q(\tilde{z}_{i}|x),\;\hat{z}_{2}\sim q(\tilde{z}_{j}|x)$ . In each SGD step we backpropagate only through the generator corresponding to the respective $(\hat{z}_{1},\hat{z}_{2})$ sample. We also note that we consider the pairs $(\hat{z}_{1},\hat{z}_{2})$ to be unordered in the losses above, because we provide both orderings to the discriminator, to ensure symmetry.

## Block 40

Overall Optimization

## Block 41

The previous sub-section described the diversity inducing adversarial loss. In addition to this, we have the likelihood, and information bottleneck loss terms, denoted together by $\mathbb{L}(\theta)$ below. Here, $\theta=(\theta_{D},\theta_{G},\Theta)$ denotes the parameters of the discriminator, the generators, and the decoders.

## Block 42

$\displaystyle\mathbb{L}(\theta)$ $\displaystyle=\sum_{i=1}^{m}\alpha_{i}\mathcal{I}(\tilde{Z}_{i},Y_{i};\theta)-\sum_{i=1}^{m}\beta_{i}\mathcal{I}(\tilde{Z}_{i},X;\theta)$

## Block 43

For notational convenience, we omit $\theta$ in subsequent discussions. The first term can be lower bounded, as in ( Alemi et al. 2016 ) :

## Block 44

$\displaystyle\mathcal{I}(\tilde{Z}_{i},Y_{i})\geq\int p(y_{i},\tilde{z}_{i})\log\frac{q(y_{i}|\tilde{z}_{i})}{p(y_{i})}\;dy_{i}d\tilde{z}_{i}$ (5) $\displaystyle=\int p(x)p(y_{i}|x)p(\tilde{z}_{i}|x)\log q(y_{i}|\tilde{z}_{i})\;dxdy_{i}d\tilde{z}_{i}+H(Y)$

## Block 45

The inequality here is a result of $KL(p(y_{i}|\tilde{z}_{i})\;||\;q(y_{i}|\tilde{z}_{i}))\geq 0$ , where $q(y_{i}|\tilde{z}_{i})$ is a variational approximation to the true distribution $p(y_{i}|\tilde{z}_{i})$ and denotes our $i^{\text{th}}$ decoder. Since the entropy of output labels $H(Y)$ is independent of $\theta$ , it can be ignored in the subsequent discussions. Formally, the second term can be formulated as

## Block 46

$\displaystyle\mathcal{I}(\tilde{Z}_{i},X)\leq\int p(\tilde{z}_{i}|x)p(x)\log\frac{p(\tilde{z}_{i}|x)}{\psi(\tilde{z}_{i})}\;d\tilde{z}_{i}dx$ (6)

## Block 47

The inequality here also results from the non-negativity of the KL divergence. The marginal $p(\tilde{z}_{j})$ has been approximated by a variational approximation $\psi(\tilde{z}_{j})$ . Following the approach in VIB ( Alemi et al. 2016 ) , to approximate $p(x,y_{i})$ in practice we can use the empirical data-distribution $p(x,y_{i})=\frac{1}{N}\sum_{n=1}^{N}\delta_{x^{n}}(x)\delta_{y^{n}_{i}}(y_{i})$ . We also note that $z^{n}=f(x^{n})$ is the shared encoder latents, where $n$ denotes the $n^{th}$ datapoint among a total of $N$ datapoints. Now, using the re-parameterization trick, we write $\tilde{z}_{i}=g_{i}(z,\epsilon)$ , where $\epsilon$ is a zero mean unit variance Gaussian noise, such that $p(\tilde{z}_{i}|z)=\mathcal{N}(\tilde{z}_{i}|g_{i}^{\mu}(z),g_{i}^{\Sigma}(z))$ . We finally obtain the following lower-bound approximation of the the loss function. The detailed derivation is in the Appendix.

## Block 48

$\displaystyle\mathbb{L}$ $\displaystyle\approx\frac{1}{N}\sum_{n=1}^{N}\Big[\mathbb{E}_{\epsilon\sim p(\epsilon)}\Big[\sum_{i=1}^{m}\alpha_{i}\log q(y_{i}^{n}\;|\;g_{i}(f(x^{n}),\epsilon))$ $\displaystyle-\sum_{i=1}^{m}\beta_{i}KL\Big(p(\tilde{z}_{i}|x^{n})\;||\;\psi(\tilde{z}_{i})\Big)\Big]\Big]$ (7)

## Block 49

In our experiments we set $\psi(\tilde{z}_{j})=\mathcal{N}(\tilde{z}_{j}|0,\mathbb{I})$ . To make predictions in classification tasks, we output the modal class of the set of class predictions by each ensemble member.

## Block 50

Similar to GANs ( Goodfellow et al. 2014 ) , the model is optimized using alternating optimization where we alternate among objectives $\max_{\theta}\mathbb{L}(\theta)$ , $\min_{\theta_{G}}L_{G}$ , and $\max_{\theta_{D}}L_{D}$ . It is important to note that we do not explicitly optimize the KL-divergence term above, but implicitly do it during the process of adversarial learning using $\mathcal{L}_{adv}$ . In Section 3.1, the case $\hat{z}_{1}\sim q(\tilde{z}_{i}|x)$ and $\hat{z}_{2}\sim r(\tilde{z})$ corresponds to minimizing this KL-divergence term. This is done similarly to ( Makhzani et al. 2015 ) .
