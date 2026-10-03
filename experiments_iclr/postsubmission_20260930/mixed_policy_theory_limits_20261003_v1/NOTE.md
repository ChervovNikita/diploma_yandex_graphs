# Binary logit coordinate convention

Section 2 uses binary class labels `y in {0,1}` and margin `t=logit(class 1)-logit(class 0)`. Section 3 writes `z=(t,0)` in the coordinate order **(class 1, class 0)**. Its target is class 1, therefore the first coordinate is the target-class logit and its CE is `f(t)=log(1+exp(-t))`. “Class 1” retains the binary label 1; it is not a statement that array coordinate index 1 is the target.

The same convention applies to the witness’s phrase “first binary class”: it denotes the first displayed logit coordinate, class 1. This note clarifies coordinate order only. The sealed report, witness, conclusions, manifest, and original checksum file remain byte-for-byte unchanged.
