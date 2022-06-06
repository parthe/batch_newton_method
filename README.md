# Batch Newton Method

We solve an optimization problem over matrix variables such that the objective function is separable with respect to the rows of the matrix variable

As an application, we use this batch newton method for computing a certain generalized proximal operator

$$\underset{X}{\rm argmin} \sum_{i=1}^n \mathsf{Loss}(X_i) + \|X\|_Q^2$$

where $\|X\|_Q^2 = {\rm trace}(X^\top X Q)$ for some positive definite matrix $Q$
