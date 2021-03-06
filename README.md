# Batch Newton Method

We solve an optimization problem over matrix variables such that the objective function is twice differentiable, and separable with respect to the rows of the matrix variable

As an application, we use this batch newton method for computing the Mahalanobis proximal operator

$$\mathrm{prox}_f(\mathbf{R};Q):=\underset{\mathbf{X}}{\rm argmin} \sum_i f(\mathbf{X}_i) + ||\mathbf{X}-\mathbf{R}||_Q^2$$

where $\mathbf{X}_i$ are rows of $\mathbf{X}\in\mathbb{R}^{n\times k}$, and the Mahalanobis-type norm $||\mathbf{X}||_Q^2 = {\rm trace}(\mathbf{X}^\top \mathbf{X} Q)$ for some positive definite matrix $Q$
