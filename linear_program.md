# Notes on the linear program

## General Problem

A linear program is an optimization problem with a linear objective and linear constraint.

For example, consider
$$ \min_{x_0, x_1} -x_0  + 4 x_1$$
such that
$$ -3x_0+x_1\leq6, \\
-x_0-2x_1 \geq -4, \\
x_1 \geq -3.
$$

We can rewrite any linear program in the general form
$$ \min_x c^Tx$$
such that
$$ A_{ub}x \leq b_{ub} \\
A_{eq} x = b_{eq} \\
l \leq x \leq u.
$$
Here, $x$ is a vector of decision variables. The first set of constraints are inequality constraints while the second are equality constraints (for general linear combinations of the decision variables). Lastly, the third set are bounds on the decision variables (e.g. non-negativity).

## MST 2018 ECMA Problem

In our setting, the linear program for the upper bound of the identified set is given by
$$ \overline{\beta^*_{fd}} \equiv \sup_{(\theta_0, \theta_1)\in\Theta}\sum_{k=1}^{K_0}\theta_{0k}\Gamma^*_0(b_{0k}) + \sum_{k=1}^{K_1} \theta_{1k}\Gamma_1^*(b_{1k})$$
subject to
$$ \sum_{k=1}^{K_0}\theta_{0k}\Gamma_{0s}(b_{0k}) +
\sum_{k=1}^{K_1}\theta_{1k}\Gamma_{1s}(b_{1k}) = \beta_s \text{ for all } s\in S.$$
Intuitively, the problem consists of
- Choosing $MTR$ functions that make the target estimand as big as possible
- Subject to the constraint that the optimizing MTRs must be consistent with the identified estimands.

Here, in this finite-dimensional version of the linear program choosing the $MTR$ functions amounts to choosing coefficients $\theta_0, \theta_1$ on the basis functions.

Recall the definition of the linear maps $\Gamma_{ds}$ for some estimand $s$:
$$ \Gamma_s(m) \equiv E\left[\int_0^1 m_0(u,X)\omega_{0,s}(u,Z)du\right] + E\left[\int_0^1 m_1(u,X)\omega_{1,s}(u,Z)du\right].$$

Under certain basis functions (like constant splines or Bernstein polynomials), $\Gamma_{ds}(b_{dk})$ can be computed analytically.

## Translation of MST problem

We need to define the following objects:
- $x$: The vector of choice variables
- $c$: The weights on the choice variables in the objective

All of the vectors are in $R^{K_0+K_1}$, while the matrices in the constraint have shape $(K_0+K_1)\times |S|$, because we have $|S|$ identified estimands.

There are no inequality constraints. Further we have $x_i\in [0,1]$ for all $x_i$.

The choice variables are given by
$$ x = (\theta_{01}, \ldots, \theta_{0K_0}, \theta_{11}, \ldots, \theta_{1K_1}).$$

The weights on the choice variable are given by
$$ c = (\Gamma^*(b_{01}) ,\ldots, \Gamma^*(b_{0K_0}),
\Gamma^*(b_{10}), \ldots, \Gamma^*(b_{1K_1})).$$

The RHS values of the equality constraint are the identified estimands, i.e.
$$ b_{eq} = (\beta_1, \ldots, \beta_{|S|}).$$

Lastly, the weight matrix in the inequality constraint collects all weights $\Gamma_s(b_{dk})$ similar to the $c$ vector for the target estimand, only that every row reflects a different identified estimand $s\in S$. Thus, we have
$$A_{eq} = \begin{bmatrix}

\Gamma_1(b_{01}) & \ldots & \Gamma_1(b_{0K_0}) & \Gamma_1(b_{11}) & \ldots & \Gamma_1(b_{1K_1})\\
\ldots & \ldots & \ldots & \ldots & \ldots & \ldots\\
\Gamma_{|S|}(b_{01}) & \ldots & \Gamma_{|S|}(b_{0K_0}) & \Gamma_{|S|}(b_{11}) & \ldots & \Gamma_{|S|}(b_{1K_1})\\

\end{bmatrix}.$$
