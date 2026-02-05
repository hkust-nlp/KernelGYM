# Gradient Variance and W(τ) Approximation: Complete Step-by-Step Derivation

## 1. The Core Problem: What is Gradient Variance?

### 1.1 The REINFORCE Gradient Estimator
In policy gradient methods, we estimate the gradient using sampled trajectories:

```
Single trajectory gradient estimate:
ĝ = ∇_θ log π_θ(τ) · R(τ)

where:
- τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...) is a sampled trajectory
- R(τ) = Σₜ rₜ is the total return
- ∇_θ log π_θ(τ) = Σₜ ∇_θ log π_θ(aₜ|sₜ) is the score function
```

### 1.2 The Variance Problem
Since we use Monte Carlo sampling, our gradient estimate has variance:

```
True gradient: ∇_θ J = E_τ~π_θ[ĝ]
Gradient estimator: ĝ = ∇_θ log π_θ(τ) · R(τ)

Variance of estimator:
Var[ĝ] = E_τ[||ĝ - ∇_θ J||²]
       = E_τ[||∇_θ log π_θ(τ) · R(τ)||²] - ||∇_θ J||²
```

**This variance is what causes unstable training!**

### 1.3 With Baseline for Variance Reduction
We introduce a baseline b to reduce variance without adding bias:

```
Modified gradient: ĝ = ∇_θ log π_θ(τ) · (R(τ) - b)
                      = ∇_θ log π_θ(τ) · A(τ)

where A(τ) = R(τ) - b is the advantage
```

**Key Question:** How do we choose b to minimize Var[ĝ]?

## 2. Finding the Optimal Baseline

### 2.1 The Optimization Problem
We want to find b that minimizes the gradient variance:

```
b* = arg min_b Var[∇_θ log π_θ(τ) · (R(τ) - b)]
   = arg min_b E[||∇_θ log π_θ(τ) · (R(τ) - b)||²]
```

### 2.2 Expanding the Variance
Let's expand this step by step:

```
E[||∇_θ log π_θ(τ) · (R(τ) - b)||²]
= E[(R(τ) - b)² · ||∇_θ log π_θ(τ)||²]
= E[(R(τ) - b)² · Σᵢ (∂log π_θ(τ)/∂θᵢ)²]
```

where i indexes over all parameters θᵢ.

### 2.3 The Key Problem
To compute this exactly, we need:
1. The full gradient vector ∇_θ log π_θ(τ) - dimension d (millions for neural networks)
2. Its squared norm ||∇_θ log π_θ(τ)||²
3. This for every sampled trajectory

**This is computationally intractable!**

## 3. The W(τ) Approximation: Step-by-Step Derivation

### 3.1 The Core Idea
Instead of computing ||∇_θ log π_θ(τ)||² exactly, we approximate it using a proxy W(τ) that:
- Is cheap to compute (only uses π(aₜ|sₜ) values)
- Correlates strongly with the true gradient norm
- Allows us to find near-optimal baselines

### 3.2 Starting Point: The Score Function
For a trajectory τ, the score function is:

```
∇_θ log π_θ(τ) = Σₜ ∇_θ log π_θ(aₜ|sₜ)
```

### 3.3 Step 1: Focus on the Output Layer
For a neural network policy, most of the gradient magnitude comes from the final layer.
Let's consider the gradient with respect to the logits (pre-softmax outputs):

```
For softmax policy: π_θ(a|s) = exp(z_a) / Σ_a' exp(z_a')
where z_a = f_θ(s,a) are the logits
```

### 3.4 Step 2: Gradient of Log Probability w.r.t. Logits
For action aₜ that was actually taken:

```
∂log π_θ(aₜ|sₜ)/∂z_a = δ_{a,aₜ} - π_θ(a|sₜ)

where δ_{a,aₜ} = 1 if a = aₜ, 0 otherwise
```

### 3.5 Step 3: Squared Gradient Norm
The squared norm of this gradient (w.r.t. logits) is:

```
||∂log π_θ(aₜ|sₜ)/∂z||² = Σ_a (δ_{a,aₜ} - π_θ(a|sₜ))²
                         = (1 - π_θ(aₜ|sₜ))² + Σ_{a≠aₜ} π_θ²(a|sₜ)
                         = 1 - 2π_θ(aₜ|sₜ) + π_θ²(aₜ|sₜ) + Σ_{a≠aₜ} π_θ²(a|sₜ)
                         = 1 - 2π_θ(aₜ|sₜ) + Σ_a π_θ²(a|sₜ)
```

### 3.6 Step 4: Define W(τ)
Therefore, we define:

```
W(τ) = Σₜ [1 - 2π_θ(aₜ|sₜ) + Σ_a π_θ²(a|sₜ)]
```

This approximates ||∇_θ log π_θ(τ)||² (up to a scaling constant).

## 4. Using W(τ) to Find the Optimal Baseline

### 4.1 Substituting W(τ) into the Optimization
Now we can approximate the gradient variance minimization:

```
Original problem:
b* = arg min_b E[(R(τ) - b)² · ||∇_θ log π_θ(τ)||²]

With W(τ) approximation:
b* ≈ arg min_b E[(R(τ) - b)² · W(τ)]
```

### 4.2 Taking the Derivative and Setting to Zero
To find the minimum, we differentiate with respect to b:

```
∂/∂b E[(R(τ) - b)² · W(τ)] = E[-2(R(τ) - b) · W(τ)]
                             = -2E[R(τ) · W(τ)] + 2b · E[W(τ)]
                             = 0
```

### 4.3 Solving for Optimal Baseline
```
2b · E[W(τ)] = 2E[R(τ) · W(τ)]

Therefore:
b* = E[R(τ) · W(τ)] / E[W(τ)]
```

This is a weighted average of returns, where **trajectories with larger gradient norms (larger W) get more weight!**

### 4.4 Intuition
- Trajectories with unlikely actions (small π(a|s)) have large W(τ)
- These trajectories have large gradient norms
- They contribute more to gradient variance
- So we weight them more when computing the baseline

## 5. How W(τ) Approximates Gradient Variance

### 5.1 The Gradient Variance We Want to Minimize
```
Var[∇_θ J] = E[||∇_θ log π_θ(τ) · A(τ)||²] - ||E[∇_θ log π_θ(τ) · A(τ)]||²
```

### 5.2 Using W(τ) as a Proxy
Since ||∇_θ log π_θ(τ)||² ≈ c · W(τ) for some constant c:

```
E[||∇_θ log π_θ(τ) · A(τ)||²] ≈ c · E[A²(τ) · W(τ)]
```

### 5.3 Comparing Different Baselines
When comparing two baselines (e.g., optimal vs RLOO):

```
For optimal baseline: Var_opt ≈ c · E[A_opt²(τ) · W(τ)]
For RLOO baseline:    Var_RLOO ≈ c · E[A_RLOO²(τ) · W(τ)]

Variance ratio: Var_opt / Var_RLOO ≈ E[A_opt²(τ) · W(τ)] / E[A_RLOO²(τ) · W(τ)]
```

The constant c cancels out in the ratio!

## 6. In Practice: Computing the Gradient Variance Approximation

### 6.1 The Practical Formula
Given trajectories with advantages A(τ) and probabilities π(aₜ|sₜ):

```
Step 1: Compute W(τ) for each trajectory
W(τ) = Σₜ [1 - 2π(aₜ|sₜ) + Σ_a π²(a|sₜ)]

Step 2: Compute gradient magnitude proxy
gradient_magnitude(τ) ≈ |A(τ)| · √W(τ)

Step 3: Compute variance across batch
gradient_variance ≈ Var[A(τ) · √W(τ)]
```

### 6.2 Why We Use √W(τ) Instead of W(τ)
```
Since: ||∇_θ log π_θ(τ)||² ≈ W(τ)
Then:  ||∇_θ log π_θ(τ)|| ≈ √W(τ)

And the gradient magnitude is:
||∇_θ J|| = ||A(τ) · ∇_θ log π_θ(τ)|| ≈ |A(τ)| · √W(τ)
```

### 6.3 Example Calculation
```python
# For a single trajectory:
π_selected = [0.1, 0.05, 0.2, 0.15]  # Probabilities of selected actions
π_squared_sum = [0.3, 0.4, 0.35, 0.38]  # Sum of squared probs over vocab

# Compute W per timestep
w_per_timestep = [1 - 2*π + π_sq for π, π_sq in zip(π_selected, π_squared_sum)]
# = [1 - 2*0.1 + 0.3, 1 - 2*0.05 + 0.4, ...]
# = [1.1, 1.3, 0.95, 1.08]

# Total W for trajectory
W_total = sum(w_per_timestep) = 4.43

# If advantage A(τ) = 2.5
gradient_magnitude ≈ 2.5 * √4.43 ≈ 5.26
```

## 7. Why the Approximation is Valid (But Not Perfect)

### 7.1 What Makes It Work
1. **Relative Comparison**: Even if absolute values are off, ratios between methods are preserved
2. **Captures Key Relationship**: Low π(a|s) → high gradient → high variance
3. **Empirically Validated**: The optimal baseline formula using W(τ) works well in practice

### 7.2 Limitations
1. **Not Exact**: W(τ) only approximates the gradient norm for the output layer
2. **Ignores Earlier Layers**: Gradients in earlier layers may behave differently
3. **Scalar Reduction**: Loses directional information about the gradient vector

### 7.3 When to Trust It Most
- Comparing methods (optimal vs RLOO) rather than absolute values
- Large vocabulary/action spaces (where output layer dominates)
- Trajectory-level advantages (constant within trajectory)

## 8. Connection to the Optimal Baseline Formula

### 8.1 The Formula in the Code
The optimal baseline in practice:
```python
b* = Σᵢ W(τᵢ) · R(τᵢ) / Σᵢ W(τᵢ)
```

### 8.2 Why This Minimizes Gradient Variance
This formula comes directly from minimizing:
```
E[(R(τ) - b)² · W(τ)]
```

Which approximates minimizing the true gradient variance:
```
E[||∇_θ log π_θ(τ) · (R(τ) - b)||²]
```

### 8.3 Verification
The fact that this baseline formula works well in practice validates that W(τ) is indeed a good proxy for gradient norm!

## 9. Summary: The Complete Picture

### 9.1 What We're Doing
1. **Goal**: Minimize gradient variance for stable training
2. **Problem**: Computing exact gradient variance is intractable
3. **Solution**: Use W(τ) as a cheap proxy for ||∇_θ log π_θ(τ)||²

### 9.2 The Approximation Chain
```
True gradient variance:
Var[∇_θ J] = Var[∇_θ log π_θ(τ) · A(τ)]

↓ (approximate gradient norm)

Approximated as:
Var[∇_θ J] ≈ Var[A(τ) · √W(τ)]

where W(τ) = Σₜ [1 - 2π(aₜ|sₜ) + Σ_a π²(a|sₜ)]
```

### 9.3 Practical Impact
- Allows efficient comparison of variance reduction between methods
- Enables finding near-optimal baselines without gradient computation
- Provides interpretable metric for optimization stability

This approximation isn't perfect, but it's computationally efficient and captures the essential relationship between action probabilities, advantages, and gradient variance.
