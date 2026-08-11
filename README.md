# AB6: Introduction to Drones — Leader-Follower Formation of Two Quadrotors

A simulation of **dual quaternion-based control for a leader-follower formation of two quadrotors**, built to reproduce and explore the ideas from:

> H. N. Marciano, D. K. D. Villa, M. Sarcinelli-Filho, J. I. Giribet, *"Dual Quaternion-Based Control for a Leader-Follower Formation of Two Quadrotors,"* 2024 International Conference on Unmanned Aircraft Systems (ICUAS), Chania, Crete, Greece, June 4–7, 2024.

This document is written so that someone with **no prior background** in quaternions, dual numbers, or formation control can follow the math and the control design step by step, and then map each concept directly onto the code in this repository.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why Dual Quaternions?](#2-why-dual-quaternions)
3. [Mathematical Building Blocks](#3-mathematical-building-blocks)
   - 3.1 [Quaternions](#31-quaternions)
   - 3.2 [Quaternions as Rotations](#32-quaternions-as-rotations)
   - 3.3 [Dual Numbers](#33-dual-numbers)
   - 3.4 [Dual Quaternions](#34-dual-quaternions)
   - 3.5 [Dual Quaternions as Rigid-Body Pose](#35-dual-quaternions-as-rigid-body-pose)
4. [Vehicle Kinematics in Dual Quaternions](#4-vehicle-kinematics-in-dual-quaternions)
5. [Pose Error and the Control Objective](#5-pose-error-and-the-control-objective)
6. [The Kinematic Control Law](#6-the-kinematic-control-law)
7. [Why Tuning Is Hard: The Linearized Error Dynamics](#7-why-tuning-is-hard-the-linearized-error-dynamics)
8. [The Pole-Placement / Gain-Selection Method](#8-the-pole-placement--gain-selection-method)
9. [From Kinematics to Quadrotor Commands](#9-from-kinematics-to-quadrotor-commands)
10. [Leader-Follower Formation Law](#10-leader-follower-formation-law)

---

## 1. Project Overview

Coordinating multiple drones is a core problem in robotics: think of a "leader" drone flying a planned path while one or more "follower" drones automatically maintain a fixed relative position and orientation with respect to it. This is the **leader-follower formation control** paradigm.

The reference paper tackles this problem for **two small quadrotors** using **dual quaternions** — a single mathematical object that represents *both* the orientation *and* the position of a rigid body at once. The paper's main contributions are:

- A **kinematic control law**, expressed purely in dual quaternions, that drives a vehicle's pose (position + attitude) to a desired, possibly time-varying, pose.
- A **systematic method for choosing controller gains** (instead of trial and error), based on analyzing the eigenvalues of the *linearized* closed-loop error dynamics.
- **Experimental validation** on two real Parrot Bebop 2 quadrotors flying a leader-follower Lemniscate ("figure-eight") trajectory, under three different gain-tuning strategies.

This project reproduces that pipeline in simulation: modeling the drones, implementing the dual-quaternion kinematic controller, tuning gains using the paper's pole-placement approach, and comparing tracking performance across the same three experimental conditions the paper uses.

---

## 2. Why Dual Quaternions?

Classical approaches represent a rigid body's **attitude** (orientation) with rotation matrices or quaternions, and its **position** separately as a 3D vector. This works, but it means:

- Position and orientation are updated and controlled with *different* mathematical tools.
- Combining/composing successive rigid-body transformations (rotate, then translate, then rotate again, etc.) requires juggling matrices and vectors together, which is computationally heavier and easier to get subtly wrong.

**Dual quaternions solve this by unifying position and orientation into a single 8-dimensional algebraic object.** Just as a unit quaternion compactly encodes a 3D rotation, a *unit dual quaternion* compactly encodes a full **rigid-body pose** (rotation **and** translation) — and, crucially, poses can be *composed* with a single quaternion-like product, exactly the way rotations compose for ordinary quaternions. This gives:

- A unified, singularity-free representation of pose (no gimbal lock, unlike Euler angles).
- Computationally efficient composition and inversion of transformations.
- A natural way to define a single **pose error** between "where the drone is" and "where it should be," combining position and attitude error into one quantity.

This is why the paper — and this project — builds the entire leader-follower controller on top of dual quaternion algebra.

---

## 3. Mathematical Building Blocks

To understand dual quaternions, we build up from ordinary quaternions and dual numbers.

### 3.1 Quaternions

Quaternions extend complex numbers to four dimensions. A quaternion is written as:

$$\bar q = q + q_0, \qquad q \in \mathbb{R}^3,\ q_0 \in \mathbb{R}$$

or, in the classical notation:

$$\bar q = q_0 + i q_1 + j q_2 + k q_3$$

where `i, j, k` are imaginary units satisfying `i² = j² = k² = ijk = -1`. Equivalently, a quaternion can be split into:

- a **vector part** `q ∈ ℝ³` (analogous to the "imaginary" components), and
- a **scalar part** `q₀ ∈ ℝ` (the "real" component).

Quaternion multiplication is **not commutative** (`p ∘ q ≠ q ∘ p` in general) — order matters, just like it does for 3D rotations. Given `p = (p, p₀)` and `q = (q, q₀)`, the quaternion product can be written compactly in matrix form as:


$$
p \circ q =
\begin{bmatrix}
S(p) + p_0 I & p \\
-p^{\mathsf{T}} & p_0
\end{bmatrix}
\begin{bmatrix}
q \\
q_0
\end{bmatrix}
$$

where `S(·)` is the **skew-symmetric matrix** built from a vector such that `S(v) w = v × w` — i.e., matrix multiplication by `S(v)` reproduces the cross product with `v`.

Other key operations:
- **Conjugate:** `q* = (-q, q₀)` (flip the sign of the vector part).
- **Norm:** `‖q‖² = q ∘ q* = q* ∘ q` (a scalar).
- A vector `p ∈ ℝ³` can always be embedded as a *pure* quaternion `p̄ = (p, 0)` (zero scalar part).

### 3.2 Quaternions as Rotations

A **unit-norm quaternion** (`‖q‖ = 1`) represents a rotation in 3D space, analogous to how a unit complex number represents a rotation in 2D. If `q̄` is the unit quaternion rotating from body frame `b` to inertial frame `i`, a vector expressed in the body frame transforms to the inertial frame via the **sandwich product**:

$$\bar p^i = \bar q \circ \bar p^b \circ \bar q^*$$

Note that `q̄` and `-q̄` represent the *same* rotation — this "double cover" is a well-known and harmless property of quaternions.

If the body rotates with angular velocity `ω` (expressed in the body frame), the quaternion evolves according to:

$$\dot{\bar q} = \tfrac{1}{2}\, \bar q \circ \bar\omega$$

This single differential equation replaces the more cumbersome update equations needed for rotation matrices or Euler angles, and it has no singularities.

### 3.3 Dual Numbers

Before combining quaternions with position, we need one more ingredient: **dual numbers**. A dual number has the form:

$$\hat\alpha = a + \varepsilon b, \qquad a, b \in \mathbb{R}$$

where `ε` is a symbol (the *dual unit*) with the defining property:

$$\varepsilon \neq 0, \qquad \varepsilon^2 = 0$$

This is conceptually similar to how `i² = -1` defines complex numbers, except here squaring the special symbol gives **zero**, not `-1`. (Important: `ε` is *not* "a small number close to zero" — it's an abstract algebraic symbol.)

Dual numbers add component-wise, and multiply using the rule `ε² = 0`:

$$(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\,\varepsilon$$

Intuitively, `a` is the "principal" or nominal value, and `b` is a first-order "perturbation" or "derivative-like" term riding along with it. This structure is exactly what's needed to piggy-back *position* information onto *orientation* information.

### 3.4 Dual Quaternions

A **dual quaternion** is obtained by applying the same trick that built complex-like dual numbers from real numbers — but starting from quaternions instead of reals:

$$Q = \bar r + \varepsilon\, \bar s, \qquad \bar r, \bar s \in \mathbb{H}$$

where `ℍ` is the set of quaternions. `Q` is an 8-dimensional object (4 real dimensions from `r̄`, 4 more from `s̄`). We call:

- `𝒫(Q) = r̄` the **principal part**,
- `𝒟(Q) = s̄` the **dual part**.

Sum, product, and conjugation extend naturally from quaternions, always remembering `ε² = 0`. The dual quaternion conjugate is:

$$Q^* = \mathcal{P}(Q)^* + \varepsilon\, \mathcal{D}(Q)^*$$

### 3.5 Dual Quaternions as Rigid-Body Pose

Just as a **unit quaternion** represents pure rotation, a **unit dual quaternion** (`Q ∘ Q* = Q* ∘ Q = 1`) represents a full rigid-body **pose** — rotation *and* translation together. Concretely:

- `Q` has unit norm **if and only if** its principal part `𝒫(Q) = q̄` is itself a unit quaternion (the attitude), and its dual part is:

$$\mathcal{D}(Q) = \tfrac{1}{2}\, \bar p \circ \mathcal{P}(Q)$$

  where `p̄ = (p, 0)` is the position vector `p ∈ ℝ³`, embedded as a pure quaternion.

- Given any unit dual quaternion `Q`, you can always **recover** the attitude and position:

$$\bar q = \mathcal{P}(Q), \qquad \bar p = 2\,\mathcal{D}(Q) \circ \mathcal{P}(Q)^*$$

So a single object `Q = q̄ + ε·½(p̄ ∘ q̄)` carries everything needed to describe "where the drone is and how it's oriented" — and dual quaternion multiplication automatically composes both the rotations *and* the translations correctly, in the right order.

---

## 4. Vehicle Kinematics in Dual Quaternions

The **twist** (linear + angular velocity) of a vehicle is also packaged into a dual quaternion:

$$\Omega(\bar\omega, \bar v), \qquad \mathcal{P}(\Omega) = \bar\omega,\quad \mathcal{D}(\Omega) = \mathcal{P}(Q)^* \circ \bar v \circ \mathcal{P}(Q)$$

and the pose evolves in time according to a beautifully compact single equation:

$$\dot Q = \tfrac{1}{2}\, Q \circ \Omega(\bar\omega, \bar v)$$

This is the dual-quaternion analogue of `q̇ = ½ q ∘ ω̄` for pure rotation — it simultaneously propagates both position and attitude, using commanded body-frame angular velocity `ω` and inertial-frame linear velocity `v`. This is the equation the simulation integrates forward in time to move each drone.

---

## 5. Pose Error and the Control Objective

Let `Q` be the vehicle's **current** pose and `Q_d` the **desired** pose. The **pose error** is defined as a single dual quaternion:

$$\delta Q = Q_d^* \circ Q = \delta \bar q + \varepsilon\, \tfrac{1}{2}\left(\overline{\delta p}^{\,b} \circ \delta \bar q\right)$$

where:
- `δq̄ = q̄_d* ∘ q̄` is the **attitude error** (a quaternion — how far the current orientation is from the desired one),
- `δp = p − p_d` is the raw **position error**, and
- `δp̄ᵇ = q̄_d* ∘ δp̄ ∘ q̄_d` is that position error expressed **in the desired body frame** — the natural frame to regulate it in.

The **control goal** is simply: drive `δQ → 1` (the identity dual quaternion), i.e., make `(δq̄, δp) → (0, 0)`, meaning the vehicle's actual pose converges to the desired pose. Both position and attitude errors are captured by this one object.

---

## 6. The Kinematic Control Law

The controller (from Giribet et al., 2021, and reproduced/analyzed here) computes the **commanded twist** `(ω, v)` sent to the vehicle as a function of the pose error, using six gain matrices — all required to be **negative definite**:

$$K_{\omega,p},\ K_{v,p},\ K_{\omega,i},\ K_{v,i},\ K_\eta,\ K_\xi \in \mathbb{R}^{3\times3}$$

The control law is:

- **Angular velocity command:**
  $$\bar\omega = \delta\bar q^* \circ \bar\omega_d \circ \delta\bar q + \big(\text{sgn}(\delta q_0)(K_{\omega,p}\,\delta q + \eta_0 K_{\omega,i}\,\eta),\ 0\big)$$
- **Linear velocity command:**
  $$\bar v = \bar v_d + \mathcal{R}(\bar q_d)\big(K_{v,p}\,\delta p^b + K_{v,i}\,\xi\big)$$
- **Attitude integral term** `η` (with a "forgetting factor" so it doesn't wind up unboundedly):
  $$\dot{\bar\eta} = \tfrac{1}{2}\bar\eta \circ \big(-|\delta q_0| K_{\omega,i}\,\delta q + \text{sgn}(\eta_0) K_\eta\,\eta,\ 0\big)$$
- **Position integral term** `ξ`:
  $$\dot\xi = -K_{v,i}\,\delta p^b + K_\xi\,\xi$$

In words:
- The **feedforward** terms (`ω_d`, `v_d`) carry the desired trajectory's own velocity.
- The **proportional** terms (`K_{ω,p}`, `K_{v,p}`) push the vehicle toward the desired pose based on the instantaneous error.
- The **integral** terms (`η`, `ξ`, weighted by `K_{ω,i}`, `K_{v,i}`, `K_η`, `K_ξ`) accumulate error over time, improving steady-state tracking and rejecting persistent disturbances or model uncertainty — much like the "I" term in a PID controller.

**Theorem (paper's Theorem 1):** if all six gain matrices are negative definite, this control law drives the full error state `(δq̄, δp, ξ, η) → 0` using a Lyapunov argument combined with LaSalle's invariance principle. This is the formal guarantee that the controller works — but it does *not* say **how well** it works (fast/slow, oscillatory/smooth). That's the tuning problem addressed next.

---

## 7. Why Tuning Is Hard: The Linearized Error Dynamics

Adding the integral terms `η` and `ξ` helps tracking performance, but doubles the number of gain matrices to choose (`K_{ω,p}`, `K_{ω,i}`, `K_η` for attitude alone — position is analogous). Picking six 3×3 matrices by trial and error is impractical, so the paper linearizes the closed-loop error dynamics around the equilibrium `(δq, η) = (0, 0)`:

$$\begin{bmatrix}\dot{\delta q} \\ \dot\eta\end{bmatrix} = \underbrace{\tfrac{1}{2}\begin{bmatrix} K_{\omega,p} & K_{\omega,i} \\ -K_{\omega,i} & K_\eta \end{bmatrix}}_{M_\omega}\begin{bmatrix}\delta q \\ \eta\end{bmatrix}$$

(An identical structure `M_v` describes the linearized *position* error dynamics.)

The **eigenvalues of `M_ω`** (and `M_v`) determine the local speed, damping, and oscillatory behavior of the tracking error near the target pose — exactly like poles determine the behavior of a linear system in classical control theory. So: **choosing gains = placing the poles of `M_ω`/`M_v` in favorable locations.**

The catch: `M_ω` is a **block matrix**, and in general there's no simple relationship between the eigenvalues of a block matrix and the eigenvalues of its individual blocks (`K_{ω,p}`, `K_{ω,i}`, `K_η`). Naively picking each block to be "nice" (e.g., diagonal, negative definite) does **not** guarantee the assembled matrix `M_ω` has eigenvalues where you want them.

---

## 8. The Pole-Placement / Gain-Selection Method

The paper's key theoretical trick: `M_ω` turns out to be **self-adjoint with respect to an indefinite inner product**. Define:

$$H = \begin{bmatrix} I & 0 \\ 0 & -I \end{bmatrix}, \qquad [x, y]_H = x^\mathsf{T} H y$$

Then `M_ω` satisfies `H M_ω = M_ωᵀ H`, i.e., it is **H-self-adjoint**. Matrices with this property have well-studied spectral bounds (Gershgorin-type theorems), which the paper adapts to this specific control problem. Concretely, letting `a₋, a₊` be the extreme eigenvalues of `½K_{ω,p}` and `d₋, d₊` those of `½K_η`:

- **General bound:** the eigenvalues of `M_ω` lie inside the union/intersection of disks centered at the eigenvalues of `½K_{ω,p}` and `½K_η`, with radii scaled by the *coupling* term `K_{ω,i}` (specifically `‖K_{ω,p}^{-1}K_{ω,i}‖` and `‖K_η^{-1}K_{ω,i}‖`).
- **Real-eigenvalue guarantee:** if the "gap" between the `K_{ω,p}` and `K_η` spectra is large enough relative to the coupling gain `K_{ω,i}` (formally, `‖½K_{ω,i}‖ < k`, a computable threshold), **all eigenvalues of `M_ω` are guaranteed to be real and negative**, landing in two explicit intervals `I₁ ∪ I₂`. Real negative eigenvalues mean **non-oscillatory, exponentially-decaying tracking error** — the best-behaved response.
- If that gap condition is *not* met, eigenvalues may become **complex**, meaning the tracking error will show oscillations (underdamped-like behavior) as it converges.

**Practical recipe used in this project (and the paper):**
1. Pick the proportional gains `K_{ω,p}`, `K_{v,p}` first (e.g. by direct experimentation on the platform).
2. Choose the integral-related gains `K_{ω,i}`, `K_η`, `K_ξ` (and their position counterparts) while checking the eigenvalue bounds above.
3. Use the real-eigenvalue condition to deliberately choose gains that keep `M_ω`/`M_v` spectra **real and negative**, avoiding oscillatory transients.

This turns an opaque 6-matrix tuning problem into a **transparent, checkable eigenvalue-placement problem** — the paper's central practical contribution.

---

## 9. From Kinematics to Quadrotor Commands

The kinematic controller above outputs a desired **twist** `(ω, v)` — it assumes the vehicle can instantly achieve any commanded velocity. Real quadrotors, however, are underactuated: they only directly control four things — collective thrust and three torques (roll, pitch, yaw), realized through four command channels `(u_φ, u_θ, u_ż, u_ψ̇)`.

- `u_ż` (vertical velocity) and `u_ψ̇` (yaw rate) map **directly** from the kinematic twist commands `(ω, v)` computed in Section 6.
- `u_φ`, `u_θ` (roll and pitch angles) are **not** velocities — they are *attitude* commands that indirectly produce horizontal acceleration by tilting the thrust vector. These must be derived from the *desired acceleration*:

$$U_p = \ddot p_{jd} + K_a(\dot p_j - v)$$

  and then converted into a required attitude via a **thrust-vector-alignment** construction — rotating the body's thrust axis to align with the direction of `U_p`:

$$\bar q'_{j\phi,\theta} = \big(n_j \times U_p,\ \langle n_j, U_p\rangle + \|U_p\|\big), \qquad \bar q_{j\phi,\theta} = \bar q'_{j\phi,\theta} / \|\bar q'_{j\phi,\theta}\|$$

  This gives the roll/pitch part of the desired attitude, independent of yaw. Yaw is controlled separately via a desired heading quaternion `q̄_{jψ}`, and the two are composed:

$$\bar q_{jd} = \bar q_{j\phi,\theta} \circ \bar q_{j\psi}$$

This two-layer structure — a kinematic dual-quaternion controller on top, and a physical thrust/attitude mapping underneath — mirrors how real multirotor autopilots are structured, and is what this simulation replicates to produce realistic, physically consistent motion.

---

## 10. Leader-Follower Formation Law

With the single-vehicle controller established, the formation law simply defines **what pose each vehicle should track**:

- **Leader:** follows an analytically defined trajectory $p_{Ld}(t)$ (position) with a freely-chosen attitude $q_{Ld}(t)$, since the platform is omnidirectional in attitude. Together these form the leader's desired dual quaternion:
  $$Q_{Ld} = \bar q_{Ld}(t) + \varepsilon\,\tfrac{1}{2}\, \bar p_{Ld}(t) \circ \bar q_{Ld}(t)$$
- **Follower:** tracks the leader's *measured* (not planned) position, offset by a fixed or time-varying vector `f(t)` that keeps the two vehicles safely apart:
  $$p_{Fd}[k] = p_L[k] + f(t)$$
  The follower's desired attitude can either be set independently or copied from the leader (the latter is what the paper's experiments use: $q̄_{Fd} = q̄_L$).

Because the follower's desired trajectory is derived by (numerically) differentiating noisy real-time measurements of the leader, the **follower's tracking is inherently noisier** than the leader's — an effect visible in both the paper's and this project's results (see Section 13).

Each vehicle (leader `L` and follower `F`) then runs its **own independent copy** of the Section 6 controller against its own desired dual quaternion:

$$\delta Q_j = Q_{jd}^* \circ Q_j, \qquad j \in \{L, F\}$$

---
