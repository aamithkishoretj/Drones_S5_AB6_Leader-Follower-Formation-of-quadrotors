<p align="center">
  <img src="https://github.com/user-attachments/assets/060f7774-a73f-4132-9413-36887ed09cfa" alt="Amrita Vishwa Vidyapeetham" width="430">
</p>

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
11. [Repository Structure](#11-repository-structure)
12. [Folder-by-Folder Walkthrough](#12-folder-by-folder-walkthrough)
    - 12.1 [dq_control/ — the math and control library](#121-dq_control--the-math-and-control-library)
    - 12.2 [envs/ — the simulator bridge](#122-envs--the-simulator-bridge)
    - 12.3 [scripts/ — command-line entry points](#123-scripts--command-line-entry-points)
    - 12.4 [utils/ — logging and metrics](#124-utils--logging-and-metrics)
    - 12.5 [configs/](#125-configs)
    - 12.6 [tests/](#126-tests)
13. [Installation & Setup](#13-installation--setup)
14. [How to Run an Experiment](#14-how-to-run-an-experiment)
15. [Visualizing Results and Metrics](#15-visualizing-results-and-metrics)
16. [Running the Tests](#16-running-the-tests)
17. [Design Notes: From Kinematic Twist to Motor Commands](#17-design-notes-from-kinematic-twist-to-motor-commands)
18. [Quick Reference](#18-quick-reference)

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

Let `Q` be the vehicle's **current** pose and $Q_d$ the **desired** pose. The **pose error** is defined as a single dual quaternion:

$$\delta Q = Q_d^* \circ Q = \delta \bar q + \varepsilon\, \tfrac{1}{2}\left(\overline{\delta p}^{\,b} \circ \delta \bar q\right)$$

where:
- $\delta \bar{q} = \bar{q}_d^* \circ \bar{q}$ is the **attitude error** (a quaternion — how far the current orientation is from the desired one),
- $\delta p = p - p_d$ is the raw **position error**, and
- $\delta \bar{p}^{\,b} = \bar{q}_d^* \circ \delta \bar{p} \circ \bar{q}_d$ is that position error expressed **in the desired body frame** — the natural frame to regulate it in.

The **control goal** is simply: drive `δQ → 1` (the identity dual quaternion), i.e., make `(δq̄, δp) → (0, 0)`, meaning the vehicle's actual pose converges to the desired pose. Both position and attitude errors are captured by this one object.

---

## 6. The Kinematic Control Law

The controller computes the **commanded twist** $(\bar{\omega}, \bar{v})$ sent to the vehicle as a function of the pose error. It uses six gain matrices:

$$K_{\omega,p},\; K_{v,p},\; K_{\omega,i},\; K_{v,i},\; K_{\eta},\; K_{\xi} \in \mathbb{R}^{3\times3}$$

For the stability result, these gain matrices are required to be **negative definite**.

### Angular velocity command

$$\bar{\omega} = \delta\bar{q}^{\,*} \circ \bar{\omega}_d \circ \delta\bar{q} + \left( \mathrm{sgn}(\delta q_0) \left( K_{\omega,p}\delta q + \eta_0 K_{\omega,i}\eta \right), \, 0 \right)$$

Here:

- $\bar{\omega}$ is the commanded angular velocity represented as a pure quaternion.
- $\bar{\omega}_d$ is the desired angular velocity.
- $\delta\bar{q}$ is the attitude-error quaternion.
- $\delta q$ is the vector part of $\delta\bar{q}$.
- $\delta q_0$ is the scalar part of $\delta\bar{q}$.
- $\eta$ is the vector part of the attitude integral state.
- $\eta_0$ is the scalar part of the attitude integral quaternion $\bar{\eta}$.

### Linear velocity command

$$\bar{v} = \bar{v}_d + \mathcal{R}(\bar{q}_d) \left( K_{v,p}\delta p^{\,b} + K_{v,i}\xi \right)$$

Here, $\delta p^{\,b}$ is the position error expressed in the desired body frame.

### Attitude integral state

$$\dot{\bar{\eta}} = \frac{1}{2} \bar{\eta} \circ \left( -|\delta q_0| K_{\omega,i}\delta q + \mathrm{sgn}(\eta_0) K_{\eta}\eta, \, 0 \right)$$

The $K_{\eta}\eta$ term acts as a **forgetting/leakage term**, preventing the integral state from growing without bound.

### Position integral state

$$\dot{\xi} = -K_{v,i}\delta p^{\,b} + K_{\xi}\xi$$

The first term accumulates the position error, while $K_{\xi}\xi$ provides the corresponding forgetting mechanism.

### Interpretation

- The **feedforward** terms $\bar{\omega}_d$ and $\bar{v}_d$ provide the velocity required by the desired trajectory.
- The **proportional** terms $K_{\omega,p}\delta q$ and $K_{v,p}\delta p^{\,b}$ respond to the instantaneous pose error.
- The **integral** terms involving $\eta$ and $\xi$ accumulate persistent tracking errors.
- The $K_{\eta}$ and $K_{\xi}$ terms prevent excessive accumulation of the integral states.

The closed-loop error states converge toward the desired equilibrium:

$$\delta\bar{q}\rightarrow\bar{1}, \qquad \delta p\rightarrow0, \qquad \eta\rightarrow0, \qquad \xi\rightarrow0$$

Here, $\bar{1}$ denotes the **identity quaternion**, representing zero attitude error.

---

## 7. Why Tuning Is Hard: The Linearized Error Dynamics

Adding the integral states $\eta$ and $\xi$ improves steady-state tracking, but it also introduces additional gain matrices.

For the attitude subsystem, the relevant gains are

$$K_{\omega,p}, \qquad K_{\omega,i}, \qquad K_{\eta}$$

The closed-loop attitude dynamics are linearized around the equilibrium

$$(\delta q,\eta)=(0,0)$$

The resulting linearized system is

$$\begin{bmatrix} \dot{\delta q}\\\\ \dot{\eta} \end{bmatrix} = M_{\omega} \begin{bmatrix} \delta q\\\\ \eta \end{bmatrix}$$

where

$$M_{\omega} = \frac{1}{2} \begin{bmatrix} K_{\omega,p} & K_{\omega,i}\\\\ -K_{\omega,i} & K_{\eta} \end{bmatrix}$$

Since each gain matrix is $3\times3$, $M_{\omega}$ is a $6\times6$ matrix.

An analogous matrix $M_v$ describes the linearized position-error dynamics.

For a linear system

$$\dot{x}=Mx$$

the eigenvalues of $M$ determine the local behavior of the system.

If

$$\lambda<0$$

the corresponding mode decays exponentially.

If

$$\lambda=\alpha\pm j\beta, \qquad \alpha<0$$

the corresponding mode decays while oscillating.

Therefore:

- **Real negative eigenvalues** → non-oscillatory exponential convergence.
- **Complex eigenvalues with negative real parts** → damped oscillatory behavior.
- **Eigenvalues with positive real parts** → instability.

Consequently, selecting the controller gains can be viewed as a structured **pole-placement problem**.

The difficulty is that $M_{\omega}$ is a block matrix. Its eigenvalues cannot, in general, be determined simply by considering the eigenvalues of the individual matrices

$$K_{\omega,p}, \qquad K_{\omega,i}, \qquad K_{\eta}$$

Therefore, making each gain matrix individually negative definite does not by itself determine the exact eigenvalues of the complete matrix $M_{\omega}$.

---

## 8. The Pole-Placement / Gain-Selection Method

The linearized matrix $M_{\omega}$ has a special spectral structure.

Define

$$H = \begin{bmatrix} I_3 & 0\\\\ 0 & -I_3 \end{bmatrix}$$

and the corresponding indefinite inner product

$$[x,y]_H = x^{\mathsf T}Hy$$

The matrix $M_{\omega}$ satisfies

$$HM_{\omega} = M_{\omega}^{\mathsf T}H$$

Therefore, $M_{\omega}$ is **$H$-self-adjoint**.

Define

$$a_- = \lambda_{\min}\left(\frac{1}{2}K_{\omega,p}\right), \qquad a_+ = \lambda_{\max}\left(\frac{1}{2}K_{\omega,p}\right)$$

and

$$d_- = \lambda_{\min}\left(\frac{1}{2}K_{\eta}\right), \qquad d_+ = \lambda_{\max}\left(\frac{1}{2}K_{\eta}\right)$$

The coupling between the proportional and integral dynamics is determined by $K_{\omega,i}$.

The spectral bounds therefore depend on quantities such as

$$\left\| K_{\omega,p}^{-1}K_{\omega,i} \right\|$$

and

$$\left\| K_{\eta}^{-1}K_{\omega,i} \right\|$$

If the coupling introduced by $K_{\omega,i}$ is sufficiently small compared with the separation between the spectra associated with $K_{\omega,p}$ and $K_{\eta}$, the eigenvalues of $M_{\omega}$ can be guaranteed to remain real and negative:

$$\lambda_i(M_{\omega})\in\mathbb{R}, \qquad \lambda_i(M_{\omega})<0$$

This produces a locally exponentially convergent and non-oscillatory response.

If the coupling becomes too strong, complex eigenvalues can appear:

$$\lambda_{1,2} = \alpha\pm j\beta, \qquad \alpha<0$$

which corresponds to damped oscillatory behavior.

### Practical Gain-Selection Procedure

1. Select the proportional gains:

   $$K_{\omega,p}, \qquad K_{v,p}$$

2. Select the integral-related gains:

   $$K_{\omega,i}, \qquad K_{\eta}, \qquad K_{v,i}, \qquad K_{\xi}$$

3. Construct the linearized attitude matrix:

   
```math
M_{\omega} = \frac{1}{2} \begin{bmatrix} K_{\omega,p} & K_{\omega,i}\\ -K_{\omega,i} & K_{\eta} \end{bmatrix}
```

4. Construct the corresponding position matrix $M_v$.

5. Compute their eigenvalues:

   $$\lambda(M_{\omega}), \qquad \lambda(M_v)$$

6. Select gains that produce eigenvalues with negative real parts. When a smooth non-oscillatory response is desired, select gains satisfying the conditions that keep the eigenvalues real and negative.

---

## 9. From Kinematics to Quadrotor Commands

The kinematic controller produces the desired twist

$$(\bar{\omega},\bar{v})$$

A real quadrotor cannot directly generate an arbitrary three-dimensional velocity. Its motion is produced through thrust and attitude control.

The command variables considered in this project are

$$u_{\phi}, \qquad u_{\theta}, \qquad u_{\dot z}, \qquad u_{\dot\psi}$$

corresponding to roll, pitch, vertical velocity, and yaw-rate commands.

The horizontal motion requires an additional mapping because a quadrotor generates horizontal acceleration by **tilting its thrust vector**.

### Desired Acceleration

$$U_p = \ddot{p}_{jd} + K_a\left(\dot{p}_j-v\right)$$

Here:

- $p_{jd}$ is the desired position of vehicle $j$.
- $\dot{p}_{jd}$ is the desired velocity.
- $\ddot{p}_{jd}$ is the desired acceleration.
- $\dot{p}_j$ is the actual velocity.
- $v$ is the velocity generated by the kinematic controller.
- $K_a$ is the acceleration-level feedback gain.

### Thrust-Vector Alignment

Let $n_j$ denote the thrust axis of vehicle $j$.

The roll-pitch quaternion is constructed as

$$\bar{q}'_{j\phi,\theta} = \left( n_j\times U_p,\, \left\langle n_j,U_p\right\rangle + \|U_p\| \right)$$

It is then normalized:

$$\bar{q}_{j\phi,\theta} = \frac{\bar{q}'_{j\phi,\theta}}{\left\|\bar{q}'_{j\phi,\theta}\right\|}$$

This quaternion represents the rotation required to align the vehicle's thrust direction with the desired acceleration direction.

### Yaw Composition

The roll-pitch orientation does not uniquely determine yaw. Therefore, yaw is specified independently using the desired yaw quaternion $\bar{q}_{j\psi}$.

The complete desired attitude is

$$\bar{q}_{jd} = \bar{q}_{j\phi,\theta} \circ \bar{q}_{j\psi}$$

The complete control structure is therefore:

```
Pose error → Kinematic controller → (ω̄, v̄) → Acceleration/attitude mapping → Quadrotor commands
```

---

## 10. Leader-Follower Formation Law

The formation controller defines the desired pose that each vehicle should track.

Let

$$j\in\{L,F\}$$

where $L$ denotes the leader and $F$ denotes the follower.

### Leader

The leader follows a predefined position trajectory

$$p_{Ld}(t)$$

and a desired attitude

$$\bar{q}_{Ld}(t)$$

The leader's desired pose is represented by the dual quaternion

$$Q_{Ld} = \bar{q}_{Ld}(t) + \varepsilon\frac{1}{2}\bar{p}_{Ld}(t) \circ \bar{q}_{Ld}(t)$$

where $\varepsilon$ is the dual unit satisfying

$$\varepsilon^2=0$$

Thus, $Q_{Ld}$ simultaneously represents the desired position and orientation of the leader.

### Follower

The follower's desired position is generated from the leader's measured position and a formation offset:

$$p_{Fd}(t) = p_L(t)+f(t)$$

Here:

- $p_L(t)$ is the measured leader position.
- $f(t)$ is the desired formation offset.
- $p_{Fd}(t)$ is the resulting desired follower position.

For example, if the follower should remain $2\,\mathrm{m}$ behind the leader:

$$f(t) = \begin{bmatrix} -2\\\\ 0\\\\ 0 \end{bmatrix}$$

The follower's desired attitude can be specified independently or chosen to follow the leader:

$$\bar{q}_{Fd}(t) = \bar{q}_L(t)$$

### Effect of Measurement Noise

Because the follower's desired position depends on the leader's measured position, measurement noise is directly transferred to the follower reference.

If

$$p_L^m(t) = p_L(t)+n(t)$$

then

$$p_{Fd}^m(t) = p_L^m(t)+f(t) = p_L(t)+f(t)+n(t)$$

Therefore, the follower's reference already contains the leader's measurement noise.

If velocity or acceleration is subsequently obtained using numerical differentiation, high-frequency measurement noise can be amplified further. This explains why the follower's trajectory can appear noisier than the leader's trajectory.

### Pose Error for Each Vehicle

Each vehicle runs its own copy of the kinematic controller.

For

$$j\in\{L,F\}$$

the dual-quaternion pose error is

$$\delta Q_j = Q_{jd}^{*} \circ Q_j$$

Therefore,

$$\delta Q_L = Q_{Ld}^{*} \circ Q_L$$

for the leader, and

$$\delta Q_F = Q_{Fd}^{*} \circ Q_F$$

for the follower.

The overall leader-follower structure is therefore:

```
Leader trajectory → Leader desired pose → Leader controller → Leader motion
```

```
Leader measured pose + Formation offset → Follower desired pose → Follower controller → Follower motion
```

---

## 11. Repository Structure

```
Dual-Quaternion-Based-Control-for-a-Leader-Follower-Formation-of-Two-Quadrotors/
├── README.md                      # this file
├── requirements.txt                # Python dependencies
├── configs/
│   └── experiment_config.yaml       # human-readable mirror of the paper's Sec. V parameters
├── dq_control/                      # core math + control library (no simulator dependency)
│   ├── __init__.py
│   ├── quaternion.py                 # Quaternion algebra (Sec. 3.1-3.2 above)
│   ├── dual_quaternion.py            # DualQuaternion / Twist, pose kinematics (Sec. 3.3-3.5, 4)
│   ├── controller.py                 # KinematicController: the eq. (5)-(9) control law (Sec. 6)
│   ├── gains.py                      # the three paper gain sets (Sec. 7-8, "proportional" / "complex_eig" / "real_eig")
│   └── trajectories.py               # LeaderTrajectory, FollowerTrajectory, PotatoChipTrajectory (Sec. 10)
├── envs/
│   ├── __init__.py
│   └── leader_follower_sim.py        # bridges dq_control to gym-pybullet-drones (Sec. 9)
├── scripts/
│   ├── run_experiment.py             # CLI entry point: runs one leader-follower experiment
│   └── plot_results.py               # reproduces the paper's tracking plots + MAE/MSE tables
├── utils/
│   ├── __init__.py
│   ├── logger.py                     # save_run / load_run (.npz persistence)
│   └── metrics.py                    # MAE/MSE position & attitude error metrics (paper Tables I-IV)
└── tests/
    └── test_dual_quaternion.py       # unit tests for the algebra + closed-loop convergence
```

The design deliberately separates **pure math/control** (`dq_control/`, no simulator dependency, fully unit-testable) from the **simulator glue** (`envs/`), so the controller itself can be reused, tested, or ported to another simulator/drone platform without touching the dual-quaternion or control-law code.

---

## 12. Folder-by-Folder Walkthrough

### 12.1 `dq_control/` — the math and control library

This package implements everything in Sections 3-8 above, independent of any simulator.

- **`quaternion.py`** — the `Quaternion` class (Sec. 3.1-3.2): Hamilton product (`*`), conjugate (`.conj()`), norm/normalization, `rotate()` (the sandwich product `q ∘ p ∘ q*`), conversions to/from rotation matrices and roll-pitch-yaw, and `from_rotvec()` for building a unit quaternion from an axis-angle rotation.
- **`dual_quaternion.py`** — the `DualQuaternion` class (Sec. 3.3-3.5): `from_pose(position, attitude)` builds a unit dual quaternion from a position vector and an attitude quaternion; `.position()` / `.attitude()` recover them back; `Twist` packages angular + linear velocity into the dual-quaternion twist object from Sec. 4; `pose_derivative()` and `integrate_pose()` implement `Q̇ = ½ Q ∘ Ω` and its one-step Euler integration.
- **`controller.py`** — `pose_error()` computes `δQ = Q_d* ∘ Q` (Sec. 5); `ControllerGains` holds the six negative-definite 3×3 gain matrices; `ControllerState` carries the integral terms `(ξ, η)` between steps; `KinematicController.compute()` evaluates the full eq. (5)-(9) control law from Sec. 6 each call and returns `(omega_cmd, v_cmd)`.
- **`gains.py`** — the three concrete gain sets used in the paper's Section V experiments (see Sec. 7-8 above): `gains_proportional()` (integral gains zeroed out), `gains_complex_eig()` (integral gains chosen so the linearized error dynamics `M_ω`, `M_v` have complex eigenvalues → oscillatory convergence), and `gains_real_eig()` (integral gains chosen so `M_ω`, `M_v` have only real eigenvalues → the paper's best-performing, non-oscillatory tuning). `get_gains(name)` looks one up by name; `EXPERIMENTS` is the `{name: builder_fn}` registry used by the CLI's `--experiment` flag.
- **`trajectories.py`** — reference trajectories (Sec. 10 above):
  - `LeaderTrajectory` — the paper's simplified Lemniscate (figure-eight), with yaw always tangent to the direction of travel.
  - `PotatoChipTrajectory` — an additional, non-paper trajectory: a circle in `(x, y)` combined with a `cos(k·θ)` ripple in `z`, tracing a saddle ("Pringle chip") shape. Useful for stress-testing the controller and the follower's formation-offset logic on a genuinely 3D, curving path.
  - `FollowerTrajectory` — builds the follower's desired pose from the leader's *measured* pose plus an offset (paper eq. 16). Supports two offset modes: `'world'` (a fixed offset vector in the world frame, exactly as in the paper — works well for trajectories that mostly move along one axis, like the lemniscate) and `'body'` (the offset is rotated into the leader's current heading so the follower always trails directly behind it — needed for a curving path like `potato_chip`). It also includes optional velocity/heading smoothing to reduce the noise introduced by numerically differentiating a measured signal.

### 12.2 `envs/` — the simulator bridge

- **`leader_follower_sim.py`** — the `LeaderFollowerSim` class connects the `dq_control` package to [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones), which provides the physics simulation and a low-level PID attitude/position controller (`DSLPIDControl`) that stands in for the real Bebop 2's onboard firmware. Each control step runs the three-stage cascade described in Section 9 above: the kinematic controller computes a twist, `integrate_pose()` turns that twist into a one-step-ahead target pose, and `DSLPIDControl` converts that target pose into motor RPMs. The file also auto-detects which generation of the `gym-pybullet-drones` API is installed (old `gym`-based vs. new `gymnasium`-based) and normalizes the differences, so the rest of the codebase doesn't need to care which version is present. `SimConfig` is the dataclass holding all simulation parameters (duration, control/physics rates, drone model, follower-offset settings, GUI on/off).

### 12.3 `scripts/` — command-line entry points

- **`run_experiment.py`** — the main CLI. Selects a gain set (`--experiment`) and a leader trajectory (`--trajectory`), builds a `LeaderFollowerSim`, runs it, and saves the resulting log to a timestamped `.npz` file via `utils.save_run`. See Section 14 below for the full flag reference.
- **`plot_results.py`** — loads a saved `.npz` run, reproduces the paper-style trajectory and per-axis tracking/error plots, prints MAE/MSE tables matching the paper's Tables I-IV, and saves the figures as PNGs next to the run file.

### 12.4 `utils/` — logging and metrics

- **`logger.py`** — `save_run(log, experiment_name, output_folder)` writes the run's logged arrays to a timestamped `.npz` file; `load_run(path)` reloads them into a plain dict.
- **`metrics.py`** — `position_error_metrics()` computes MAE/MSE of the Euclidean position error (paper Tables I-II); `attitude_error_metrics()` computes per-axis (roll/pitch/yaw) MAE/MSE of angle-wrapped attitude error (paper Tables III-IV).

### 12.5 `configs/`

- **`experiment_config.yaml`** — a human-readable reference mirroring the numeric parameters actually used in `dq_control/gains.py` and `dq_control/trajectories.py` (trajectory shapes, simulation rates, and a one-line description of each of the three experiments). The Python modules are the source of truth; this file exists for bookkeeping and as a template for anyone wiring up their own config-driven CLI.

### 12.6 `tests/`

- **`test_dual_quaternion.py`** — `pytest` unit tests covering: a 90° quaternion rotation about `z`, that a unit quaternion's conjugate is its inverse, a position/attitude round-trip through `DualQuaternion.from_pose()`, pure-translation integration via `integrate_pose()`, closed-loop convergence of `KinematicController` to a static setpoint for all three gain sets, and that `LeaderTrajectory` is periodic with the expected period `2π/w_d`.

---

## 13. Installation & Setup

### 13.1 Requirements

The core math/control library (`dq_control/`) only needs the packages in `requirements.txt`:

```
numpy>=1.23,<2.0
scipy>=1.9
pyyaml>=6.0
matplotlib>=3.6
pandas>=1.5
pytest>=7.2
```

Install them with:

```bash
pip install -r requirements.txt
```

### 13.2 `gym-pybullet-drones` (only needed to run full simulations)

`envs/leader_follower_sim.py` — and therefore `scripts/run_experiment.py` — depends on [`gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) for the physics simulation and low-level PID controller. It is **not** a `pip`-installable dependency listed in `requirements.txt`; install it separately, either as a sibling directory next to this repository or as an editable package:

```bash
# from the parent directory of this repo
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones
pip install -e .
```

`leader_follower_sim.py` looks for `gym-pybullet-drones` first as a normally importable package, and falls back to a `../gym-pybullet-drones` sibling directory relative to this repo if it isn't found — so either the editable-install method above or simply cloning it next to this project's folder works.

The bridge code also auto-detects whether the old (`gym`-based, package version `0.6.0`/`v1.0.0`) or new (`gymnasium`-based, `>=2.0`) API of `gym-pybullet-drones` is installed and adapts automatically (differing `reset()`/`step()` signatures, action-space container types, etc.), so either generation of the dependency should work without code changes.

### 13.3 Verifying the install

The math/control library can be exercised without the simulator dependency at all:

```bash
pytest tests/
```

If these tests pass, `dq_control/` is working correctly; `gym-pybullet-drones` is only required for the next step.

---

## 14. How to Run an Experiment

The main entry point is `scripts/run_experiment.py`, run from the repository root:

```bash
python scripts/run_experiment.py --experiment real_eig --trajectory lemniscate
```

This reproduces the paper's best-performing (real-eigenvalue) gain tuning on the figure-eight Lemniscate trajectory, for 30 simulated seconds, and saves the results to `results/real_eig_lemniscate_<timestamp>.npz`.

### 14.1 Key flags

| Flag | Choices / type | Default | Meaning |
|---|---|---|---|
| `--experiment` | `proportional`, `complex_eig`, `real_eig` | *(required)* | Which of the paper's three Sec. V gain sets to use (see Sections 7-8 above). |
| `--trajectory` | `lemniscate`, `potato_chip` | `lemniscate` | Shape the leader flies: the paper's figure-eight, or the additional saddle/"potato chip" curve. |
| `--duration` | float (s) | `30.0` | Simulation length. |
| `--pyb_freq` | int (Hz) | `240` | Physics-engine step rate. |
| `--ctrl_freq` | int (Hz) | `48` | Kinematic-controller / low-level-PID step rate. |
| `--gui` | flag | off | Show the live PyBullet visualization. |
| `--output` | path | `results/` | Where the `.npz` run file is written. |
| `--x_offset` | float (m) | `1.85` | Follower's formation offset magnitude (paper eq. 16). |
| `--follower_offset_mode` | `world`, `body`, `auto` | `auto` | `world` = fixed offset in the world frame (paper-exact, best for the lemniscate); `body` = offset rotated into the leader's current heading so the follower trails directly behind it (needed for curved paths); `auto` picks `world` for `lemniscate` and `body` for `potato_chip`. |
| `--follower_heading_source` | `velocity`, `attitude` | `velocity` | (only with `body` mode) whether the follower's trailing direction is derived from the leader's smoothed measured velocity (robust to yaw-tracking lag) or its measured yaw directly. |
| `--start_x`, `--start_y`, `--start_z` | float (m) | trajectory default | Rigidly shift the whole trajectory so the leader starts at an exact point, without changing its size, speed, or shape. |

Trajectory-shape flags are grouped separately:

- **Lemniscate** (`--trajectory lemniscate`): `--r_x`, `--r_y`, `--w_d`, `--x0`, `--y0`, `--z0` — matches the parameters in Section 10 above.
- **Potato chip** (`--trajectory potato_chip`): `--chip_r`, `--chip_w`, `--chip_z_amp`, `--chip_k`, `--chip_phase`, `--chip_x0`, `--chip_y0`, `--chip_z0`.

Run `python scripts/run_experiment.py --help` for the complete, authoritative list (including help text) directly from the CLI.

### 14.2 Reproducing all three paper experiments

```bash
python scripts/run_experiment.py --experiment proportional --trajectory lemniscate
python scripts/run_experiment.py --experiment complex_eig  --trajectory lemniscate
python scripts/run_experiment.py --experiment real_eig     --trajectory lemniscate
```

Each run is saved as its own timestamped `.npz` file, so all three can be plotted and compared afterward.

### 14.3 Trying the non-paper "potato chip" trajectory

```bash
python scripts/run_experiment.py --experiment real_eig --trajectory potato_chip --gui
```

`--follower_offset_mode` defaults to `auto`, which switches to `body` mode here automatically, since a fixed world-frame offset would not keep the follower trailing the leader around a curving, non-axis-aligned path.

---

## 15. Visualizing Results and Metrics

Once a run has been saved, pass its `.npz` path to `plot_results.py`:

```bash
python scripts/plot_results.py --run results/real_eig_lemniscate_20260101_120000.npz
```

This will:

1. Plot the leader's and follower's actual vs. desired `(X, Y)` ground tracks on one figure.
2. Plot per-axis (`X`, `Y`, `Z`) desired-vs-actual position and tracking error, for the leader and the follower separately (reproducing the paper's Fig. 6-14 style plots).
3. Print MAE/MSE tables to the console: Euclidean position error for each vehicle (paper Tables I-II), and per-axis roll/pitch/yaw attitude error (paper Tables III-IV).
4. Save every figure as a `.png` next to the run file.

Add `--no-show` to skip the interactive `plt.show()` window (useful when generating plots on a headless machine or in a batch script).

---

## 16. Running the Tests

```bash
pytest tests/
```

covers the quaternion/dual-quaternion algebra, pose integration, closed-loop convergence of the controller for all three gain sets, and periodicity of the leader trajectory. These tests only depend on `dq_control/` and `numpy`/`pytest` — no simulator install is required.

---

## 17. Design Notes: From Kinematic Twist to Motor Commands

`envs/leader_follower_sim.py` implements the three-stage cascade summarized in Section 9 above, concretely as:

1. **`dq_control.controller.KinematicController.compute(...)`** evaluates the paper's eq. (5)-(9) control law and returns a kinematic twist command `(omega_cmd, v_cmd)`.
2. **`dq_control.dual_quaternion.integrate_pose(...)`** integrates that twist forward by one control timestep (eq. 3) to produce a next-step target position and attitude. This stands in for the instantaneous position/attitude setpoint that the real Bebop 2's onboard firmware would track, given the same velocity/attitude-rate commands.
3. **`gym_pybullet_drones.control.DSLPIDControl.computeControlFromState(...)`** consumes that target pose (plus the commanded velocity and body rates as feed-forward) and outputs the four motor RPMs, standing in for the Bebop's own low-level controller.

This mirrors how real multirotor autopilots are typically layered — a high-level kinematic/guidance controller on top, and a physical attitude/thrust controller underneath — and is what lets this simulation produce realistic, physically consistent motion despite the paper's controller itself being purely kinematic (Section 6 above notes it assumes the vehicle can achieve any commanded twist instantaneously).

Because the follower's desired trajectory (`FollowerTrajectory`) is built from the leader's *measured*, noisy state rather than an analytic function of time, its velocity feed-forward requires differentiating a noisy signal — which is why `FollowerTrajectory` supports the `vel_smoothing`, `heading_smoothing`, and `offset_mode`/`heading_source` options described in Section 12.1, all aimed at controlling how much of that measurement noise propagates into the follower's control commands.

---

## 18. Quick Reference

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# 2. Install the simulator dependency (sibling directory or editable install)
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones && pip install -e . && cd ..

# 3. Sanity-check the math/control library (no simulator needed)
pytest tests/

# 4. Run the paper's best-performing experiment on the figure-eight trajectory
python scripts/run_experiment.py --experiment real_eig --trajectory lemniscate --gui

# 5. Plot the results and print the MAE/MSE tables
python scripts/plot_results.py --run results/real_eig_lemniscate_<timestamp>.npz
```
