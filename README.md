<p align="center">
  <img src="https://github.com/user-attachments/assets/060f7774-a73f-4132-9413-36887ed09cfa" alt="Amrita Vishwa Vidyapeetham" width="430">
</p>

# AB6: Introduction to Drones — Leader-Follower Formation of Two Quadrotors

**Data-driven control is now the default command-line workflow.** Simulator
state/action/next-state recordings train a separate response model for each
backend. A predictive optimizer uses that model and dual-quaternion pose error
to choose commands during every flight step. The original controller is retained
as `--controller analytic` for data collection and comparisons.
See [training, results, limitations, and launch commands](docs/data_driven.md).

**Desktop simulation UI:** Launch the interactive control panel with
`python scripts/run_ui.py`. The compact panel launches separate simulator
windows; it no longer attempts to embed native windows. PyBullet, MuJoCo and
Gazebo offer learned control with a trained response model. Kinematic mode
uses analytic control and a separate 3D reference plot. ArduPilot documentation
and results will be added after its separate integration is merged.
Leader speed, follower gap, route radius, and flight altitude can be adjusted
while the simulation runs; applied values are displayed as they change.
See [desktop control details](docs/ui_controls.md). On Windows, Gazebo and ROS run inside WSL (Ubuntu 24.04 by
default); see [Gazebo setup](gazebo/README.md) for the required packages and
WSL settings.
Train first with `python scripts/train_data_driven.py --simulator pybullet`,
then run `python scripts/run_experiment.py --simulator pybullet --gui`.

**Supporting course concepts:** See [the runnable guide](docs/course_concepts.md)
for B-spline and LERP/SLERP references, optional position Kalman filtering,
CVXPY path smoothing, hover stability/prediction, and Taylor/RK45 examples.
The original dual-quaternion controller and paper gain sets are retained.
Use `--simulator kinematic --controller analytic` for an ideal control-law demonstration without a
physics engine; the existing physics backends remain available.

**Follow the leader's actual route (new default):** the follower targets a point
0.8 m behind along the leader's recorded track, instead of a translated copy
of the trajectory. GUI runs draw red leader and blue follower trails.
Use `--follow_distance 0.8` to set the along-path gap, `--no-trails` to hide
lines, or `--follower_offset_mode world` for the original paper formation.
See [path-following details](docs/path_following.md).

Sections 3–9 retain the mathematical foundation and **original analytic baseline**.
Section 10 explains the current path-following reference; Section 21 derives the
**learned predictive controller**. Measured simulator results are collected at the end in Section 23. The project began as a simulation of
**dual quaternion-based control for a leader-follower formation of two quadrotors**, built to reproduce and explore the ideas from:

> H. N. Marciano, D. K. D. Villa, M. Sarcinelli-Filho, J. I. Giribet, *"Dual Quaternion-Based Control for a Leader-Follower Formation of Two Quadrotors,"* 2024 International Conference on Unmanned Aircraft Systems (ICUAS), Chania, Crete, Greece, June 4–7, 2024.

This document is written so that someone with **no prior background** in quaternions, dual numbers, or formation control can follow the math and the control design step by step, and then map each concept directly onto the code in this repository.

This expanded edition additionally works every core equation through **fully numeric, hand-checkable examples** (verified with a small NumPy script) and includes **diagrams and plots** generated directly from the same formulas, so you can see — not just read — what each piece of math does.

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
    - 12.2 [simulators/ — the simulator bridge](#122-simulators--the-simulator-bridge)
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
19. [Appendix A: Full End-to-End Numeric Walkthrough](#19-appendix-a-full-end-to-end-numeric-walkthrough)
20. [Appendix B: Notation Glossary](#20-appendix-b-notation-glossary)
21. [Data-Driven Dual-Quaternion Control](#21-data-driven-dual-quaternion-control)
22. [Current Workflow and Course Concepts](#22-current-workflow-and-course-concepts)
23. [Results by Simulation](#23-results-by-simulation)

---

## 1. Project Overview

Coordinating multiple drones is a core problem in robotics: think of a "leader" drone flying a planned path while one or more "follower" drones automatically maintain a fixed relative position and orientation with respect to it. This is the **leader-follower formation control** paradigm.

The reference paper tackles this problem for **two small quadrotors** using **dual quaternions** — a single mathematical object that represents *both* the orientation *and* the position of a rigid body at once. The paper's main contributions are:

- A **kinematic control law**, expressed purely in dual quaternions, that drives a vehicle's pose (position + attitude) to a desired, possibly time-varying, pose.
- A **systematic method for choosing controller gains** (instead of trial and error), based on analyzing the eigenvalues of the *linearized* closed-loop error dynamics.
- **Experimental validation** on two real Parrot Bebop 2 quadrotors flying a leader-follower Lemniscate ("figure-eight") trajectory, under three different gain-tuning strategies.

The original implementation reproduced the analytic control pipeline in simulation. The current project extends it into **simulator-trained predictive control**: recorded state–command–next-state transitions identify a response model, and an optimizer uses that model with dual-quaternion pose error to choose commands during flight. Both vehicles use this controller in learned mode.

The leader still follows a chosen mathematical reference. The follower instead targets an earlier pose on the leader's **measured path from the current flight**, normally 0.8 m behind by arc length. This is different from replaying a prerecorded training route. Red and blue trails display actual leader and follower movement.

| Component | Original baseline | Current implementation |
|---|---|---|
| High-level commands | Analytic proportional/integral law with selected gains | Learned response prediction plus bounded optimization |
| Training | No identification stage | Four simulator flights fit the response model; a fifth checks prediction |
| Pose mathematics | Dual quaternions | Preserved, including relative pose error |
| Follower reference | Spatially offset leader pose | Chronological leader-path history with interpolation |
| Interaction | Command-line experiment | CLI plus a live control panel and separate viewers |
| Validation | Analytic examples and tests | Model-prediction checks, controller comparisons, tracking logs and tests |

This is **data-driven high-level control**, not an entirely learned flight stack. Physics, low-level stabilization, pose kinematics, objective weights and some command constraints are designed. Training happens offline; flight commands are recalculated online. The paper's real-drone experiments are background evidence, not experiments performed by this repository.

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

> **Worked comparison — why "one object" actually matters in practice**
>
> Suppose you need to compose three successive rigid-body transforms (e.g., "world → leader body → camera mount → gimbal"). With the classical representation you carry a rotation matrix $R \in  SO(3)$ **and** a translation vector $t \in  \mathbb R^{3}$ for each transform, and composing two transforms $(R_{1},t_{1})$ then $(R_{2},t_{2})$ requires the (easy to get wrong) rule:
>
> $$
> (R_2, t_2) \circ (R_1, t_1) = (R_2 R_1,\; R_2 t_1 + t_2)
> $$
>
> — a matrix-vector rule that is *different* from ordinary matrix multiplication. With unit dual quaternions, composing $Q_{2}$ after $Q_{1}$ is **exactly one dual-quaternion product**, $Q_{2} \circ  Q_{1}$, using the *same* multiplication rule you'd use to compose pure rotations. There is nothing special to remember, and inverting a transform is just the conjugate, $Q^{-1} = Q*$, instead of $(R,t)^{-1} = (R^{\mathsf T}, -R^{\mathsf T}t)$. This is exactly what `DualQuaternion.__mul__` and `.inverse()` (or equivalent) implement in `dq_control/dual_quaternion.py`.
>
> | | Rotation matrix + vector | Unit dual quaternion |
> |---|---|---|
> | Storage | 9 + 3 = 12 numbers (with 6 constraints) | 8 numbers (with 2 constraints: $\Vert \bar{q}\Vert =1$ and the dual-part orthogonality condition) |
> | Compose two poses | $(R_{2}R_{1}, R_{2}t_{1}+t_{2})$ — two different operations | $Q_{2} \circ  Q_{1}$ — one product |
> | Invert a pose | $(R^{\mathsf T}, -R^{\mathsf T}t)$ | $Q*$ |
> | Interpolate poses | Needs separate schemes for rotation (e.g. SLERP) and translation (linear) | Screw Linear Interpolation (ScLERP) handles both together |
> | Singularities | None for matrices; Euler angles have gimbal lock | None (like plain quaternions) |

---

## 3. Mathematical Building Blocks

To understand dual quaternions, we build up from ordinary quaternions and dual numbers.

### 3.1 Quaternions

Quaternions extend complex numbers to four dimensions. A quaternion is written as:

$$
\bar q = q + q_0, \qquad q \in \mathbb{R}^3,\ q_0 \in \mathbb{R}
$$

or, in the classical notation:

$$
\bar q = q_0 + i q_1 + j q_2 + k q_3
$$

where $i, j, k$ are imaginary units satisfying $i^{2} = j^{2} = k^{2} = ijk = -1$. Equivalently, a quaternion can be split into:

- a **vector part** $q \in  \mathbb R^{3}$ (analogous to the "imaginary" components), and
- a **scalar part** $q_{0} \in  \mathbb R$ (the "real" component).

Quaternion multiplication is **not commutative** ($p \circ  q \ne  q \circ  p$ in general) — order matters, just like it does for 3D rotations. Given $p = (p, p_{0})$ and $q = (q, q_{0})$, the quaternion product can be written compactly in matrix form as:


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

where $S(\cdot )$ is the **skew-symmetric matrix** built from a vector such that $S(v) w = v \times  w$ — i.e., matrix multiplication by $S(v)$ reproduces the cross product with $v$.

Other key operations:
- **Conjugate:** $q* = (-q, q_{0})$ (flip the sign of the vector part).
- **Norm:** $\Vert q\Vert ^{2} = q \circ  q* = q* \circ  q$ (a scalar).
- A vector $p \in  \mathbb R^{3}$ can always be embedded as a *pure* quaternion $\bar{p} = (p, 0)$ (zero scalar part).

> **Worked example — multiplying two quaternions by hand**
>
> Let $\bar{p} = (p, p_{0})$ with vector part $p = (0,0,0.7071)$ and scalar part $p_{0} = 0.7071$ (this happens to be the quaternion for a 90° rotation about $z$ — more on that in §3.2). Using the closed-form Hamilton product
>
> $$
> p \circ q = \big(p_0 q + q_0 p + p \times q,\;\; p_0 q_0 - p\cdot q\big)
> $$
>
> compute $\bar{p} \circ  \bar{p}$ (composing the rotation with itself):
>
> - vector part: $p_{0}\cdot p + p_{0}\cdot p + p\times p = 0.7071\cdot (0,0,0.7071) + 0.7071\cdot (0,0,0.7071) + 0 = (0, 0, 1.0)$ (the cross product of any vector with itself is zero)
> - scalar part: $p_{0}\cdot p_{0} - p\cdot p = 0.7071^{2} - 0.7071^{2} = 0$
>
> So $\bar{p} \circ  \bar{p} = ((0,0,1.0), 0)$. Running this through NumPy confirms it exactly:
>
> ```text
> >>> q_z90 ∘ q_z90 = (array([0., 0., 1.]), 0.0)
> >>> expected q_z180              = (array([0., 0., 1.]), 0.0)   # matches!
> ```
>
> This double-check is exactly what you'd hope: composing a 90° rotation about $z$ with itself gives the 180° rotation about $z$ (vector part $(0,0,1)$ is $sin(90^\circ )=1$, scalar part $cos(90^\circ )=0$) — confirming that quaternion multiplication really does compose rotations by *adding* their angles when the axis is shared, just like multiplying two unit complex numbers $e^{i\theta _{1}}e^{i\theta _{2}}=e^{i(\theta _1+\theta _2)}$ adds angles in 2D.

### 3.2 Quaternions as Rotations

A **unit-norm quaternion** ($\Vert q\Vert  = 1$) represents a rotation in 3D space, analogous to how a unit complex number represents a rotation in 2D. If $\bar{q}$ is the unit quaternion rotating from body frame $b$ to inertial frame $i$, a vector expressed in the body frame transforms to the inertial frame via the **sandwich product**:

$$
\bar p^i = \bar q \circ \bar p^b \circ \bar q^*
$$

Note that $\bar{q}$ and $-\bar{q}$ represent the *same* rotation — this "double cover" is a well-known and harmless property of quaternions.

If the body rotates with angular velocity $\omega $ (expressed in the body frame), the quaternion evolves according to:

$$
\dot{\bar q} = \tfrac{1}{2}\, \bar q \circ \bar\omega
$$

This single differential equation replaces the more cumbersome update equations needed for rotation matrices or Euler angles, and it has no singularities.

> **Worked example — rotating a vector with the sandwich product**
>
> Take $\bar{q} = (0, 0, sin45^\circ , cos45^\circ ) = (0, 0, 0.7071, 0.7071)$, the unit quaternion for a 90° rotation about $z$, and rotate the body-frame vector $\bar{p}^b = (1, 0, 0)$ (a pure quaternion, scalar part zero):
>
> $$
> \bar p^i = \bar q \circ \bar p^b \circ \bar q^*
> $$
>
> Step by step:
> 1. $\bar{q} \circ  \bar{p}^b$ — multiply the rotation quaternion by the pure vector quaternion.
> 2. Multiply that result by $\bar{q}* = (0,0,-0.7071,0.7071)$ (flip the sign of the vector part).
> 3. The scalar part of the final result is always $0$ (rotating a pure vector keeps it pure); the vector part is the rotated vector.
>
> Doing this arithmetic (verified numerically) gives:
>
> ```text
> rotate (1,0,0) by 90° about z ->  [0.  1.  0.]
> ```
>
> i.e. the $x$-axis is carried onto the $y$-axis, exactly matching the right-hand rule for a positive (counter-clockwise, viewed from $+z$) rotation about $z$:
>
> <p align="center"><img src="assets/fig1_quaternion_rotation.png" width="480" alt="Quaternion sandwich product rotating the x-axis onto the y-axis"></p>
>
> This is precisely what `Quaternion.rotate()` in `dq_control/quaternion.py` computes, and it's the same operation used, e.g., to express the leader's velocity or the follower's offset in a different frame throughout the rest of this document (see §5, §10).

### 3.3 Dual Numbers

Before combining quaternions with position, we need one more ingredient: **dual numbers**. A dual number has the form:

$$
\hat\alpha = a + \varepsilon b, \qquad a, b \in \mathbb{R}
$$

where $\varepsilon $ is a symbol (the *dual unit*) with the defining property:

$$
\varepsilon \neq 0, \qquad \varepsilon^2 = 0
$$

This is conceptually similar to how $i^{2} = -1$ defines complex numbers, except here squaring the special symbol gives **zero**, not $-1$. (Important: $\varepsilon $ is *not* "a small number close to zero" — it's an abstract algebraic symbol.)

Dual numbers add component-wise, and multiply using the rule $\varepsilon ^{2} = 0$:

$$
(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\,\varepsilon
$$

Intuitively, $a$ is the "principal" or nominal value, and $b$ is a first-order "perturbation" or "derivative-like" term riding along with it. This structure is exactly what's needed to piggy-back *position* information onto *orientation* information.

> **Worked example — dual numbers are automatic differentiation, for free**
>
> A nice way to build intuition for *why* $\varepsilon ^{2} = 0$ is useful is to notice that dual numbers automatically compute derivatives. Take $f(x) = x^{2} + 3x$ and evaluate it at $x = 2 + 1\cdot \varepsilon $ (i.e., set $a = 2$, the point of interest, and $b = 1$):
>
> $$
> f(2+\varepsilon) = (2+\varepsilon)^2 + 3(2+\varepsilon) = \underbrace{4}_{a^2} + \underbrace{4\varepsilon}_{2ab\,\varepsilon} + \underbrace{6}_{3a} + \underbrace{3\varepsilon}_{3b\,\varepsilon} \;(\text{using } \varepsilon^2=0) = 10 + 7\varepsilon
> $$
>
> Reading off the result: the **principal part is $f(2) = 10$** and the **dual part is $f'(2) = 7$** — and indeed $f'(x) = 2x+3$, so $f'(2) = 7$. Verified numerically:
>
> ```text
> f(2)=10.0, f'(2) via dual numbers=7.0   (analytic f'(x)=2x+3 -> 7)
> ```
>
> This is exactly the same "carry a first-order perturbation along for free" mechanism that lets a dual *quaternion* carry position information (the "derivative-like" part) alongside orientation (the "principal" part) in §3.5 below — the dual part isn't derived by extra bookkeeping, it falls out of the algebra automatically, the same way $f'(2)$ fell out above.

### 3.4 Dual Quaternions

A **dual quaternion** is obtained by applying the same trick that built complex-like dual numbers from real numbers — but starting from quaternions instead of reals:

$$
Q = \bar r + \varepsilon\, \bar s, \qquad \bar r, \bar s \in \mathbb{H}
$$

where $\mathbb H$ is the set of quaternions. $Q$ is an 8-dimensional object (4 real dimensions from $\bar{r}$, 4 more from $\bar{s}$). We call:

- $\mathcal P(Q) = \bar{r}$ the **principal part**,
- $\mathcal D(Q) = \bar{s}$ the **dual part**.

Sum, product, and conjugation extend naturally from quaternions, always remembering $\varepsilon ^{2} = 0$. The dual quaternion conjugate is:

$$
Q^* = \mathcal{P}(Q)^* + \varepsilon\, \mathcal{D}(Q)^*
$$

> **Worked example — multiplying two dual quaternions**
>
> Given $Q_{1} = \bar{r}_{1} + \varepsilon  \bar{s}_{1}$ and $Q_{2} = \bar{r}_{2} + \varepsilon  \bar{s}_{2}$, apply the dual-number product rule $(a+\varepsilon b)(c+\varepsilon d) = ac + \varepsilon (ad+bc)$ from §3.3, term by term, but using the quaternion product $\circ $ instead of ordinary multiplication:
>
> $$
> Q_1 \circ Q_2 = \underbrace{\bar r_1 \circ \bar r_2}_{\text{principal part}} \; + \; \varepsilon\underbrace{\left(\bar r_1 \circ \bar s_2 + \bar s_1 \circ \bar r_2\right)}_{\text{dual part}}
> $$
>
> Note that **order matters twice over**: quaternion multiplication is non-commutative (§3.1), *and* the dual-part sum $\bar{r}_{1}\circ \bar{s}_{2} + \bar{s}_{1}\circ \bar{r}_{2}$ keeps each factor in its original left/right position. This single formula is what `DualQuaternion.__mul__` implements, and it's also what makes $Q \circ  Q*$ collapse to the identity for any unit dual quaternion — checked numerically for the pose built in the next section:
>
> ```text
> Q ∘ Q* principal part (should be (0,0,0),1): (array([0., 0., 0.]), 1.0)
> Q ∘ Q* dual part      (should be (0,0,0),0): (array([0., 0., 0.]), 0.0)
> ```

### 3.5 Dual Quaternions as Rigid-Body Pose

Just as a **unit quaternion** represents pure rotation, a **unit dual quaternion** ($Q \circ  Q* = Q* \circ  Q = 1$) represents a full rigid-body **pose** — rotation *and* translation together. Concretely:

- $Q$ has unit norm **if and only if** its principal part $\mathcal P(Q) = \bar{q}$ is itself a unit quaternion (the attitude), and its dual part is:

$$
\mathcal{D}(Q) = \tfrac{1}{2}\, \bar p \circ \mathcal{P}(Q)
$$

  where $\bar{p} = (p, 0)$ is the position vector $p \in  \mathbb R^{3}$, embedded as a pure quaternion.

- Given any unit dual quaternion $Q$, you can always **recover** the attitude and position:

$$
\bar q = \mathcal{P}(Q), \qquad \bar p = 2\,\mathcal{D}(Q) \circ \mathcal{P}(Q)^*
$$

So a single object $Q = \bar{q} + \varepsilon \cdot \frac12 (\bar{p} \circ  \bar{q})$ carries everything needed to describe "where the drone is and how it's oriented" — and dual quaternion multiplication automatically composes both the rotations *and* the translations correctly, in the right order.

> **Worked example — building a pose and recovering it back**
>
> Let the drone be at position $p = (1, 2, 3)$ m with attitude $\bar{q} = (0,0,0.7071,0.7071)$ (90° about $z$, from §3.2). Build the unit dual quaternion:
>
> $$
> Q = \bar q + \varepsilon\, \tfrac12\bar p \circ \bar q, \qquad \bar p = (p, 0)
> $$
>
> Computing $\frac12  \bar{p} \circ  \bar{q}$ gives the dual part $((1.0607, 0.3536, 1.0607), -1.0607)$. This does **not** look like $p$ at all by itself — position is encoded *jointly* with orientation, which is exactly the point of the algebra: you cannot read off the position without also using the attitude. Recovering it with the paper's formula,
>
> $$
> \bar p = 2\,\mathcal{D}(Q)\circ\mathcal{P}(Q)^*
> $$
>
> gives back **exactly** $(1, 2, 3)$ (vector part) with $0$ scalar part:
>
> ```text
> principal part P(Q) = qbar          = (array([0.  , 0.  , 0.7071]), 0.7071)
> dual part D(Q) = 1/2 pbar∘qbar       = (array([1.0607, 0.3536, 1.0607]), -1.0607)
> recovered attitude:                    (array([0.  , 0.  , 0.7071]), 0.7071)
> recovered position (should be [1,2,3]): [1. 2. 3.]   scalar part (should be ~0): 0.0
> ```
>
> Visually, $Q$ is simultaneously "an arrow from the world origin to the drone" **and** "a rotated triad attached to the drone":
>
> <p align="center"><img src="assets/fig2_dual_quaternion_pose.png" width="480" alt="A unit dual quaternion encoding both position and orientation"></p>
>
> This round-trip (`from_pose()` → `.position()`/`.attitude()`) is directly exercised by `test_dual_quaternion.py`'s "position/attitude round-trip" test mentioned in §12.6.

---

## 4. Vehicle Kinematics in Dual Quaternions

The **twist** (linear + angular velocity) of a vehicle is also packaged into a dual quaternion:

$$
\Omega(\bar\omega, \bar v), \qquad \mathcal{P}(\Omega) = \bar\omega,\quad \mathcal{D}(\Omega) = \mathcal{P}(Q)^* \circ \bar v \circ \mathcal{P}(Q)
$$

and the pose evolves in time according to a beautifully compact single equation:

$$
\dot Q = \tfrac{1}{2}\, Q \circ \Omega(\bar\omega, \bar v)
$$

This is the dual-quaternion analogue of $\dot{q} = \frac12  q \circ  \bar{\omega}$ for pure rotation — it simultaneously propagates both position and attitude, using commanded body-frame angular velocity $\omega $ and inertial-frame linear velocity $v$. This is the equation the simulation integrates forward in time to move each drone.

> **Worked example — one Euler integration step**
>
> Suppose the drone starts at the identity pose ($p=(0,0,0)$, no rotation) and is commanded a constant angular velocity $\omega  = (0,0,1)$ rad/s about $z$ and a constant linear velocity $v = (1,0,0)$ m/s in the inertial frame. Over a small step $dt = 0.05$ s, the (first-order / Euler) update
>
> $$
> Q_{k+1} \approx Q_k + dt\cdot\dot Q_k = Q_k + dt \cdot \tfrac12\, Q_k \circ \Omega(\bar\omega,\bar v)
> $$
>
> moves the drone roughly $0.05$ m along $x$ (from $v$) while its heading rotates roughly $0.05$ rad (≈ 2.9°) about $z$ (from $\omega $) — after which `integrate_pose()` **re-normalizes** the result back onto the unit-dual-quaternion manifold (since a first-order Euler step of a nonlinear ODE drifts off it slightly). Chaining this update at the controller's rate (`--ctrl_freq`, default 48 Hz) is exactly how each simulated drone's pose advances every control cycle in `simulators/leader_follower.py` (as a target pose for physics backends; their actual motion comes from physics).

---

## 5. Pose Error and the Control Objective

Let $Q$ be the vehicle's **current** pose and $Q_d$ the **desired** pose. The **pose error** is defined as a single dual quaternion:

$$
\delta Q = Q_d^* \circ Q = \delta \bar q + \varepsilon\, \tfrac{1}{2}\left(\overline{\delta p}^{\,b} \circ \delta \bar q\right)
$$

where:
- $\delta \bar{q} = \bar{q}_d^* \circ \bar{q}$ is the **attitude error** (a quaternion — how far the current orientation is from the desired one),
- $\delta p = p - p_d$ is the raw **position error**, and
- $\delta \bar{p}^{\,b} = \bar{q}_d^* \circ \delta \bar{p} \circ \bar{q}_d$ is that position error expressed **in the desired body frame** — the natural frame to regulate it in.

The **control goal** is simply: drive $\delta Q \to  1$ (the identity dual quaternion), i.e., make the attitude-error vector and position error approach zero (the full error quaternion approaches the identity), meaning the vehicle's actual pose converges to the desired pose. Both position and attitude errors are captured by this one object.

> **Worked example — computing a pose error**
>
> Say the drone's **current** pose is $p = (1, 2, 3)$ m, $\bar{q} = (0,0,0.7071,0.7071)$ (90° about $z$), and the **desired** pose is $p_d = (1, 2, 2)$ m with **no** rotation, $\bar{q}_d = (0,0,0,1)$ (identity). Then:
>
> - **Attitude error** $\delta \bar{q} = \bar{q}_d* \circ  \bar{q}$. Since $\bar{q}_d$ is the identity, its conjugate is also the identity, so $\delta \bar{q} = \bar{q} = (0,0,0.7071,0.7071)$ — the drone is 90° off from the desired heading, exactly as expected.
> - **Raw position error** $\delta p = p - p_d = (0, 0, 1)$ m — the drone is 1 m too high in $z$.
> - **Position error in the desired body frame** $\delta p^b = \bar{q}_d* \circ  \delta \bar{p} \circ  \bar{q}_d$. Because $\bar{q}_d$ is the identity here, the "desired body frame" coincides with the world frame, so $\delta p^b = \delta p = (0,0,1)$ unchanged. (Had $\bar{q}_d$ been non-trivial, $\delta p^b$ would differ from $\delta p$ — this is the step that makes the position-error gains act along the *desired heading's* axes rather than the world axes, which matters once the leader is turning.)
>
> ```text
> attitude error delta_qbar               = (array([0.    , 0.    , 0.7071]), 0.7071)
> raw position error delta_p              = [0. 0. 1.]
> position error in desired body frame    = [0. 0. 1.]
> ```
>
> This $(\delta \bar{q}, \delta p^b)$ pair is exactly the input the control law in §6 consumes.

---

## 6. The Kinematic Control Law

> **Analytic baseline:** this section preserves the original derivation and examples. Learned mode uses the predictive formulation in Section 21; these gain-based stability arguments do not prove stability of the learned controller.

The controller computes the **commanded twist** $(\bar{\omega}, \bar{v})$ sent to the vehicle as a function of the pose error. It uses six gain matrices:

$$
K_{\omega,p},\; K_{v,p},\; K_{\omega,i},\; K_{v,i},\; K_{\eta},\; K_{\xi} \in \mathbb{R}^{3\times3}
$$

For the stability result, these gain matrices are required to be **negative definite**.

### Angular velocity command

$$
\bar{\omega} = \delta\bar{q}^{\,*} \circ \bar{\omega}_d \circ \delta\bar{q} + \left( \mathrm{sgn}(\delta q_0) \left( K_{\omega,p}\delta q + \eta_0 K_{\omega,i}\eta \right), \, 0 \right)
$$

Here:

- $\bar{\omega}$ is the commanded angular velocity represented as a pure quaternion.
- $\bar{\omega}_d$ is the desired angular velocity.
- $\delta\bar{q}$ is the attitude-error quaternion.
- $\delta q$ is the vector part of $\delta\bar{q}$.
- $\delta q_0$ is the scalar part of $\delta\bar{q}$.
- $\eta$ is the vector part of the attitude integral state.
- $\eta_0$ is the scalar part of the attitude integral quaternion $\bar{\eta}$.

### Linear velocity command

$$
\bar{v} = \bar{v}_d + \mathcal{R}(\bar{q}_d) \left( K_{v,p}\delta p^{\,b} + K_{v,i}\xi \right)
$$

Here, $\delta p^{\,b}$ is the position error expressed in the desired body frame.

### Attitude integral state

$$
\dot{\bar{\eta}} = \frac{1}{2} \bar{\eta} \circ \left( -|\delta q_0| K_{\omega,i}\delta q + \mathrm{sgn}(\eta_0) K_{\eta}\eta, \, 0 \right)
$$

The $K_{\eta}\eta$ term acts as a **forgetting/leakage term**, preventing the integral state from growing without bound.

### Position integral state

$$
\dot{\xi} = -K_{v,i}\delta p^{\,b} + K_{\xi}\xi
$$

The first term accumulates the position error, while $K_{\xi}\xi$ provides the corresponding forgetting mechanism.

### Interpretation

- The **feedforward** terms $\bar{\omega}_d$ and $\bar{v}_d$ provide the velocity required by the desired trajectory.
- The **proportional** terms $K_{\omega,p}\delta q$ and $K_{v,p}\delta p^{\,b}$ respond to the instantaneous pose error.
- The **integral** terms involving $\eta$ and $\xi$ accumulate persistent tracking errors.
- The $K_{\eta}$ and $K_{\xi}$ terms prevent excessive accumulation of the integral states.

The closed-loop error states converge toward the desired equilibrium:

$$
\delta\bar{q}\rightarrow\bar{1}, \qquad \delta p\rightarrow0, \qquad \eta\rightarrow0, \qquad \xi\rightarrow0
$$

Here, $\bar{1}$ denotes the **identity quaternion**, representing zero attitude error.

> **Worked example — evaluating the control law for one time step**
>
> Continue the pose-error example from §5 ($\delta q = (0,0,0.7071)$, $\delta q_{0} = 0.7071$, $\delta p^b = (0,0,1)$), and — to isolate the proportional action — assume the integral states start at zero ($\eta =0, \xi =0$), there is no feedforward ($\bar{\omega}_d = \bar{v}_d = 0$), and use simple diagonal gains $K_{\omega ,p} = -2I_{3}$, $K_{v,p} = -1.5I_{3}$.
>
> **Angular velocity command** (only the proportional term survives since $\eta =0$):
>
> $$
> \bar\omega = \mathrm{sgn}(\delta q_0)\, K_{\omega,p}\,\delta q = (+1)\cdot(-2)\cdot(0,0,0.7071) = (0,0,-1.4142)
> $$
>
> **Linear velocity command** (only the proportional term survives since $\xi =0$, and here $\bar{q}_d$= identity so $𝓡(\bar{q}_d)$ is the identity rotation):
>
> $$
> \bar v = K_{v,p}\,\delta p^{\,b} = (-1.5)\cdot(0,0,1) = (0,0,-1.5)
> $$
>
> ```text
> omega_cmd (feedforward=0): [ 0.      0.     -1.4142]
> v_cmd:                     [ 0.   0.  -1.5]
> ```
>
> Both signs make physical sense: the drone needs to rotate back (negative about $z$, undoing the 90° attitude error) and descend (negative $z$-velocity, since it's currently 1 m too high). Notice the **sign-of-$\delta q_{0}$** factor in the angular term — it exists because $\delta \bar{q}$ and $-\delta \bar{q}$ represent the same rotation (§3.2's "double cover"); without correcting for it, the controller could occasionally command a rotation the "long way around" when the error crosses through $\delta q_{0} = 0$.

---

## 7. Why Tuning Is Hard: The Linearized Error Dynamics

> **Analytic baseline:** this section preserves the original derivation and examples. Learned mode uses the predictive formulation in Section 21; these gain-based stability arguments do not prove stability of the learned controller.

Adding the integral states $\eta$ and $\xi$ improves steady-state tracking, but it also introduces additional gain matrices.

For the attitude subsystem, the relevant gains are

$$
K_{\omega,p}, \qquad K_{\omega,i}, \qquad K_{\eta}
$$

The closed-loop attitude dynamics are linearized around the equilibrium

$$
(\delta q,\eta)=(0,0)
$$

The resulting linearized system is

$$
\begin{bmatrix} \dot{\delta q}\\ \dot{\eta} \end{bmatrix} = M_{\omega} \begin{bmatrix} \delta q\\ \eta \end{bmatrix}
$$

where

$$
M_{\omega} = \frac{1}{2} \begin{bmatrix} K_{\omega,p} & K_{\omega,i}\\ -K_{\omega,i} & K_{\eta} \end{bmatrix}
$$

Since each gain matrix is $3\times3$, $M_{\omega}$ is a $6\times6$ matrix.

An analogous matrix $M_v$ describes the linearized position-error dynamics.

For a linear system

$$
\dot{x}=Mx
$$

the eigenvalues of $M$ determine the local behavior of the system.

If

$$
\lambda<0
$$

the corresponding mode decays exponentially.

If

$$
\lambda=\alpha\pm j\beta, \qquad \alpha<0
$$

the corresponding mode decays while oscillating.

Therefore:

- **Real negative eigenvalues** → non-oscillatory exponential convergence.
- **Complex eigenvalues with negative real parts** → damped oscillatory behavior.
- **Eigenvalues with positive real parts** → instability.

Consequently, selecting the controller gains can be viewed as a structured **pole-placement problem**.

The difficulty is that $M_{\omega}$ is a block matrix. Its eigenvalues cannot, in general, be determined simply by considering the eigenvalues of the individual matrices

$$
K_{\omega,p}, \qquad K_{\omega,i}, \qquad K_{\eta}
$$

Therefore, making each gain matrix individually negative definite does not by itself determine the exact eigenvalues of the complete matrix $M_{\omega}$.

> **Worked example — real vs. complex eigenvalues in a toy 2×2 case**
>
> The full $M_\omega$ is $6\times6$ (three coupled $3\times3$ blocks), but the essential behavior already shows up in a **scalar toy version** where each $3\times3$ gain matrix is replaced by a single number, so $M_\omega$ shrinks to $2\times2$:
>
> $$
> M_\omega = \begin{bmatrix} a & c \\ -c & d\end{bmatrix}, \qquad a=\tfrac12 K_{\omega,p}=-2,\quad d = \tfrac12 K_\eta = -0.5,\quad c = \tfrac12 K_{\omega,i}
> $$
>
> The characteristic polynomial is $\lambda^2-(a+d)\lambda+(ad+c^2)=0$, whose discriminant is $(a-d)^2-4c^2$. So the eigenvalues stay **real** as long as the coupling isn't too large:
>
> $$
> |c| \le \frac{|a-d|}{2} = 0.75 \quad\text{(here)}
> $$
>
> Sweeping $c$ from $0$ upward and plotting the resulting eigenvalues on the complex plane makes the transition completely visible:
>
> <p align="center"><img src="assets/fig3_eigenvalue_placement.png" width="760" alt="Eigenvalue placement: real poles for small coupling, complex poles for large coupling"></p>
>
> Numerically, right around the predicted threshold of $0.75$:
>
> ```text
> coupling c=K_i/2= 0.300 -> eigenvalues: [-1.9374 -0.5626]      # real, well separated
> coupling c=K_i/2= 0.750 -> eigenvalues: [-1.25+0.j   -1.25-0.j]  # real, repeated (the boundary case)
> coupling c=K_i/2= 1.500 -> eigenvalues: [-1.25+1.299j -1.25-1.299j]  # complex conjugate pair
> ```
>
> This is the precise mechanism behind the paper's (and this repo's) three named gain sets: `gains_proportional()` effectively sets the coupling $K_{\omega ,i}=0$ (no integral coupling at all — the simplest, always-real case); `gains_complex_eig()` intentionally picks a large $K_{\omega ,i}$ relative to the spread of $K_{\omega ,p}$ and $K_\eta $, landing past the threshold into complex territory (visible as overshoot/ringing in the tracking plots); `gains_real_eig()` picks $K_{\omega ,i}$ small enough to stay below threshold, giving smooth, non-oscillatory convergence — the paper's best-performing tuning.

---

## 8. The Pole-Placement / Gain-Selection Method

> **Analytic baseline:** this section preserves the original derivation and examples. Learned mode uses the predictive formulation in Section 21; these gain-based stability arguments do not prove stability of the learned controller.

The linearized matrix $M_{\omega}$ has a special spectral structure.

Define

$$
H = \begin{bmatrix} I_3 & 0\\ 0 & -I_3 \end{bmatrix}
$$

and the corresponding indefinite inner product

$$
[x,y]_H = x^{\mathsf T}Hy
$$

The matrix $M_{\omega}$ satisfies

$$
HM_{\omega} = M_{\omega}^{\mathsf T}H
$$

Therefore, $M_{\omega}$ is **$H$-self-adjoint**.

Define

$$
a_- = \lambda_{\min}\left(\frac{1}{2}K_{\omega,p}\right), \qquad a_+ = \lambda_{\max}\left(\frac{1}{2}K_{\omega,p}\right)
$$

and

$$
d_- = \lambda_{\min}\left(\frac{1}{2}K_{\eta}\right), \qquad d_+ = \lambda_{\max}\left(\frac{1}{2}K_{\eta}\right)
$$

The coupling between the proportional and integral dynamics is determined by $K_{\omega,i}$.

The spectral bounds therefore depend on quantities such as

$$
\left\| K_{\omega,p}^{-1}K_{\omega,i} \right\|
$$

and

$$
\left\| K_{\eta}^{-1}K_{\omega,i} \right\|
$$

If the coupling introduced by $K_{\omega,i}$ is sufficiently small compared with the separation between the spectra associated with $K_{\omega,p}$ and $K_{\eta}$, the eigenvalues of $M_{\omega}$ can be guaranteed to remain real and negative:

$$
\lambda_i(M_{\omega})\in\mathbb{R}, \qquad \lambda_i(M_{\omega})<0
$$

This produces a locally exponentially convergent and non-oscillatory response.

If the coupling becomes too strong, complex eigenvalues can appear:

$$
\lambda_{1,2} = \alpha\pm j\beta, \qquad \alpha<0
$$

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

> **Worked example — applying the procedure to the toy 2×2 case**
>
> Continuing the scalar example from §7 ($a = -2$, $d = -0.5$):
>
> 1. **Proportional gains** are fixed at $K_{\omega ,p} = -4$ (so $a=K_{\omega,p}/2=-2$) — this sets how aggressively the attitude error itself is corrected.
> 2. **Integral-related gains**: $K_\eta  = -1$ (so $d=K_\eta/2=-0.5$) is the forgetting rate of the integral state; $K_{\omega ,i}$ (equivalently $c=K_{\omega,i}/2$) is the free parameter being tuned.
> 3–5. **Build $M_\omega $ and sweep its eigenvalues** as $c$ varies — exactly the sweep plotted in §7's figure.
> 6. **Pick a gain**: the threshold derived above, $\vert c\vert  \le  0.75$, i.e. $\vert K_{\omega ,i}\vert  \le  1.5$, is the *boundary* of the non-oscillatory region. Choosing, say, $K_{\omega ,i} = -1.0$ ($c = 0.5$, comfortably inside the boundary) guarantees real, negative eigenvalues and therefore smooth convergence with no overshoot — this is the spirit of how `dq_control/gains.py`'s `gains_real_eig()` chooses its integral gains relative to its proportional and forgetting gains, just carried out on the full $3\times3$ (so $6\times6$) blocks using matrix norms like $\lVert K_{\omega,p}^{-1}K_{\omega,i}\rVert$ instead of the scalar ratio used here for intuition.
>
> The general lesson generalizes directly to the $3\times3$/$6\times6$ case: **more integral coupling buys faster disturbance rejection but risks oscillation**, and the paper's contribution is turning that qualitative trade-off into a concrete, checkable inequality instead of leaving it to trial and error.

---

## 9. From Kinematics to Quadrotor Commands

> **Analytic baseline:** this section preserves the original derivation and examples. Learned mode uses the predictive formulation in Section 21; these gain-based stability arguments do not prove stability of the learned controller.

The kinematic controller produces the desired twist

$$
(\bar{\omega},\bar{v})
$$

A real quadrotor cannot directly generate an arbitrary three-dimensional velocity. Its motion is produced through thrust and attitude control.

The command variables considered in this project are

$$
u_{\phi}, \qquad u_{\theta}, \qquad u_{\dot z}, \qquad u_{\dot\psi}
$$

corresponding to roll, pitch, vertical velocity, and yaw-rate commands.

The horizontal motion requires an additional mapping because a quadrotor generates horizontal acceleration by **tilting its thrust vector**.

### Desired Acceleration

$$
U_p = \ddot{p}_{jd} + K_a\left(\dot{p}_j-v\right)
$$

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

$$
\bar{q}'_{j\phi,\theta} = \left( n_j\times U_p,\, \left\langle n_j,U_p\right\rangle + \|U_p\| \right)
$$

It is then normalized:

$$
\bar{q}_{j\phi,\theta} = \frac{\bar{q}'_{j\phi,\theta}}{\left\|\bar{q}'_{j\phi,\theta}\right\|}
$$

This quaternion represents the rotation required to align the vehicle's thrust direction with the desired acceleration direction.

### Yaw Composition

The roll-pitch orientation does not uniquely determine yaw. Therefore, yaw is specified independently using the desired yaw quaternion $\bar{q}_{j\psi}$.

The complete desired attitude is

$$
\bar{q}_{jd} = \bar{q}_{j\phi,\theta} \circ \bar{q}_{j\psi}
$$

The complete control structure is therefore:

```
Pose error → Kinematic controller → (ω̄, v̄) → Acceleration/attitude mapping → Quadrotor commands
```

Visually, as a block diagram matching the code layering in `simulators/leader_follower.py` (see also §17):

<p align="center"><img src="assets/fig6_control_cascade.png" width="820" alt="Control cascade: pose error, kinematic controller, acceleration/attitude mapping, low-level PID"></p>

> **Worked example — aligning the thrust vector**
>
> Suppose the body thrust axis is currently straight up, $n_j = (0,0,1)$, and the desired acceleration (already including gravity compensation, from $U_p = \ddot{p}_{jd} + K_a(\dot p_j - v)$) works out to $U_p = (1, 0, 9.8)$ m/s² — i.e., mostly hovering thrust, plus a little push in $+x$. Then:
>
> $$
> \bar q'_{j\phi,\theta} = \big(n_j\times U_p,\;\; \langle n_j,U_p\rangle + \lVert U_p\rVert\big) = \big((0,\,1,\,0),\;\; 9.8+9.8509\big) = \big((0,1,0),\;19.6509\big)
> $$
>
> Normalizing (dividing by $\Vert q'\Vert  \approx  19.6763$) gives $\bar{q}_{jφ,\theta } \approx  (0, 0.0508, 0, 0.9987)$ — a **small** rotation (tiny vector part), as expected for a small tilt away from vertical. Checking that this quaternion really does rotate $n_j$ onto the (normalized) desired direction:
>
> ```text
> rotate n by q_align -> [0.1015 0.     0.9948]
> compare to Up/||Up||    [0.1015 0.     0.9948]     # matches!
> ```
>
> The rotation angle here is `2·arcsin(0.0508) ≈ 5.8°` — a small, sensible tilt to accelerate gently sideways while (mostly) still fighting gravity, exactly the kind of roll/pitch command a real quadrotor executes to move horizontally. This computation is what turns the kinematic controller's abstract $\bar{v}$ into something a rotor-actuated vehicle can physically do.

---

## 10. Leader-Follower Formation Law

### 10.1 Current default: follow the leader's measured route

Let the measured leader positions be $p_L^0,p_L^1,\ldots,p_L^k$. Accumulated distance along the recorded trail is

$$
s_k=s_{k-1}+\left\|p_L^k-p_L^{k-1}\right\|_2.
$$

For following distance $d$, the desired trail coordinate is

$$
s_F^{\mathrm{target}}=\max(0,s_k-d).
$$

Choose the chronological neighboring samples whose distances bracket that target. Their interpolation fraction and the follower's desired position are

$$
\alpha=\frac{s_F^{\mathrm{target}}-s_a}{s_b-s_a},\qquad
p_{F,d}=(1-\alpha)p_L^a+\alpha p_L^b.
$$

Orientation is interpolated on the quaternion sphere:

$$
q_{F,d}=\mathrm{SLERP}(q_L^a,q_L^b,\alpha).
$$

The resulting position and attitude form the follower's desired dual quaternion. This reference is passed to the same selected controller as the leader's reference. At a figure-eight crossing, chronological history determines which segment to follow; the code does not jump to the nearest unrelated crossing segment.

> **Worked example — following a corner**
>
> Suppose the leader has traveled from $(0,0,1)$ to $(1,0,1)$ and then to $(1,1,1)$. With $d=1\,\mathrm{m}$, the follower's target is $(1,0,1)$: one metre back along the actual turn. A fixed world offset would generally give a different point.

`dq_control/path_following.py` implements this behavior. Initial history contains a seed segment between the two starting positions. Live-control runs retain extra history so increasing the gap can recover older points. Until sufficient history exists, the target is limited to the available segment. The gap is a reference arc length, not guaranteed physical separation or collision avoidance.

### 10.2 Original offset formulation and worked examples

The following leader equations remain applicable. The fixed-offset follower derivation and examples are retained for the explicit `world` formation mode; they are no longer the default follower reference.

The formation controller defines the desired pose that each vehicle should track.

Let

$$
j\in\{L,F\}
$$

where $L$ denotes the leader and $F$ denotes the follower.

### Leader

The leader follows a predefined position trajectory

$$
p_{Ld}(t)
$$

and a desired attitude

$$
\bar{q}_{Ld}(t)
$$

The leader's desired pose is represented by the dual quaternion

$$
Q_{Ld} = \bar{q}_{Ld}(t) + \varepsilon\frac{1}{2}\bar{p}_{Ld}(t) \circ \bar{q}_{Ld}(t)
$$

where $\varepsilon$ is the dual unit satisfying

$$
\varepsilon^2=0
$$

Thus, $Q_{Ld}$ simultaneously represents the desired position and orientation of the leader.

### Follower

The follower's desired position is generated from the leader's measured position and a formation offset:

$$
p_{Fd}(t) = p_L(t)+f(t)
$$

Here:

- $p_L(t)$ is the measured leader position.
- $f(t)$ is the desired formation offset.
- $p_{Fd}(t)$ is the resulting desired follower position.

For example, if the follower should remain $2\,\mathrm{m}$ behind the leader:

$$
f(t) = \begin{bmatrix} -2\\ 0\\ 0 \end{bmatrix}
$$

The follower's desired attitude can be specified independently or chosen to follow the leader:

$$
\bar{q}_{Fd}(t) = \bar{q}_L(t)
$$

> **Worked example — computing the follower's desired position**
>
> If the leader is momentarily at $p_L = (3, 4, 0)$ m and the formation offset is $f = (-2, 0, 0)$ m (follower 2 m behind, in the *world* $x$-direction), then simply:
>
> $$
> p_{Fd} = p_L + f = (3,4,0) + (-2,0,0) = (1, 4, 0)\ \text{m}
> $$
>
> This is the **`world`** offset mode: $f$ is a fixed vector regardless of which way the leader is pointed. It works well when the leader's heading stays roughly constant (as it mostly does along the long axis of the Lemniscate), but breaks down on a tightly curving path — a fixed world-frame offset can end up *beside* or even *ahead of* the leader instead of behind it. The **`body`** mode instead rotates $f$ into the leader's current heading before adding it, so the follower always trails directly behind, however the leader is turning. The figure below shows both modes on the paper's Lemniscate:
>
> <p align="center"><img src="assets/fig4_leader_follower_trajectory.png" width="520" alt="Leader-follower formation on the Lemniscate: world-frame vs body-frame offset"></p>
>
> and on the (much more sharply curving) non-paper "potato chip" trajectory, where a body-frame offset aligns separation with heading, while current path mode follows the previously traveled curve:
>
> <p align="center"><img src="assets/fig5_potato_chip_trajectory.png" width="480" alt="Leader-follower formation on the potato-chip trajectory using a body-frame offset"></p>

### Effect of Measurement Noise

Because the follower's desired position depends on the leader's measured position, measurement noise is directly transferred to the follower reference.

If

$$
p_L^m(t) = p_L(t)+n(t)
$$

then

$$
p_{Fd}^m(t) = p_L^m(t)+f(t) = p_L(t)+f(t)+n(t)
$$

Therefore, the follower's reference already contains the leader's measurement noise.

If velocity or acceleration is subsequently obtained using numerical differentiation, high-frequency measurement noise can be amplified further. This explains why the follower's trajectory can appear noisier than the leader's trajectory.

> **Worked example — noise transfer and why differentiation makes it worse**
>
> Continuing the numbers above, suppose the leader's position sensor has measurement noise $n = (0.05, -0.02, 0.01)$ m, so $p_L^m = p_L + n = (3.05, 3.98, 0.01)$. Then the follower's noisy desired position is
>
> $$
> p_{Fd}^m = p_L^m + f = (3.05,3.98,0.01) + (-2,0,0) = (1.05,\,3.98,\,0.01) = p_{Fd} + n
> $$
>
> — the *exact same noise vector* $n$ appears in the follower's reference, unfiltered, confirming the algebra above. Now suppose the follower's velocity feedforward is obtained by naive finite differencing at the controller's default 48 Hz rate ($dt \approx  0.0208$ s), and each axis of $n$ has standard deviation $\sigma  = 0.05$ m. Since differencing two independent noisy samples roughly doubles the variance, the resulting **velocity** noise has standard deviation
>
> $$
> \sigma_v \approx \frac{\sqrt{2}\,\sigma}{dt} = \frac{\sqrt2 \times 0.05}{0.0208} \approx 3.4\ \text{m/s}
> $$
>
> — a **large** amount of noise for a drone that might only be flying at 1-2 m/s! This is precisely why `FollowerTrajectory` in `dq_control/trajectories.py` exposes `vel_smoothing` and `heading_smoothing` options (§12.1): low-pass filtering the measured leader state *before* differentiating it is what keeps this amplification from injecting jittery, high-frequency commands into the follower's controller.

### Pose Error for Each Vehicle

Each vehicle runs its own copy of the kinematic controller.

For

$$
j\in\{L,F\}
$$

the dual-quaternion pose error is

$$
\delta Q_j = Q_{jd}^{*} \circ Q_j
$$

Therefore,

$$
\delta Q_L = Q_{Ld}^{*} \circ Q_L
$$

for the leader, and

$$
\delta Q_F = Q_{Fd}^{*} \circ Q_F
$$

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

```text
dq_control/
  quaternion.py, dual_quaternion.py   # pose algebra and integration
  controller.py, gains.py            # analytic baseline
  learned_control.py                 # identification and predictive controller
  path_following.py                  # measured leader-history reference
  trajectories.py, course_trajectories.py
  estimation.py, optimization.py     # optional filtering and path smoothing
simulators/
  base.py, registry.py                # common state/command interface
  leader_follower.py                  # shared experiment loop and live controls
  gym_pybullet.py, mujoco.py, gazebo.py, kinematic.py
  gazebo_wsl_worker.py                # Windows-to-WSL protocol
scripts/
  run_ui.py, run_experiment.py        # desktop and CLI entry points
  train_data_driven.py                # collection, fitting and evaluation
  plot_results.py, analyze_course_concepts.py
models/                              # saved response matrices and metadata
data/                                # identification and evaluation recordings
configs/                             # reference trajectory definitions
results/                             # timestamped experiment logs
assets/                              # retained mathematical illustrations
utils/                               # logging and metrics
tests/                               # algebra, control and integration checks
```

## 12. Folder-by-Folder Walkthrough

### 12.1 `dq_control/` — the math and control library

- `quaternion.py` and `dual_quaternion.py` implement the algebra, pose conversion and integration explained above.
- `controller.py` and `gains.py` retain the analytic baseline and the paper-inspired gain sets.
- `learned_control.py` defines state features, ridge fitting, saved-model validation, dual-quaternion error and bounded predictive command selection.
- `path_following.py` records measured leader poses and constructs the follower's delayed-by-distance target.
- `course_trajectories.py` supplies B-spline and interpolated-pose references. `estimation.py` supplies optional position Kalman filtering; `optimization.py` smooths B-spline control points with CVXPY.

### 12.2 `simulators/` — the simulator bridge

`LeaderFollowerSimulation` reads both drone states, builds their desired poses, calls their controllers, integrates each twist into a one-step target, applies commands, and logs actual motion. `DroneState` contains position, attitude, world velocity and body angular velocity. `VehicleCommand` supplies position/orientation targets plus linear and angular velocity commands.

| Backend | How commands produce movement | Viewer |
|---|---|---|
| PyBullet | gym-pybullet-drones low-level controller and motor dynamics | Native PyBullet window |
| MuJoCo | Bounded world-force translation, roll/pitch stabilization and bounded yaw tracking | Native MuJoCo window |
| Gazebo | ROS 2 actuator messages, motor model and quadrotor physics | Gazebo / WSLg window |
| Kinematic | Direct ideal twist integration, without motors or forces | Separate 3D reference plot in UI mode |

MuJoCo is not equivalent to the rotor-actuated backends. Cross-simulator numbers must not be interpreted as rankings of controller quality on identical physical systems.

### 12.3 `scripts/` — command-line entry points

`train_data_driven.py` collects four training flights and a separate validation flight, fits one backend-specific model, and compares analytic/learned control on additional trajectories. `run_experiment.py` loads a model in learned mode and saves a completed run. `run_ui.py` supports long-running interactive sessions; its callback path does not save the full CLI experiment log. Use the CLI when collecting results for a report.

### 12.4 `utils/` — logging and metrics

`logger.py` saves and loads NumPy archives. `metrics.py` supports the plotting script's MAE/MSE tables. The training report separately records Euclidean position RMSE, rotation-angle RMSE, altitude extrema and optimizer saturation counts. These measures have different meanings and must not be substituted for each other.

### 12.5 `configs/`

The course JSON files define B-spline control points/knots and interpolated-pose endpoints. `experiment_config.yaml` documents baseline parameters; Python modules and CLI parsing remain the authoritative execution settings.

### 12.6 `tests/`

Tests cover quaternion algebra, pose integration, analytic convergence, learned-model fitting and compatibility checks, absence of analytic fallback in learned mode, chronological path following, history retention, live parameter propagation and Windows-to-WSL trail forwarding.

---

## 13. Installation & Setup

### 13.1 Python environments

Use the existing Windows `.venv` for PyBullet, MuJoCo and the desktop panel. For a fresh Windows checkout, from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install mujoco
```

Optional course-path smoothing dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-course.txt
```

The requirements file selects NumPy versions according to Python version; do not replace those environment markers with an old hard-coded version list. Keep Windows and Linux environments separate. Do not recreate a Linux environment over the Windows `.venv`.

### 13.2 PyBullet dependency

The PyBullet backend needs the external `gym-pybullet-drones` package. If its repository is already beside this project:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ../gym-pybullet-drones
```

If it is absent, clone that dependency into the sibling location first:

```powershell
git clone https://github.com/utiasDSL/gym-pybullet-drones.git ../gym-pybullet-drones
```

PyBullet wheel availability depends on the Python/platform combination. A source build can require a C++ compiler; installing only the high-level drone package is not sufficient if PyBullet itself fails to install.

### 13.3 Gazebo through Ubuntu / WSL

The current setup uses Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic. Use the existing [Gazebo setup guide](gazebo/README.md) for system installation. In an Ubuntu terminal at this project root:

```bash
source /opt/ros/jazzy/setup.bash
source .venv-gazebo/bin/activate
```

CLI Gazebo flights run there. The Windows UI instead forwards commands to `gazebo_wsl_worker.py` in WSL, whose current launcher uses `python3` after sourcing ROS. Ensure the worker's Python has its dependencies; installing only into a different virtual environment does not make them visible to it. The worker also receives trail requests so markers are sent in the active Gazebo transport partition.

---

## 14. How to Run an Experiment

### 14.1 Train, then run learned control

In Windows PowerShell, from this project folder:

```powershell
.\.venv\Scripts\python.exe -B scripts/train_data_driven.py --simulator pybullet
.\.venv\Scripts\python.exe -B scripts/run_experiment.py --simulator pybullet --controller learned --trajectory lemniscate --follow_distance 0.8 --duration 60 --gui
```

Replace `pybullet` with `mujoco` in both commands for MuJoCo. Training is headless. Existing compatible model files can be reused; retrain after changing backend dynamics or command semantics. A missing or incompatible model fails explicitly rather than silently selecting analytic control.

In the prepared Ubuntu environment:

```bash
python -B scripts/train_data_driven.py --simulator gazebo
env -u QT_QUICK_BACKEND QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
  python -B scripts/run_experiment.py --simulator gazebo --controller learned \
  --trajectory lemniscate --follow_distance 0.8 --duration 60 --gui \
  --gazebo_render_engine ogre
```

### 14.2 Desktop panel

```powershell
.\.venv\Scripts\python.exe -B scripts/run_ui.py
```

Select the backend, controller and leader route, then start. The panel stays separate from the native viewer. Speed, radius, altitude and gap sliders update the running simulation; gradual transitions avoid discontinuous targets, and the **applied** values are displayed separately. Simulator/controller/route selections are locked until the run stops. Kinematic mode selects analytic control and opens an ideal-motion plot, so it needs no trained model.

MuJoCo starts with its side menus hidden and a cleaner camera view. Its updated controller follows commanded yaw, and the response model was retrained after this dynamics change. Red denotes the leader and blue the follower. Closing the control panel requests simulator cleanup; the UI is not simply a window embedding host.

### 14.3 Main options

| Flag | Default | Purpose |
|---|---|---|
| `--simulator` | `pybullet` | Backend selection |
| `--controller` | `learned` | Learned predictive controller or `analytic` baseline |
| `--model` | Backend response file in `models/` | Explicit compatible learned model |
| `--experiment` | `real_eig` | Baseline gain selection; not a learned-model tuning option |
| `--trajectory` | `lemniscate` | Also `potato_chip`, `bspline`, `interpolated` |
| `--duration` | 30 seconds | CLI flight duration |
| `--ctrl_freq` | 48 Hz | Must match model control timestep |
| `--physics_freq` | 240 Hz | Keep consistent with training/backend configuration |
| `--follower_offset_mode` | `path` | Recorded path; explicit `world`/`body` modes retain offset behavior |
| `--follow_distance` | 0.8 m | Arc-length target gap in path mode |
| `--gui` | Off | Display backend viewer |
| `--no-trails` | Off | Disable actual-motion trails |
| `--position_noise_std` | 0 | Inject measurement noise per position axis |
| `--kalman` | Off | Optional position/velocity estimation |
| `--optimize_path` | Off | CVXPY B-spline control-point smoothing |
| `--output` | `results/` | Completed CLI run location |

Run `python scripts/run_experiment.py --help` in the selected environment for all shape and configuration options. `SimConfig` retains an analytic API default for compatibility; the CLI defaults to learned mode. Explicitly select the desired controller in Python scripts.

### 14.4 Baseline and alternative references

```powershell
.\.venv\Scripts\python.exe -B scripts/run_experiment.py --simulator pybullet --controller analytic --experiment real_eig --follower_offset_mode world --duration 30 --gui
.\.venv\Scripts\python.exe -B scripts/run_experiment.py --simulator pybullet --controller learned --trajectory bspline --duration 30 --gui
.\.venv\Scripts\python.exe -B scripts/run_experiment.py --simulator mujoco --controller learned --trajectory potato_chip --duration 60 --gui
.\.venv\Scripts\python.exe -B scripts/run_experiment.py --simulator kinematic --controller analytic --duration 30
```

The first command explicitly selects the baseline formation; alternative gain sets are `proportional` and `complex_eig`. The last command is headless ideal integration, not a quadrotor physics flight.

---

## 15. Visualizing Results and Metrics

Completed CLI logs include actual/desired poses and the run settings; learned runs also record the model hash. Filenames include simulator, controller, experiment, trajectory and timestamp. Plot an actual saved file with:

```powershell
$run = Get-ChildItem results/pybullet_learned_*.npz |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe -B scripts/plot_results.py --run $run.FullName
```

For position tracking, the training report uses

$$
\mathrm{RMSE}_p=\sqrt{\frac{1}{N}\sum_{k=1}^{N}\|p_k-p_{d,k}\|_2^2}.
$$

For a relative orientation quaternion with vector part $q_v$ and scalar part $q_0$, the shortest rotation-angle error and its RMSE are

$$
\theta_k=2\mathrm{atan2}(\|q_v\|_2,|q_0|),\qquad
\mathrm{RMSE}_{\theta}=\sqrt{\frac{1}{N}\sum_{k=1}^{N}\theta_k^2}.
$$

Position error is measured against each vehicle's own desired pose. It does not by itself prove exact geometric retracing, a guaranteed inter-drone distance, or collision avoidance. All measured simulator results are collected in Section 23 at the end.

## 16. Running the Tests

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q
```

The latest full check during this update reported **38 passed, 1 skipped**, with one CVXPY backend-fallback warning. Passing numerical/unit tests does not establish GUI rendering on every machine or stability at every live-slider setting.

## 17. Design Notes: From Kinematic Twist to Motor Commands

The high-level controller returns body angular rate and world linear velocity. Known dual-quaternion kinematics integrate these into a short-horizon target pose. Each backend then implements that target differently: motor control in PyBullet/Gazebo, simplified force and yaw-torque control in MuJoCo, or direct integration in the ideal reference.

The learned response therefore identifies the **closed-loop backend**, including its existing low-level behavior. It does not identify a universal bare-airframe model. A model trained in one simulator must not be reused by renaming its file for another simulator. Backend and control timestep checks reject incompatible saved models. Other physics and low-level changes still require deliberate retraining.

## 18. Quick Reference

```powershell
# Open the live control panel
.\.venv\Scripts\python.exe -B scripts/run_ui.py

# Generate training data and fit the response model
.\.venv\Scripts\python.exe -B scripts/train_data_driven.py --simulator pybullet

# Refit existing matching recordings and rerun evaluation
.\.venv\Scripts\python.exe -B scripts/train_data_driven.py --simulator pybullet --reuse-data

# Run the trained controller and save a flight
.\.venv\Scripts\python.exe -B scripts/run_experiment.py --simulator pybullet --controller learned --duration 60 --gui
```

See [the command PDF](output/pdf/simulator_running_commands.pdf) for the earlier command reference and [current UI notes](docs/ui_controls.md) for panel behavior. The README and source reflect the latest yaw and UI updates.

---

## 19. Appendix A: Full End-to-End Numeric Walkthrough

This appendix chains together every worked example above into a single, self-contained pass through the pipeline — an illustrative analytic-baseline control step, not the learned optimizer. All numbers below were computed with NumPy and can be reproduced with the ~40-line script summarized at the end of this appendix.

| Stage | Section | Input | Output |
|---|---|---|---|
| 1. Attitude & pose construction | §3.2, §3.5 | $p=(1,2,3)$, 90° about $z$ | $Q = (0,0,0.7071,0.7071) + \varepsilon (1.0607,0.3536,1.0607,-1.0607)$ |
| 2. Pose error | §5 | current $Q$ above, desired $p_d=(1,2,2)$, identity attitude | $\delta q=(0,0,0.7071,0.7071)$, $\delta p^b=(0,0,1)$ |
| 3. Kinematic control law | §6 | $\delta q$, $\delta p^b$, $K_{\omega ,p}=-2I$, $K_{v,p}=-1.5I$, zero integral state | $\bar{\omega}_cmd=(0,0,-1.4142)$, $\bar{v}_cmd=(0,0,-1.5)$ |
| 4. Gain-selection sanity check | §7-8 | toy scalar $a=-2, d=-0.5$, coupling $c=0.5$ | eigenvalues real & negative → non-oscillatory ✓ |
| 5. Acceleration → attitude mapping | §9 | $n=(0,0,1)$, $U_p=(1,0,9.8)$ | $\bar{q}_{φ,\theta }\approx (0,0.0508,0,0.9987)$ (≈5.8° tilt) |
| 6. Leader-follower offset | §10 | $p_L=(3,4,0)$, $f=(-2,0,0)$ | $p_{Fd}=(1,4,0)$; noise $n$ propagates unfiltered into $p_{Fd}^m$ |

Reading the table top to bottom is exactly the "Pose error → Kinematic controller → Acceleration/attitude mapping → Quadrotor commands" cascade from §9's diagram, plus the formation layer from §10 that generates the desired pose fed into row 2 for a follower vehicle.

**Reproducing these numbers.** Everything above follows from three small building blocks: the quaternion Hamilton product $p\circ q = (p_{0}q + q_{0}p + p\times q,\ p_{0}q_{0}-p\cdot q)$ (§3.1), the pose-construction/recovery formulas of §3.5, and the control law of §6. A minimal, self-contained script implementing just those (no repo dependency) looks like:

```python
import numpy as np

def qmul(p, q):
    pv, p0 = p; qv, q0 = q
    return (p0*qv + q0*pv + np.cross(pv, qv), p0*q0 - np.dot(pv, qv))

def qconj(p):
    pv, p0 = p
    return (-pv, p0)

def qrotate(q, v):
    r = qmul(qmul(q, (np.array(v, dtype=float), 0.0)), qconj(q))
    return r[0]

# Attitude: 90 deg about z
q = (np.array([0, 0, np.sin(np.pi/4)]), np.cos(np.pi/4))
print("rotate (1,0,0):", qrotate(q, [1, 0, 0]))   # -> [0. 1. 0.]
```

Running the equivalent (fuller) script against this repository's actual `dq_control.quaternion.Quaternion` and `dq_control.dual_quaternion.DualQuaternion` classes can be used to check the rounded values in this appendix within numerical tolerance, and is a good first sanity check after cloning the repo, before moving on to a full simulated run.

---

## 20. Appendix B: Notation Glossary

| Symbol | Meaning | First used in |
|---|---|---|
| $\bar q, \bar p, \bar\omega, \dots$ | A quaternion (vector part + scalar part, an over-bar marks "this is a quaternion") | §3.1 |
| $q, q_0$ | Vector part and scalar part of quaternion $\bar q$ | §3.1 |
| $\circ$ | Quaternion (or dual quaternion) product — the non-commutative Hamilton product | §3.1 |
| $S(\cdot)$ | Skew-symmetric matrix operator, $S(v)w = v\times w$ | §3.1 |
| $q^*$ | Conjugate (flips the sign of the vector part) | §3.1 |
| $\varepsilon$ | The dual unit, an abstract symbol with $\varepsilon^2=0$ (not "a small number") | §3.3 |
| $Q$ | A dual quaternion, $Q=\bar r + \varepsilon\bar s$ | §3.4 |
| $\mathcal P(Q), \mathcal D(Q)$ | Principal part / dual part of a dual quaternion | §3.4 |
| $\Omega(\bar\omega,\bar v)$ | The twist dual quaternion packaging angular velocity $\bar\omega$ and linear velocity $\bar v$ | §4 |
| $\delta Q, \delta\bar q, \delta p, \delta p^{\,b}$ | Pose error, attitude error, raw position error, position error expressed in the desired body frame | §5 |
| $K_{\omega,p}, K_{v,p}$ | Proportional gain matrices (attitude, position) | §6 |
| $K_{\omega,i}, K_{v,i}$ | Integral-coupling gain matrices (attitude, position) | §6 |
| $K_\eta, K_\xi$ | Integral-state "forgetting"/leakage gain matrices | §6 |
| $\eta, \xi$ | Attitude and position integral (error-accumulator) states | §6 |
| $M_\omega, M_v$ | Linearized closed-loop error-dynamics matrices (attitude, position) | §7 |
| $\lambda(\cdot)$ | Eigenvalue(s) of a matrix | §7 |
| $H$-self-adjoint | $M$ satisfies $HM=M^{\mathsf T}H$ for an indefinite $H$; governs the structure of $M_\omega$'s spectrum | §8 |
| $U_p$ | Desired (feedback-corrected) acceleration fed into the thrust-alignment step | §9 |
| $n_j$ | Body thrust axis of vehicle $j$ | §9 |
| $L, F$ | Subscripts denoting the Leader and Follower vehicle | §10 |
| $f(t)$ | Time-varying formation offset (Leader → Follower) | §10 |
| $p^m, n(t)$ | Measured (noisy) position, and the measurement-noise vector itself | §10 |


---

## 21. Data-Driven Dual-Quaternion Control

### 21.1 What is learned, and why it changes the implementation

The original controller calculates commands from selected gains and pose error. The current learned controller first measures how a simulator responds to commands, fits that response, and uses the fitted model inside every command calculation. Removing the model prevents learned mode from starting: it is not an optional graph or a parallel classroom demonstration.

The leader's route remains a chosen reference. The training recordings describe vehicle response; they are not a prerecorded route that the follower replays. The follower uses the current flight's measured leader history described in Section 10.

Let $R$ rotate body-frame vectors into the world frame. The eight response-state components and six command components are

$$
x_k=\begin{bmatrix}R_k^{\mathsf T}v_k^{w}\\\omega_k^{b}\\(R_k^{\mathsf T}e_z)_{1:2}\end{bmatrix}\in\mathbb R^8,
\qquad
u_k=\begin{bmatrix}R_k^{\mathsf T}v_{k,\mathrm{cmd}}^{w}\\\omega_{k,\mathrm{cmd}}^{b}\end{bmatrix}\in\mathbb R^6.
$$

Here $e_z=(0,0,1)^{\mathsf T}$ is the unit world vertical. Its first two body-frame components describe tilt; these are not acceleration measurements. Position is represented separately by the dual-quaternion pose.

The learned one-step response is

$$
\widehat x_{k+1}=Ax_k+Bu_k+c,
\qquad A\in\mathbb R^{8\times8},\quad B\in\mathbb R^{8\times6},\quad c\in\mathbb R^8.
$$

- $A$ describes how the present response state carries into the next step.
- $B$ describes how commands affect the next response state, including cross-axis effects.
- $c$ is the fitted constant bias.

There are 120 fitted scalar coefficients. These are matrices and a vector, not just three numbers. Command limits are also derived partly from observed command magnitudes, with designed minimum limits and a Gazebo-specific roll/pitch cap.

> **Worked example — the meaning of A, B and c**
>
> Consider a single illustrative velocity axis with coefficients $A=0.8$, $B=0.2$, $c=0.01$. If current velocity is $0.5\,\mathrm{m/s}$ and the velocity command is $1\,\mathrm{m/s}$, the model predicts
>
> $$
> \widehat v_{k+1}=0.8(0.5)+0.2(1)+0.01=0.61\,\mathrm{m/s}.
> $$
>
> These example coefficients are for explanation only. The actual saved model uses all eight state components and six commands together.

### 21.2 Collecting and fitting simulator data

`scripts/train_data_driven.py` flies the analytic baseline with seeded command excitation and records aligned state, command and measured next-state samples. By default, four 20-second training episodes at 48 Hz with two drones provide 7,680 transitions. A separate fifth episode provides 1,920 validation transitions. Two additional reference cases are then flown once with each controller, producing four evaluation flights. Thus the default workflow performs nine flights; only four fit the model. These counts change with training arguments.

Whole episodes are separated for prediction validation. The fitting target is the measured next state, not the analytic controller's action. This is supervised system identification, not reinforcement learning or a neural network.

For feature matrix $\Phi=[X\ U]$, columns are standardized using training means $\mu$ and scales $\sigma$. With $Y$ containing measured next states and $\bar y$ their mean, ridge regression solves

$$
Z_{ij}=\frac{\Phi_{ij}-\mu_j}{\max(\sigma_j,10^{-5})},\qquad
W_z=(Z^{\mathsf T}Z+\lambda I)^{-1}Z^{\mathsf T}(Y-\mathbf1\bar y),\qquad\lambda=10^{-3}.
$$

The implementation uses a linear-system solve, not an explicit inverse. After dividing the rows of $W_z$ by their feature scales, the resulting matrix $W$ is partitioned as

$$
A=W_{1:8,:}^{\mathsf T},\qquad B=W_{9:14,:}^{\mathsf T},\qquad c=\bar y^{\mathsf T}-W^{\mathsf T}\mu.
$$

`LearnedResponse.save()` stores the fitted arrays, command bounds and metadata in a backend-specific NPZ file. Reports record dataset and model hashes, feature rank, validation error and evaluation metrics. `--reuse-data` refits compatible recordings; it does not generate new flights for the identification split.

### 21.3 Dual quaternions remain central

The current and desired poses are unit dual quaternions. Learned control uses the current-frame relative pose

$$
Q_{\mathrm{rel}}=Q^*\circ Q_d.
$$

This order differs from the desired-frame analytic error in Section 5. They must not be interchanged without changing the frame and sign conventions. The relative position and shortest rotation vector form the six-component error

$$
e=\begin{bmatrix}p_{\mathrm{rel}}\\\theta a\end{bmatrix},\qquad
\theta a=2\mathrm{atan2}(\|q_v\|,q_0)\frac{q_v}{\|q_v\|}.
$$

The quaternion sign is selected so $q_0\ge0$; the zero-vector limit returns zero. `dual_pose_error()` implements this calculation. The learned model predicts response; dual-quaternion algebra still represents poses, composes transformations and supplies the tracking error.

### 21.4 Choosing a command during flight

At every control step, `LearnedDQController.compute()` predicts 24 future response steps, normally half a second at 48 Hz. It holds one candidate command constant across that prediction window:

$$
\widehat x_{j+1}=A\widehat x_j+Bu+c,\qquad\widehat x_0=x_k.
$$

Let $\widehat\nu_j$ denote the first six predicted state components and $\nu_d$ the reference twist expressed in the current frame. A local first-order pose forecast integrates predicted twist:

$$
\widehat e_j=e+j\Delta t\,\nu_d-\Delta t\sum_{i=1}^{j}\widehat\nu_i.
$$

The bounded least-squares problem is

$$
\min_{-\ell\le u\le\ell}\;
\sum_{j=1}^{24}\left(\|W_p\widehat e_j\|_2^2+\|W_v(\widehat\nu_j-\nu_d)\|_2^2\right)
+\|0.12(u-\nu_d)\|_2^2+\|0.2(u-u_{\mathrm{previous}})\|_2^2.
$$

The diagonal residual weights are

$$
W_p=\mathrm{diag}(4,4,5,0.5,0.5,1.5),\qquad
W_v=\mathrm{diag}(0.3,0.3,0.4,0.05,0.05,0.15).
$$

These weights, horizon and regularization values are designed parameters, not learned coefficients. SciPy's bounded least-squares solver chooses six command values. Only the current command is applied; the next measured state triggers a new solve. This is a constant-input receding-horizon controller with a local pose approximation, not a full nonlinear dual-quaternion optimization. A failed solve raises an error instead of silently switching to analytic control.

## 22. Current Workflow and Course Concepts

### 22.1 One complete control cycle

1. Read both drones' measured positions, attitudes and velocities from the backend.
2. Apply optional measurement noise and position filtering when enabled.
3. Evaluate the leader reference and add its measured pose to chronological path history.
4. Interpolate the follower target at the selected distance back along that history.
5. Build desired dual-quaternion poses and compute each drone's relative pose error.
6. Use the backend's trained response model to optimize each drone's next command.
7. Integrate a short pose target and send it to the backend's low-level control.
8. Update actual-position trails, status and any enabled recording; repeat.

The CLI defaults to learned control, whereas the reusable simulation configuration retains an analytic default. Select the controller explicitly in scripts when reproducibility matters. The UI's ideal kinematic demonstration uses analytic control.

### 22.2 Live controls and visualization

The desktop panel provides simulator, route and controller selection before launch. During a flight, speed, follower gap, radius and altitude controls update the running simulation with gradual transitions. Phase is accumulated so a speed change does not restart the route. Applied values show the transition, rather than only the requested slider value.

Simulator viewers open separately. PyBullet and MuJoCo use their native viewers; Gazebo runs through ROS in Ubuntu/WSL; the ideal demonstration has a separate Matplotlib view. Red is the leader's actual trail and blue is the follower's actual trail. Bounded trail history avoids unlimited growth. The UI callback path is intended for interaction and does not save the same full flight log as a command-line experiment.

MuJoCo now tracks yaw with bounded torque and uses a cleaner camera/viewer layout. Its translational force model and roll/pitch stabilization remain simplified. Gazebo trail commands are implemented, but visibility in the latest GUI still needs direct visual confirmation.

### 22.3 How the supplied course topics are used

| Course concept | Role in the current project |
|---|---|
| Quaternion operations and dual quaternions | Core pose representation, transforms, integration and tracking error |
| LERP / SLERP | Position and orientation interpolation along recorded leader history; optional reference trajectories |
| Differential equations and first-order approximation | Pose integration and the predictive controller's local pose forecast |
| Linear state prediction | Fitted response model used in every learned command calculation |
| Constrained optimization | Bounded predictive command selection through SciPy |
| B-splines and knot vectors | Optional smooth trajectory references and an evaluation case |
| Kalman filtering | Optional position/velocity estimation with noisy position measurements |
| CVXPY optimization | Optional trajectory smoothing; separate from the learned command solver |
| Taylor examples, nonlinear RK45 and eigenvalue stability | Supporting demonstrations and analytic-baseline analysis; not prerequisites for the learned controller |

The optional Gaussian measurement-noise setting defaults to zero. Training excitation is a separate command perturbation used to identify response. Neither should be confused with learning the leader's route. The project does not require every course topic to appear in the main flight loop.

### 22.4 Interpretation and current limits

The learned model approximates the simulator plus its low-level controller near the collected operating conditions. It is fixed during flight and should be retrained after relevant physics or low-level changes. A fitted model is required separately for each backend; matching backend and timestep checks do not guarantee a model remains suitable after every code change.

The analytic eigenvalue derivation is retained for comparison, not as a proof of stability of the learned controller. Command bounds and path gaps do not guarantee collision avoidance. Recorded improvements are simulation measurements, not real-flight evidence. Gazebo's roll/pitch bound was tuned using development tracking evaluations, so those evaluations are not an untouched final test set. ArduPilot integration documentation and results are deferred to a later update.

---

## 23. Results by Simulation

Results are grouped here at the end. Position RMSE measures actual position against each drone's own desired position; the follower reference is its point on the leader's recorded trail. Lower values indicate closer reference tracking, not necessarily safer separation. Values below are copied from saved reports and rounded, not newly rerun for this documentation update.

$$
\mathrm{RMSE}_p=\sqrt{\frac1N\sum_{k=1}^{N}\|p_k-p_{d,k}\|_2^2}.
$$

Each comparison below uses 20-second evaluation flights. Each learned vehicle performs 960 optimization steps. The two reference cases differ from the training recordings, but the reported development comparisons are not a universal benchmark or a ranking of simulator realism.

### 23.1 PyBullet

| Reference | Controller | Leader position RMSE (m) | Follower position RMSE (m) |
|---|---|---:|---:|
| Lemniscate | Analytic | 0.00995 | 0.01430 |
| Lemniscate | Learned | 0.00896 | 0.01123 |
| B-spline | Analytic | 0.00781 | 0.01005 |
| B-spline | Learned | 0.00813 | 0.00972 |

The learned controller improves follower position RMSE in both recorded cases. B-spline leader error is slightly higher than the analytic baseline; improvement is not uniform across every metric.

Source: [saved PyBullet evaluation report](data/identification/pybullet/report.json).

### 23.2 MuJoCo

| Reference | Controller | Leader position RMSE (m) | Follower position RMSE (m) |
|---|---|---:|---:|
| Lemniscate | Analytic | 0.01108 | 0.01963 |
| Lemniscate | Learned | 0.00381 | 0.00536 |
| B-spline | Analytic | 0.00835 | 0.00930 |
| B-spline | Learned | 0.00232 | 0.00227 |

These are the latest saved results after yaw tracking was added and the response model retrained. They supersede the older fixed-yaw MuJoCo identification report. The simplified force-controlled backend should not be interpreted as motor-level validation. The latest viewer changes have not been visually validated by these numerical runs.

Source: [saved MuJoCo evaluation report](data/yaw_tracking/mujoco/report.json).

### 23.3 Gazebo

| Reference | Controller | Leader position RMSE (m) | Follower position RMSE (m) |
|---|---|---:|---:|
| Lemniscate | Analytic | 0.03928 | 0.10123 |
| Lemniscate | Learned | 0.01693 | 0.03051 |
| B-spline | Analytic | 0.03273 | 0.08863 |
| B-spline | Learned | 0.01486 | 0.02093 |

These measurements use the tuned roll/pitch command cap. In the lemniscate case, the learned follower reaches a command bound on 810 of 960 steps, so the improvement comes with substantial constraint activity. GUI launch and trail appearance are separate from numerical tracking validation; latest trail visibility remains to be checked.

Source: [saved Gazebo evaluation report](data/identification/gazebo/report.json).

### 23.4 Ideal kinematic simulation

This backend provides a reference for pose integration and path-following behavior without motor dynamics. The UI uses the analytic controller and a separate 3D plot. There is no comparable saved identification/evaluation report presented here, so no physics-performance RMSE is claimed. Algebra, integration and history behavior are covered by automated tests.

### 23.5 Longer runs and verification status

The historical [60-second validation summary](results/learned_validation/summary.json) records learned position RMSE of 0.01060 m / 0.01404 m for PyBullet and 0.02136 m / 0.03425 m for Gazebo (leader / follower). These runs predate the latest UI changes and do not validate the current viewer appearance. Older MuJoCo values in that summary predate yaw retraining and are not the current MuJoCo result.

The recorded automated-check status before this README edit was **38 passed, 1 skipped**, with a CVXPY backend-fallback warning. This documentation update does not constitute a new simulation or test run. Real-flight validation and ArduPilot results are not included.
