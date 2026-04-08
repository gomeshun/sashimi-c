# Title: Orbit-averaged, physically motivated spatial-distribution model for subhalo ensembles in a spherical host potential

Purpose and scope
- This note lays out a complete, rigorous theoretical framework for augmenting a semi-analytical subhalo inference model (e.g., SASHIMI-C) with spatial (radial) distributions, without integrating individual orbits.
- We define all symbols and normalizations, state physical assumptions, assemble foundational relations, and derive the governing equation for the ensemble’s evolution in integral-of-motion space.
- The construction follows the physics used in SatGen (e.g., infall at the virial boundary; Chandrasekhar dynamical friction; tidal truncation and disruption) but recasts them in an orbit-averaged, deterministic continuum formulation suited to SASHIMI’s noise-free, high-resolution philosophy.

## 0. Notation and conventions
- Units: We keep $G$ explicit. Time $t$ and redshift $z$ are inter-convertible via cosmology:
  $$
  \frac{dt}{dz} \;=\; -\,\frac{1}{(1+z)\,H(z)} \,,
  $$
  with $H(z)$ the Hubble rate.
- Spherical host halo at time $t$:
  - $M_{200}(t)$: mass enclosed within $R_{200}(t)$, where mean density is $200\,\rho_c(t)$ (critical density).
  - $R_{200}(t) \;=\; \left[\dfrac{3\,M_{200}(t)}{4\pi \times 200 \times \rho_c(t)}\right]^{1/3}$.
  - $c(t) = c\!\left(M_{200}(t),\,t\right)$: concentration (from a chosen $c$–$M$–$z$ relation).
  - $\Phi(r;\,t)$: gravitational potential with $\Phi(\infty;\,t)=0$. Negative for bound orbits.
  - $M_{\rm host}(<r;\,t)$: enclosed mass profile of the host.
- Subhalo (satellite) internal structure:
  - At time $t$, subhalo has $\big(m(t),\,r_s(t),\,\rho_s(t)\big)$, where $m(t)$ is the mass inside its tidal radius $r_t(t)$.
  - The internal density profile can be modeled (e.g., NFW) and updated via a mass-loss model.
- Kinematics:
  - $r$: spherical radius.
  - $v$: speed; $v_r$: radial component; $v_t$: tangential component; $v^2 = v_r^2 + v_t^2$.
  - $E$: specific orbital energy, $E = \Phi(r;\,t) + \dfrac{v^2}{2}$ ($E<0$ for bound orbits).
  - $L$: specific angular momentum, $L = \lvert\mathbf{r} \times \mathbf{v}\rvert = r\,v_t$.
  - Effective potential: $V_{\rm eff}(r;\,L,\,t) = \Phi(r;\,t) + \dfrac{L^2}{2\,r^2}$.
  - Radial motion: $v_r^2(r;\,E,\,L,\,t) = 2\,\big[E - V_{\rm eff}(r;\,L,\,t)\big]$.
  - Turning points (pericenter $r_p$ and apocenter $r_a$) are roots of $v_r^2=0$ with $r_p \le r \le r_a$.
- Ensemble:
  - We represent the subhalo population as a continuum distribution in $(E, L)$ at time $t$ with density $N(E, L, t)$ (defined precisely in §4).

## 1. Host-halo model (spherical; NFW example)
- Host density (NFW):
  $$
  \rho_{\rm host}(r;\,t) \;=\; \frac{\rho_0(t)}{\big(r/r_{s,{\rm h}}\big)\,\big(1 + r/r_{s,{\rm h}}\big)^2}\,,
  \qquad
  r_{s,{\rm h}}(t) \;=\; \frac{R_{200}(t)}{c(t)}\,.
  $$
  $$
  \rho_0(t) \;=\; \frac{M_{200}(t)}{4\pi\,r_{s,{\rm h}}^3(t)\,\Big[\ln(1+c) - \frac{c}{1+c}\Big]} \,.
  $$
- Enclosed mass:
  $$
  M_{\rm host}(<r;\,t) \;=\; M_{200}(t)\,
  \frac{\ln(1+x) - \dfrac{x}{1+x}}{\ln(1+c) - \dfrac{c}{1+c}} \,,
  \qquad x \equiv \frac{r}{r_{s,{\rm h}}(t)} \,.
  $$
- Potential with $\Phi(\infty)=0$:
  $$
  \Phi(r;\,t) \;=\; -\,\frac{G\,M_{200}(t)}{R_{200}(t)\,\Big[\ln(1+c) - \dfrac{c}{1+c}\Big]}\;
  \frac{\ln(1+x)}{x}\,,
  \qquad x \equiv \frac{r}{r_{s,{\rm h}}(t)} \,.
  $$

## 2. Orbital dynamics and orbit-averaged kernels
2.1. Radial period and time-fraction (PDF) along the orbit
- For a given $(E, L, t)$, define the radial period:
  $$
  T_r(E, L, t) \;=\; 2 \int_{r_p}^{r_a} \frac{dr}{\lvert v_r(r;\,E,\,L,\,t)\rvert}\,.
  $$
- Time-average probability density in radius (per unit volume).
  Over one full radial period, the time-averaged fraction in shell $[r, r+dr]$ gives:
  $$
  p_V(r \mid E, L, t) \;=\;
  \begin{cases}
  \dfrac{2}{T_r(E, L, t)\;4\pi r^2\;\lvert v_r(r;\,E,\,L,\,t)\rvert}\,, & r_p \le r \le r_a\,,\\[6pt]
  0\,, & \text{otherwise.}
  \end{cases}
  $$
- Normalization:
  $$
  \int p_V(r \mid E, L, t)\,dV \;=\;
  \int_{r_p}^{r_a} \frac{2}{T_r}\frac{dr}{\lvert v_r\rvert} \;=\; 1\,.
  $$

2.2. Mapping from $(E, L)$ ensemble to radial density
- Let $N(E, L, t)$ be the ensemble’s number (weight) density per unit $E$ and $L$.
  Then the 3D number density of subhalos at radius $r$ is
  $$
  n(r, t) \;=\; \iint N(E, L, t)\; p_V(r \mid E, L, t)\; dE\,dL\,.
  $$
- Consistency:
  $$
  \int n(r, t)\,dV \;=\; \iint N(E, L, t)\,dE\,dL\,.
  $$

## 3. Physical source and sink processes
3.1. Infall (source at the virial boundary)
- Physical picture: New satellites are injected when they cross the host virial boundary for the first time (r ≈ Rvir(tacc)), with an initial distribution of velocities characterized by literature-based P(V/Vvir) and P(Vr/V) at infall (e.g., Jiang+2015; Li+2020 families).
- Transformation to (E, L): Given host Φ and Vcirc at Rvir, sampled (V, Vr, Vt) define E and L, hence a joint P(E, L | tacc).
- Injection term for a group $g$:
  $$
  S_{\rm infall}(E, L, t) \;=\; \sum_g W_g\; P_g(E, L)\; \delta\!\big(t - t_{{\rm acc},g}\big)\,,
  \qquad \iint P_g(E,L)\,dE\,dL \;=\; 1\,.
  $$

3.2. Dynamical friction (deterministic drift in $E$ and $L$)
- Instantaneous Chandrasekhar formula (for an object of mass m in a background with local density ρ, approximately Maxwellian velocities):
  $$
  \mathbf{a}_{\rm df} \;=\; -\,4\pi\,G^2\,m\,\ln\Lambda\;\rho(<v)\;\frac{\mathbf{v}}{v^3}\,,
  $$
  with $X \equiv \dfrac{v}{\sqrt{2}\,\sigma}$ and $\dfrac{\rho(<v)}{\rho} \simeq \operatorname{erf}(X) - \dfrac{2X}{\sqrt{\pi}}e^{-X^2}$ for a Maxwellian.
- Energy-loss rate and angular-momentum-loss rate:
  $$
  \frac{dE}{dt} \;=\; \mathbf{a}_{\rm df}\!\cdot\!\mathbf{v} \;=\; -\,4\pi\,G^2\,m\,\ln\Lambda\;\frac{\rho(<v)}{v}\,,
  $$
  $$
  \frac{dL}{dt} \;\approx\; -\,4\pi\,G^2\,m\,\ln\Lambda\;\rho(<v)\;\frac{L}{v^3}\,.
  $$
- Orbit-averaged drift coefficients (quasi-static and adiabatic approximation):
  $$
  V_E(E, L, t) \;=\; \int p_V(r \mid E, L, t)\;\Big[\frac{dE}{dt}(r;E,L,t)\Big]\; dV\,,
  $$
  $$
  V_L(E, L, t) \;=\; \int p_V(r \mid E, L, t)\;\Big[\frac{dL}{dt}(r;E,L,t)\Big]\; dV\,.
  $$

3.3. Tidal truncation and disruption (sink)
- Tidal (Jacobi) radius at instantaneous position $r$:
  $$
  r_t(r;\,t) \;\approx\; r\;\left[\frac{m_{\rm sub}(<r_t;\,t)}{\big(2 - \dfrac{d\ln M_{\rm host}}{d\ln r}\big)\,M_{\rm host}(<r;\,t)}\right]^{1/3}.
  $$
- Subhalo internal mass (NFW subhalo with $\big(r_s(t), \rho_s(t)\big)$):
  $$
  m_{\rm sub}(<r_t;\,t) \;=\; 4\pi\,\rho_s(t)\,r_s^3(t)\;\Big[\ln\big(1+x_t\big) - \frac{x_t}{1+x_t}\Big]\,,
  \qquad x_t \equiv \frac{r_t}{r_s(t)}\,.
  $$
- Disruption criterion:
  $$
  c_t(r;\,t) \;\equiv\; \frac{r_t(r;\,t)}{r_s(t)}\,,
  \qquad \text{disrupt if } c_t < c_{t,{\rm th}}\,.
  $$
- Orbit-averaged disruption rate:
  Pericenter approximation:
  $$
  \Gamma_{\rm disr}(E, L, t) \;=\;
  \begin{cases}
  \dfrac{1}{T_r(E,L,t)}\,, & c_t\!\big(r_p(E,L,t);\;t\big) < c_{t,{\rm th}},\\[6pt]
  0\,, & \text{otherwise.}
  \end{cases}
  $$
  Time-fraction approximation:
  $$
  \Gamma_{\rm disr}(E, L, t) \;=\; \frac{1}{T_r}\;\oint dt\;\mathbf{1}\!\big[c_t(r(t);t)<c_{t,{\rm th}}\big]
  \;=\; \frac{1}{T_r}\;\int p_V(r \mid E, L, t)\;\mathbf{1}\!\big[c_t(r;t)<c_{t,{\rm th}}\big]\; dV\,.
  $$

3.4. Mass evolution and structural response (external ODE)
- We assume the subhalo’s mass m(t) and structural parameters (rs(t), rhos(t)) follow an external, physically calibrated evolution model (e.g., SASHIMI’s perturbative solution “pert2_shanks” or an ODE solver), which already encapsulates average tidal stripping and profile change (profile_change option).
- This external evolution feeds back into VE, VL (through the factor m(t) and the speed-dependent ρ(<v)/v factors) and into Γ_disr (through m_sub(<rt) in ct and rs(t)).

## 4. Ensemble description and normalizations
- Ensemble density $N(E, L, t)$:
  - $N$ is defined such that $N(E, L, t)\,dE\,dL$ equals the expected effective number of surviving subhalos whose invariants lie in $[E, E+dE]\times[L, L+dL]$ at time $t$.
  - Total surviving weight:
    $$
    N_{\rm tot}(t) \;=\; \iint N(E, L, t)\,dE\,dL\,.
    $$
- Radial number density:
  $$
  n(r, t) \;=\; \iint N(E, L, t)\;p_V(r \mid E, L, t)\; dE\,dL\,.
  $$
  For any shell $[r_i, r_{i+1}]$,
  $$
  N_{\rm shell} \;=\; \int_{r_i}^{r_{i+1}} n(r, t)\;4\pi r^2\,dr\,.
  $$
- Infall injection distributions:
  $$
  S_g(E, L, t) \;=\; W_g\,P_g(E, L)\,\delta\!\big(t - t_{{\rm acc},g}\big)\,,
  \qquad \iint P_g\,dE\,dL = 1\,.
  $$

## 5. Governing equation (derivation)
5.1. Conservation form in $(E, L)$ space
- Continuity (drift-only Fokker–Planck):
  $$
  \frac{\partial N}{\partial t} \;+\; \frac{\partial\big(V_E\,N\big)}{\partial E} \;+\; \frac{\partial\big(V_L\,N\big)}{\partial L}
  \;=\; S_{\rm infall}(E, L, t) \;-\; \Gamma_{\rm disr}(E, L, t)\,N\,.
  $$

5.2. Expressions for the coefficients
- Drift velocities:
  $$
  V_E(E, L, t) \;=\; \int p_V(r \mid E, L, t)\;\Big[ -\,4\pi\,G^2\,m(t)\,\ln\Lambda\;\frac{\rho(<v)}{v}\Big]\; dV\,,
  $$
  $$
  V_L(E, L, t) \;=\; \int p_V(r \mid E, L, t)\;\Big[ -\,4\pi\,G^2\,m(t)\,\ln\Lambda\;\rho(<v)\;\frac{L}{v^3}\Big]\; dV\,.
  $$
- Disruption:
  $$
  \Gamma_{\rm disr}(E, L, t) \;\text{ as in §3.3 with current subhalo structure.}
  $$

5.3. Boundary and initial conditions
- $L$ domain: $0 \le L \le L_{\rm circ}(E;\,t)$, where $L_{\rm circ}$ is angular momentum of the circular orbit with energy $E$ at time $t$.
- $E$ domain: $E_{\rm min}(t) < E < 0$ for bound orbits ($E=0$ at the escape boundary $\Phi(\infty)=0$).

## 6. From the PDE to observables
- Solve the PDE for $N(E, L, t)$ from early times to $t_0$ $(z=0)$.
- Compute $n(r, t_0) \;=\; \iint N(E, L, t_0)\, p_V(r \mid E, L, t_0)\, dE\,dL$.
- Shell counts, cumulative counts, and projected surface density $\Sigma(R)$ follow:
  $$
  N(<r) \;=\; \int_{0}^{r} n(r', t_0)\;4\pi\,r'^2\,dr'\,,
  $$
  $$
  \Sigma(R) \;=\; 2 \int_{R}^{\infty} \frac{n(r, t_0)\,r\,dr}{\sqrt{r^2 - R^2}}\,.
  $$
- Optional: If needed for annihilation/decay signals, combine with subhalo internal $J$ factors computed from $\big(m, r_s, \rho_s\big)$ at $t_0$, aggregated over radius via $n(r, t_0)$.

## 7. Assumptions, validity, and extensions
- Spherical host, quasi-static evolution:
  - The host is treated as spherically symmetric at each time step; its evolution is slow compared to orbital periods (adiabatic approximation).
- No diffusion terms:
  - The minimal model includes only deterministic drifts $V_E, V_L$; no stochastic diffusion in $E$ or $L$. This is justified if subhalo–subhalo encounters and random shocks are subdominant to systematic dynamical friction and tidal effects. If needed, a diffusion term can be added in standard Fokker–Planck form:
    $$
    \frac{\partial N}{\partial t}
    + \frac{\partial\!\big(V_E N\big)}{\partial E}
    + \frac{\partial\!\big(V_L N\big)}{\partial L}
    \;=\;
    \frac{\partial^2\!\big(D_{EE} N\big)}{\partial E^2}
    + 2\,\frac{\partial^2\!\big(D_{EL} N\big)}{\partial E\,\partial L}
    + \frac{\partial^2\!\big(D_{LL} N\big)}{\partial L^2}
    + S_{\rm infall} - \Gamma_{\rm disr}\,N\,,
    $$
    with diffusion coefficients $D$ computed from a specified stochastic process. We avoid this to minimize model dependence.
- Infall distributions:
  - $P(E, L \mid t_{\rm acc})$ derives from physically motivated $P(V/V_{\rm vir}, V_r/V)$ at the virial boundary; a single literature-based choice with fixed hyperparameters can be adopted to avoid freedom creep.
- Coulomb logarithm:
  - Take $\ln\Lambda$ by a physical prescription (e.g., $\ln(1+M_{\rm host}/m)$) and keep it fixed unless calibrated for a specific application.
- Tidal disruption criterion:
  - Use the same $c_{t,{\rm th}}$ as in SASHIMI to ensure consistency. Both pericenter and time-fraction forms are admissible and parameter-free beyond $c_{t,{\rm th}}$.

## 8. Algorithmic pathway (for implementation; not part of the derivation)
- Time grid $t_k$ (e.g., uniform in cosmic time or adaptive to host evolution).
- For each $t_k$:
  1) Set host $\big(M_{200}, R_{200}, c\big)$ and compute $\Phi(r;\,t_k)$, $V_c(r;\,t_k)$, $\sigma(r;\,t_k)$.
  2) Precompute orbit kernels on an $(E, L)$ grid:
     - $r_p, r_a, T_r(E, L, t_k),\; p_V(r \mid E, L, t_k)$.
     - $V_{E,{\rm per\;mass}}(E, L, t_k)$, $V_{L,{\rm per\;mass}}(E, L, t_k)$ by orbit-averaging instantaneous Chandrasekhar rates with $m=1$.
  3) For each SASHIMI group $g$:
     - Update $m_g(t_k)$, $\big(r_{s,g}, \rho_{s,g}\big)(t_k)$ from the external mass-loss/structure ODE.
     - If $t_k = t_{{\rm acc},g}$, inject $W_g\,P_g(E, L)$ into $N_g(E, L, t_k^{+})$.
  4) Evolve $N_g$ via upwind conservative scheme:
     $$
     N_g^{k+1} \;=\; N_g^{k}
     - \Delta t \left[\frac{\partial\!\big(V_{E,g}\,N_g\big)}{\partial E}
     + \frac{\partial\!\big(V_{L,g}\,N_g\big)}{\partial L}\right]
     - \Delta t\;\Gamma_g\,N_g
     + \Delta t\;S_g\,,
     $$
     where $V_{E,g} = m_g(t_k)\,V_{E,{\rm per\;mass}}$, $V_{L,g} = m_g(t_k)\,V_{L,{\rm per\;mass}}$, $\Gamma_g$ from §3.3 using $\big(r_{s,g}, \rho_{s,g}\big)$.
- At $t_0$: Sum over $g$ to get $N(E, L, t_0)$, then map to $n(r, t_0)$.

## 9. Symbol glossary
- $t$: cosmic time; $z$: redshift; $H(z)$: Hubble rate; $\rho_c(z)$: critical density.
- $M_{200}, R_{200}, c$: host spherical-overdensity mass, radius, concentration.
- $r_{s,{\rm h}}$: host scale radius; $\rho_0$: host characteristic density.
- $\Phi(r; t)$: host potential (spherical).
- $E, L$: specific orbital energy and angular momentum.
- $V_{\rm eff}$: effective potential.
- $v_r, v_t$: radial and tangential speeds; $v$: total speed.
- $r_p, r_a, T_r$: pericenter, apocenter, radial period.
- $p_V(r \mid E, L, t)$: orbit-averaged presence probability density per volume.
- $N(E, L, t)$: ensemble density in invariant space (effective-number density).
- $n(r, t)$: 3D number density in physical space.
- $S_{\rm infall}(E, L, t)$: source by infall at $R_{\rm vir}$ at $t_{\rm acc}$.
- $\mathbf{a}_{\rm df}$: Chandrasekhar dynamical-friction acceleration.
- $\rho(<v)$: background density with speeds less than $v$ (from Maxwellian).
- $V_E, V_L$: orbit-averaged drift rates in $E$ and $L$.
- $\ln\Lambda$: Coulomb logarithm.
- $r_t$, $c_t = r_t/r_s$: tidal radius and dimensionless tidal parameter.
- $\Gamma_{\rm disr}$: orbit-averaged disruption rate.
- $m(t), r_s(t), \rho_s(t)$: subhalo mass and structural parameters (from external ODE).

## Appendix A: Core formulae (ready-to-use)
- Host NFW:
  $$
  r_{s,{\rm h}} \;=\; \frac{R_{200}}{c}\,,
  \qquad
  \rho_0 \;=\; \frac{M_{200}}{4\pi\,r_{s,{\rm h}}^3 \left[\ln(1+c) - \dfrac{c}{1+c}\right]}\,,
  $$
  $$
  M_{\rm host}(<r) \;=\; M_{200}\,
  \frac{\ln(1+x) - \dfrac{x}{1+x}}{\ln(1+c) - \dfrac{c}{1+c}}\,,
  \quad x \equiv \frac{r}{r_{s,{\rm h}}}\,,
  $$
  $$
  \Phi(r) \;=\; -\,\frac{G\,M_{200}}{R_{200}\,\left[\ln(1+c) - \dfrac{c}{1+c}\right]}\;\frac{\ln(1+x)}{x}\,.
  $$
- Orbit:
  $$
  V_{\rm eff}(r) \;=\; \Phi(r) + \frac{L^2}{2 r^2}\,,
  \qquad
  v_r(r;\,E,L) \;=\; \sqrt{2\,[E - V_{\rm eff}(r)]}\,,
  $$
  $$
  T_r(E, L) \;=\; 2 \int_{r_p}^{r_a} \frac{dr}{\lvert v_r\rvert}\,,
  \qquad
  p_V(r \mid E, L) \;=\; \frac{2}{T_r \; 4\pi r^2 \lvert v_r\rvert} \;\; (r_p \le r \le r_a)\,.
  $$
- Dynamical friction (instantaneous):
  $$
  \mathbf{a}_{\rm df} \;=\; - 4\pi G^2 m \ln\Lambda \;\rho(<v)\;\frac{\mathbf{v}}{v^3}\,,
  \qquad
  \frac{dE}{dt} \;=\; - 4\pi G^2 m \ln\Lambda \;\frac{\rho(<v)}{v}\,,
  $$
  $$
  \frac{dL}{dt} \;\approx\; - 4\pi G^2 m \ln\Lambda \;\rho(<v)\;\frac{L}{v^3}\,.
  $$
- Orbit-averaged drift:
  $$
  V_E(E, L) \;=\; \int p_V\,dV \;\Big[ - 4\pi G^2 m \ln\Lambda \;\frac{\rho(<v)}{v} \Big]\,,
  \qquad
  V_L(E, L) \;=\; \int p_V\,dV \;\Big[ - 4\pi G^2 m \ln\Lambda \;\rho(<v)\;\frac{L}{v^3} \Big]\,.
  $$
- Tidal radius and disruption:
  $$
  r_t \;=\; r \left[ \frac{ m_{\rm sub}(<r_t) }{ \big(2 - d \ln M_{\rm host}/d \ln r\big)\, M_{\rm host}(<r) } \right]^{1/3},
  \qquad
  c_t \;=\; \frac{r_t}{r_s},
  $$
  $$
  \Gamma_{\rm disr}(E, L) \;=\; \begin{cases} 1/T_r\,, & c_t(r_p) < c_{t,{\rm th}},\\ 0\,, & \text{otherwise.}\end{cases}
  $$
- Governing PDE (minimal, no diffusion):
  $$
  \frac{\partial N}{\partial t} + \frac{\partial(V_E N)}{\partial E} + \frac{\partial(V_L N)}{\partial L}
  \;=\; S_{\rm infall} - \Gamma_{\rm disr}\,N\,.
  $$
- Radial density:
  $$
  n(r, t) \;=\; \iint N(E, L, t)\, p_V(r \mid E, L, t)\, dE\,dL\,.
  $$

## Appendix B: Cosmology relations
- $$
  \rho_c(z) \;=\; \frac{3\,H^2(z)}{8\pi G}\,,
  \qquad
  R_{200}(z) \;=\; \left[\frac{3\,M_{200}(z)}{4\pi \times 200 \times \rho_c(z)}\right]^{1/3},
  \qquad
  \frac{dt}{dz} \;=\; -\frac{1}{(1+z)\,H(z)}\,.
  $$

## Appendix C: Validity and limits
- Timescale separation:
  - The orbit-averaging assumes dynamical friction and mass loss are slow compared to $T_r$, and host evolution is slow compared to $T_r$.
- Spherical symmetry:
  - Non-sphericity and disk potentials can be included by replacing the spherical kernels with axisymmetric ones at the cost of greater complexity; the present theory is for spherical hosts.
- Choice of $\rho(<v)$:
  - Using a Maxwellian with $\sigma(r) \approx V_c(r)/\sqrt{2}$ is a standard approximation and keeps the model parameter-minimal.

This document provides all definitions, assumptions, and derivations necessary to implement a physically anchored, orbit-averaged spatial-distribution solver that is fully compatible with SASHIMI’s architecture while remaining close in spirit to SatGen’s physical prescriptions.