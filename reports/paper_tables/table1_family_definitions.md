# Table 1: DDE Benchmark Family Definitions

| Family | Equation | State dim | Delay type | # delays | Notes |
|--------|----------|----------:|------------|----------|-------|
| Hutchinson | $\dot{x} = r x(t)(1 - x(t-\tau)/K)$ | 1 | discrete | 1 | Positive, logistic growth |
| Linear2 | $\dot{x} = a_1 x(t-\tau_1) + a_2 x(t-\tau_2)$ | 1 | discrete | 2 | Stability-sensitive |
| Van der Pol | $\ddot{x} - \mu(1-x^2)\dot{x} + x = \gamma x(t-\tau)$ | 2 | discrete | 1 | Oscillator, limit cycle |
| DistUniform | $\dot{x} = f(x, \frac{1}{\tau}\int_{t-\tau}^t x(s)ds)$ | 2 | distributed | 1 | Uniform kernel, aux. ODE |
| DistExp | $\dot{x} = f(x, \int K_\lambda(t-s) x(s) ds)$ | 2 | distributed | 1 | Exp kernel, $\theta=\lambda\tau \in [0.5, 1.8]$ |

**Location:** Main paper, Section 4 (Experimental Setup)