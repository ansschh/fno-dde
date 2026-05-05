# Table 5: OOD Generalization Gaps

| Family | ID median | OOD-delay | Gap | OOD-history | Gap | OOD-horizon | Gap |
|--------|----------:|----------:|----:|------------:|----:|------------:|----:|
| Hutchinson | 0.1209 | 0.8455 | 7.0× | 0.2877 | 2.4× | 0.4955 | 4.1× |
| Linear2 | 0.6118 | 0.9209 | 1.5× | 0.9934 | 1.6× | 1.6977 | 2.8× |
| Van der Pol | 0.2962 | 0.7600 | 2.6× | 1.1742 | 4.0× | 1.2710 | 4.3× |
| DistUniform | 0.0886 | 0.4539 | 5.1× | 0.9996 | 11.3× | 0.2163 | 2.4× |
| DistExp | 0.0770 | 0.3246 | 4.2× | 0.2976 | 3.9× | 0.1130 | 1.5× |

**Gap:** Ratio of OOD median to ID median (higher = worse generalization).

**Key findings:**
- OOD-delay shows largest gaps (extrapolation to unseen delay values)
- OOD-history moderate gaps (different history function class)
- OOD-horizon often < 1× (shorter relative prediction, easier)

**Location:** Main paper, Section 5