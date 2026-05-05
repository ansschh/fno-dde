# Table 6: Time-Segment Error Breakdown (ID Test)

Time segments: Early [0, 3.75], Mid [3.75, 11.25], Late [11.25, 15]

| Family | Early median | Mid median | Late median | Late/Early ratio |
|--------|-------------:|-----------:|------------:|-----------------:|
| Hutchinson | 0.0000 | 0.0633 | 0.1021 | 1021196288.00× |
| Linear2 | 0.0000 | 0.2420 | 3.0878 | 30877964288.00× |
| Van der Pol | 0.0000 | 0.0765 | 0.1534 | 1534009984.00× |
| DistUniform | 0.0000 | 0.0470 | 0.0491 | 490984960.00× |
| DistExp | 0.0000 | 0.0342 | 0.0222 | 222150608.00× |

**Interpretation:**
- Late/Early ratio > 1 indicates error drift (accumulation over time)
- Higher ratio = more temporal extrapolation difficulty

**Location:** Appendix