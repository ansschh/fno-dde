"""
Solver Health Logging and Statistics

Tracks solver performance metrics:
- Success/failure rates
- Fallback usage
- Timing statistics
- Rejection reasons
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import time


@dataclass
class SolveAttempt:
    """Record of a single solve attempt."""
    sample_id: int
    success: bool
    attempt_id: int  # Which solver in fallback ladder (1-4, 0 = all failed)
    retcode: str
    wall_time: float
    max_state: float
    min_state: float
    n_steps: Optional[int] = None
    rejection_reason: Optional[str] = None


@dataclass 
class SolverHealthLog:
    """
    Accumulates solver health data during dataset generation.
    """
    family: str
    split: str
    attempts: List[SolveAttempt] = field(default_factory=list)
    rejection_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def log_success(
        self,
        sample_id: int,
        attempt_id: int,
        wall_time: float,
        max_state: float,
        min_state: float,
        n_steps: Optional[int] = None,
    ):
        """Log a successful solve."""
        self.attempts.append(SolveAttempt(
            sample_id=sample_id,
            success=True,
            attempt_id=attempt_id,
            retcode="Success",
            wall_time=wall_time,
            max_state=max_state,
            min_state=min_state,
            n_steps=n_steps,
        ))
    
    def log_failure(
        self,
        sample_id: int,
        rejection_reason: str,
        wall_time: float = 0.0,
    ):
        """Log a failed solve."""
        self.attempts.append(SolveAttempt(
            sample_id=sample_id,
            success=False,
            attempt_id=0,
            retcode="Failed",
            wall_time=wall_time,
            max_state=float('nan'),
            min_state=float('nan'),
            rejection_reason=rejection_reason,
        ))
        self.rejection_counts[rejection_reason] += 1
    
    def log_qc_rejection(self, reason: str):
        """Log a QC rejection (sample generated but failed QC)."""
        self.rejection_counts[f"qc_{reason}"] += 1
    
    def save(self, path: Path):
        """Save log to JSON file."""
        data = {
            "family": self.family,
            "split": self.split,
            "attempts": [asdict(a) for a in self.attempts],
            "rejection_counts": dict(self.rejection_counts),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "SolverHealthLog":
        """Load log from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        log = cls(family=data["family"], split=data["split"])
        log.attempts = [SolveAttempt(**a) for a in data["attempts"]]
        log.rejection_counts = defaultdict(int, data["rejection_counts"])
        return log


def compute_solver_stats(log: SolverHealthLog) -> Dict:
    """
    Compute summary statistics from solver health log.
    
    Returns:
        Dictionary with:
        - acceptance_rate
        - fallback_distribution
        - timing_stats (p50, p95, p99)
        - state_bounds
        - rejection_taxonomy
    """
    if len(log.attempts) == 0:
        return {"error": "No attempts logged"}
    
    successes = [a for a in log.attempts if a.success]
    n_total = len(log.attempts)
    n_success = len(successes)
    
    # Acceptance rate
    acceptance_rate = n_success / n_total if n_total > 0 else 0.0
    
    # Fallback distribution
    fallback_counts = defaultdict(int)
    for a in successes:
        fallback_counts[a.attempt_id] += 1
    
    fallback_distribution = {
        f"attempt_{k}": v / n_success if n_success > 0 else 0.0
        for k, v in sorted(fallback_counts.items())
    }
    
    # Timing stats
    times = [a.wall_time for a in successes if a.wall_time > 0]
    if times:
        times = np.array(times)
        timing_stats = {
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "mean": float(np.mean(times)),
            "total": float(np.sum(times)),
        }
    else:
        timing_stats = {}
    
    # State bounds
    max_states = [a.max_state for a in successes if np.isfinite(a.max_state)]
    min_states = [a.min_state for a in successes if np.isfinite(a.min_state)]
    
    state_bounds = {}
    if max_states:
        state_bounds["max_state_p95"] = float(np.percentile(max_states, 95))
        state_bounds["max_state_max"] = float(np.max(max_states))
    if min_states:
        state_bounds["min_state_p5"] = float(np.percentile(min_states, 5))
        state_bounds["min_state_min"] = float(np.min(min_states))
    
    # Rejection taxonomy
    rejection_taxonomy = dict(log.rejection_counts)
    
    return {
        "n_total": n_total,
        "n_success": n_success,
        "acceptance_rate": acceptance_rate,
        "fallback_distribution": fallback_distribution,
        "timing_stats": timing_stats,
        "state_bounds": state_bounds,
        "rejection_taxonomy": rejection_taxonomy,
    }


def aggregate_health_logs(logs: List[SolverHealthLog]) -> Dict:
    """Aggregate stats across multiple logs (e.g., all shards)."""
    combined = SolverHealthLog(family="combined", split="all")
    
    for log in logs:
        combined.attempts.extend(log.attempts)
        for reason, count in log.rejection_counts.items():
            combined.rejection_counts[reason] += count
    
    return compute_solver_stats(combined)
