"""Search space definition for parallel configuration grid search."""
from __future__ import annotations

from dataclasses import dataclass, field

from zrt.training.spec.strategy import (
    CPKind, OffloadPolicy, OptKind, PPSched, RecomputePolicy, TPOverlap,
)


@dataclass
class SearchSpace:
    """Defines the dimensions and constraints for parallel config search."""

    tp_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    cp_values: list[int] = field(default_factory=lambda: [1])
    pp_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    ep_values: list[int] = field(default_factory=lambda: [1, 8])
    dp_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    zero_stages: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    pp_schedules: list[PPSched] = field(default_factory=lambda: [PPSched.ONE_F_ONE_B, PPSched.INTERLEAVED, PPSched.DUALPIPE])
    recompute_policies: list[str] = field(default_factory=lambda: ["none", "selective", "full"])
    vpp_chunks_values: list[int] = field(default_factory=lambda: [1, 2, 4])

    micro_batch: int = 1
    global_batch: int = 0
    max_memory_gb: float = 80.0

    def strategies(self, world_size: int) -> list["Strategy"]:
        """Generate all valid Strategy instances for the given world_size."""
        from zrt.training.spec.strategy import Strategy

        results = []
        seen = set()

        for tp in self.tp_values:
            for cp in self.cp_values:
                for pp in self.pp_values:
                    for ep in self.ep_values:
                        total = tp * cp * pp * ep
                        if world_size % total != 0:
                            continue
                        dp = world_size // total
                        if dp not in self.dp_values and dp > 0:
                            continue

                        for zero_stage in self.zero_stages:
                            if zero_stage >= 1 and dp <= 1:
                                continue

                            for sched in self.pp_schedules:
                                for rc in self.recompute_policies:
                                    if sched == PPSched.INTERLEAVED:
                                        for vc in self.vpp_chunks_values:
                                            s = self._make_strategy(
                                                tp, cp, pp, ep, dp, zero_stage,
                                                sched, rc, vc,
                                            )
                                            key = (tp, cp, pp, ep, dp, zero_stage, sched, rc, vc)
                                            if key not in seen:
                                                seen.add(key)
                                                results.append(s)
                                    else:
                                        s = self._make_strategy(
                                            tp, cp, pp, ep, dp, zero_stage,
                                            sched, rc, 1,
                                        )
                                        key = (tp, cp, pp, ep, dp, zero_stage, sched, rc, 1)
                                        if key not in seen:
                                            seen.add(key)
                                            results.append(s)

        return results

    def _make_strategy(self, tp, cp, pp, ep, dp, zero_stage, sched, rc, vpp_chunks):
        from zrt.training.spec.strategy import Strategy, RecomputePolicy

        rc_policy = RecomputePolicy()
        if rc == "selective":
            rc_policy.per_layer = {"moe": {"attn"}, "dense": {"attn"}}
        elif rc == "full":
            rc_policy.per_layer = {"moe": {"full"}, "dense": {"full"}}

        return Strategy(
            tp=tp, cp=cp, pp=pp, ep=ep, dp=dp,
            micro_batch=self.micro_batch,
            global_batch=self.global_batch,
            zero_stage=zero_stage,
            pp_schedule=sched,
            vpp_chunks=vpp_chunks,
            recompute=rc_policy,
        )
