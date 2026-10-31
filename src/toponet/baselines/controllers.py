"""Baselines / controllers.

Computational definitions preserved from the selected experiment; package imports
and obsolete command-line entry points were reorganized. See provenance.
"""

from dataclasses import dataclass, asdict
import math
import numpy as np


def checked_observation(obs):
    mask = np.asarray(obs["action_mask"], dtype=bool)
    rsrp = np.asarray(obs["rsrp_dbm"], dtype=float)
    rates = np.asarray(obs["candidate_user_rate_mbps"], dtype=float)
    serving = obs["serving_cell"]
    t = float(obs["time_s"])
    if (
        mask.ndim != 1
        or not len(mask)
        or rsrp.shape != mask.shape
        or rates.shape != mask.shape
    ):
        raise ValueError("Invalid observation shapes.")
    if (
        isinstance(serving, bool)
        or not isinstance(serving, (int, np.integer))
        or not 0 <= serving < len(mask)
        or not mask[serving]
    ):
        raise ValueError("Serving fallback missing.")
    if (
        not math.isfinite(t)
        or t < 0
        or not np.isfinite(rsrp).all()
        or not np.isfinite(rates).all()
        or (rates < 0).any()
    ):
        raise ValueError("Invalid observation values.")
    return mask, rsrp, rates, int(serving), t


class ImmediateSignalController:
    name = "immediate_signal"

    def reset(self):
        pass

    def act(self, obs):
        mask, rsrp, _, _, _ = checked_observation(obs)
        return int(np.argmax(np.where(mask, rsrp, -np.inf)))

    def configuration(self):
        return {"rule": "Immediately choose strongest allowed signal; diagnostic only."}


@dataclass(frozen=True)
class A3Settings:
    hysteresis_db: float = 3.0
    time_to_trigger_s: float = 0.64


class A3StyleController:
    name = "a3_style"

    def __init__(self, settings=None):
        self.settings = A3Settings() if settings is None else settings
        if (
            not math.isfinite(self.settings.hysteresis_db)
            or self.settings.hysteresis_db < 0
            or not math.isfinite(self.settings.time_to_trigger_s)
            or self.settings.time_to_trigger_s < 0
        ):
            raise ValueError("Invalid A3 settings.")
        self.reset()

    def reset(self):
        self.target = None
        self.condition_since_s = None
        self.last_serving = None
        self.last_time_s = None

    def act(self, obs):
        mask, rsrp, _, serving, t = checked_observation(obs)
        if self.last_time_s is not None and t < self.last_time_s:
            raise ValueError("Reset controller between episodes.")
        if self.last_serving != serving:
            self.target = None
            self.condition_since_s = None
        self.last_serving = serving
        self.last_time_s = t
        candidates = mask.copy()
        candidates[serving] = False
        if not candidates.any():
            self.target = None
            self.condition_since_s = None
            return serving
        target = int(np.argmax(np.where(candidates, rsrp, -np.inf)))
        if rsrp[target] <= rsrp[serving] + self.settings.hysteresis_db:
            self.target = None
            self.condition_since_s = None
            return serving
        if self.target != target:
            self.target = target
            self.condition_since_s = t
        if t - self.condition_since_s + 1e-12 >= self.settings.time_to_trigger_s:
            self.target = None
            self.condition_since_s = None
            return target
        return serving

    def configuration(self):
        return {
            **asdict(self.settings),
            "rule": "Strongest allowed neighbor > serving + hysteresis at consecutive decision samples; target-specific elapsed-time TTT.",
            "sampling_note": "At dt=0.5s, TTT=0.64s first triggers 1.0s after the first qualifying sample if the condition persists.",
            "limits": "No LTE measurement filtering, cell offsets, report latency, protocol or reconnection procedures. No extra weak-signal rescue rule.",
        }


@dataclass(frozen=True)
class LoadAwareSettings:
    switch_margin_mbps: float = 1.0
    minimum_dwell_s: float = 3.0
    weak_signal_threshold_dbm: float = -120.0
    rescue_weak_serving: bool = True


class LoadAwareController:
    name = "load_aware"

    def __init__(self, settings=None):
        self.settings = LoadAwareSettings() if settings is None else settings
        f = self.settings
        if (
            not all(
                math.isfinite(x)
                for x in (
                    f.switch_margin_mbps,
                    f.minimum_dwell_s,
                    f.weak_signal_threshold_dbm,
                )
            )
            or f.switch_margin_mbps < 0
            or f.minimum_dwell_s < 0
            or not isinstance(f.rescue_weak_serving, bool)
        ):
            raise ValueError("Invalid load-aware settings.")
        self.reset()

    def reset(self):
        self.last_time_s = None

    def act(self, obs):
        mask, rsrp, rates, serving, t = checked_observation(obs)
        if self.last_time_s is not None and t < self.last_time_s:
            raise ValueError("Reset controller between episodes.")
        self.last_time_s = t
        candidates = mask.copy()
        candidates[serving] = False
        if not candidates.any():
            return serving
        # Explicit safety exception: while serving signal is weak, choose the
        # strongest allowed target without waiting for dwell or throughput margin.
        if (
            self.settings.rescue_weak_serving
            and rsrp[serving] < self.settings.weak_signal_threshold_dbm
        ):
            return int(np.argmax(np.where(candidates, rsrp, -np.inf)))
        age = obs["time_since_handover_s"]
        age = t if age is None else float(age)
        if not math.isfinite(age) or age < 0:
            raise ValueError("Invalid handover age.")
        if age + 1e-12 < self.settings.minimum_dwell_s:
            return serving
        target = int(np.argmax(np.where(candidates, rates, -np.inf)))
        if rates[target] > rates[serving] + self.settings.switch_margin_mbps:
            return target
        return serving

    def configuration(self):
        return {
            **asdict(self.settings),
            "rule": "Highest CURRENT estimated candidate user rate, strictly exceeding serving rate + margin, after dwell; explicit optional weak-signal rescue.",
            "limits": "Uses simulator-provided current rate estimates, not actual operator measurements; no future traffic, rate persistence or prediction. Candidate rates do not include switching interruption.",
        }


def self_test():
    def obs(
        t=0.0,
        signal=(-90.0, -86.0, -80.0),
        rates=(10.0, 15.0, 20.0),
        mask=(True, True, False),
        serving=0,
        age=None,
    ):
        return {
            "time_s": t,
            "rsrp_dbm": np.array(signal),
            "candidate_user_rate_mbps": np.array(rates),
            "action_mask": np.array(mask),
            "serving_cell": serving,
            "time_since_handover_s": age,
        }

    a = A3StyleController()
    assert a.act(obs()) == 0
    assert a.act(obs(0.5)) == 0 and a.act(obs(1.0)) == 1
    a.reset()
    assert a.act(obs()) == 0
    assert a.act(obs(0.5, signal=(-90.0, -88.0, -80.0))) == 0
    assert a.act(obs(1.0)) == 0 and a.act(obs(1.5)) == 0 and a.act(obs(2.0)) == 1
    a.reset()
    a.act(obs())
    assert a.act(obs(0.5, mask=(True, False, False))) == 0
    assert a.act(obs(1.0)) == 0
    zero = A3StyleController(A3Settings(time_to_trigger_s=0.0))
    assert zero.act(obs()) == 1
    assert zero.act(obs(0.5, signal=(-90.0, -87.0, -80.0))) == 0  # strict > margin
    l = LoadAwareController()
    assert l.act(obs()) == 0 and l.act(obs(2.5)) == 0 and l.act(obs(3.0)) == 1
    assert l.act(obs(3.5, serving=1, mask=(True, True, True), age=0.5)) == 1
    l.reset()
    assert l.act(obs(3.0, rates=(10.0, 11.0, 20.0))) == 0
    l.reset()
    assert l.act(obs(0.0, signal=(-121.0, -100.0, -80.0))) == 1
    l.reset()
    assert (
        l.act(obs(0.0, signal=(-121.0, -100.0, -80.0), mask=(True, False, False))) == 0
    )
    no_rescue = LoadAwareController(LoadAwareSettings(rescue_weak_serving=False))
    assert no_rescue.act(obs(0.0, signal=(-121.0, -100.0, -80.0))) == 0
    assert ImmediateSignalController().act(obs()) == 1
    # Candidate changes reset A3 timer; changing serving clears stale trigger.
    a.reset()
    a.act(obs(mask=(True, True, True)))
    assert a.act(obs(0.5, signal=(-90.0, -80.0, -86.0), mask=(True, True, True))) == 0
    assert a.act(obs(1.0, signal=(-90.0, -80.0, -86.0), mask=(True, True, True))) == 0
    assert a.act(obs(1.5, signal=(-90.0, -80.0, -86.0), mask=(True, True, True))) == 1
    assert (
        a.act(
            obs(
                2.0,
                serving=1,
                signal=(-80.0, -90.0, -100.0),
                mask=(True, True, True),
                age=0.5,
            )
        )
        == 1
    )
    print(
        "Baseline tests passed: masks, elapsed-time TTT, strict margins, condition/target resets, dwell, serving changes and explicit weak-signal rescue."
    )
