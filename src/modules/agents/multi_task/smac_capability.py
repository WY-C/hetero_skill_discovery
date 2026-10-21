"""Verified SC2-derived static capability profiles for HiSSD offline training.

Offline experience does not contain live SC2 Unit objects. Export one profile for
all source/target maps and only use the profile when the agent-id -> unit-type
ordering is verified stable. Do not infer new map rosters from names.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

FEATURES = (
    'smac_shoot_range', 'smac_sight_range',
    'sc2_ground_range', 'sc2_air_range',
    'movement_speed', 'armor', 'ground_dps', 'air_dps',
    'can_attack_ground', 'can_attack_air',
)
# Fixed scales preserve cross-map comparability. Do not fit to held-out maps.
SCALES = (10., 20., 10., 10., 10., 5., 30., 30., 1., 1.)
CAPABILITY_DIM = len(FEATURES)


def _weapon_stats(unit_type_data: Any, target: int) -> tuple[float, float]:
    """TargetType: Ground=1, Air=2, Any=3. Return max(range), max(DPS)."""
    relevant = [w for w in getattr(unit_type_data, 'weapons', ())
                if int(w.type) in (target, 3)]
    if not relevant:
        return 0., 0.
    attack_range = max(float(w.range) for w in relevant)
    dps = max(
        (float(w.damage) * max(1, int(w.attacks))) / max(float(w.speed), 1e-6)
        for w in relevant
    )
    return attack_range, dps


def snapshot_env(env: Any) -> dict:
    """Call after env.reset(); env must be real SMAC/PySC2 (not offline replay)."""
    if not getattr(env, 'agents', None) or getattr(env, '_controller', None) is None:
        raise RuntimeError('Call env.reset() before extracting capabilities')
    data = {int(unit_id): d for unit_id, d in env._controller.data().unit_stats.items()}
    entries = []
    n_agents = int(env.n_agents)
    for agent_id in range(n_agents):
        unit = env.get_unit_by_id(agent_id)
        unit_type_id = int(unit.unit_type)
        if unit_type_id not in data:
            raise RuntimeError(f'SC2 data() lacks unit_type={unit_type_id}; cannot fabricate stats')
        d = data[unit_type_id]
        ground_range, ground_dps = _weapon_stats(d, 1)
        air_range, air_dps = _weapon_stats(d, 2)
        values = (
            float(env.unit_shoot_range(agent_id)),
            float(env.unit_sight_range(agent_id)),
            ground_range, air_range,
            float(d.movement_speed), float(d.armor),
            ground_dps, air_dps,
            float(ground_range > 0.), float(air_range > 0.),
        )
        entries.append({
            'agent_id': agent_id, 'unit_type_id': unit_type_id,
            'unit_name': str(getattr(d, 'name', '')),
            'raw': list(values),
        })
    return {'agent_count': n_agents, 'agents': entries}


def check_same_roster(first: dict, next_: dict, *, map_name: str) -> None:
    a = [(r['agent_id'], r['unit_type_id']) for r in first['agents']]
    b = [(r['agent_id'], r['unit_type_id']) for r in next_['agents']]
    if a != b:
        raise RuntimeError(
            f'{map_name}: agent->unit_type changed across resets; static map profile '
            'would mislabel heterogeneous agents. Store capabilities per episode instead. '
            f'First={a}; next={b}'
        )
    for left, right in zip(first['agents'], next_['agents']):
        if any(abs(x-y)>1e-5 for x,y in zip(left['raw'],right['raw'])):
            raise RuntimeError(f'{map_name}: capability values changed across resets')


class CapabilityRegistry:
    def __init__(self, manifest_path: str | Path):
        self.path = Path(manifest_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Capability manifest not found: {self.path}')
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if data.get('version') != 1 or tuple(data.get('features', [])) != FEATURES:
            raise ValueError('Incompatible capability manifest schema')
        self.profiles = data['maps']
        self._cache: dict[tuple, torch.Tensor] = {}

    def features_for(self, task: str, *, batch_size: int, n_agents: int,
                     token_count: int, reference: torch.Tensor) -> torch.Tensor:
        if task not in self.profiles:
            raise KeyError(f'Missing capability profile for task={task!r} in {self.path}')
        profile = self.profiles[task]
        if int(profile['agent_count']) != n_agents:
            raise ValueError(f'{task}: expected {n_agents} agents, manifest has {profile["agent_count"]}')
        agents = profile['agents']
        if [int(a['agent_id']) for a in agents] != list(range(n_agents)):
            raise ValueError(f'{task}: noncontiguous or unsorted agent IDs')
        if reference.shape[0] != batch_size * n_agents or reference.ndim != 3:
            raise ValueError(f'Expected common tokens [B*A,T,D]; got {tuple(reference.shape)}')
        if reference.shape[1] != token_count:
            raise ValueError('Token count mismatch')
        key = (task, reference.device, reference.dtype)
        if key not in self._cache:
            raw = torch.tensor([a['raw'] for a in agents], device=reference.device,
                               dtype=reference.dtype)
            if raw.shape != (n_agents, CAPABILITY_DIM) or not torch.isfinite(raw).all():
                raise ValueError(f'{task}: malformed capability values')
            scales = torch.tensor(SCALES, device=reference.device, dtype=reference.dtype)
            self._cache[key] = raw / scales
        features = self._cache[key]
        return (features.unsqueeze(0).expand(batch_size, -1, -1)
                .reshape(batch_size*n_agents, 1, CAPABILITY_DIM)
                .expand(-1, token_count, -1))
