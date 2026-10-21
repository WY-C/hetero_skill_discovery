#!/usr/bin/env python3
"""CPU-safe unit smoke tests, using actual V3 math and actual profile manifest.

Does not import StarCraftII, PyMARL runtime, or run episodes. Run this before
launching any V3 experiment.  End-to-end SMAC GPU integration is a separate step.
"""
from pathlib import Path
import importlib.util
import sys
import types
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

# Build a minimal package layout without importing any unrelated PyMARL deps.
for name in ('modules', 'modules.agents', 'modules.agents.multi_task',
             'learners', 'learners.multi_task'):
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
cap = load('modules.agents.multi_task.smac_capability',
           ROOT/'src/modules/agents/multi_task/smac_capability.py')

# The V3 adapter's direct parent is the existing V2 capability class. Supply
# only the constructor needed for this unit test; the live installer keeps and
# imports the real V2 implementation without alteration.
class StubV2Adapter(nn.Module):
    def __init__(self, observation_dim, skill_dim, hidden_dim=128,
                 capability_embed_dim=32, capability_manifest_path=None):
        super().__init__()
        self.observation_dim = observation_dim
        self.skill_dim = skill_dim
        self.registry = cap.CapabilityRegistry(capability_manifest_path)
        self.capability_encoder = nn.Sequential(nn.Linear(cap.CAPABILITY_DIM, capability_embed_dim),
                                                nn.ReLU(), nn.Linear(capability_embed_dim, capability_embed_dim))
        self.task = None
        self.n_agents = None
    def set_task(self, task, n_agents):
        self.task = str(task)
        self.n_agents = int(n_agents)

stub = types.ModuleType('modules.agents.multi_task.hissd_capability_agent')
stub.CapabilityConditionedCommonSkillAdapter = StubV2Adapter
stub.HISSDCapabilityAgent = nn.Module
sys.modules[stub.__name__] = stub
team = load('modules.agents.multi_task.hissd_capability_team_agent',
            ROOT/'src/modules/agents/multi_task/hissd_capability_team_agent.py')

manifest = ROOT/'src/config/capability_profiles.json'
registry = cap.CapabilityRegistry(manifest)
source = registry.profiles['2s3z']['agents']
target = registry.profiles['1s3z']['agents']
assert [row['unit_name'] for row in source] == ['Stalker_RL']*2 + ['Zealot_RL']*3
assert [row['unit_name'] for row in target] == ['Stalker_RL'] + ['Zealot_RL']*3
assert len(source) != len(target)

model = team.TeamConditionedCommonSkillAdapter(
    observation_dim=64, skill_dim=64, capability_manifest_path=str(manifest), max_agent=15)
model.set_task('2s3z', 5)
common = torch.randn(2 * 5, 10, 64, requires_grad=True)
obs = torch.randn_like(common)
adapted, delta = model(common, obs)
assert adapted.shape == (10, 10, 64)
assert torch.allclose(adapted, common, atol=1e-7), 'Zero residual init must equal HiSSD'
assert torch.count_nonzero(delta) == 0
assert model.last_delta_rms is None
model.collect_metrics = True
adapted, delta = model(common, obs)
assert model.last_delta_rms is not None and torch.isfinite(model.last_delta_rms)
loss = adapted.square().mean()
loss.backward()
assert model.network[-1].weight.grad is not None
assert model.network[-1].weight.grad.abs().sum() > 0

# A changed roster produces a changed team summary, without ever fitting
# normalization on unseen maps.
for name, n in [('1s3z', 4), ('4s6z', 10), ('2s4z', 6)]:
    model.set_task(name, n)
    c = torch.randn(n, 2, 64)
    out, _ = model(c, torch.zeros_like(c))
    assert out.shape == c.shape
print('PASS: V3 team encoder, manifest rosters, identity init, adapter gradients')
