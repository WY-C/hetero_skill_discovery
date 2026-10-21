"""HiSSD V3: team-conditioned common-skill residual adapter.

No changes to the original HiSSD or capability adapter are needed.  The
original planner's hook binds task context before every call, and this module
replaces only the adapter it calls.  Team statistics use the VERIFIED fixed
roster, not map-name inference or held-out trajectories.
"""
import torch as th
import torch.nn as nn

from .hissd_capability_agent import (
    HISSDCapabilityAgent,
    CapabilityConditionedCommonSkillAdapter,
)
from .smac_capability import CAPABILITY_DIM


class TeamConditionedCommonSkillAdapter(CapabilityConditionedCommonSkillAdapter):
    """Shared residual conditioned on own unit and aggregate team capabilities.

    g(team) = MLP([mean(k), sum(k)/max_agent, N/max_agent]).
    k and all aggregates use the original fixed per-feature capability scales.
    The same team embedding is broadcast to each agent/entity token.  Entity-
    specific inputs c_ij and h_ij remain distinct; the capability k_i belongs
    to the observing agent, not the entity being observed.
    """

    def __init__(self, observation_dim, skill_dim, hidden_dim=128,
                 capability_embed_dim=32, team_embed_dim=32,
                 capability_manifest_path=None, max_agent=15,
                 residual_scale=1.0, use_observation_embedding=True):
        super().__init__(
            observation_dim=observation_dim, skill_dim=skill_dim,
            hidden_dim=hidden_dim, capability_embed_dim=capability_embed_dim,
            capability_manifest_path=capability_manifest_path,
        )
        if max_agent < 1:
            raise ValueError('max_agent must be >=1')
        if residual_scale <= 0:
            raise ValueError('residual_scale must be positive')
        self.max_agent = int(max_agent)
        self.residual_scale = float(residual_scale)
        self.use_observation_embedding = bool(use_observation_embedding)
        self.team_encoder = nn.Sequential(
            nn.Linear(2 * CAPABILITY_DIM + 1, team_embed_dim),
            nn.ReLU(),
            nn.Linear(team_embed_dim, team_embed_dim),
        )
        self.network = nn.Sequential(
            nn.Linear((observation_dim if self.use_observation_embedding else 0) + skill_dim + capability_embed_dim + team_embed_dim,
                      hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, skill_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.collect_metrics = False
        self.last_delta_rms = None
        self.last_common_rms = None
        self.last_team_norm = None

    def forward(self, common_skill, observation_embedding):
        if common_skill.shape != observation_embedding.shape or common_skill.ndim != 3:
            raise ValueError('Expected matching [B*A,T,D] common and observation tensors')
        if common_skill.shape[-1] != self.skill_dim:
            raise ValueError('Unexpected embedding dimension')
        if self.task is None or self.n_agents is None:
            raise RuntimeError('Planner task context must be set before adapter forward')
        if common_skill.shape[0] % self.n_agents:
            raise ValueError('Incorrect flattened batch order / agent count')
        if self.n_agents > self.max_agent:
            raise ValueError('n_agents exceeds configured max_agent')

        batch_size = common_skill.shape[0] // self.n_agents
        token_count = common_skill.shape[1]
        capability = self.registry.features_for(
            self.task, batch_size=batch_size, n_agents=self.n_agents,
            token_count=token_count, reference=common_skill,
        )
        cap_embedding = self.capability_encoder(capability)

        # reshape preserves original [batch, agent, entity_token] ordering
        per_agent = capability.reshape(batch_size, self.n_agents, token_count, CAPABILITY_DIM)[:, :, 0]
        team_mean = per_agent.mean(dim=1)
        team_sum = per_agent.sum(dim=1) / self.max_agent
        team_size = per_agent.new_full((batch_size, 1), self.n_agents / self.max_agent)
        team_features = th.cat((team_mean, team_sum, team_size), dim=-1)
        team_embedding = self.team_encoder(team_features)
        team_tokens = (team_embedding[:, None, None, :]
                       .expand(batch_size, self.n_agents, token_count, -1)
                       .reshape(batch_size * self.n_agents, token_count, -1))

        adapter_parts = ((common_skill, observation_embedding, cap_embedding, team_tokens)
                         if self.use_observation_embedding
                         else (common_skill, cap_embedding, team_tokens))
        adapter_input = th.cat(adapter_parts, dim=-1)
        delta = self.residual_scale * self.network(adapter_input)
        # Metric reductions launch extra CUDA kernels.  The learner enables
        # them only on steps that are actually logged.
        if self.collect_metrics:
            self.last_delta_rms = delta.detach().square().mean().sqrt()
            self.last_common_rms = common_skill.detach().square().mean().sqrt()
            self.last_team_norm = team_embedding.detach().square().mean().sqrt()
        return common_skill + delta, delta


class HISSDCapabilityTeamAgent(HISSDCapabilityAgent):
    """V3 agent. Preserves the original HiSSD module and controller contracts."""

    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents,
                 decomposer, args):
        super().__init__(task2input_shape_info, task2decomposer,
                         task2n_agents, decomposer, args)
        if not self.common_skill_adapter_enabled:
            raise ValueError('V3 requires common_skill_adapter_enabled=True')
        # The V2 hook is already registered and dynamically uses the current
        # planner.common_skill_adapter, so no second hook is installed.
        with th.random.fork_rng(devices=[]):
            self.planner.common_skill_adapter = TeamConditionedCommonSkillAdapter(
                observation_dim=self.planner.entity_embed_dim,
                skill_dim=self.planner.entity_embed_dim,
                hidden_dim=int(args.common_skill_adapter_hidden_dim),
                capability_embed_dim=int(getattr(args, 'capability_embed_dim', 32)),
                team_embed_dim=int(getattr(args, 'team_embed_dim', 32)),
                capability_manifest_path=args.capability_manifest_path,
                max_agent=int(args.max_agent),
                residual_scale=float(getattr(args, 'v3_residual_scale', 1.0)),
                use_observation_embedding=bool(getattr(args, 'v3_use_observation_embedding', True)),
            )

