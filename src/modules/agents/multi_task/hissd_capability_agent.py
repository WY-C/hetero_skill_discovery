"""SMAC HiSSD agent with one shared agent-conditioned residual adapter.

The original planner emits a structured common representation consisting of
self, enemy, and ally tokens.  This variant applies the same residual MLP to
every token, conditioned on the corresponding current-observation embedding:

    c'_i = c_i + A([c_i, h_i])

The adapted tokens are consumed by both the action decoder and the planner's
next-observation reconstruction loss.  No adapter-specific loss is added.
"""

import torch as th
import torch.nn as nn

from .hissd_agent import HISSDAgent
from .smac_capability import CapabilityRegistry, CAPABILITY_DIM


class AgentConditionedCommonSkillAdapter(nn.Module):
    """Shared zero-initialized residual adapter."""

    def __init__(self, observation_dim, skill_dim, hidden_dim=128):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.skill_dim = int(skill_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.observation_dim + self.skill_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.skill_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, common_skill, observation_embedding):
        if common_skill.shape != observation_embedding.shape:
            raise ValueError(
                "Common-skill and observation shapes must match, got "
                f"{tuple(common_skill.shape)} and "
                f"{tuple(observation_embedding.shape)}"
            )
        if common_skill.shape[-1] != self.skill_dim:
            raise ValueError(
                f"Expected skill_dim={self.skill_dim}, "
                f"got {common_skill.shape[-1]}"
            )
        adapter_input = th.cat((common_skill, observation_embedding), dim=-1)
        delta = self.network(adapter_input)
        return common_skill + delta, delta



class CapabilityConditionedCommonSkillAdapter(nn.Module):
    """
    Shared capability-conditioned residual adapter.

    c'_ij = c_ij + A([c_ij, h_ij, f(k_i)])

    All agents share the same parameters.
    """

    def __init__(
        self,
        observation_dim,
        skill_dim,
        hidden_dim=128,
        capability_embed_dim=32,
        capability_manifest_path=None,
    ):
        super().__init__()

        self.observation_dim = int(observation_dim)
        self.skill_dim = int(skill_dim)

        # Actual SC2 capability profiles
        self.registry = CapabilityRegistry(
            capability_manifest_path
        )

        self.task = None
        self.n_agents = None

        # Capability Encoder
        self.capability_encoder = nn.Sequential(
            nn.Linear(
                CAPABILITY_DIM,
                capability_embed_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                capability_embed_dim,
                capability_embed_dim,
            ),
        )

        # Shared Residual Adapter
        self.network = nn.Sequential(
            nn.Linear(
                observation_dim
                + skill_dim
                + capability_embed_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                skill_dim,
            ),
        )

        # Zero initialization
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def set_task(self, task, n_agents):
        self.task = str(task)
        self.n_agents = int(n_agents)

    def forward(
        self,
        common_skill,
        observation_embedding,
    ):
        if common_skill.shape != observation_embedding.shape:
            raise ValueError(
                "Common/observation shapes differ"
            )

        if (
            common_skill.ndim != 3
            or common_skill.shape[-1] != self.skill_dim
        ):
            raise ValueError(
                f"Expected common tokens [B*A,T,{self.skill_dim}]"
            )

        if self.task is None or self.n_agents is None:
            raise RuntimeError(
                "Missing planner task context"
            )

        if common_skill.shape[0] % self.n_agents:
            raise ValueError(
                "Incompatible flattened batch / agent count"
            )

        # [B*A, T, capability_dim]
        capability = self.registry.features_for(
            self.task,
            batch_size=common_skill.shape[0] // self.n_agents,
            n_agents=self.n_agents,
            token_count=common_skill.shape[1],
            reference=common_skill,
        )

        # [B*A, T, capability_embed_dim]
        cap_embedding = self.capability_encoder(
            capability
        )

        # Common Skill + Observation + Capability
        adapter_input = th.cat(
            (
                common_skill,
                observation_embedding,
                cap_embedding,
            ),
            dim=-1,
        )

        # Residual
        delta = self.network(adapter_input)

        adapted_common = common_skill + delta

        return adapted_common, delta


class HISSDCapabilityAgent(HISSDAgent):
    """HiSSD agent whose planner outputs pass through the shared adapter."""

    def __init__(
        self,
        task2input_shape_info,
        task2decomposer,
        task2n_agents,
        decomposer,
        args,
    ):
        super().__init__(
            task2input_shape_info,
            task2decomposer,
            task2n_agents,
            decomposer,
            args,
        )
        self.common_skill_adapter_enabled = bool(
            getattr(args, "common_skill_adapter_enabled", True)
        )
        if not self.common_skill_adapter_enabled:
            return

        hidden_dim = int(args.common_skill_adapter_hidden_dim)
        if hidden_dim <= 0:
            raise ValueError("common_skill_adapter_hidden_dim must be positive")
        # Do not perturb initialization of modules constructed after the agent
        # (notably the learner's mixer). Thus a same-seed ON/OFF comparison
        # differs only by the adapter parameters and their computation.
        with th.random.fork_rng(devices=[]):
            adapter = CapabilityConditionedCommonSkillAdapter(
                observation_dim=self.planner.entity_embed_dim,
                skill_dim=self.planner.entity_embed_dim,
                hidden_dim=hidden_dim,
                capability_manifest_path=args.capability_manifest_path,
            )
        self.planner.common_skill_adapter = adapter
        self.planner.register_forward_pre_hook(
            self._capability_planner_pre_hook,
            with_kwargs=True,
        )
        self.planner.last_adapter_delta_rms = 0.0


    def _capability_planner_pre_hook(
        self,
        planner,
        args,
        kwargs,
    ):
        """
        Bind the current task and agent count
        before each Planner forward pass.
        """

        task = kwargs.get("task") if kwargs else None

        if task is None and len(args) >= 4:
            task = args[3]

        if task is None:
            raise RuntimeError(
                "Cannot determine planner task"
            )

        task = str(task)

        if task not in planner.task2n_agents:
            raise KeyError(
                f"Unknown task: {task}"
            )

        planner.common_skill_adapter.set_task(
            task,
            planner.task2n_agents[task],
        )
