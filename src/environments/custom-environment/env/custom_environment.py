from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from typing import (
    override,
)


AGENT_SIZE = 1.0
MAX_AMMO = 30
NUM_TEAMS = 2
RELOAD_COOLDOWN = 4.0
SHOOT_COOLDOWN = 0.08
STEP_RATE = 60
TEAM_SIZE = 2
WORLD_SIZE = 10


AgentID = np.int8

# Agents always observe their own weapon (ammo and cooldown).
# For now, aiming and movement is instantaneous, so no need
# to observe things like orientation or velocity.
AgentSelfObservation = spaces.Dict({
    "ammo": spaces.Discrete(n=MAX_AMMO),
    "cooldown": spaces.Box(low=0, high=RELOAD_COOLDOWN, shape=(1,), dtype=np.float32),
})

# Aim observation is relative to observer aim. TODO: could observe aim relative to allies.
# Weapon cooldown is observable.
# Direction of observed agent relative to observer aim.
# Whether agent is an enemy of the observer is observable.
AgentOtherObservation = spaces.Dict({
    "aim": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "self": AgentSelfObservation,
    "direction": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "is_enemy": spaces.MultiBinary(n=1),
})

# Agents can aim, move, reload, and shoot.
# Aiming and moving are instantaneous for now.
# Can only reload when ammo < MAX_AMMO.
# Can only shoot when ammo > 0 && cooldown == 0.
ActionType = spaces.Dict
ACTION_SPACE = spaces.Space[ActionType(spaces={
    "aim": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "move": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    # 0: No-op, 1: shoot, 2: reload
    "weapon": spaces.MultiDiscrete(nvec=[3], dtype=np.int8),
})]()

# An agent always observes itself and everything their allies observe.
# Observable entities include self and other agents.
# TODO: the local environment should be observable (LIDAR?).
ObsType = spaces.Dict
OBSERVATION_SPACE = spaces.Space[ObsType(spaces={
    "agents": spaces.Sequence(space=AgentOtherObservation, stack=True),
    "self": AgentSelfObservation,
})]()


class CustomEnvironment(ParallelEnv[AgentID, ObsType, ActionType]):
    action_spaces: dict[AgentID, spaces.Space[ActionType]]
    agents: list[AgentID]
    observation_spaces: dict[AgentID, spaces.Space[ObsType]]
    possible_agents: list[AgentID]


    @override
    def __init__(self):
        self.rng: np.random.Generator = np.random.default_rng()
        self.possible_agents = list(map(AgentID, range(NUM_TEAMS * TEAM_SIZE)))
        self.agents = self.possible_agents[:]
        self.action_spaces = {agent_id: ACTION_SPACE for agent_id in self.agents}
        self.observation_spaces = {agent_id: OBSERVATION_SPACE for agent_id in self.agents}
        self._reset_agents()


    @override
    def action_space(self, agent: AgentID):
        return ACTION_SPACE


    @override
    def close(self):
        pass


    def _get_observations(self):
        return {agent_id: ObsType(spaces={}) for agent_id in self.agents}


    @override
    def observation_space(self, agent: AgentID):
        return OBSERVATION_SPACE


    @override
    def render(self):
        pass


    @override
    def reset(self, seed: int | None = None, options: dict[object, object] | None = None):
        self._reset_agents()
        self.agents = self.possible_agents[:]
        observations = self._get_observations()
        # Dummy infos required for parallel_to_aec conversion
        infos = {agent_id: {None: None} for agent_id in self.agents}
        return observations, infos


    def _reset_agents(self):
        self._agent_aims: NDArray[np.float32] = self.rng.uniform(
            low=-1,
            high=1,
            size=(NUM_TEAMS * TEAM_SIZE, 2),
        ).astype(dtype=np.float32)

        self._agent_ammos: NDArray[np.int64] = np.full(
            shape=NUM_TEAMS * TEAM_SIZE,
            fill_value=MAX_AMMO,
            dtype=np.int64,
        )

        self._agent_cooldowns: NDArray[np.float32] = np.zeros(
            shape=NUM_TEAMS * TEAM_SIZE,
            dtype=np.float32,
        )

        self._agent_positions: NDArray[np.float32] = self.rng.uniform(
            low=AGENT_SIZE / 2,
            high=WORLD_SIZE - AGENT_SIZE / 2,
            size=(NUM_TEAMS * TEAM_SIZE, 2),
        ).astype(dtype=np.float32)

        self._agent_team_ids: NDArray[np.int8] = np.arange(
            NUM_TEAMS * TEAM_SIZE,
            dtype=np.int8,
        ) % NUM_TEAMS


    @override
    def state(self):
        return np.empty([])


    @override
    def step(self, actions: dict[AgentID, ActionType]):
        


        observations = {agent_id: ObsType() for agent_id in self.agents}
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        terminations = {agent_id: False for agent_id in self.agents}
        truncations = {agent_id: False for agent_id in self.agents}
        infos = {agent_id: {None: None} for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos
