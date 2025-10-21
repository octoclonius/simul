from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from typing import (
    override,
)


AGENT_SIZE = 1
MAX_AMMO = 30
MOVE_SPEED = 5
NUM_TEAMS = 2
RELOAD_COOLDOWN = 4
SHOOT_COOLDOWN = 0.08
STEP_RATE = 60
TEAM_SIZE = 2
WORLD_SIZE = 10


AgentID = np.int8

# Agents always observe their own weapon (ammo and cooldown).
# For now, aiming and movement is instantaneous, so no need
# to observe things like orientation or velocity.
AgentSelfObservation = spaces.Space[spaces.Dict({
    "ammo": spaces.Discrete(n=MAX_AMMO + 1),
    "cooldown": spaces.Box(low=0, high=RELOAD_COOLDOWN, shape=(1,), dtype=np.float32),
})]

# Aim observation is relative to observer aim. TODO: could observe aim relative to allies.
# Weapon cooldown is observable.
# Direction of observed agent relative to observer aim.
# Whether agent is an enemy of the observer is observable.
AgentOtherObservation = spaces.Space[spaces.Dict({
    "aim": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "direction": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "is_enemy": spaces.MultiBinary(n=1),
    "self": AgentSelfObservation(),
})]

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
    "agents": spaces.Sequence(space=AgentOtherObservation(), stack=True),
    "self": AgentSelfObservation(),
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
        return {
            agent_id: {
                "agents": [
                    {
                        "aim": self._agent_aims[other_agent_id],
                        "direction": self._agent_aims[other_agent_id],
                        "is_enemy": self._agent_team_ids[agent_id] != self._agent_team_ids[other_agent_id],
                        "self": {
                            "ammo": self._agent_ammos[agent_id],
                            "cooldown": self._agent_cooldowns[agent_id],
                        },
                    }
                    for other_agent_id in self.agents[:agent_id] + self.agents[agent_id + 1:]
                ],
                "self": {
                    "ammo": self._agent_ammos[agent_id],
                    "cooldown": self._agent_cooldowns[agent_id],
                }
            }
            for agent_id in self.agents
        }


    @override
    def observation_space(self, agent: AgentID):
        return OBSERVATION_SPACE


    @override
    def render(self):
        pass


    @override
    def reset(self, seed: int | None = None, options: dict[object, object] | None = None):
        # Method "reset" overrides class "ParallelEnv" in an incompatible manner
        # Return type mismatch: base method returns type "tuple[dict[AgentID, ObsType], dict[AgentID, dict[Unknown, Unknown]]]", override returns type "tuple[dict[AgentID, dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]], dict[AgentID, dict[None, None]]]"
        # "tuple[dict[AgentID, dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]], dict[AgentID, dict[None, None]]]" is not assignable to "tuple[dict[AgentID, ObsType], dict[AgentID, dict[Unknown, Unknown]]]"
        # Tuple entry 1 is incorrect type
        # "dict[AgentID, dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]]" is not assignable to "dict[AgentID, ObsType]"
        # Type parameter "_VT@dict" is invariant, but "dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]" is not the same as "ObsType"
        # Consider switching from "dict" to "Mapping" which is covariant in the value type [reportIncompatibleMethodOverride]

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

        self._agent_alive: NDArray[np.bool_] = np.ones(
            shape=NUM_TEAMS * TEAM_SIZE,
            dtype=np.bool_,
        )

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
        # Method "state" overrides class "ParallelEnv" in an incompatible manner
        # Return type mismatch: base method returns type "ndarray[Unknown, Unknown]", override returns type "dict[str, NDArray[float32] | NDArray[bool_] | NDArray[int64] | NDArray[int8]]"
        # "dict[str, NDArray[float32] | NDArray[bool_] | NDArray[int64] | NDArray[int8]]" is not assignable to "ndarray[Unknown, Unknown]" [reportIncompatibleMethodOverride]

        return {
            "aims": self._agent_aims.copy(),
            "alive": self._agent_alive.copy(),
            "ammos": self._agent_ammos.copy(),
            "cooldowns": self._agent_cooldowns.copy(),
            "positions": self._agent_positions.copy(),
            "team_ids": self._agent_team_ids.copy(),
        }


    @override
    def step(self, actions: dict[AgentID, ActionType]):
        # Method "step" overrides class "ParallelEnv" in an incompatible manner
        # Return type mismatch: base method returns type "tuple[dict[AgentID, ObsType], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict[Unknown, Unknown]]]", override returns type "tuple[dict[AgentID, dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict[None, None]]]"
        # "tuple[dict[AgentID, dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict[None, None]]]" is not assignable to "tuple[dict[AgentID, ObsType], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict[Unknown, Unknown]]]"
        # Tuple entry 1 is incorrect type
        # "dict[AgentID, dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]]" is not assignable to "dict[AgentID, ObsType]"
        # Type parameter "_VT@dict" is invariant, but "dict[str, list[dict[str, Any | dict[str, Any]]] | dict[str, Any]]" is not the same as "ObsType"
        # Consider switching from "dict" to "Mapping" which is covariant in the value type [reportIncompatibleMethodOverride]

        observations = self._get_observations()
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        terminations = {agent_id: False for agent_id in self.agents}
        truncations = {agent_id: False for agent_id in self.agents}
        infos = {agent_id: {None: None} for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos
