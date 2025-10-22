from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Space,
)
import numpy as np
from numpy.typing import NDArray
from pettingzoo import ParallelEnv
from typing import override


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
AgentSelfObservation = Space[Dict({
    "ammo": Discrete(n=MAX_AMMO + 1),
    "cooldown": Box(low=0, high=RELOAD_COOLDOWN, shape=(1,), dtype=np.float32),
})]

# Aim observation is relative to observer aim. TODO: could observe aim relative to allies.
# Weapon cooldown is observable.
# Direction of observed agent relative to observer aim.
# Whether agent is an enemy of the observer is observable.
AgentOtherObservation = Space[Dict({
    "aim": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "direction": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "is_enemy": MultiBinary(n=1),
    "self": AgentSelfObservation(),
})]

# Agents can aim, move, reload, and shoot.
# Aiming and moving are instantaneous for now.
# Can only reload when ammo < MAX_AMMO.
# Can only shoot when ammo > 0 && cooldown == 0.
ActionType = Dict
ACTION_SPACE = Space[ActionType(spaces={
    "aim": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    "move": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
    # 0: No-op, 1: shoot, 2: reload
    "weapon": MultiDiscrete(nvec=[3], dtype=np.int8),
})]()

# An agent always observes itself and everything their allies observe.
# Observable entities include self and other agents.
# TODO: the local environment should be observable (LIDAR?).
ObsType = Dict
OBSERVATION_SPACE = Space[ObsType(spaces={
    "agents": Sequence(space=AgentOtherObservation(), stack=True),
    "self": AgentSelfObservation(),
})]()


class CustomEnvironment(ParallelEnv[AgentID, ObsType, ActionType]):
    action_spaces: dict[AgentID, Space[ActionType]]
    agents: list[AgentID]
    observation_spaces: dict[AgentID, Space[ObsType]]
    possible_agents: list[AgentID]


    @override
    def __init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng()
        self.possible_agents = list(map(AgentID, range(NUM_TEAMS * TEAM_SIZE)))
        self.action_spaces = {agent_id: ACTION_SPACE for agent_id in self.agents}
        self.observation_spaces = {agent_id: OBSERVATION_SPACE for agent_id in self.agents}
        self._reset_agents()


    @override
    def action_space(self, agent: AgentID) -> Space[ActionType]:
        return ACTION_SPACE


    @override
    def close(self) -> None:
        pass


    def _get_observations(self) -> dict[AgentID, ObsType]:
        displacements = self._agent_positions[np.newaxis, :, :] - self._agent_positions[:, np.newaxis, :]
        denom = np.max(np.abs(displacements), axis=2, keepdims=True)
        unit_square_directions = np.divide(displacements, denom, out=np.zeros_like(displacements), where=denom != 0)
        return {
            agent_id: {
                "agents": [
                    {
                        "aim": self._agent_aims[other_agent_id],
                        "direction": unit_square_directions[agent_id, other_agent_id],
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
    def observation_space(self, agent: AgentID) -> Space[ObsType]:
        return OBSERVATION_SPACE


    @override
    def render(self) -> None:
        pass


    @override
    def reset(self, seed: int | None = None, options: dict[object, object] | None = None) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict[None, None]]]:
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
    def state(self) -> NDArray[np.float32]:
        return np.concatenate((
            self._agent_aims.ravel(),
            self._agent_alive.astype(np.float32),
            self._agent_ammos.astype(np.float32),
            self._agent_cooldowns,
            self._agent_positions.ravel(),
            self._agent_team_ids.astype(np.float32),
        ))


    @override
    def step(self, actions: dict[AgentID, ActionType]) -> tuple[dict[AgentID, ObsType], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict[None, None]]]:
        self._agent_cooldowns[:] = np.maximum(self._agent_cooldowns - 1 / STEP_RATE, 0)
        shooters: list[AgentID] = []
        for agent_id, action in actions.items():
            if not self._agent_alive[agent_id]:
                continue
            # Make sure to resolve physics (clipping into world bounds or other players)
            self._agent_positions[agent_id] += action["move"] * MOVE_SPEED / STEP_RATE
            self._agent_aims[agent_id] = action["aim"]

            if self._agent_cooldowns[agent_id] == 0:
                match action["weapon"]:
                    case 1:
                        self._agent_ammos[agent_id] -= 1
                        self._agent_cooldowns[agent_id] = SHOOT_COOLDOWN
                        shooters.append(agent_id)
                    case 2:
                        self._agent_ammos[agent_id] = MAX_AMMO
                        self._agent_cooldowns[agent_id] = RELOAD_COOLDOWN
                    case _:
                        pass

        # Calculate all hits simulatenously without biasing towards low agent_id.
        # TODO: Use numpy and vectorize this entire operation.
        hits: list[AgentID] = []
        for shooter in shooters:
            best_tgt = None
            best_dot = -1.0
            aim_dir = self._agent_aims[shooter]
            for target in self.agents:
                if target == shooter or not self._agent_alive[target]:
                    continue
                vec = self._agent_positions[target] - self._agent_positions[shooter]
                dist = np.linalg.norm(vec)
                if dist == 0:
                    continue
                dir_norm = vec / dist
                dot = float(np.dot(aim_dir, dir_norm))
                if dot > best_dot:
                    best_dot = dot
                    best_tgt = target
            if best_tgt is not None and best_dot > 0.0:
                hits.append(best_tgt)
        for tgt in hits:
            self._agent_alive[tgt] = False

        # Reward agents as a team whenever their team eliminates an enemy agent
        # TODO: Use numpy and vectorize this entire operation.
        team_rewards = {team: 0.0 for team in range(NUM_TEAMS)}
        for shooter in shooters:
            team = self._agent_team_ids[shooter]
            for tgt in hits:
                if self._agent_team_ids[tgt] != team:
                    team_rewards[team] += 1.0
        rewards = {
            agent_id: team_rewards[self._agent_team_ids[agent_id]]
            for agent_id in self.agents
        }

        self.agents = [agent_id for agent_id in self.agents if self._agent_alive[agent_id]]

        observations = self._get_observations()
        terminations = dict(zip(self.agents, self._agent_alive[np.array(self.agents, dtype=np.int8)]))
        truncations = {agent_id: False for agent_id in self.agents}
        infos = {agent_id: {None: None} for agent_id in self.agents}
        return observations, rewards, terminations, truncations, infos
