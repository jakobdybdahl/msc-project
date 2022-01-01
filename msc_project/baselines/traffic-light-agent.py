import argparse
from statistics import mean

import gym
import numpy as np

# constants taken from tjc-gym
ENV_WIDTH = 1
GRASS_WIDTH = ENV_WIDTH * 0.425

VERTICAL_LANE = 0
HORIZONTAL_LANE = 1


class TrafficLightAgent:
    def __init__(self, n_agents, env):
        self.n_agents = n_agents
        self.env = env.env

        self.green_lane = HORIZONTAL_LANE

        self.steps_green = 100
        self.steps_to_clear = 25
        self.speed = 0.75

        self.steps_since_switch = 0
        self.is_clearing = False
        self.num_steps_clearing = 0

    def get_actions(self):
        if self.is_clearing:
            self.num_steps_clearing += 1

            actions = [0.0] * self.n_agents
            new_green_lane = HORIZONTAL_LANE if self.green_lane == VERTICAL_LANE else VERTICAL_LANE
            agents_to_move = [
                agent
                for agent in self.env._agents
                if self._on_the_road(agent)
                and not self._at_stop_line(agent)
                and not self._at_stop_line(agent, new_green_lane)
            ]
            for a in agents_to_move:
                actions[a.index] = self.speed

            if self.num_steps_clearing > self.steps_to_clear:
                self.is_clearing = False
                self.num_steps_clearing = 0
                self.green_lane = HORIZONTAL_LANE if self.green_lane == VERTICAL_LANE else VERTICAL_LANE
                self.steps_since_switch = 0

            return actions

        actions = [self.speed] * self.n_agents
        self.steps_since_switch += 1

        if self.steps_since_switch > self.steps_green:
            self.is_clearing = True

        for i, agent in enumerate(self.env._agents):
            if self._on_the_road(agent) and not self._agent_at_green_lane(agent):
                if self._at_stop_line(agent):
                    actions[i] = 0.0

        return actions

    def _at_stop_line(self, agent, green_lane_override=None):
        green_lane = self.green_lane if green_lane_override == None else green_lane_override
        at_line = lambda pos: pos >= stop_line - 0.01 and pos <= stop_line + 0.01
        if green_lane != HORIZONTAL_LANE:
            if agent.state.direction == (1, 0):
                stop_line = GRASS_WIDTH
                return at_line(agent.state.position[0])
            else:
                stop_line = ENV_WIDTH - GRASS_WIDTH
                return at_line(agent.state.position[0])
        else:
            if agent.state.direction == (0, 1):
                stop_line = GRASS_WIDTH
                return at_line(agent.state.position[1])
            else:
                stop_line = ENV_WIDTH - GRASS_WIDTH
                return at_line(agent.state.position[1])

    def _agent_at_green_lane(self, agent, green_lane_override=None):
        green_lane = self.green_lane if green_lane_override == None else green_lane_override
        if agent.state.direction == (1, 0) or agent.state.direction == (-1, 0):
            return green_lane == HORIZONTAL_LANE
        else:
            return green_lane == VERTICAL_LANE

    def _on_the_road(self, agent):
        return agent.state.on_the_road and not agent.state.done


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = gym.make("tjc_gym:TrafficJunctionContinuous6-v0")
    env.seed(args.seed)
    agent = TrafficLightAgent(env.n_agents, env)

    results = []

    for ep_i in range(args.episodes):
        dones = [False] * env.n_agents
        env.reset()
        score = 0
        num_steps = [0] * env.n_agents
        while not all(dones):
            env.render()
            actions = agent.get_actions()
            _, _, dones, info = env.step(actions)
            num_steps += info["took_step"]
        print(f"Episode {ep_i}/{args.episodes}: {mean(num_steps)}")
        results.append(mean(num_steps))

    print(f"Final average result: {mean(results)}")
