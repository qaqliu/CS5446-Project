"""Dependency-based deadlock heuristic adapted from jbr-ai-labs/mamba.

Source: https://github.com/jbr-ai-labs/mamba/blob/2c97258f71bf1c421c40ce14fd2f7cc3fe7fe19f/env/flatland/DeadlockChecker.py
Only the active dependency checker is retained. Flatland 4 configuration/state
fields replace the 2.x fields; observation-builder graph mutations are omitted.
The occupancy-based detector is a heuristic, not a general deadlock proof.
See rewards.py for the champion reward formula and integration details.

MIT License

Copyright (c) 2022 jbr-ai-labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import defaultdict

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position


class DeadlockChecker:
    def __init__(self):
        pass

    def reset(self, env):
        self._is_deadlocked = np.zeros(len(env.agents))
        self._is_far_deadlocked = np.zeros(len(env.agents))
        self._old_deadlock = np.zeros(len(env.agents))
        self.env = env

        self.agent_positions = defaultdict(lambda: -1)
        self.far_dep = defaultdict(list)
        self.simple_dep = dict()

    def is_deadlocked(self, handle):
        return self._is_deadlocked[handle]

    def is_far_deadlocked(self, handle):
        return self._is_far_deadlocked[handle]

    def old_deadlock(self, handle):
        return self._old_deadlock[handle]

    def _check_blocked(self, handle):
        agent = self.env.agents[handle]
        transitions = self.env.rail.get_transitions(agent.current_configuration)
        self.checked[handle] = 1

        for direction, transition in enumerate(transitions):
            if transition == 0:
                continue # no road

            new_position = get_new_position(agent.current_configuration[0], direction)
            handle_opp_agent = self.agent_positions[new_position]
            if handle_opp_agent == -1:
                self.checked[handle] = 2
                return False # road is free

            if self._is_deadlocked[handle_opp_agent]:
                continue # road is blocked

            if self.checked[handle_opp_agent] == 0:
                self._check_blocked(handle_opp_agent)

            if self.checked[handle_opp_agent] == 2 and not self._is_deadlocked[handle_opp_agent]:
                self.checked[handle] = 2
                return False # road may become free

            self.dep[handle].append(handle_opp_agent)

            continue # road is blocked. cycle

        if not self.dep[handle]:
            self.checked[handle] = 2
            if len(list(filter(lambda t: t != 0, transitions))) == 0:
                return False # dead-end is not deadlock
            self._is_deadlocked[handle] = 1
            return True
        return None # shrug


    def fix_deps(self):
        any_changes = True
        # might be slow, but in practice won't
        while any_changes:
            any_changes = False
            for handle, agent in enumerate(self.env.agents):
                if self.checked[handle] == 1:
                    cnt = 0
                    for opp_handle in self.dep[handle]:
                        if self.checked[opp_handle] == 2:
                            if self._is_deadlocked[opp_handle]:
                                cnt += 1
                            else:
                                self.checked[handle] = 2
                                any_changes = True
                    if cnt == len(self.dep[handle]):
                        self.checked[handle] = 2
                        self._is_deadlocked[handle] = True
                        any_changes = True
        for handle, agent in enumerate(self.env.agents):
            if self.checked[handle] == 1:
                self._is_deadlocked[handle] = True
                self.checked[handle] = 2


    def update_deadlocks(self):
        self.agent_positions.clear()
        self.dep = [list() for _ in range(len(self.env.agents))]
        for handle, agent in enumerate(self.env.agents):
            self._old_deadlock[handle] = self._is_deadlocked[handle]
            if agent.state.is_on_map_state():
                self.agent_positions[agent.current_configuration[0]] = handle
        self.checked = [0] * len(self.env.agents)
        for handle, agent in enumerate(self.env.agents):
            if agent.state.is_on_map_state() and not self._is_deadlocked[handle] \
                    and not self.checked[handle]:
                self._check_blocked(handle)

        self.fix_deps()

