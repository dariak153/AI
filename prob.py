# prob.py
# This is

import random
import numpy as np
from gridutil import *
from itertools import combinations
import queue
from typing import List, Tuple, Sequence, Union, Set
from gridutil import *

best_turn = {('N', 'E'): 'turnright',
             ('N', 'S'): 'turnright',
             ('N', 'W'): 'turnleft',
             ('E', 'S'): 'turnright',
             ('E', 'W'): 'turnright',
             ('E', 'N'): 'turnleft',
             ('S', 'W'): 'turnright',
             ('S', 'N'): 'turnright',
             ('S', 'E'): 'turnleft',
             ('W', 'N'): 'turnright',
             ('W', 'E'): 'turnright',
             ('W', 'S'): 'turnleft'}


class LocAgent:

    def __init__(self, size: int, pit_prob: float):
        self.size = size
        # location of agent as tuple (x, y)
        self.loc = (0, 0)
        # direction of agent as a string from set {'N', 'E', 'W', 'S'}
        self.dir = 'N'
        # map where breeze was perceived
        self.breeze = np.zeros([size, size])
        # set of visited locations
        self.vis = set()
        # set of frontier locations, i.e. not visited locations adjacent to visited ones
        self.front = set()
        # prior probability that a location will contain a pit
        self.pit_prob = pit_prob


    def __call__(self, percept: List[str]) -> str:
        # update info
        # add current location to visited
        self.vis.add(self.loc)
        # remove current location from frontiers
        self.front.discard(self.loc)
        # add adjacent locations to frontiers
        for nh_dir in ['N', 'E', 'W', 'S']:
            # nh, _ = self.forward(self.loc, nh_dir)
            nh = nextLoc(self.loc, nh_dir)
            if legalLoc(nh, self.size) and nh not in self.vis:
                self.front.add(nh)

        # if 'breeze' perceived update map
        if 'breeze' in percept:
            self.breeze[self.loc[0], self.loc[1]] = True

        # map from frontier locations to probability that they contain pit
        self.front_to_prob = {}
        for cur_loc in self.front:
            # compute using brute force
            P_q_giv_k_b_norm_comp = self.prob_bf(cur_loc, self.vis)
            self.front_to_prob[cur_loc] = P_q_giv_k_b_norm_comp
            #
            # compute using decomposition
            # TODO PUT YOUR CODE HERE
            best_dests = [self.loc]
            best_dests_prob = 2.0
            for dest, prob in self.front_to_prob.items():
                # if new probability very close to the best one so far
                if abs(prob - best_dests_prob) < 1e-5:
                    # add to list
                    best_dests.append(dest)
                # if new probability is better
                elif prob < best_dests_prob:
                    # empty the list and add new element
                    best_dests = [dest]
                    best_dests_prob = prob

        # among ones with the lowest probability find the one with the shortest path
        shortest_dest = self.loc
        shortest_dest_cmds_len = 1e6
        shortest_dest_cmds = ['forward']
        # among ones with the lowest probability find the one with the shortest path

        for dest in best_dests:
            # find path to location
            cmds = self.path_to_loc(dest)
            # if path shorter than the shortest so far
            if len(cmds) < shortest_dest_cmds_len:
                shortest_dest = dest
                shortest_dest_cmds_len = len(cmds)
                shortest_dest_cmds = cmds

        # ------------------

        # return command
        next_cmd = shortest_dest_cmds[0]
        if next_cmd == 'forward':
            next_loc = nextLoc(self.loc, self.dir)
            if legalLoc(next_loc, self.size):
                self.loc = next_loc
            return 'forward'
        elif next_cmd == 'turnleft':
            self.dir = leftTurn(self.dir)
            return 'turnleft'
        elif next_cmd == 'turnright':
            self.dir = rightTurn(self.dir)
            return 'turnright'
        else:
            next_loc = nextLoc(self.loc, self.dir)
            if legalLoc(next_loc, self.size):
                self.loc = next_loc
            return 'forward'

    def get_posterior(self) -> np.array:
        P = np.zeros((self.size, self.size))
        for loc, prob in self.front_to_prob.items():
            print(loc, prob)
            P[loc[0], loc[1]] = prob

        return P

    # checks whether breeze observations are consistent with set of pits
    def check_breeze(self,
                     pits: Union[Sequence[Tuple[int, int]], Set[Tuple[int, int]]],
                     vis: Union[Sequence[Tuple[int, int]], Set[Tuple[int, int]]]) -> float:
        breeze_comp = np.zeros([self.size, self.size])
        for pit in pits:
            for nh_dir in ['N', 'E', 'W', 'S']:
                # nh, _ = self.forward(pit, nh_dir)
                nh = nextLoc(pit, nh_dir)
                if nh != pit and nh in vis:
                    breeze_comp[nh[0], nh[1]] = True
        if np.array_equal(self.breeze, breeze_comp):
            return 1.0
        else:
            return 0.0

    # finds the shortest path to destination in terms of number of commands, using only visited locations
    def path_to_loc(self, dest: Tuple[int, int]) -> List[str]:
        # BFS algorithm
        # state encoded as tuple (x, y, dir)
        start_state = self.loc + (self.dir,)
        # command that was used to reach that state
        prev_cmd = {(loc + (cur_dir,)): '' for loc in self.vis.union([dest]) for cur_dir in ['N', 'E', 'W', 'S']}
        # queue with states
        q = queue.Queue()
        # adding current state
        q.put(start_state)
        prev_cmd[start_state] = 'none'
        # state in which arrived to destination
        dest_state = None
        while not q.empty():
            cur_state = q.get()
            cur_loc = cur_state[0: 2]
            cur_dir = cur_state[2]
            # if reached destination
            if cur_loc == dest:
                dest_state = cur_state
                break

            # all possible moves
            for cmd in ['forward', 'turnleft', 'turnright']:
                # if cmd == 'forward':
                #     next_loc, next_dir = self.forward(cur_loc, cur_dir)
                # elif cmd == 'turnleft':
                #     next_loc, next_dir = self.turnleft(cur_loc, cur_dir)
                # elif cmd == 'turnright':
                #     next_loc, next_dir = self.turnright(cur_loc, cur_dir)
                next_loc = cur_loc
                next_dir = cur_dir
                if cmd == 'forward':
                    next_loc_fwd = nextLoc(cur_loc, cur_dir)
                    if legalLoc(next_loc_fwd, self.size):
                        next_loc = next_loc_fwd
                elif cmd == 'turnleft':
                    next_dir = leftTurn(cur_dir)
                elif cmd == 'turnright':
                    next_dir = rightTurn(cur_dir)

                next_state = next_loc + (next_dir,)
                # if it is visited location and wasn't yet reached
                if next_state in prev_cmd and prev_cmd[next_state] == '':
                    prev_cmd[next_state] = cmd
                    q.put(next_state)

        # if reached destination
        if prev_cmd[dest_state] != '':
            cur_state = dest_state
            cmds = []
            # going backwards and collecting commands
            while cur_state != start_state:
                cmds.append(prev_cmd[cur_state])
                if prev_cmd[cur_state] == 'forward':
                    cur_loc = nextLoc(cur_state[0: 2], nextDirection(cur_state[2], 2))
                    cur_dir = cur_state[2]
                elif prev_cmd[cur_state] == 'turnleft':
                    cur_loc = cur_state[0: 2]
                    cur_dir = rightTurn(cur_state[2])
                    # cur_loc, cur_dir = self.turnright(cur_state[0: 2], cur_state[2])
                elif prev_cmd[cur_state] == 'turnright':
                    cur_loc = cur_state[0: 2]
                    cur_dir = leftTurn(cur_state[2])
                    # cur_loc, cur_dir = self.turnleft(cur_state[0: 2], cur_state[2])
                cur_state = cur_loc + (cur_dir,)

            # commands were in a reverse order
            cmds.reverse()
            return cmds
        else:
            return []

    # calculates probability of query location containing pits given breeze and known (visited) locations
    def prob_bf(self, query: Tuple[int, int], vis: Union[Sequence[Tuple[int, int]], Set[Tuple[int, int]]]):
        frontier = self.front
        unknown = {(x, y) for x in range(self.size) for y in range(self.size)} - vis - {query} - frontier

        P_q_giv_k_b = 0
        P_nq_giv_k_b = 0

        for num_pits in range(len(frontier) + 1):
            for cur_pits in combinations(frontier, num_pits):
                pits_with_query = set(cur_pits).union({query})
                pits_without_query = set(cur_pits)

                P_breeze_with_query = self.check_breeze(pits_with_query, vis.union({query}))
                P_breeze_without_query = self.check_breeze(pits_without_query, vis)

                P_q_giv_k_b += P_breeze_with_query * self.pit_prob ** len(pits_with_query) * (1 - self.pit_prob) ** (
                            len(unknown) + len(frontier) - len(pits_with_query))
                P_nq_giv_k_b += P_breeze_without_query * self.pit_prob ** len(pits_without_query) * (
                            1 - self.pit_prob) ** (len(unknown) + len(frontier) - len(pits_without_query))

        # normalize, so sum is equal to 1
        P_q_giv_k_b_normalized = P_q_giv_k_b / (P_q_giv_k_b + P_nq_giv_k_b)

        return P_q_giv_k_b_normalized