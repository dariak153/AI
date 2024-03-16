#!/usr/bin/env python

"""code template"""

import random
import numpy as np

from graphics import *
from gridutil import *

import agents


class LocWorldEnv:
    actions = "turnleft turnright forward".split()

    def __init__(self, size, walls, gold, pits, eps_perc, eps_move, start_loc, start_dir, use_dirs):
        self.size = size
        self.walls = walls
        self.gold = gold
        self.pits = pits
        self.action_sensors = []
        self.locations = {*locations(self.size)}.difference(self.walls)
        self.eps_perc = eps_perc
        self.eps_move = eps_move
        self.start_loc = start_loc
        self.lifes = 3
        self.start_dir = start_dir
        self.use_dirs = use_dirs
        self.reset()
        self.finished = False

    def reset(self):
        self.agentLoc = self.start_loc
        self.agentDir = self.start_dir

    def getPercept(self):
        p = self.action_sensors
        self.action_sensors = []
        for dir in ORIENTATIONS:
            nh = nextLoc(self.agentLoc, dir)
            prob = 0.0 + self.eps_perc
            if (not legalLoc(nh, self.size)) or nh in self.walls:
                prob = 1.0 - self.eps_perc
            if random.random() < prob:
                p.append(dir)

        for dir in ORIENTATIONS:
            nh = nextLoc(self.agentLoc, dir)
            if nh in self.pits and 'breeze' not in p:
                p.append('breeze')

        if self.agentLoc in self.pits:
            p.append('pit')

        return p

    def doAction(self, action):
        points = -1

        if self.use_dirs:
            if action == 'forward':
                loc = nextLoc(self.agentLoc, self.agentDir)
                if legalLoc(loc, self.size) and (loc not in self.walls):
                    self.agentLoc = loc
                else:
                    self.action_sensors.append("bump")
            elif action == 'turnleft':
                self.agentDir = leftTurn(self.agentDir)
            elif action == 'turnright':
                self.agentDir = rightTurn(self.agentDir)
        else:
            rand_val = random.random()
            if rand_val < self.eps_move:
                action = nextDirection(action, -1)
            elif rand_val < 2 * self.eps_move:
                action = nextDirection(action, 1)

            loc = nextLoc(self.agentLoc, action)
            if legalLoc(loc, self.size) and (loc not in self.walls):
                self.agentLoc = loc
            else:
                self.action_sensors.append("bump")

        if self.agentLoc in self.pits:
            self.lifes -= 1
            points -= 10
            if self.lifes == 0:
                self.finished = True
            print('You stepped into a pit')

        if self.agentLoc == self.gold:
            points += 20
            self.finished = True
            print('You found gold!')

        return points  # cost/benefit of action

    # def finished(self):
    #     return False


class LocView:
    # LocView shows a view of a LocWorldEnv. Just hand it an env, and
    #   a window will pop up.

    Size = .2
    Points = {'N': (0, -Size, 0, Size), 'E': (-Size, 0, Size, 0),
              'S': (0, Size, 0, -Size), 'W': (Size, 0, -Size, 0)}

    color = "black"

    def __init__(self, state, height=800, title="Loc World"):
        xySize = state.size
        win = self.win = GraphWin(title, 1.33 * height, height, autoflush=False)
        win.setBackground("gray99")
        win.setCoords(-.5, -.5, 1.33 * xySize - .5, xySize - .5)
        cells = self.cells = {}
        self.cells_prob = {}
        for x in range(xySize):
            for y in range(xySize):
                cells[(x, y)] = Rectangle(Point(x - .5, y - .5), Point(x + .5, y + .5))
                cells[(x, y)].setWidth(2)
                cells[(x, y)].draw(win)
        for x in range(xySize):
            for y in range(xySize):
                self.cells_prob[(x, y)] = Circle(Point(x, y), .25)
                self.cells_prob[(x, y)].setWidth(2)
                self.cells_prob[(x, y)].draw(win)
        self.agt = None
        self.arrow = None
        ccenter = 1.167 * (xySize - .5)
        # self.time = Text(Point(ccenter, (xySize - 1) * .75), "Time").draw(win)
        # self.time.setSize(36)
        # self.setTimeColor("black")

        self.agentName = Text(Point(ccenter, (xySize - 1) * .5), "").draw(win)
        self.agentName.setSize(20)
        self.agentName.setFill("Orange")

        self.info = Text(Point(ccenter, (xySize - 1) * .25), "").draw(win)
        self.info.setSize(20)
        self.info.setFace("courier")
        self.policy_arrows = []

        self.update(state)

    def setAgent(self, name):
        self.agentName.setText(name)

    # def setTime(self, seconds):
    #     self.time.setText(str(seconds))

    def setInfo(self, info):
        self.info.setText(info)

    def update(self, state, P=None, pi=None):
        # View state in exiting window
        for loc, cell in self.cells.items():
            if loc in state.walls:
                cell.setFill("black")
            elif loc == state.gold:
                cell.setFill("yellow")
            elif loc in state.pits:
                cell.setFill("gray")
            else:
                cell.setFill("white")

        if P is not None:
            for loc, cell in self.cells_prob.items():
                c = int(round(P[loc[0], loc[1]] * 255))
                cell.setFill('#ff%02x%02x' % (255 - c, 255 - c))

        if self.agt:
            self.agt.undraw()
        if state.agentLoc:
            self.agt = self.drawArrow(state.agentLoc, state.agentDir, 10, self.color)

        if pi:
            for a in self.policy_arrows:
                a.undraw()
            self.policy_arrows = []
            for loc, cell in self.cells.items():
                if loc in pi:
                    self.policy_arrows.append(self.drawArrow(loc, pi[loc], 3, 'green'))

    def drawArrow(self, loc, heading, width, color):
        x, y = loc
        dx0, dy0, dx1, dy1 = self.Points[heading]
        p1 = Point(x + dx0, y + dy0)
        p2 = Point(x + dx1, y + dy1)
        a = Line(p1, p2)
        a.setWidth(width)
        a.setArrow('last')
        a.setFill(color)
        a.draw(self.win)
        return a

    def pause(self):
        self.win.getMouse()

    # def setTimeColor(self, c):
    #     self.time.setTextColor(c)

    def close(self):
        self.win.close()


def main():
    random.seed(13)
    # rate of executing actions
    rate = 1
    # chance that perception will be wrong
    eps_perc = 0.0
    # chance that the agent will not move forward despite the command
    eps_move = 0.0
    # probability that a location contains a pit
    pit_prob = 0.2
    # number of actions to execute
    n_steps = 40
    # size of the environment
    env_size = 4

    # build the list of walls locations
    start_loc = (0, 0)
    start_dir = 'N'
    walls = []
    locs = list({*locations(env_size)}.difference(walls).difference([start_loc]))
    gold = (3, 3)
    pits = []
    for i in range(env_size):
        for j in range(env_size):
            if i != 0 and j != 0:
                if random.random() < pit_prob:
                    pits.append((j, env_size - i - 1))

    # create the environment and viewer
    env = LocWorldEnv(env_size, walls, gold, pits, eps_perc, eps_move, start_loc, start_dir, use_dirs=True)
    view = LocView(env)

    # create the agent
    agent = agents.prob.LocAgent(env.size, pit_prob)
    for t in range(n_steps):
        print('step %d' % t)

        percept = env.getPercept()
        action = agent(percept)
        # get what the agent thinks of the environment
        P = agent.get_posterior()

        print('Percept: ', percept)
        print('Action ', action)

        view.update(env, P)
        update(rate)
        # uncomment to pause before action
        view.pause()

        env.doAction(action)

    # pause until mouse clicked
    view.pause()


if __name__ == '__main__':
    main()
