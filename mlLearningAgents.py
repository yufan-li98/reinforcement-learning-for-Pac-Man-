# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import itertools
import pandas as pd
import numpy as np
import game
import util


class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Initialise an empty dictionary saving action values indexed by state and action pair
        # In subsequent operations, fill this dictionary with Pacman's position and legal actions
        self.Q_matrix = {}

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        print "Legal moves: ", legal
        print "Pacman position: ", state.getPacmanPosition()
        print "Ghost positions:", state.getGhostPositions()
        print "Food locations: "
        print state.getFood()
        print "Score: ", state.getScore()

        # Now pick what action to take. For now a random choice among
        # the legal moves

        pac_state = state.getPacmanPosition()  # get Pacman's position in the map

        # if map contains ghost, use coordinate, else stationary coordinate (0,0)
        if state.getGhostPositions()[0]:
            ghost=state.getGhostPositions()[0]
        else:
            ghost = (0,0)

        # combine Pacman's position and ghost's position to a set
        # define this set as Q-matrix's keys
        key_dic=(pac_state,ghost)

        for j in range(self.getNumTraining()):
            for i in legal:
                nextState = state.generatePacmanSuccessor(i)  # generate pacman's next state after executing action i
                pac_np = nextState.getPacmanPosition()  # get Pacman's position in the map
                ghost_np= state.getGhostPositions()[0]  # get the ghost's position
                key_dic_next=(pac_np,ghost_np) # combine Pacman's next position and ghost's next position to a set
                print "next state: ", nextState
                print "pac next positions:", pac_np
                print "Ghost next positions:",ghost_np
                reward = nextState.getScore() - state.getScore()  # define action reward through scores
                try:
                    Q_value_max_next = max(self.Q_matrix[key_dic_next].values())  # get the maximum Q-value of the next step
                # since the Q matrix we defined may empty, in this case,
                # allocate 0 to the maximum Q-value of the next pixel
                except:
                    Q_value_max_next = 0

                # if the combination of Pacman's position and ghost's position haven't defined in Q matrix before,
                # we should initialise a sub-dictionary, then fill this sub-dictionary in subsequent operations
                if key_dic not in self.Q_matrix.keys():
                    self.Q_matrix[key_dic] = {}
                    self.Q_matrix[key_dic][i] = 0

                # if the Q-value in Pacman's position for each direction haven't defined,
                # set Q-value to 0
                elif i not in self.Q_matrix[key_dic].keys():
                    self.Q_matrix[key_dic][i] = 0
                # update Q(state,action) values in Q matrix through Q-Learning algorithm
                self.Q_matrix[key_dic][i] = (1 - self.getAlpha()) * self.Q_matrix[key_dic][i] + \
                                              self.getAlpha() * (reward + self.getGamma() * Q_value_max_next)

        print self.Q_matrix
        # find the action with the maximum Q-value for the current coordinates of Pac-Man
        max_Q = list(self.Q_matrix[key_dic].keys())[
            list(self.Q_matrix[key_dic].values()).index(max(self.Q_matrix[key_dic].values()))]

        # epsilon-greedy learning
        if np.random.random() > self.epsilon:  # do exploitation with probability epsilon
            pick = max_Q
        else:  # do exploration with probability epsilon
            pick = random.choice(legal)
        # return an action
        return pick

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        print "A game just ended!"

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)




        # def registerIntialState(self,state):
        #     row = range(1, state.getWalls().height - 1)
        #     column = range(1, state.getWalls().width - 1)
        #     all_state = list(itertools.product(row, column))
        #     all_move = ['North', 'South', 'East', 'West', 'Stop']  # pd.MultiIndex.from_tuples
        #     self.Q_matrix = pd.DataFrame(columns=all_move, index=pd.MultiIndex.from_tuples(all_state))
        #     self.Q_matrix.loc[:, :] = 0

        # for j in range(self.getNumTraining()):
        #     for i in legal:
        #         nextState = state.generatePacmanSuccessor(i)
        #         l = nextState.getPacmanPosition()
        #         reward = nextState.getScore() - state.getScore()
        #         Q_value_max_next = np.max(self.Q_matrix[:].loc[l[0], l[1]])
        #         self.Q_matrix[i].loc[pac_state[0], pac_state[1]] = (1 - self.getAlpha()) * self.Q_matrix[i].loc[pac_state[0], pac_state[1]] + \
        #                                                            self.getAlpha() * (reward + self.getGamma() * Q_value_max_next)
        #        #print type(self.Q_matrix[:].loc[pac_state[0], pac_state[1]])
        # max_Q = self.Q_matrix[legal].loc[pac_state[0], pac_state[1]].idxmax()
