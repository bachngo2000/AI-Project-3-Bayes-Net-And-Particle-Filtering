# inference.py
# ------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.

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


import itertools
import random
import busters
import game
import numpy as np
from itertools import product

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing."""
        "*** YOUR CODE HERE ***"
        sum = self.total()
        if sum != 0:
            for key, value in self.items():
                self[key] = value/sum




    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.
        """
        "*** YOUR CODE HERE ***"
        # pick a rqandom number between 0 and 1 unifromly [0, 1) upper bound not included
        #rand_num = np.random.uniform(0, 1, 1)
        #accum_sum_prob = 0
        #for key, value in self.items():
            # add the probability of the current key/state to the sum
            #accum_sum_prob += value
            #if (rand_num < accum_sum_prob):
                #return key
        sum = self.total()
        rand_sample = random.uniform(0, sum)
        initVal = 0
        for key, value in self.items():
            if initVal + value >= rand_sample:
                return key
            initVal += value


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    # Question 1
    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE *** Bach"
        if ghostPosition == jailPosition:
            if noisyDistance == None:
                return 1
            else:
                return 0
        elif noisyDistance == None:
            if ghostPosition == jailPosition:
                return 1
            else:
                return 0
        else:
            distance = manhattanDistance(pacmanPosition, ghostPosition)
            prob = busters.getObservationProbability(noisyDistance, distance)
            return prob

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    # Question 2
    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        # beliefs represent the probability that the ghost is at a particular location, and are stored as a
        # DiscreteDistribution object in a field called self.beliefs, which we will update.
        # current position of Pacman
        currPacmanPos = gameState.getPacmanPosition()
        # jail position (only consider positions that are in self.allPositions)
        jailPos = self.getJailPosition()
        # a list of the possible ghost positions, including all legal positions plus the special jail position
        allPossibleGhostPos = self.allPositions

        # iterate updates over the variable allPossibleGhostPos = self.allPositions
        for ghostPos in allPossibleGhostPos:
            # current belief/probability distribution at the current position
            prevProb = self.beliefs[ghostPos]
            # utilize the function self.getObservationProb that returns the probability of an observation given
            # Pacman’s current position, a potential ghost position, and the jail position.
            # observation is the noisy Manhattan distance to the ghost we are tracking.
            probObservation = self.getObservationProb(noisyDistance=observation, pacmanPosition=currPacmanPos,
                                                      ghostPosition=ghostPos, jailPosition=jailPos)
            # update the belief at every position on the map after receiving a sensor reading
            self.beliefs[ghostPos] = probObservation * prevProb
        self.beliefs.normalize()

    # Question 3
    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"



    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    # Question 5
    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        "*** YOUR CODE HERE *** passed all tests"
        # declare a list for storing the particles where each particle is a pair (x,y) for the ghost position
        self.particles = []

        """  Global Initialization:
        when the initial state is unknown, B(X0) is initialized by a uniform distribution 
        over the space of all legal positions in the map. B(X0) = 1 / number of legal positions."""

        # take a random number between 0 and 1 from a uniform distribution
        randNumList = np.random.uniform(0, 1, self.numParticles)
        """ Map the random number to a legal position such that if random number is 
        in [0, 1/number of legal positions) it is mapped to the first state in the 
        self.legalPositions list. If it is in [1/number of legal positions, 2/number of legal positions)
        it is mapped to the second state, and so on..."""
        # the interval between zero and 1 assigned to each state is 1/n  for n states
        interval = 1 / len(self.legalPositions)
        # loop through the list of random numbers
        for rn in randNumList:
            # find the corresponding state to that random number rn by finding how many full intervals are included in rn
            # round index up or down to a full integer whichever closer
            state_idx = round((rn - (rn % interval))/interval)
            # add the sampled state to the list of particles
            self.particles.append(self.legalPositions[state_idx])


    # Question 6
    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """

        """ Use self.getObservationProb to find the probability of an observation 
        given Pacman’s position, a potential ghost position, and the jail position."""

        "*** YOUR CODE HERE ***"
        # create the current belief distribution by calling self.getBeliefDistribution()
        beliefDist = self.getBeliefDistribution()
        # declare a distribution for particles weights by creating a DiscreteDistribution object
        # keys are the states and weights are the probability of the observation for a specific ghost given its state
        weightedDist = DiscreteDistribution()
        # initialized weightedDist values to zero weight for all states/keys
        for state in self.legalPositions:
            weightedDist[state] = 0

        # observation is a noisyDistance to a ghost
        # ghostPosition is represented by a number of particles
        for particle in self.particles:
            weight_i = self.getObservationProb(observation, gameState.getPacmanPosition(), particle, self.getJailPosition())
            # un-normalized distribution: combine likelihood weighting with the current belief distribution of ghost state
            weightedDist[particle] = beliefDist[particle] * weight_i

        # normalize the updated belief dist
        weightedDist.normalize()

        # check for case when all particles receive zero weight
        # If true the list of particles should be reinitialized by calling initializeUniformly.
        if weightedDist.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # resample from the weighted distribution to construct the new list of particles.
            # The sample method of the DiscreteDistribution is used.
            newParticleList = [weightedDist.sample() for i in range(self.numParticles)]
            # assign the new list to the field self.particles
            self.particles = newParticleList

    # Question 7
    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        # declare a list for storing the particles where each particle is a pair (x,y) for the ghost position
        newParticleList = []
        # loop through all the particles or previous ghost positions
        for particle in self.particles:
            # get the new position distribution given the previous position using the sensor model
            newPosDist = self.getPositionDistribution(gameState, particle)
            # add the new particle/sample to the list
            newParticleList.append(newPosDist.sample())
        # assign the new list to the field self.particles
        self.particles = newParticleList

    # Question 5
    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        # create the current belief distribution by calling self.getBeliefDistribution()
        beliefDist = DiscreteDistribution()
        # initialized beliefDist to zero probability for all states
        for state in self.legalPositions:
            beliefDist[state] = 0

        # populate the belief distribution by the occurrence frequency of each particle
        # assume the probability of each sampled state = number of with value of that state / num particles
        # the distribution is normalized
        for state in self.particles:
            beliefDist[state] += 1 / self.numParticles
        # return the distribution
        return beliefDist



class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    # Question 8
    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        "*** YOUR CODE HERE ***"

        """ itertools.product(list, repeat= --) returns the cartesian product of the provided list with itself
         for the number of times specified by keyword “repeat”. For the case of tracking multimple ghosts 
         Each particle is a tuple of possible states for all ghosts which has self.numGhosts elements such as:
         (Sg1, Sg2, Sg3, Sg4, ....Sgself.numGhosts). 
         Random particles are drawn from a pool of permutations of all the legal states in this tuple"""

        # declare a list for storing the particles where each particle is a tuple of all ghosts' states
        self.particles = []
        # create a pool/list of all possible arrangements of ghosts legal states in a tuple using Cartesian Product
        pool = list(product(self.legalPositions, repeat=self.numGhosts))
        # for the number of particles (already selected) repeat the following:
        for i in range(self.numParticles):
            # randomly choose a tuple from the pool by randomly picking an index
            # randIndex is a random integer from the “discrete uniform” distribution in [low, high].
            randIndex = np.random.random_integers(0, len(pool)-1, size=1)[0]
            # take the state at the selected index and add it to the list of particle
            self.particles.append(pool[randIndex])

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    # Question 9
    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # JointParticleFilter extends the ParticleFilter class; it has access to the same class functions.
        # create the current belief distribution by calling self.getBeliefDistribution()
        beliefDist = self.getBeliefDistribution()
        # declare a distribution for particles weights by creating a DiscreteDistribution object
        # keys are the states and weights are the probability of the observation for a specific ghost given its state
        weightedDist = DiscreteDistribution()
        # initialize the values of weightedDist to zero for all states keys
        for state in self.legalPositions:
            weightedDist[state] = 0

        # Note: observation is a "list" of noisyDistances to ghosts each corresponding to a ghost
        # ghostPositions are represented by particles
        # loop through all the particles or current ghost positions
        for particle in self.particles:
            # initialize the weight of each particle to 1
            weight_particle = 1
            # loop through the number of ghosts
            for i in range(self.numGhosts):
                # calculate the weight of each particle as the product of all weights each corresponding
                # to a probability of specific ghost' observation given its state
                weight_particle = weight_particle * self.getObservationProb(observation[i], gameState.getPacmanPosition(), particle[i],
                                               self.getJailPosition(i))
            # create un-normalized distribution by combining the likelihood weighting of each particle with its current belief dist.
            weightedDist[particle] = beliefDist[particle] * weight_particle

        # normalize the distribution
        weightedDist.normalize()

        # check for case when all particles receive zero weight
        # If true the list of particles should be reinitialized by calling initializeUniformly.
        if weightedDist.total() == 0:
            self.initializeUniformly(gameState)
        else:
            # resample from the weighted distribution to construct the new list of particles.
            # The sample method of the DiscreteDistribution class is used.
            newParticleList = [weightedDist.sample() for i in range(self.numParticles)]
            # assign the new list to the field self.particles
            self.particles = newParticleList

    # Question 10
    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        # declare a list for storing the particles where each particle in a tuple of all ghosts' states
        newParticles = []
        # ghostPositions are represented by particles
        # loop through all the particles or previous ghost positions
        for oldParticle in self.particles:
            # A list instead of tuple for new ghost positions
            newParticle = []
            # convert the previous position tuple to a position list
            prevGhostPositions = list(oldParticle)
            # repeat the following for the number of ghosts:
            # Note: each ghost draws a new position given the previous positions of all the ghosts.
            for i in range(self.numGhosts):
                # get the distribution for the new state of each ghost from the transition model
                newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])
                # sample the new state transition to obtain a new particle and add it to the list of new particles
                newParticle.append(newPosDist.sample())
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
