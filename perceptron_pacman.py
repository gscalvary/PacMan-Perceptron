# Christopher Oliver
# CS5100 - Foundations of AI
# Fall 2014
# -----------
# perceptron_pacman.py
# --------------------
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


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                # initialize variables
                bestScore = float("-inf")
                bestAction = ""
                # for each data point spin through each possible action
                for action in trainingData[i][1]:
                    # compute a score for each action
                    actionScore = 0
                    # for each action account for each feature change in the
                    # score
                    for f in self.features:
                        actionScore += self.weights[f] * trainingData[i][0][action][f]
                    # save the best score and its action
                    if actionScore > bestScore:
                        bestScore = actionScore
                        bestAction = action
                # if the wrong action was chosen increase the weight of the
                # features associated with the correct action and decrease
                # the weight of the features associated with the incorrect
                # action.
                if bestAction != trainingLabels[i]:
                    # there is a weight for each feature like food count
                    for f in self.features:
                        self.weights[f] += trainingData[i][0][trainingLabels[i]][f]
                        self.weights[f] -= trainingData[i][0][bestAction][f]