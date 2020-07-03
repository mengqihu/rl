import numpy as np
from scipy.stats import beta


class Bandit(object):
    def __init__(self, theta):
        self.theta = theta
        self.numArms = len(self.theta)

        self.numWins = np.zeros(self.numArms)
        self.numTries = np.zeros(self.numArms)
        self.winProbabilities = (self.numWins + 1) / (self.numTries + 2)

    def reset(self):
        self.numWins = np.zeros(self.numArms)
        self.numTries = np.zeros(self.numArms)
        self.winProbabilities = (self.numWins + 1) / (self.numTries + 2)

    def pull(self, i):
        return np.random.rand() < self.theta[i]

    def update(self, i, success):
        self.numTries[i] += 1
        if success:
            self.numWins[i] += 1
        self.winProbabilities[i] = (
            self.numWins[i] + 1) / (self.numTries[i] + 2)

    def simulate(self, policy, steps):
        wins = np.zeros(steps)

        for step in range(steps):
            i = policy.arm(self)
            win = self.pull(i)
            self.update(i, win)
            wins[step] = wins[max(0, step-1)] + win

        return wins


class EpsGreedyPolicy(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def arm(self, bandit):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(range(bandit.numArms))
        else:
            return np.argmax(bandit.winProbabilities)


class SoftmaxPolicy(object):
    def __init__(self, lamb):
        self.lamb = lamb

    def arm(self, bandit):
        prob = np.exp(self.lamb * bandit.winProbabilities)
        prob = prob / sum(prob)
        return np.random.choice(range(bandit.numArms), p=prob)


class IntervalPolicy(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def arm(self, bandit):
        intervals = [beta(bandit.numWins[i]+1, bandit.numTries[i]-bandit.numWins[i]+1).ppf(self.alpha)
                     for i in range(bandit.numArms)]
        return np.argmax(intervals)
