import numpy as np

class IndepBnet(object):
    def __init__(self):
        self.probs = None
    
    def train(self, data):        
        N, _ = data.shape
        counts = np.sum(data, dtype=float, axis=0)
        counts[counts != N] += 1 # dont add +1 when its value is equal to N
        counts[counts == N] -= 1 # subtract a token 1 since it cause numerical
        counts /= N # probability of var = 1
        self.probs = counts

    def log_prob(self, datavec):
        logprob = 0
        for i in range(len(datavec)):
            # print(i, logprob, datavec[i], self.probs[i])
            if datavec[i] == 1:
                logprob += np.log(self.probs[i])
            else:
                logprob += np.log(1 - self.probs[i])
        return logprob

