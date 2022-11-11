import random
import math


# use this class to represent the process to calculate the theoretical and
# practical value of one single probabilistic binary neuron
class ComputeProbNN:
    def __init__(self):
        self.random_min = 1
        self.random_max = 6
        self.x = [1, 1, 2, -1, -1]
        self.w = [1, 2, -2, 1, -2]
        self.alpha = 1

    def compute_prob(self):
        s_hat = 0
        for i in range(len(self.x)):
            s_hat += self.x[i] * self.w[i]
        prob = (1 + math.exp(-self.alpha * s_hat)) ** -1
        return prob

    def throw_die(self, prob):
        num = random.randint(self.random_min, self.random_max)
        if num >= self.random_max * prob:
            return 1
        else:
            return 0


prob_neuron = ComputeProbNN()
theoretical_value = prob_neuron.compute_prob()
count_one = 0
test_num = 100
for i in range(test_num):
    if prob_neuron.throw_die(theoretical_value) == 1:
        count_one += 1
practical_value = count_one / test_num
print(practical_value, theoretical_value)