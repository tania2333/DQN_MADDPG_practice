import numpy as np
import matplotlib.pyplot as plt
class OU_noise(object):
	def __init__(self, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.05, decay_period = 0):  #0.2
		self.mu = mu
		self.theta = theta
		self.sigma = max_sigma
		self.max_sigma = max_sigma
		self.min_sigma = min_sigma
		self.decay_period = decay_period#630000
		self.num_actions = [8, 14]
		self.reset()

	def reset(self):
		self.state = np.zeros(self.num_actions)

	def state_update(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions[0],self.num_actions[1])
		self.state = x + dx

	def get_noise(self, training_step):
		self.state_update()
		state = self.state
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, training_step / self.decay_period)
		return state

def main():
	ou_noise = OU_noise(mu=np.zeros(1))
	plt.figure('data')
	y = []
	t = np.linspace(0, 100, 1000)
	for i in t:
		y.append(ou_noise.get_noise(i))
	plt.plot(t, y)
	plt.show()


if __name__ == "__main__":
	main()
