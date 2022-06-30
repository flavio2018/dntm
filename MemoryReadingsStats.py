import torch


class MemoryReadingsStats:
	def __init__(self):
		self.readings = None
		self.readings_variance = None
		self.kl_divergence = {}
		self.random_projections = None


	def update_memory_readings(self, batch_readings):
		return


	def compute_readings_variance(self):
		return


	def compute_readings_kl_divergence(self, from_dist='uniform'):
		assert from_dist in ['uniform', 'white_noise']
		return


	def compute_random_projections(self):
		return


	def get_stats(self):
		return "Something"


	def __repr__(self):
		return self.get_stats()
