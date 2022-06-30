import torch


class MemoryReadingsStats:
	def __init__(self):
		self.memory_readings = None  # will have shape (dataset_size, memory_size)
		self.readings_variance = None
		self.kl_divergence = None
		self.random_projections = None
		self.random_matrix = None


	def update_memory_readings(self, batch_readings):
		if self.memory_readings is None:
			self.memory_readings = batch_readings
		else:
			self.memory_readings = torch.concat((self.memory_readings, batch_readings))


	def compute_readings_variance(self):
		self.readings_variance = torch.var(self.memory_readings, dim=0, unbiased=False)
		return self.readings_variance


	def compute_readings_kl_divergence(self,):
		kl_div = torch.nn.functional.kl_div
		sample = torch.rand(self.memory_readings.shape)
		self.kl_divergence = kl_div(self.memory_readings, sample)
		return self.kl_divergence


	def init_random_matrix(self, memory_size):
		if self.random_matrix is None:
			self.random_matrix = torch.rand((memory_size, 2))


	def compute_random_projections(self):
		assert self.random_matrix is not None
		self.random_projections = self.memory_readings @ self.random_matrix
		return self.random_projections


	def get_stats(self):
		var = f"Readings variance: {self.readings_variance}\n"
		kl = f"Readings KL divergence from uniform distribution: {self.kl_divergence}\n"
		return var + kl


	def __repr__(self):
		return self.get_stats()
