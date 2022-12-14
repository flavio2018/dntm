import torch
from torch import nn
import torch.nn.functional as F
import logging


class MemoryHead(nn.Module):
    def __init__(self, n_locations, location_size, batch_size, hidden_size):
        super(MemoryHead, self).__init__()

        self.register_buffer("exp_mov_avg_similarity", torch.zeros(size=(n_locations, batch_size)))
        self.W_hat_hidden = nn.Parameter(torch.zeros(size=(n_locations, hidden_size)))

        # query vector MLP parameters (W_k, b_k)
        self.W_query = nn.Parameter(torch.zeros(size=(n_locations, location_size)), requires_grad=True)
        self.b_query = nn.Parameter(torch.zeros(size=(location_size, 1)), requires_grad=True)

        # sharpening parameters (u_beta, b_beta)
        self.u_sharpen = nn.Parameter(torch.zeros(size=(hidden_size, 1)), requires_grad=True)
        self.b_sharpen = nn.Parameter(torch.zeros(1), requires_grad=True)

        # LRU parameters (u_gamma, b_gamma)
        self.b_lru = nn.Parameter(torch.zeros(1))
        self.u_lru = nn.Parameter(torch.zeros(size=(hidden_size, 1)))

    def address(self, h, M):
        self.projected_hidden_state = self.W_hat_hidden @ h
        self.query = self.W_query.T @ self.projected_hidden_state + self.b_query
        self.sharpening_beta = F.softplus(self.u_sharpen.T @ h + self.b_sharpen) + 1
        
        # compute similarity
        self.similarity_vector = self.sharpening_beta * F.cosine_similarity(M, self.query.T[:, None, :],
                                                                            eps=1e-7, dim=-1).T
        # lru mechanism
        self.lru_gamma = torch.sigmoid(self.u_lru.T @ h + self.b_lru)
        self.lru_similarity_vector = F.softmax(self.similarity_vector - self.lru_gamma * self.exp_mov_avg_similarity, dim=0)
        with torch.no_grad():
            self.exp_mov_avg_similarity = 0.1 * self.exp_mov_avg_similarity + 0.9 * self.similarity_vector
        return self.lru_similarity_vector

    def forward(self, h, M):
        return self.address(h, M)



class DynamicNeuralTuringMachineMemory(nn.Module):
    def __init__(self, n_locations, content_size, address_size, controller_input_size, controller_hidden_state_size, batch_size):
        """Instantiate the memory.
        n_locations: number of memory locations
        content_size: size of the content part of memory locations
        address_size: size of the address part of memory locations"""
        super(DynamicNeuralTuringMachineMemory, self).__init__()

        self.register_buffer("memory_contents", torch.zeros(size=(batch_size, n_locations, content_size)))
        self.memory_addresses = nn.Parameter(torch.zeros(size=(batch_size, n_locations, address_size)), requires_grad=True)
        self.overall_memory_size = content_size + address_size

        self.read_head = MemoryHead(
            n_locations=n_locations,
            location_size=self.overall_memory_size,
            batch_size=batch_size,
            hidden_size=controller_hidden_state_size)

        self.write_head =  MemoryHead(
            n_locations=n_locations,
            location_size=self.overall_memory_size,
            batch_size=batch_size,
            hidden_size=controller_hidden_state_size)

        # erase parameters (generate e_t)
        self.W_erase = nn.Parameter(torch.zeros(size=(content_size, controller_hidden_state_size)))
        self.b_erase = nn.Parameter(torch.zeros(size=(content_size, 1)))

        # writing parameters (W_m, W_h, alpha)
        self.W_content_hidden = nn.Parameter(torch.zeros(size=(content_size, controller_hidden_state_size)))
        self.W_content_input = nn.Parameter(torch.zeros(size=(content_size, controller_input_size)))
        self.u_input_content_alpha = nn.Parameter(torch.zeros(size=(1, controller_input_size)))
        self.u_hidden_content_alpha = nn.Parameter(torch.zeros(size=(1, controller_hidden_state_size)))
        self.b_content_alpha = nn.Parameter(torch.zeros(1))

    def read(self, controller_hidden_state):
        logging.debug("Reading memory")
        self.read_weights = self.read_head(controller_hidden_state, self._full_memory_view())
        # this implements the memory NO-OP at reading phase
        # M is (BS, R, C), r_w is (BS, R), we want (BS, C)
        return (self._full_memory_view()[:, :-1].mT @ self.read_weights[:-1].T.unsqueeze(2)).squeeze().T
        # TODO add in tests NO-OP

    def update(self, controller_hidden_state, controller_input):
        logging.debug("Updating memory")
        self.write_weights = self.write_head(controller_hidden_state, self._full_memory_view())
        self.erase_vector = self.W_erase @ controller_hidden_state + self.b_erase  # TODO MLP

        self.alpha = (self.u_input_content_alpha @ controller_input +
                 self.u_hidden_content_alpha @ controller_hidden_state + self.b_content_alpha)

        self.candidate_content_vector = F.relu(self.W_content_hidden @ controller_hidden_state +
                                               torch.mul(self.alpha, self.W_content_input @ controller_input))

        # this implements the memory NO-OP at writing phase
        self.memory_contents[:, :-1, :] = (self.memory_contents[:, :-1, :]
                                           - self.write_weights[:-1, :] @ self.erase_vector.T
                                           + self.write_weights[:-1, :] @ self.candidate_content_vector.T)

    def _full_memory_view(self):
        return torch.cat((self.memory_addresses, self.memory_contents), dim=2)

    def _reset_memory_content(self):
        """This method exists to implement the memory reset at the beginning of each episode."""
        self.memory_contents.fill_(0)
        self.memory_contents.detach_()
        # self.memory_contents = torch.zeros_like(self.memory_contents)  # alternative

    def _reshape_and_reset_exp_mov_avg_sim(self, batch_size, device):
        with torch.no_grad():
            n_locations = self.memory_addresses.shape[0]
        self.register_buffer("exp_mov_avg_similarity", torch.zeros(size=(n_locations, batch_size), device=device))

    # def reshape_and_reset_read_write_weights(self, shape):
    #     self.read_weights = nn.Parameter(torch.zeros(size=shape))
    #     self.write_weights = nn.Parameter(torch.zeros(size=shape))

    def forward(self, x):
        raise RuntimeError("It makes no sense to call the memory module on its own. "
                           "The module should be accessed by the controller "
                           "either by addressing, reading or updating the memory.")
