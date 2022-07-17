"""This script contains the implementation of a Dynamic-Neural Turing Machine.

By convention, tensors whose name starts with a 'W' are bidimensional (i.e. matrices), 
while tensors whose name starts with a 'u' or a 'b' are one-dimensional (i.e. vectors).
Usually, these parameters are part of linear transformations implementing a multi-input perceptron,
thereby representing the weights and biases of these operations.

The choice that was made in this implementation is to decouple the external memory of the model
as a separate PyTorch Module. The full D-NTM model is thus composed of a controller module and a 
memory module."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from .CustomGRU import CustomGRU
from .DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


class DynamicNeuralTuringMachine(nn.Module):
    def __init__(self, memory, controller_hidden_state_size, controller_input_size, controller_output_size=10):
        super(DynamicNeuralTuringMachine, self).__init__()
        self.add_module("memory", memory)
        self.controller = CustomGRU(input_size=controller_input_size,
                                    hidden_size=controller_hidden_state_size,
                                    memory_size=memory.overall_memory_size)
        self.W_output = nn.Parameter(torch.zeros(controller_output_size, self.memory.memory_contents.shape[1]))
        self.b_output = nn.Parameter(torch.zeros(controller_output_size, 1))

        self._init_parameters(init_function=nn.init.xavier_uniform_)

    def forward(self, input):
        if len(input.shape) == 2:
            return self.step_on_batch_element(input)
        elif len(input.shape) == 3:
            return self.step_on_batch(input)

    def step_on_batch(self, batch):
        """Note: the batch is assumed to conform to the batch_first convention of PyTorch, i.e. the first dimension of the batch
        is the batch size, the second one is the sequence length and the third one is the feature size."""
        logging.debug(f"Looping through image pixels")
        batch_size, seq_len, feature_size = batch.shape
        
        for i_seq in range(seq_len):
            batch_element = batch[:, i_seq, :].reshape(feature_size, batch_size)
            controller_hidden_state, output = self.step_on_batch_element(batch_element)
        return controller_hidden_state, output

    def step_on_batch_element(self, x):
        self.memory_reading = self.memory.read(self.controller_hidden_state)
        self.memory.update(self.controller_hidden_state, x)
        self.controller_hidden_state = self.controller(x, self.controller_hidden_state, self.memory_reading)
        address_size = self.memory.memory_addresses.shape[1] 
        self.memory_reading = self.memory_reading[:, address_size:]
        output = F.log_softmax(self.W_output @ self.memory_reading + self.b_output, dim=0)
        return self.controller_hidden_state, output

    def _init_parameters(self, init_function):
        logging.info(f"Initialization method: {init_function.__name__}")
        # Note: the initialization method is not specified in the original paper
        for name, parameter in self.named_parameters():
            if len(parameter.shape) > 1:
                logging.info(f"Initializing parameter {name}")
                if name in ("memory_addresses", "W_query", "b_query"):
                    init_function(parameter, gain=1)
                elif name in ("u_sharpen", "W_content_hidden", "W_content_input"):
                    init_function(parameter, gain=torch.nn.init.calculate_gain("relu"))
                elif name == "u_lru":
                    init_function(parameter, gain=torch.nn.init.calculate_gain("sigmoid"))
                else:
                    init_function(parameter)
                logging.debug(f"{name}: {parameter}")
            if name == 'b_output':
                logging.info("Initializing bias b_output")
                torch.nn.init.constant_(parameter, 0.1)
                logging.debug(f"{name}: {parameter}")

    def prepare_for_batch(self, batch, device):
        self.memory._reset_memory_content()
        self._reshape_and_reset_hidden_states(batch_size=batch.shape[0], device=device)
        self.memory._reshape_and_reset_exp_mov_avg_sim(batch_size=batch.shape[0], device=device)
        self.controller_hidden_state = self.controller_hidden_state.detach()

    def _reshape_and_reset_hidden_states(self, batch_size, device):
        with torch.no_grad():
            controller_hidden_state_size = self.controller.W_ir.shape[0]
        self.register_buffer("controller_hidden_state", torch.zeros(size=(controller_hidden_state_size, batch_size), device=device))


def build_dntm(cfg, device):
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=cfg.model.n_locations,
        content_size=cfg.model.content_size,
        address_size=cfg.model.address_size,
        controller_input_size=cfg.model.controller_input_size,
        controller_hidden_state_size=cfg.model.controller_hidden_state_size
    )

    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=cfg.model.controller_hidden_state_size,
        controller_input_size=cfg.model.controller_input_size,
        controller_output_size=cfg.model.controller_output_size
    ).to(device)

    if cfg.model.ckpt is not None:
        logging.info(f"Reloading from checkpoint: {cfg.model.ckpt}")
        state_dict = torch.load(cfg.model.ckpt)
        batch_size_ckpt = state_dict['controller_hidden_state'].shape[1]
        dntm.memory._reset_memory_content()
        dntm._reshape_and_reset_hidden_states(batch_size=batch_size_ckpt, device=device)
        dntm.memory._reshape_and_reset_exp_mov_avg_sim(batch_size=batch_size_ckpt, device=device)
        dntm.load_state_dict(state_dict)
    return dntm
