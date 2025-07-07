This notebook provides a comprehensive, from-scratch implementation of an LSTM (Long Short-Term Memory) neural network using PyTorch. The focus is on building the LSTM cell manually, layer-stacking logic, and forward computation, rather than relying on nn.LSTM. This approach offers a deeper understanding of LSTM internals, including gate computations, cell state updates, and normalization techniques.

# Overview of LSTM Architecture
LSTM networks are a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequence data. Unlike standard RNNs, LSTMs use internal gating mechanisms to control the flow of information and gradients, making them resilient to the vanishing gradient problem.
The architecture includes:

**Forget gate fₜ** -decides what information from the previous cell state should be discarded.

**Input gate iₜ**-controls how much of the new candidate information should be added to the cell state.

**Candidate value gₜ**-is the new information, generated from the current input and previous hidden state, that could be added to the cell state.

**Output gate oₜ**-determines how much of the updated cell state should be exposed as the hidden state.

**Cell state update cₜ**-serves as the memory of the network, carrying accumulated and filtered information across time steps

**Hidden state update hₜ**-is the output of the LSTM at the current time step, used for both final predictions and passing to the next layer or time step.

# LSTM Cell Equations

Given input vector `xₜ`, previous hidden state `hₜ₋₁`, and previous cell state `cₜ₋₁`, the LSTM cell performs the following computations:

1. **Gates and Candidate Value:**

fₜ = sigmoid(Wf · xₜ + Uf · hₜ₋₁ + bf)     # Forget gate

iₜ = sigmoid(Wi · xₜ + Ui · hₜ₋₁ + bi)     # Input gate

gₜ = tanh(Wg · xₜ + Ug · hₜ₋₁ + bg)        # Candidate value

oₜ = sigmoid(Wo · xₜ + Uo · hₜ₋₁ + bo)     # Output gate

2. **Cell State Update:**

cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ

3. **Hidden State Update:**

hₜ = oₜ ⊙ tanh(cₜ)

Where:
- `sigmoid()` and `tanh()` are activation functions.
- `⊙` denotes element-wise (Hadamard) product.
- `W*`, `U*`, and `b*` are weight matrices and bias vectors learned during training.

# Code Structure

**LSTMCell**

This class defines a single LSTM cell that performs the gate computations and state updates. Two separate linear layers (hidden_lin, input_lin) are used to project the previous hidden state and current input respectively. Optional Layer Normalization is applied to each gate and the cell state to stabilize learning.

If layer_norm=True, nn.LayerNorm is used.
If False, identity layers are used instead, making it behave like a standard LSTM

Layer normalization is often preferred over batch normalization in recurrent neural networks like LSTMs because it normalizes across features within a single data point rather than across the entire batch. This is important because in sequential models, each time step is processed individually and may have different batch sizes (especially in variable-length sequences), making batch statistics unstable or unavailable. Layer normalization avoids this issue by computing statistics (mean and variance) across the hidden units of a single input sample, making it more stable and suitable for time-series and NLP tasks. It also works better when batch sizes are small or when the model needs to generalize across varying conditions, which is common in RNN-based model



**LSTM**

This class stacks multiple LSTMCell layers into a deep network. It handles input sequences shaped as (sequence_length, batch_size, input_size) and loops over both time steps and LSTM layers.

The hidden and cell states are initialized to zeros if not provided.

Each layer’s output at time step t becomes the input for the next layer.

The top layer's outputs are collected across all time steps and returned along with final states.

# Output Format

The model returns:

out: the output sequence from the final LSTM layer (shape: [sequence_length, batch_size, hidden_size])

(h, c): the final hidden and cell states for each LSTM layer (each of shape [num_layers, batch_size, hidden_size])

# Dependencies

torch

typing (for type hints like Optional, Tuple)







