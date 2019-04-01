"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
np.random.seed(42)

# data I/O
data = open('simple_pattern.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size = len(data)
vocab_size = len(chars)
print('data has %d total characters and %d unique characters.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 3 # size of hidden layer of neurons
seq_length = 4  # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
input_weights_U = np.random.randn(hidden_size, vocab_size) * 0.1     # input to hidden
hidden_weights_W = np.random.randn(hidden_size, hidden_size) * 0.1   # hidden to hidden
hidden_bias = np.zeros((hidden_size, 1)) # hidden bias

output_weights_V = np.random.randn(vocab_size, hidden_size) * 0.1    # hidden to output
output_bias = np.zeros((vocab_size, 1)) # output bias

def step(inputs, targets, hidden_state_prev):

  xs, hidden_states, outputs, probabilities = {}, {}, {}, {}
  hidden_states[-1] = np.copy(hidden_state_prev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    # one-hot-encoding the input character
    xs[t] = np.zeros((vocab_size,1)) 
    character = inputs[t]
    target = targets[t]
    xs[t][character] = 1
    # Compute hidden state
    hidden_states[t] = np.tanh(input_weights_U @ xs[t] + hidden_weights_W @ hidden_states[t-1] + hidden_bias) 
    # Compute output and probabilities
    outputs[t] = output_weights_V @ hidden_states[t] + output_bias
    probabilities[t] = np.exp(outputs[t]) / np.sum(np.exp(outputs[t]))
    #Compute cross-entropy loss
    loss += -np.log(probabilities[t][target,0]) # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  input_weights_U_grad = np.zeros_like(input_weights_U)
  hidden_weights_W_grad = np.zeros_like(hidden_weights_W)
  hidden_bias_grad = np.zeros_like(hidden_bias)
  output_weights_V_grad = np.zeros_like(output_weights_V)
  output_bias_grad = np.zeros_like(output_bias)

  hidden_state_next_grad = np.zeros_like(hidden_states[0])

  for t in reversed(range(len(inputs))):
    output_grad = np.copy(probabilities[t])
    output_grad[targets[t]] -= 1 
    output_weights_V_grad += output_grad @ hidden_states[t].T
    output_bias_grad += output_grad
    dh = output_weights_V.T @ output_grad + hidden_state_next_grad # backprop into h
    dhraw = (1 - hidden_states[t] * hidden_states[t]) * dh # backprop through tanh nonlinearity
    hidden_bias_grad += dhraw
    input_weights_U_grad += dhraw @ xs[t].T
    hidden_weights_W_grad += dhraw @ hidden_states[t-1].T
    hidden_state_next_grad = hidden_weights_W.T @ dhraw

  for param_grad in [input_weights_U_grad, hidden_weights_W_grad, output_weights_V_grad, hidden_bias_grad, output_bias_grad]:
    np.clip(param_grad, -5, 5, out=param_grad) # clip to mitigate exploding gradients


  return loss, input_weights_U_grad, hidden_weights_W_grad, output_weights_V_grad, hidden_bias_grad, output_bias_grad, hidden_states[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is hidden memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(input_weights_U @ x + hidden_weights_W @ h + hidden_bias)
    y = output_weights_V @ h + output_bias
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hidden_state_prev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hidden_state_prev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, input_weights_U_grad, hidden_weights_W_grad, output_weights_V_grad, hidden_bias_grad, output_bias_grad, hidden_state_prev = step(inputs, targets, hidden_state_prev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: 
      print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, param_grad in zip([input_weights_U, hidden_weights_W, output_weights_V, hidden_bias, output_bias], 
                                [input_weights_U_grad, hidden_weights_W_grad, output_weights_V_grad, hidden_bias_grad, output_bias_grad]):
    param += -learning_rate * param_grad 

  p += seq_length # move data pointer
  n += 1 # iteration counter 