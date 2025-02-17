## Recurrent Neural Networks

Traditional neural networks cannot use their understanding of previously seen data on current data.

RNNs address this issue. They are networks with loops in them, allowing information to persist.

A recurrent neural network can be thought of multiple copies of the same network. Each passing a message to a successor.

Natural architecture for sequences and lists of data.

Traditional RNNs are good at handling short sequences of data, but not so much long pieces.

As the gap between the relevant context of the input data and the data that the model is tasked with predicting grows, RNNs become unable to learn to connect the information.

LSTMS - Long Short Term Memory networks, were created to help address this context memory issue in RNNs
* the LSTM is designed to handle longer memory context
* does this by incoporating a unique structure in the repeating module using Gates
* The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.
* Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation.

Three gates:
1. forget gate layer
2. input gate layer
3. update gate

GRU - Gated Recurrent Unit, combines the forget and input gates into a single “update gate.” It also merges the cell state and hidden state, and makes some other changes.

Another potential improvement to make to RNNs: Attention! Let every step of an RNN pick information to look at from some larger collection of information.


Sources:
https://colah.github.io/posts/2015-08-Understanding-LSTMs/