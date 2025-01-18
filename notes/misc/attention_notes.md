## Attention

Attention in neural networks refers to the most important details one should focus on (or attend to) in order to solve a given task.

Goal of introducing attention in deep learning is to teach the machine where to pay attention to, given its purpose and context.

Building an Attention neural network:
* very simple Attention netowrk
* has an Embedding layer for the context (this is where the network learns how contexts affect Attention)
* a Linear layer that computes the output from the attention glimpse

Once you have attention weights, they are multiplied element-wise multiplied by the input passed to a model, emphasizing or de-emphasizing certain parts of the sequence based on the context.

Scaled dot product attention mechanism is a common attention mechanism seen in transformer models.
* computes a weighted sum of the input items
* weights are acquired during training and reflect the relevance of eaach input piece
Works with vectors: Keys, values, and queries
* Query (Q) - program wants to know something specific
* Key (K) - these are like the pieces of inormation it has. Each piece has its key.
* Value (V) - this is the actual information associated with each key

--> Goal is to determine which piece of the information are most significant to the inquiry. Accomplished by determining how similar the question (Q) is to each item of information (K). 
* dot product of the query and information component

Dot Product Attention - attention scors calculated as the dot product of the queries ane keys.

Multi-Head Attention

Multiple attention heads capture different aspects of the input sequence. Each head calculates its own set of attention scores, and the results are concatenated and transformed to produce the final attention weights.

Used extensively in Transformers architecture. Enables the model to atten dto different parts of the input sequence concurrently, capturing diverse characteristics or patterns.

Each expert, or head, poses specific inquiry regarding the incoming data. Based on their experience each expert extracts the most relevant information. They foucs on their designated aspect while ignoring the rest.

Attention can be applied to many deep learning architectures and tasks. The capacity to choose and focus on relevant information adds to improved deep-learning model performance.


Self-Attention vs Attention

Attention refers to mechanism in which a model calcs attention scores between different parts of an input and another part of the input or external memory.

Self-Attention the model calculates attention scores between different parts of the input sequence without using external memory. Self-attention lets the model figure out how important each part of the series is, determine how the parts depend on each other and make predictions based on that.

```python
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted
```
* query (Q) -> transforms the input into queries, asking "what am I looking for" for each element in sequence
* key (K) -> transforms the input into keys, represent "what is contained here" for each element
* value (V) -> transforms input to values, represent the actual information we wnat to use or pass forward

Dot product between a query and a key measures their similarity (how well they "match)

The core idea behind the Transformer model is the attention mechanism, an innovation that was originally envisioned as an enhancement for the encoder-decoder RNNs applied to sequence-to-sequence applications.
* The intuition behind attention is that rather than compressing the input, it might be better for the decoder to revisit the input sequence at every step

What is attention? The attention mechanism describes a weighted average of (sequence) elements with the weights dynamically computed based on an input query and elements' keys. 

Query, Key, and Values are just linear layers. Each taking the input in.
* they leanr different weight matrices based on their interactions with each other and the task specific supervision

Queries compared with keys to calculate attention scores

These scores determine how values are weighted

The model learns to align these weights (queries, keys, and values) such that:
* queries focus on the most relevant parts of the sequence
* keys encode information about which parts are relevant
* values contain the information needed for the final output


Sources:
https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html
https://www.freecodecamp.org/news/what-are-attention-mechanisms-in-deep-learning/
https://spotintelligence.com/2023/01/31/self-attention/