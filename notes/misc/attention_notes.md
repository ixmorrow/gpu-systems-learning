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

Attention is one component to a neural network's architecture.

3Blue1Brown:

Token converted to an embedding which represents a location in latent space of the token.
* initial token embedding is simply a lookup to get a corresponding vector embedding

Next step in a transformer is the attention block which gives the surrounding tokens a chance to provide context to each other.

Tokens with different meanings given different contexts would have different locations in the embedding space. It is the job of an attention block to calculate what it needs to add to the generic embedding, as a function of its context, to move it to one of those specific directions.

This transfer of information from the embedding of one token to another can occur over potentially large distances and can involve information that's much richer than just a single word.

Initial embedding for each word is some high-dimensional vector that contains no reference to the context.
* actually contain positional encodings as well

Example:

"A fluffy blue creature roamed the verdant forest."

Want the adjectives in the phrase to adjust the embeddings of the nouns they describe.
Our goal is to have a series of computations produce a new refined set of embeddings E', where, in our case, those corresponding to the nouns have ingested the meaning from their corresponding adjectives.

Queries - For first layer in attention block, can imagine each noun in the sentence asking the question:
"Hey, are there any adjectives sitting in front of me?"
* these questions are somehow encoded as another vector we call the Query
* to compute this query, we multiply the vector by the embedding of the specific word we are querying
* multiply query vector Wq by all of the embeddings in the context and produce one query vector for each token
* values in the vector Wq are the parameters of the model, meaning they have to be learned through training
Specifically, we’re imagining that this Wq matrix maps the embeddings of nouns to a certain direction in this smaller query space that (somehow) encodes the notion of looking for adjectives in preceding positions.

Keys - There is also a key matrix that the input context is passed into. This is made up of parameters the model will learn as well, just like the queries vector.
* this vector is matrix multiplied with the input context
* the product is a second sequence of vectors for each token we call the keys
* Conceptually, we want to think of these keys as potential answers to the queries.
* We want to think of the keys matching the queries when they closely align with each other
* In our made up example, we might imagine that the key matrix maps the adjectives like fluffy and blue to vectors that are closely aligned with the query produced by the word creature.
* The way we measure how closely queries/keys align is by taking their dot product. The larger the dot product result, the closer they are in the Query/Key vector space
* after this dot product across all of the query/key values, we are left with a table of query/key pairing results that display a score of how relevant each word is to updating the meaning of every other word
* we want each col to add up to 1, so we need to normalize the data (currently ranges b/t -inf - inf) - so we compute a softmax of each col to normalize data between 0 and 1
* after normalizing, we store the normalized data back in a table. This data is called the Attention Pattern
* At this point, we're safe to think about each column as giving weights according to how relevant the word on the left is to the corresponding value at the top.

The model is run on a given text example where it is tasked with predicting the next word. The model's weights are adjusted to either reward or punish it based on the probability it assigns to the true next word.
* The model simultaneously predicts every possible next word given a sequence of words, not just the next word following a string of text.

The size of the attention pattern is equal to the square of the context size!
* this is why context size is a significant limitation in large language models!
* there are newer ideas to try to expand context windows, (Google TITAN paper!)[https://arxiv.org/abs/2501.00663]


Values - Next steps once you have the attention patterns is to actually update the embeddings of each word given the context. For example, we would want the embedding of fluffy to somehow cause a change to the embedding of creature, one that moves it to a different part of this high-dimensional embedding space that more specifically encodes a fluffy creature.

To do this, we use a third matrix called the values matrix. 

Value vector is multiplied by every single token in the context to product a vector of values for each token.

For each column in our grid, we would multiply each of the value vectors by the corresponding weight in that column.
* in the context of our example, this would mean under the column for "creature" multiplying the value vector by the key vectors of each token. The key tokens for blue and fluffly would be much larger than the ones for the other words/tokens

Then, we add the result of this multiplication to each value in the embedding vector foe each token.

This whole process is a single head of attention.

A full attention block contains what is called multi-headed attention, where a lot of these ops are run in parallel, each with its own query, key, value maps.
* GPT-3, for example, uses 96 attention heads inside each block.

What this means is that for every single token in the context, each of these attention heads produces a proposed change to the token embedding. Then, all of these proposed changes would be added up, one for each head, and the result is added to the original embedding of that position.

The overall idea is that by running many distinct heads in parallel, the model is given the capacity to learn many distinct ways that context changes meaning.

A large reason for the success of the attention mechanism is not so much any specific kind of behavior that it enables, but the fact that it's extremely parallelizable, meaning that it can run a huge number of computations in a short time using GPUs. Given that one of the big lessons about deep learning in the last decade or two has been that scale alone seems to give huge qualitative improvements in model performance, there's a huge advantage to parallelizable architectures that allow for scaling.


Sources:
https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html
https://www.freecodecamp.org/news/what-are-attention-mechanisms-in-deep-learning/
https://spotintelligence.com/2023/01/31/self-attention/
https://www.3blue1brown.com/lessons/attention