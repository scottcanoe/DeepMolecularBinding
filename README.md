# Deep Molecular Binding

This notebook describes my (current) submission to the [Leash Bio kaggle competition](https://www.kaggle.com/competitions/leash-BELKA/overview). In short, the goal is to predict whether small molecules bind with one of three possible proteins. We are given a rather large dataset (~100 million unique molecules) along with information about how each binds to each of the three proteins of interest. I wanted to try a transformer-based approach, and this repo contains my attempt so far.
<br /><br />
# Approach
Molecules (and their building blocks) are given in [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) format, a linear text-based description of molecular structure, and so this project seemed like a good candidate for a transformer-based appraoch. We are not given the structure of the three protein binding targets. My basic idea was to use a pre-trained embedding followed by a feedforward network that terminates with a 3-unit layer that would given binding probabilities to the three targets.
<br /><br />
**Embeddings**: There are a handful of options for embedding SMILES, but for this attempt I used [MoLFormer-XL](https://huggingface.co/ibm/MoLFormer-XL-both-10pct). According to its documentation, "MoLFormer-XL leverages masked language modeling and employs a linear attention Transformer combined with rotary embeddings." It's fast, and it's available on HuggingFace. It produces better results than my previous attempt which used [MolecularTransformerEmbeddings](https://github.com/mpcrlab/MolecularTransformerEmbeddings). The embedding space is 768-dimensional. I've written a class (`RandomBatchLoader`) that wraps the tokenizer and transformer to deliver the pooled output of the transformer to train a feedforward network. This is far more efficient than my previous attempt using MolecularTransformerEmbeddings which used the non-pooled embedding to train the feedforward model (after flattening).
<br /><br />
**Feedforward Network**: The layout of the feedforward network (located in `model.py` is currently as follows:
  * Flatten
  * Dense(768 units; ReLU activation)
  * Dense(300 units; ReLU activation)
  * Dropout(10% probability)
  * Dense(3 units; sigmoid activation)

Each of the units in the final layer corresponds to one of the target proteins (BRD4, HSA, sEH).
<br /><br />
**Training**: The first step was getting the training data into pairs of SMILES and 3-vectors (the dataset actually has ~300 million rows, with each row only indicating binding to one protein each). After some testing, I determined that 1,000 was an efficient batch size. I used Adam optimization with a learning rate of 0.001 to minimize binary cross-entropy loss.
<br />&nbsp;&nbsp;&nbsp;&nbsp;It takes about 1 second to train a single batch on my machine (with an RTX 4090).Unfortunately, I'd have to tie up my computer for >100 hrs to see all of the data in the training set once. This isn't really feasible for me, so I took a slightly different approach to generating test and validation batches. At the start, I randomly shuffled the provided training set and set aside 5% for a validation set. Each batch was then randomly sampled (with replacement) from the training set. I set up the `RandomBatchLoader` class (see `model.py`) to handle the training/validation splits, random sampling, and the embedding step. Also note that each batch was balanced--50% of the samples bound to at least one protein, and 50% did not. This was important since the total training dataset is very unbalanced (very few binders).
<br /><br />
## Results
After 15,000 iterations, the loss seemed to level out at ~0.15. Note that is a pretty small fraction of the training data (6.66% at most assuming no repeats were drawn). The validation loss tracked the training loss quite closely, and there was clearly little risk here of overtraining. Even though the network didn't seem to be improving (with respect to loss) after 12,000 or so iterations, it's hard to know a priori what the underlying distributions are in the training set and whether I'd might see gains as the network saw a wider range of training data or got a chance to train on the same data multiple times. As of this writing, my score on the kaggle competition is 0.45. I may or may not continue training this network, depending on the amount of compute I have available.
<br /><br />
## Discussion

There are quite a few potential things to try differently here.
  * The biggest and most obvious drawback of this approach is that it does not explicitly draw on the chemistry of the molecules or proteins. Other entries into this competition seem to be making good use of molecular fingerprinting schemes and/or graph-based approaches.
  * The transformer was trained on a much wider range of molecules, which might help my network generalize, but it might also be unnecessarily general given the small-molecule constraints. However, I suspect that this might be less of a drawback than the fact that this is an entirely SMILES-based approach (no molecular fingerprints involved).
  * I'm using the full molecular SMILES directly and not the three building blocks that make up the molecule (which are supplied in the dataset). I did this because it's really the full molecule itself that matters, but it might be possible to embed the subunits individually for a performance increase since the number of tokens in each subunit is quite a bit smaller than in the full molecule. One might feasibly create an embedding de novo trained on the smaller building blocks, and then use this transformer to embed the blocks individually.
  * The unlikelihood of seeing the same data more than once during training might really be hurting me here. There might be a way to reduce the size of the training set without overfitting, but this is hard to know in advance or without more domain-specific knowledge (I'm not a molecular biologist).
  * Non-transformer approaches (convolutional, RNN) might be better. It'll be interesting to see what the leaders publish when the competition is over. However, my goal with this project was to get some practice with transformers, so I'll not be taking vastly different approaches anytime soon.


