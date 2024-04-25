# Deep Molecular Binding

This notebook describes my (frist) submission to the [Leash Bio kaggle competition](https://www.kaggle.com/competitions/leash-BELKA/overview). In short, the goal is to predict whether small molecules bind with one of three possible proteins. We are given a rather large dataset (~100 million unique molecules) along with information about how each binds to each of the three proteins of interest.
<br /><br />
# Approach
Molecules (and their building blocks) are given in [SMILES](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) format, a linear text-based description of molecular structure, and so this project seemed like a good candidate for a transformer-based appraoch. We are not given the structure of the three protein binding targets. My basic idea was to use a pre-trained embedding followed by a feedforward network that terminates with a 3-unit layer that would given binding probabilities to the three targets.
<br /><br />
**Embedding Network**: There are a handful of options for embedding SMILES, but ultimately I settled on [MolecularTransformerEmbeddings](https://github.com/mpcrlab/MolecularTransformerEmbeddings). This embedding was trained to translate between SMILES and IAUAC (another text-based molecular description format), and the [publication](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01212) that presented this embedding indicated that it might be an appropriate choice for predicting binding affinity. The embedding space is 512-dimensional, and the longest SMILES in the dataset is 142 characters long. My embedder class (in `embedding.py`) essentially wraps the encoder side of the transformer provided by the authors and produces tensors of shape (n_molecules, 150, 512) to be fed into the feedforward network. Here, *n_molecules* is typically the batch size.
<br /><br />
**Feedforward Network**: The layout of the feedforward network (located in `model.py` is currently as follows:
  * Flatten
  * Dense(76,800 units; ReLU activation)
  * Dense(500 units; ReLU activation)
  * Dropout(20% probability)
  * Dense(250 units; ReLU activation)
  * Dense(3 units; sigmoid activation)

Each of the units in the final layer corresponds to one of the target proteins (BRD4, HSA, sEH).
<br /><br />
**Training**: The first step was getting the training data into pairs of SMILES and 3-vectors (the dataset actually has ~300 million rows, with each row only indicating binding to one protein each). After some testing, I determined that 1,000 was an efficeint batch size. I used Adam optimization with a learning rate of 0.001 to minimize binary cross-entropy loss (details in `training.py`.
<br />&nbsp;&nbsp;&nbsp;&nbsp;It takes about 4 seconds to train a single batch on my machine (with an RTX 4090). A lot of this time is eaten up by the embedding step (which does not undergo training), but I suspect I might be able to speed this up somehow. Unfortunately, I'd have to tie up my computer for >100 hrs to see all of the data in the training set even just once. This isn't really feasible for me, so I took a slightly different approach to generating test and validation batches. At the start, I randomly shuffled the provided training set and set aside 5% for a validation set. Each batch was then randomly sampled (with replacement) from the training set. I set up the `RandomBatchLoader` class (see `data_loader.py`) to handle the training/validation splits, random sampling, and the embedding step. Also note that each batch was balanced--50% of the samples bound to at least one protein, and 50% did not. This was important since the total training dataset is very unbalanced (very few binders).
<br /><br />
## Results
After 15,000 iterations, the loss seemed to level out at ~0.15. Note that is a pretty small fraction of the training data (6.66% at most assuming no repeats were drawn). The validation loss tracked the training loss quite closely, and there was clearly little risk here of overtraining. Even though the network didn't seem to be improving (with respect to loss) after 12,000 or so iterations, it's hard to know a priori what the underlying distributions are in the training set and whether I'd might see gains as the network saw a wider range of training data or got a chance to train on the same data multiple times. As of this writing, my score on the kaggle competition is 0.44. I may or may not continue training this network, depending on the amount of compute I have available.
<br /><br />
## Discussion

There are quite a few potential things to try differently here.
  * The embedding I used might not be the best for the task. It was trained on a much wider range of molecules, which might help my network generalize, but it might also be unnecessarily general given the small-molecule constraints. However, I suspect that a transformer trained to translate between SMILES and a more detailed geometric representation of moleculer structure would probably be more appropriate.
  * There might be a way to speed up the embedding/encoder step which would enable faster training and model iteration. Cacheing the embeddings is really not feasible given the vast number of unique inputs, but there might be other things to try.
  * I could try fine-tuning the encoder on the training set to improve the ability to adjust representations in such a way that is more explicitly designed to predict binding likelihoods for these particular proteins. Training my own embedding from scratch is probably not feasible at the moment.
  * I'd like to try some different things in the feedfoward network as well. A convolutional layer or two might be able to pick out promising structural components of the molecule that are prone to binding. However, it's pretty unclear to me as to whether this would work after flattening the embeddings.
  * I'm using the full molecular SMILES directly and not the three building blocks that make up the molecule (which are supplied in the dataset). I did this because it's really the full molecule itself that matters, but it might be possible to embed the subunits individually for a performance increase since the number of tokens in each subunit is quite a bit smaller than in the full molecule. One might be able to feasibly create an embedding de novo (and with limited compute) with this reduced-size transformer.
  * The unlikelihood of seeing the same data more than once during training might really be hurting me here. There might be a way to reduce the size of the training set without overfitting, but this is hard to know in advance or without more domain-specific knowledge.
  * Non-transformer approaches (convolutional, RNN) might be better. It'll be interesting to see what the leaders publish when the competition is over. However, my goal with this project was to get some practice with transformers, so I'll not be taking vastly different approaches anytime soon.

All in all, it's a pretty interesting project. I'll update this notebook as I try different things.
