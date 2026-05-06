# Deep Learning Series: RNN, LSTM, Attention, BERT, VAE, GAN, GCN

This repository contains implementations for 5 deep learning labs covering sequence models, attention mechanisms, transformer fine-tuning, generative models, and graph neural networks. All code is implemented in Python using TensorFlow/Keras and PyTorch.


## 📁 Files in This Repository

 File Name | Description |
-----------|-------------|
 `RNN vs LSTM vs GRU.ipynb` | Compare RNN, LSTM, GRU on IMDB sentiment classification |
 `LSTM Text Classification+Seq2Seq.ipynb` | LSTM text classification + Basic Seq2Seq model |
 `Attention Heatmaps in Seq2Seq.ipynb` | Seq2Seq with Attention + Attention heatmap visualization |
 `Fine-tuning BERT.ipynb` | Fine-tune DistilBERT for sentiment analysis + QA pipeline |
 `VAE + GAN + GCN.ipynb` | VAE latent space visualization, GAN training, GCN on Cora |

---

## 🧪 Lab 4: RNN vs LSTM vs GRU Performance (`RNN vs LSTM vs GRU.ipynb`)

### What This Code Does
- Loads IMDB dataset (5000 vocabulary, 100 sequence length)
- Builds three models: SimpleRNN, LSTM, GRU with Embedding layer
- Trains each model for 3 epochs
- Compares accuracy and training time

### Key Results (from your output)
| Model | Test Accuracy | Training Time |
|-------|---------------|----------------|
| RNN   | 80.20%        | 69.18 seconds  |
| LSTM  | 85.01%        | 182.74 seconds |
| GRU   | 85.68%        | 217.02 seconds |



## Lab 5: LSTM for Text Classification + Seq2Seq (LSTM Text Classification+Seq2Seq.ipynb)
Part A: LSTM Text Classification (IMDB)
Code features:

Loads IMDB with num_words=5000, max_len=100
Model: Embedding(5000,128) → LSTM(64) → Dense(1, sigmoid)
5 epochs training
Final test accuracy: 84.34%

Part B: Seq2Seq Model (Basic Example)
Code features:

Random dummy data (1000 samples, length 10)
Encoder LSTM returns state
Decoder LSTM uses encoder states as initial state
Dense output layer with linear activation
Trains for 5 epochs

Note: This is a basic demonstration; for real translation, use actual sentence pairs.



## Lab 6: Attention Heatmaps in Seq2Seq (Attention Heatmaps in Seq2Seq.ipynb)
What This Code Does
Implements English-to-French translation with attention

Uses custom vocabulary (4 example sentences)
GRU-based Encoder
Bahdanau (additive) Attention mechanism
Decoder attends to encoder outputs at each step
Generates attention heatmap using seaborn

Example Translation
text
Input: "i am happy"
Output: ['je', 'suis', 'content']  ✓
Attention Heatmap Generated
X-axis: Input words (i, am, happy)
Y-axis: Output words (je, suis, content)
Bright cells show where model focuses

Key Classes
Class	Purpose
Encoder	GRU encoder, returns outputs and hidden state
Attention	Computes attention weights using tanh(linear)
Decoder	GRU decoder with attention context vector
Training Output (from your notebook)
text
Epoch 0, Loss 36.43
Epoch 50, Loss 0.07
Epoch 100, Loss 0.02
Epoch 150, Loss 0.01



## Lab 7: Fine-tuning BERT for QA / Sentiment (Fine-tuning BERT.ipynb)
Part 1: Sentiment Analysis (DistilBERT)
What the code does:

Loads IMDB dataset (500 train samples, 200 test samples for speed)
Tokenizer: distilbert-base-uncased
Model: distilbert-base-uncased with sequence classification head (2 labels)
Training: 1 epoch, batch size 16, FP16 if GPU available
Final prediction on "This movie is awesome!" → Returns 1 (Positive)

Part 2: Question Answering Pipeline
What the code does:

Loads pre-trained QA pipeline: distilbert-base-cased-distilled-squad
Context: "KIET is located in Ghaziabad and it offers BTech programs."
Question: "Where is KIET located?"
Output: "Ghaziabad"
Code Highlights
python

# Sentiment training
trainer = Trainer(model=model, args=training_args, 
                  train_dataset=train_data, eval_dataset=test_data)
trainer.train()

# QA pipeline
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
result = qa(question=question, context=context)




## Lab 8: VAE + GAN + GCN (VAE + GAN + GCN.ipynb)
8.1 VAE (Variational Autoencoder) on MNIST
Architecture:

Encoder: 784 → 400 → 2 (mu and logvar)
Latent dimension: 2 (for visualization)
Decoder: 2 → 400 → 784 (sigmoid output)
Loss Function: BCE + KL Divergence

python
BCE = binary_cross_entropy(recon_x, x, reduction='sum')
KLD = -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp())

Output:
text
Epoch 0, Loss 16481.83
Epoch 1, Loss 16039.74
Epoch 2, Loss 14847.66
Latent Space Plot: 2D scatter plot colored by digit class (0-9)


8.2 GAN (Generative Adversarial Network) on MNIST
Generator:

Input: 100-dim noise
Layers: Linear(100,256) → ReLU → Linear(256,784) → Tanh
Discriminator:
Input: 784-dim image
Layers: Linear(784,256) → LeakyReLU(0.2) → Linear(256,1) → Sigmoid
Training Loop (3 epochs):
text
Epoch 0 Loss D: 1.05, Loss G: 0.81
Epoch 1 Loss D: 0.90, Loss G: 0.99
Epoch 2 Loss D: 0.80, Loss G: 1.50


8.3 GCN (Graph Convolutional Network) on Cora Dataset
Dataset:

2708 scientific papers (nodes)
5429 citations (edges)
7 classes (Machine Learning, NLP, CV, etc.)
Model Architecture:
text
GCNConv(1433 → 16) → ReLU → Dropout → GCNConv(16 → 7) → LogSoftmax
Training: 50 epochs, lr=0.01
Final Test Accuracy: 76.10%



📊 Expected Outputs Summary
Lab	File Name	Model	Accuracy/Task	Output
4	RNN vs LSTM vs GRU.ipynb	RNN	80.20%	Training logs + comparison
4	RNN vs LSTM vs GRU.ipynb	LSTM	85.01%	Training logs + comparison
4	RNN vs LSTM vs GRU.ipynb	GRU	85.68%	Training logs + comparison
5A	LSTM Text Classification+Seq2Seq.ipynb	LSTM	84.34%	Final accuracy
5B	LSTM Text Classification+Seq2Seq.ipynb	Seq2Seq	MSE loss	Training logs
6	Attention Heatmaps in Seq2Seq.ipynb	Attention Seq2Seq	Translation	Heatmap image
7	Fine-tuning BERT.ipynb	DistilBERT	Sentiment	1 (Positive)
7	Fine-tuning BERT.ipynb	QA Pipeline	QA	"Ghaziabad"
8A	VAE + GAN + GCN.ipynb	VAE	2D latent space	Scatter plot
8B	VAE + GAN + GCN.ipynb	GAN	Image generation	Loss values
8C	VAE + GAN + GCN.ipynb	GCN	76.10%	Node classification accuracy
