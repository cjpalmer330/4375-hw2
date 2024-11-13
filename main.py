import torch


sample_data = [
    (["this", "is", "good"], 4),
    (["this", "is", "bad"], 0),
    (["amazing", "service"], 4),
    (["poor", "experience"], 0)
]

# Assuming `RNN` class and `make_vocab` function are available in the same scope
vocab = make_vocab(sample_data)
_, word2index, _ = make_indices(vocab)

# Load or mock word embeddings (for testing, just set random ones)
word_embedding = {word: torch.randn(50) for word in word2index}
word_embedding[unk] = torch.randn(50)  # for unknown tokens

# Convert sample data to vector representation
test_data = []
for words, label in sample_data:
    vectors = [word_embedding.get(word, word_embedding[unk]) for word in words]
    vectors = torch.stack(vectors).view(len(vectors), 1, -1)
    test_data.append((vectors, torch.tensor([label])))

# Initialize the model
model = RNN(input_dim=50, h=10)  # adjust hidden dim `h` as needed

# Run a forward pass for each sample in the test data
model.eval()  # set to evaluation mode
for vectors, label in test_data:
    with torch.no_grad():  # disable gradients for testing
        output = model(vectors)
        predicted_label = torch.argmax(output, dim=1)
        print(f"Predicted: {predicted_label.item()}, True: {label.item()}")
