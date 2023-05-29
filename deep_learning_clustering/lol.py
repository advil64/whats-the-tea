import torch
import torch.nn as nn
import spacy

# Load the spacy model for tokenization
nlp = spacy.load("en_core_web_lg")

# Define the tweet topic classifier model
class TweetTopicClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TweetTopicClassifier, self).__init__()
        self.conv1 = nn.Conv1d(300, 64, kernel_size=3)  # Assuming word embeddings are of size 300
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 78, num_classes)  # Adjusted the input size of the fully connected layer

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Adjust tensor shape for convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the trained model
model = TweetTopicClassifier(num_classes=42)
# model.load_state_dict(torch.load("path_to_trained_model_weights.pth"))
model.eval()

# Tokenize and preprocess the tweets
def preprocess_tweet(tweet):
    tokens = nlp(tweet.lower())
    # Perform additional preprocessing if needed
    return tokens

def tokenize_tweets(tweets):
    tokenized_tweets = []
    for tweet in tweets:
        tokens = preprocess_tweet(tweet)
        tokenized_tweets.append(tokens)
    return tokenized_tweets

# Convert tokenized tweets to input tensors
def get_input_tensors(tokenized_tweets):
    tensors = []
    for tokens in tokenized_tweets:
        # Assuming word embeddings are available for each token
        # and represented as a 300-dimensional vector
        token_tensors = torch.tensor([token.vector for token in tokens])
        tensors.append(token_tensors)
    return tensors

# Classify the topics of the tweets
def classify_tweets(tweets):
    tokenized_tweets = tokenize_tweets(tweets)
    input_tensors = get_input_tensors(tokenized_tweets)

    predictions = []
    with torch.no_grad():
        for tensor in input_tensors:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            output = model(tensor)
            _, predicted_class = torch.max(output, 1)
            predictions.append(predicted_class.item())

    return predictions

# Example usage
tweets = [
    "I love going to the beach on weekends",
    "Just watched an amazing movie",
    "Working on a new project",
    "Excited about the upcoming game"
]

predictions = classify_tweets(tweets)
print(predictions)
