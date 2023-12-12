import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your social media data (replace 'Tweets.csv' with your actual file)
df = pd.read_csv('Tweets.csv')

# Replace NaN values in the 'tweet_id' column with an empty string or any suitable placeholder
df['tweet_id'].fillna('', inplace=True)

# Assuming you have a 'tweets_id' column in your dataset
X = df['tweet_id']
y_str = df['airline_sentiment']  # Replace 'airline_sentiment' with the actual column containing sentiment labels

# Convert string labels to numerical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores and convert them to labels
def get_sentiment_label(text):
    sentiment_score = sia.polarity_scores(str(text))['compound']
    return 1 if sentiment_score >= 0 else 0  # Assuming 1 for positive and 0 for negative

# Apply the sentiment analysis function to the training and testing sets
y_train_pred = X_train.apply(get_sentiment_label)
y_test_pred = X_test.apply(get_sentiment_label)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
