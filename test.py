from predictor import predict_sentiment

test_text = """
I like the main theme except that song
in the movie but the experience was nice.
"""

prediction = predict_sentiment(
    test_text
)

print("Review:")
print(test_text)

print(
    "\nPredicted Sentiment:",
    prediction
)