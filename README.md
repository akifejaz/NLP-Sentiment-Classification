# NLP-Sentiment-Classification
Sentiment analysis is the process of determining the sentiment or emotion expressed in a piece of text. This project aims to build a sentiment analysis model that can classify text into different categories such as positive, negative, or neutral.

## Requirements
1. Python 3.6 or higher
2. TensorFlow 2.x
3. pandas
4. numpy
5. scikit-learn
6. NLTK
7. matplotlib

## Data
The dataset used in this project is the IMDB Reviews dataset which contains 50,000 movie reviews with labels indicating whether the review is positive or negative.

## Model
The model is a deep learning model that uses a combination of an Embedding layer and LSTM layers. The Embedding layer is used to convert the words in the reviews into a dense vector representation, and the LSTM layers are used to process the sequence of vectors. The final output of the model is a single sigmoid neuron, which is used to classify the review as positive or negative.

## Training
The model is trained for 10 epochs using the Adam optimizer and binary cross-entropy loss. The training set is divided into a training set and a validation set, with a 80-20 split respectively.

## Evaluation
The model is evaluated on the test set, and the accuracy, precision, recall and F1-score are calculated.

## Usage
Clone this repo to your local machine and install the requirements. You can use the train.py script to train the model and the predict.py script to classify new reviews.

## Example usage:
```
python train.py
python predict.py --review "I loved this movie. It was so well-made and engaging."
```

## Conclusion
The model is able to classify movie reviews with an accuracy of around 85%. This model can be improved further by fine-tuning the hyperparameters or by using a different dataset. The model can be used to classify reviews on any topic such as product reviews, news articles and more.
