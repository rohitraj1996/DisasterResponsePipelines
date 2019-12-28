# import libraries
import re
import sys
import pickle
import pandas as pd
from sqlalchemy import create_engine

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download(['punkt', 'wordnet','stopwords'])

def load_data(database_filepath):
    """
        - Load cleaned data from database into dataframe
        - Extract X, Y data
        
        Args:
            database_filepath (str): File path of database
            
        Returns:
            X (pandas dataframe): Contains messages
            Y (pandas dataframe): Contains categories of disaster
            category_names (list): Name of categories
    """
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    engine.dispose()
    
    # Extract X and Y
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
    """
        - Tokenize text
        
        Args:
            text (str): Message string
            
        Returns:
            tokens (list): List of tokens
    """
    
    # Convert to lower case
    text = text.lower()
    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    # Lemmatize
    tokens = [WordNetLemmatizer().lemmatize(word).strip() for word in tokens]
    
    return tokens


def build_model():
    """
        - Build pipeline for creating model
        
        Returns:
            cv (GridSearchCV): Machine learning model
    """
    
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])
    
    # Parameter selection using GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__learning_rate': [0.01, 0.1, 1]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        - Evaluate model
        
        Args:
            model (Predicter model): Machine learning model (Predicter)
            X_test (pandas series): Test data set of X
            Y_test (pandas dataframe): Test data set of Y
            category_names (list): Name of categories
    """
    
    # Predict test data
    predict_y = model.predict(X_test)
    
    # Evaluate
    for i in range(len(category_names)):
        category = category_names[i]
        print(category)
        print(classification_report(Y_test[category], predict_y[:, i]))


def save_model(model, model_filepath):
    """
        - Saves model in pickle file
        
        Args:
            model (Predicter model): Machine learning model (Predicter)
            model_filepath (str): Path where model will save.
    """
    
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()