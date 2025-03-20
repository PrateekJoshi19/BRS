import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import uuid
import os
import ast
import warnings

warnings.filterwarnings('ignore')

# Load the books dataset
@st.cache_data
def load_data():
    return pd.read_csv('BooksDatasett.csv')

books_df = load_data()

# First Layer: User Segmentation Layer
class UserSegmentLayer:
    def __init__(self):
        self.questions = [
            "How often do you read books? (1-5): ",
            "Rate your interest in fiction books (1-5): ",
            "Rate your interest in non-fiction books (1-5): ",
            "Rate your interest in mystery books (1-5): ",
            "Rate your interest in science fiction books (1-5): ",
            "Rate your interest in fantasy books (1-5): ",
            "How important is the book's length to you? (1-5): ",
            "How much do you care about book ratings? (1-5): ",
            "Do you prefer books in English? (1-5): ",
            "How important is the author's reputation to you? (1-5): ",
            "Do you prefer recent books or classics? (1-5, 1 for classics, 5 for recent): ",
            "How much do you care about the book's publication year? (1-5): ",
            "Rate your interest in foreign language books (1-5): ",
            "How important are book reviews to you? (1-5): ",
            "Do you prefer reading physical books or e-books? (1-5, 1 for e-books, 5 for physical books): "
        ]
        self.scaler = StandardScaler()

    def get_user_answers(self):
        answers = []
        for question in self.questions:
            answer = st.slider(question, 1, 5)
            answers.append(answer)
        return answers

    def segment_user(self, answers):
        avg_score = sum(answers) / len(answers)
        if avg_score < 2:
            return 0
        elif avg_score < 3:
            return 1
        elif avg_score < 4:
            return 2
        else:
            return 3

    def generate_user_id(self):
        return str(uuid.uuid4())

    def store_user_data(self, user_id, answers, segment):
        user_data = {
            'user_id': user_id,
            'answers': answers,
            'segment': int(segment)
        }
        os.makedirs('user_data', exist_ok=True)
        with open(f'user_data/{user_id}.pkl', 'wb') as f:
            pickle.dump(user_data, f)

    def process_user(self):
        answers = self.get_user_answers()
        segment = self.segment_user(answers)
        user_id = self.generate_user_id()
        self.store_user_data(user_id, answers, segment)
        return user_id

# Second Layer: Recommendation Layer
class RecommendationLayer:
    def __init__(self, books_df):
        self.books_df = books_df
        self.prepare_data()
        self.model = RandomForestRegressor()
        self.train_model()

    def prepare_data(self):
        # Parse genres from the 'Category' column
        def parse_genres(category_str):
            try:
                genres = ast.literal_eval(category_str)
                if isinstance(genres, list) and all(isinstance(g, str) for g in genres):
                    return genres
                elif isinstance(genres, str):
                    return [genres]
                else:
                    return []
            except:
                return []

        self.books_df['genres'] = self.books_df['Category'].fillna('[]').apply(parse_genres)

        self.genre_list = ['Fiction', 'Non-Fiction', 'Mystery', 'Science Fiction', 'Fantasy']
        for genre in self.genre_list:
            self.books_df[f'genre_{genre}'] = self.books_df['genres'].apply(lambda x: 1 if genre in x else 0)

        # Convert publication year to numeric
        self.books_df['publication_year'] = pd.to_numeric(self.books_df['Publish Date (Year)'], errors='coerce')

        # Normalize numerical features
        numerical_features = ['Price Starting With ($)', 'publication_year']
        for feature in numerical_features:
            self.books_df[feature] = (self.books_df[feature] - self.books_df[feature].min()) / (self.books_df[feature].max() - self.books_df[feature].min())

        self.features = [col for col in self.books_df.columns if col.startswith('genre_')] + ['Price Starting With ($)', 'publication_year']

        self.X = self.books_df[self.features].fillna(0)
        self.y = self.books_df['Price Starting With ($)']

    def train_model(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        y_train_pred = self.model.predict(self.X_train)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)

        y_test_pred = self.model.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)

        st.write(f"Training MSE: {train_mse:.4f}")
        st.write(f"Training R^2 score: {train_r2:.4f}")
        st.write(f"Testing MSE: {test_mse:.4f}")
        st.write(f"Testing R^2 score: {test_r2:.4f}")

    def get_user_segment(self, user_id):
        with open(f'user_data/{user_id}.pkl', 'rb') as f:
            user_data = pickle.load(f)
        return user_data['segment'], user_data['answers']

    def recommend_books(self, user_id, top_n=10):
        _, user_answers = self.get_user_segment(user_id)

        user_profile = {
            'read_frequency': user_answers[0],
            'genre_preferences': user_answers[1:6],
            'length_importance': user_answers[6],
            'rating_importance': user_answers[7],
            'english_preference': user_answers[8],
            'author_importance': user_answers[9],
            'recency_preference': user_answers[10],
            'publication_year_importance': user_answers[11],
            'foreign_preference': user_answers[12],
            'reviews_importance': user_answers[13],
            'format_preference': user_answers[14]
        }

        self.books_df['user_score'] = 0

        for i, genre in enumerate(self.genre_list):
            self.books_df['user_score'] += self.books_df[f'genre_{genre}'] * user_profile['genre_preferences'][i]

        self.books_df['user_score'] += self.books_df['Price Starting With ($)'] * user_profile['length_importance']
        self.books_df['user_score'] += self.books_df['publication_year'] * user_profile['recency_preference']

        self.books_df['user_score'] = (self.books_df['user_score'] - self.books_df[
            'user_score'].min()) / (self.books_df['user_score'].max() - self.books_df['user_score'].min())

        self.books_df['final_score'] = 0.7 * self.books_df['user_score'] + 0.3 * self.books_df['Price Starting With ($)']

        recommended_books = self.books_df.sort_values('final_score', ascending=False).head(top_n)
        return recommended_books[['Title', 'final_score', 'genres', 'Price Starting With ($)']]

# Streamlit app
def main():
    st.title("Book Recommendation System")

    # Display dataset overview
    st.header("Dataset Overview")
    st.subheader("First few rows of the dataset:")
    st.dataframe(books_df.head())

    st.subheader("Last rows of the dataset:")
    st.dataframe(books_df.tail())

    st.subheader("Columns in the dataset:")
    st.write(books_df.columns.tolist())

    # Instantiate user segmentation and recommendation layers
    user_segment_layer = UserSegmentLayer()
    recommendation_layer = RecommendationLayer(books_df)

    st.header("User Preferences")
    user_id = user_segment_layer.process_user()

    if st.button("Get Recommendations"):
        st.write(f"User ID: {user_id}")
        recommendations = recommendation_layer.recommend_books(user_id)
        st.header("Top 10 Recommended Books:")
        st.dataframe(recommendations)

    # Option to add another user
    if st.button("Add Another User"):
        st.rerun()

    st.write("Thank you for using the book recommendation system!")

if __name__ == "__main__":
    main()