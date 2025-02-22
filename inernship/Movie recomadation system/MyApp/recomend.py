import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from math import sqrt

class MovieRecommender:
    def __init__(self):
        # Load and preprocess data
        self.movies = pd.read_csv('movies.csv')
        self.ratings = pd.read_csv('ratings.csv')
        
        # Create user-item matrix
        self.user_item_matrix = self.ratings.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Calculate user similarity matrix using manual implementation
        self.user_similarity = self._calculate_similarity()
        
    def _calculate_similarity(self):
        """Calculate cosine similarity between users"""
        # Normalize ratings
        normalized_matrix = self.user_item_matrix.sub(
            self.user_item_matrix.mean(axis=1), axis=0
        )
        
        # Calculate similarity
        sim = np.dot(normalized_matrix, normalized_matrix.T)
        norms = np.array([np.sqrt(np.diagonal(sim))])
        return sim / norms.T / norms
    
    def get_recommendations(self, user_id, n_recommendations=20):
        if user_id not in self.user_item_matrix.index:
            return []
            
        # Get similar users
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similar_users = self.user_similarity[user_idx]
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unwatched_movies = user_ratings[user_ratings == 0].index
        
        # Calculate predicted ratings
        predictions = []
        for movie_id in unwatched_movies:
            movie_ratings = self.user_item_matrix[movie_id]
            # Calculate weighted average rating
            weighted_sum = np.sum(movie_ratings * similar_users)
            similarity_sum = np.sum(np.abs(similar_users))
            
            if similarity_sum != 0:
                pred_rating = weighted_sum / similarity_sum
                predictions.append((movie_id, pred_rating))
        
        # Sort and get top recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_movies = predictions[:n_recommendations]
        
        # Get movie details
        results = []
        for movie_id, pred_rating in recommended_movies:
            movie_info = self.movies[self.movies['movieId'] == movie_id].iloc[0]
            results.append({
                'title': movie_info['title'],
                'predicted_rating': round(pred_rating, 2)
            })
            
        return results
        
    def evaluate_recommendations(self, test_users=10):
        # Split ratings into train and test
        test_ratings = []
        for user_id in self.ratings['userId'].unique()[:test_users]:
            user_ratings = self.ratings[self.ratings['userId'] == user_id]
            test_ratings.extend(user_ratings.sample(min(5, len(user_ratings))).values)
        
        test_ratings = pd.DataFrame(test_ratings, columns=self.ratings.columns)
        
        # Calculate RMSE
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            # Get similar users
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            similar_users = self.user_similarity[user_idx]
            
            # Predict rating
            movie_ratings = self.user_item_matrix[movie_id]
            weighted_sum = np.sum(movie_ratings * similar_users)
            similarity_sum = np.sum(np.abs(similar_users))
            
            if similarity_sum != 0:
                pred_rating = weighted_sum / similarity_sum
                predictions.append(pred_rating)
                actuals.append(actual_rating)
            
        rmse = sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        return rmse

class RecommenderUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommender System")
        self.root.geometry("800x600")
        
        # Create loading label
        self.loading_label = ttk.Label(
            self.root,
            text="Loading recommender system...",
            font=('Arial', 12)
        )
        self.loading_label.pack(pady=20)
        
        # Schedule the loading of the recommender system
        self.root.after(100, self.load_recommender)
        
    def load_recommender(self):
        try:
            self.recommender = MovieRecommender()
            self.loading_label.destroy()
            self.create_widgets()
        except FileNotFoundError:
            self.loading_label.config(
                text="Error: Please ensure 'movies.csv' and 'ratings.csv' are in the current directory",
                foreground="red"
            )
        except Exception as e:
            self.loading_label.config(
                text=f"Error: {str(e)}",
                foreground="red"
            )
        
    def create_widgets(self):
        # User input frame
        input_frame = ttk.LabelFrame(self.root, text="Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(input_frame, text="User ID:").pack(side="left")
        self.user_id_entry = ttk.Entry(input_frame)
        self.user_id_entry.pack(side="left", padx=5)
        
        ttk.Button(
            input_frame,
            text="Get Recommendations",
            command=self.show_recommendations
        ).pack(side="left", padx=5)
        
        ttk.Button(
            input_frame,
            text="Evaluate System",
            command=self.evaluate_system
        ).pack(side="left", padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Recommendations", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create treeview for recommendations
        self.tree = ttk.Treeview(
            results_frame,
            columns=("Title", "Predicted Rating"),
            show="headings"
        )
        
        self.tree.heading("Title", text="Movie Title")
        self.tree.heading("Predicted Rating", text="Predicted Rating")
        
        self.tree.column("Title", width=400)
        self.tree.column("Predicted Rating", width=100)
        
        self.tree.pack(fill="both", expand=True)
        
    def show_recommendations(self):
        try:
            user_id = int(self.user_id_entry.get())
            recommendations = self.recommender.get_recommendations(user_id)
            
            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Show new recommendations
            for rec in recommendations:
                self.tree.insert(
                    "",
                    "end",
                    values=(rec['title'], rec['predicted_rating'])
                )
                
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid user ID")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def evaluate_system(self):
        rmse = self.recommender.evaluate_recommendations()
        messagebox.showinfo(
            "Evaluation Results",
            f"Root Mean Square Error (RMSE): {rmse:.3f}"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderUI(root)
    root.mainloop()