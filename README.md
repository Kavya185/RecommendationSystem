Recommendation System - Two Tower Model
=
Dataset : Movielens 100k dataset.

Setup and Imports: TensorFlow Recommenders and TensorFlow Datasets are installed and imported. Essential libraries like TensorFlow and NumPy are also imported.

Data Loading and Preparation: The Movielens 100k dataset is loaded, focusing on ratings and movie titles. Features such as movie_title and user_id are extracted from the datasets.

Vocabulary Building: Vocabularies are created to convert user IDs and movie titles into integer indices for embedding layers.

Model Definition: A custom model (TwoTowerModel) is defined using TensorFlow Recommenders. It includes separate models for users and movies, and a retrieval task to compute loss.

Model Training: The model is compiled with Adagrad optimizer and trained for three epochs on the ratings dataset.

Retrieval Setup: Brute-force search is used to set up retrieval, indexing the movie dataset with the trained user model.

Recommendation Generation: Top movie recommendations are generated for a specific user ("42") using the trained model. This process outlines the steps involved in building and training a recommendation system using TensorFlow Recommenders with the Movielens 100k dataset, focusing on data preparation, model setup, training, and recommendation generation.

Recommendation System - wide & deep model
=
Dataset : Movielens 100k dataset.

Setup and Imports: TensorFlow Recommenders and TensorFlow Datasets are installed and imported. Necessary libraries such as TensorFlow and NumPy are imported for data manipulation and modeling.

Data Loading and Preparation: The Movielens 100k dataset is loaded from TensorFlow Datasets, focusing on ratings and movie titles. Features such as movie_title and user_id are extracted from the datasets to capture user-movie interactions.

Vocabulary Building: StringLookup layers are used to create vocabularies that map user IDs and movie titles to integer indices. These vocabularies are adapted using the ratings and movies datasets to prepare them for embedding layers.

Model Definition: A custom model class MovieLensModel is defined, inheriting from tfrs.Model. It takes three main components: user_model: Sequential model for users, including a StringLookup layer, an Embedding layer, and a Dense layer for embedding transformation. movie_model: Sequential model for movies, similarly structured with StringLookup, Embedding, and Dense layers. task: Retrieval task defined using tfrs.tasks.Retrieval, which computes the loss based on user and movie embeddings and evaluates metrics like FactorizedTopK.

Model Compilation and Training: The model is compiled with the Adagrad optimizer (tf.keras.optimizers.Adagrad) with a learning rate of 0.5. It is trained for three epochs (epochs=3) using the ratings dataset batched into sizes of 4096.

Retrieval Setup: After training, a brute-force search setup (tfrs.layers.factorized_top_k.BruteForce) is established using the user model. The movie dataset is indexed with the movie model to set up efficient retrieval of movie recommendations.

Recommendation Generation: Using the trained retrieval setup, recommendations are generated for user "42". The top 5 movie titles recommended for user "42" are printed to demonstrate the model's ability to provide personalized recommendations based on learned user preferences.

Recommendation System - LightFM model
=
Dataset : Movielens 100k dataset.

Prepare Interactions Matrix:
Converts the ratings data into a format suitable for LightFM. It uses TensorFlow Datasets to load movie ratings and maps them to user IDs, item IDs (movie titles), and ratings.

Build LightFM Dataset:
Converts the interactions (user IDs, item IDs, ratings) into a format compatible with LightFM. It builds the dataset required for training the model.

Model Initialization:
Initializes the LightFM model with specific parameters (no_components=64 and loss='warp'). Here, no_components defines the dimensionality of the user and item embeddings, and loss='warp' specifies the type of loss function used for training (Weighted Approximate-Rank Pairwise loss).

Fit the Model:
Trains the LightFM model on the interactions matrix for 10 epochs (epochs=10). This step optimizes the model parameters (user and item embeddings) to predict ratings or preferences accurately.

Predictions and Recommendations:
Once the model is trained, it predicts scores for all items for a given user (user_index). These scores represent the model's estimated preferences for each item.
It then identifies the top 3 items (movies) with the highest predicted scores (np.argsort(-scores)[:3]), which are recommended to the user.
The movie titles of these top recommendations are retrieved using the vocabulary from movie_titles_vocabulary.

Output:
Finally, it prints the top 3 recommended movie titles for the user.
