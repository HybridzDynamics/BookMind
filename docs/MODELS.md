# Model Documentation

## 1. Popularity model

### Purpose
The popularity model provides a baseline recommendation strategy when user history is missing or weak. It is useful for cold-start situations and is easy to explain in a viva.

### Inputs
- cleaned book catalog
- average rating per book
- total number of ratings per book

### Processing
The model calculates a Bayesian weighted score using the average rating and the global mean rating. It then scales the result by the logarithm of the rating count so highly reviewed books are rewarded without ignoring quality.

### Output
A sorted list of books ranked by `popularity_score`.

### Use case
Good for new users or users with minimal interaction history.

### Limitation
It does not capture individual preferences; it only reflects broad catalog popularity.

---

## 2. Content-based model

### Purpose
This model recommends books similar to a selected book using metadata and textual features.

### Inputs
- title
- author
- genres
- description
- language code
- combined content text

### Processing
The project builds a TF-IDF representation of each book and uses cosine-based nearest-neighbor search to find similar books. A diversity limit of two books per author is enforced to avoid repetitive results.

### Output
Books with a similarity score and a reason explaining the relationship to the selected book.

### Use case
Useful when the user selects a book they already like and wants more similar titles.

### Limitation
It depends heavily on metadata quality and is less personalized than collaborative filtering.

---

## 3. Collaborative filtering model

### Purpose
This model uses user-item interaction patterns to learn latent preferences.

### Inputs
- user_id
- book_id
- rating

### Processing
The ratings matrix is transformed into a sparse matrix and decomposed with `TruncatedSVD`. The model compares a user vector with book vectors to identify books the user may enjoy, while excluding books already rated by that user.

### Output
A ranked list with `predicted_score` and a reason referencing similar readers.

### Use case
This is the main personalized recommendation signal when enough interaction history is available.

### Limitation
For new users or users with insufficient ratings, the model falls back to popularity instead of pretending there is strong collaborative signal.

---

## 4. Hybrid model

### Purpose
The hybrid model combines multiple recommendation signals to provide the final ranking.

### Inputs
- collaborative score
- content score
- readability match score
- popularity score

### Processing
The model normalizes each component to a comparable scale and combines them with the project weights:

- Collaborative: 35%
- Content: 30%
- Readability match: 15%
- Popularity: 20%

The readability score compares each book's difficulty estimate (derived from the share of 1-2 star ratings in the catalog) against the user's preferred reading difficulty, computed from the books they rated 4+ stars.

### Output
A final hybrid score and final ranked list with component breakdowns.

### Use case
This is the strongest general recommendation strategy when the user has enough history to support collaborative behavior.

### Limitation
The ensemble still depends on the quality of the underlying base models and the data available for each user.
