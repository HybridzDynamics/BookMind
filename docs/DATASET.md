# Dataset Documentation

## Source

BookMind uses the Goodreads-style Goodbooks-10k dataset, which is a well-known public recommendation dataset for academic and prototype recommender systems. The project keeps the original dataset metadata and attribution files in the [../dataset](../dataset) directory, while the processed and project-specific cleaned versions live under [../data](../data).

## Important files

- [../dataset/books.csv](../dataset/books.csv): original catalog metadata
- [../dataset/ratings.csv](../dataset/ratings.csv): raw user-item interactions
- [../dataset/book_tags.csv](../dataset/book_tags.csv): tag assignments per book
- [../dataset/tags.csv](../dataset/tags.csv): tag name lookup table
- [../data/raw/books.csv](../data/raw/books.csv): project input copy used by processing
- [../data/processed/books_clean.csv](../data/processed/books_clean.csv): cleaned catalog used by models
- [../data/processed/user_ratings.csv](../data/processed/user_ratings.csv): filtered and deduplicated ratings
- [../data/processed/evaluation_results.csv](../data/processed/evaluation_results.csv): offline evaluation metrics

## Columns used

The main BookMind pipeline relies on:

- `book_id` / `bookId`: book identifier
- `title`: title text
- `authors` / `author`: author names
- `average_rating` / `rating`: average book rating
- `ratings_count` / `numRatings`: number of ratings received
- `user_id`: user identifier
- `rating`: user rating on a 1–5 scale
- `genres`: cleaned genre labels from tag metadata
- `description`: available descriptive text used in content features

## Preprocessing applied

The project performs the following cleaning steps before model training:

1. Normalizes column names and text whitespace.
2. Converts IDs to strings consistently.
3. Filters invalid or missing values.
4. Keeps only ratings within the valid 1–5 range.
5. Removes duplicate user-book interactions by averaging multiple ratings.
6. Uses tag metadata to build a compact genre description for TF-IDF content modeling.
7. Builds a composite `content` field from title, author, genres, description, and language metadata.

## Licensing and attribution

The dataset source is distributed under the dataset license included in [../dataset/LICENSE](../dataset/LICENSE), which is a Creative Commons Attribution-ShareAlike 4.0 license. The project itself is distributed under the MIT license in [../LICENSE](../LICENSE).

## Limitations

- The dataset is a public benchmark and not a live recommendation service.
- Ratings are sparse, and many books/users are cold-start cases.
- Some metadata can be noisy or incomplete, especially description text and author formatting.
- Content-based similarity is limited by the quality of the generated metadata features.

## Data provenance

The original dataset is associated with Goodreads-style book data and is commonly packaged as `goodbooks-10k`.
