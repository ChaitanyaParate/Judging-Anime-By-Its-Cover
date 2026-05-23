# %% [markdown]
# # Phase 2: Exploratory Data Analysis (EDA)
# In this notebook-style script, we will load the scraped MyAnimeList dataset, clean it, and establish our baseline metrics.

# %%
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import os

# Set plotting style
sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

# %% [markdown]
# ## 1. Load Data
# Load the raw anime data from SQLite into a Pandas DataFrame.

# %%
conn = sqlite3.connect("anime_data.db")
df = pd.read_sql_query("SELECT * FROM anime", conn)
conn.close()

print(f"Total raw records: {len(df)}")
df.head()

# %% [markdown]
# ## 2. Data Cleaning
# Remove records that lack a downloaded cover image and those with fewer than 1000 votes, as their scores are statistically noisy.

# %%
# Filter out missing images
df_clean = df[df['local_image_path'].notna()].copy()
# Also filter out non-image files
df_clean = df_clean[df_clean['local_image_path'].str.endswith('.jpg')]

print(f"Records with local images: {len(df_clean)}")

# Filter out low-vote entries
df_clean = df_clean[df_clean['scored_by'] >= 1000]
print(f"Records after removing <1000 votes: {len(df_clean)}")

# Drop rows with missing scores (if any)
df_clean = df_clean.dropna(subset=['score'])
print(f"Final clean dataset size: {len(df_clean)}")

# %% [markdown]
# ## 3. Score Distribution
# Let's see the distribution of MAL scores in our cleaned dataset.

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['score'], bins=30, kde=True, color='purple')
plt.title("Distribution of MAL Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.axvline(df_clean['score'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df_clean["score"].mean():.2f}')
plt.legend()
plt.tight_layout()
plt.savefig("plots/score_distribution.png")
# plt.show()

# %% [markdown]
# ## 4. Genre Analysis
# Check if certain genres inherently score higher or lower. We explode the comma-separated genres column to analyze this.

# %%
# Explode genres
df_genres = df_clean.copy()
df_genres['genres'] = df_genres['genres'].str.split(', ')
df_genres = df_genres.explode('genres')

# Get top 20 most common genres
top_genres = df_genres['genres'].value_counts().nlargest(20).index
df_top_genres = df_genres[df_genres['genres'].isin(top_genres)]

plt.figure(figsize=(12, 8))
# using hue and legend=False as recommended by newer seaborn versions
sns.boxplot(y='genres', x='score', data=df_top_genres, order=top_genres, hue='genres', palette="viridis", legend=False)
plt.title("Score Distribution by Genre (Top 20 Genres)")
plt.xlabel("Score")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig("plots/score_by_genre.png")
# plt.show()

# %% [markdown]
# ## 5. Baseline Evaluation
# To evaluate our future computer vision model, we need a baseline. The simplest baseline is always predicting the global mean score. We'll compute the Mean Absolute Error (MAE) for this approach.

# %%
global_mean = df_clean['score'].mean()
y_true = df_clean['score']
y_pred = [global_mean] * len(y_true)

mae_baseline = mean_absolute_error(y_true, y_pred)
print(f"Baseline MAE (Predicting Mean): {mae_baseline:.4f}")
print("Our future CV model must achieve an MAE lower than this to be considered useful!")
