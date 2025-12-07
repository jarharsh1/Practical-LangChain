from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv() # Load environment variables

# Initialize Embeddings
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=300
)

# FIXED: Use triple quotes (""") for multi-line strings containing other quotes
documents = [
    """Royal Enfield Bullet: The "Longest Running" Legend
    The Royal Enfield Bullet holds the world record for the longest-running motorcycle model in continuous production. While it originated in the UK, its survival is credited to a 1949 order from the Indian Army for border patrol use, which led to the factory being established in Madras (now Chennai) in 1955.""",

    """Yamaha RX100: The "Pocket Rocket" Anomaly
    Despite having a small 98.2cc engine, the Yamaha RX100 became a drag-racing legend in India due to its incredible power-to-weight ratio. It weighed only around 103 kg but produced 11 bhp, allowing it to accelerate faster than many larger 4-stroke motorcycles of its time.""",

    """Bajaj Pulsar: The "Digital" Pioneer
    When the "UG-3" (Upgrade 3) version of the Bajaj Pulsar was launched in 2006, it became one of the first Indian motorcycles to feature a fully digital speedometer (with no analog needle for speed), a feature that was considered a futuristic luxury at the time.""",

    """TVS Apache: Born on the Racetrack
    The TVS Apache series is explicitly developed using data from the *TVS Racing* factory team, which has been competing for over 40 years. This "track-to-road" philosophy led to it being the first in its segment to introduce "Glide Through Technology" (GTT), which allows the bike to move in traffic without using the throttle, preventing stalling.""",

    """Hero Splendor: The Sales Behemoth
    The Hero Splendor is so popular that its individual sales figures often surpass the entire annual two-wheeler sales of many countries. In October 2025 alone, it sold over 3.4 lakh units, maintaining its position as India's best-selling motorcycle for decades."""
]

query = "Which bike is more race savvy?"

# Generate Embeddings
document_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate Similarity
# Note: cosine_similarity expects 2D arrays. We wrap query_embedding in a list.
similarities = cosine_similarity([query_embedding], document_embedding)

# Get the index of the highest score
best_index = np.argmax(similarities)
max_score = np.max(similarities)

# Output results
print("Similarity Scores:", similarities)
print("\n") # Fixed: used \n for new line, not /n

# Fixed: Used an f-string for cleaner formatting
print(f"Similarity score: {max_score:.4f}, Most Similar Document: {documents[best_index]}")