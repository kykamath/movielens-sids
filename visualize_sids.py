import pandas as pd
import plotly.express as px
import umap
from datasets import load_dataset
from models import HUB_SIDS_DATASET_ID # Import the final dataset ID

def visualize_semantic_ids():
    """
    Loads the final dataset containing embeddings and Semantic IDs,
    and generates an interactive UMAP visualization.
    """
    print(f"Loading final dataset from Hugging Face Hub: {HUB_SIDS_DATASET_ID}")
    try:
        hub_dataset = load_dataset(HUB_SIDS_DATASET_ID, split="train")
    except Exception as e:
        print(f"Could not load dataset from Hub. Error: {e}")
        print("Please ensure the dataset exists and you have the correct permissions.")
        return

    # --- 1. Prepare Data for Plotting ---
    # Filter for items that have both an embedding and a semantic ID
    valid_items = [
        item for item in hub_dataset 
        if item.get('all_mpnet_base_v2_embedding') and item.get('semantic_id')
    ]

    if not valid_items:
        print("No valid items with both embeddings and SIDs found in the dataset.")
        return

    # Extract the data we need
    embeddings = [item['all_mpnet_base_v2_embedding'] for item in valid_items]
    titles = [item['title'] for item in valid_items]
    genres = [', '.join(item['genres']) for item in valid_items]
    sids = [item['semantic_id'] for item in valid_items]
    
    # The first token (T1) is the first element of each SID list
    t1_tokens = [sid[0] for sid in sids]
    
    # The number of layers (tokens) in the SID
    num_layers = len(sids[0])

    print(f"Found {len(valid_items)} valid items to visualize.")

    # --- 2. Run UMAP ---
    print("Running UMAP for dimensionality reduction... (This may take a minute)")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # --- 3. Create a DataFrame for Plotting ---
    df = pd.DataFrame({
        'title': titles,
        'genres': genres,
        'umap_x': embeddings_2d[:, 0],
        'umap_y': embeddings_2d[:, 1],
        'T1': t1_tokens,
        'SID': [' '.join([f"<T{i+1}:{token:04d}>" for i, token in enumerate(sid)]) for sid in sids]
    })

    # --- 4. Create the Interactive Visualization ---
    print("Generating interactive plot with Plotly...")
    fig = px.scatter(
        df,
        x='umap_x',
        y='umap_y',
        color=df['T1'].astype(str),  # Color by the first token (T1)
        hover_name='title',         # Show movie title on hover
        hover_data=['genres', 'SID'], # Show genres and full SID in the hover tooltip
        title="UMAP Visualization of Movie Embeddings, Colored by Semantic ID Token 1 (T1)"
    )

    fig.update_layout(
        legend_title_text='First Semantic Token (T1)',
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.8))

    # Save to an HTML file
    output_filename = "semantic_id_visualization.html"
    fig.write_html(output_filename)
    print(f"\n✅ Visualization saved to '{output_filename}'. Open this file in your browser.")

if __name__ == '__main__':
    visualize_semantic_ids()
