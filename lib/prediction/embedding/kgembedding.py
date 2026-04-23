from prediction.embedding.embeddingsubmodule import ObservationEmbeddingSubModule
from prediction.utils import obsToList
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

class ObservationKGEmbedding(ObservationEmbeddingSubModule):
    def __init__(self, predictionModule, kb_embedding_path, net_embedding_path, aggregation_method="sum"):
        """
        Base class for KG-based embeddings. Supports aggregation methods ('sum' or 'average').
        """
        super().__init__(self.computeEmbedding, predictionModule, self._loadEmbeddingsInternal)
        self.kb_embedding_path = kb_embedding_path
        self.net_embedding_path = net_embedding_path
        if aggregation_method not in {"sum", "average"}:
            raise ValueError("aggregation_method must be either 'sum' or 'average'")
        self.aggregation_method = aggregation_method

    def get_embedding(self, node_id, df, cache):
        """
        Retrieve the embedding for a given node ID using cache. Falls back to DataFrame lookup if not in cache.
        """
        if node_id in cache:
            return cache[node_id]

        try:
            # Retrieve the embedding string
            embedding_str = df.loc[df['Node'] == node_id, 'Embedding'].values[0]
            
            # Parse the string to a Python list and convert to numpy array
            embedding_list = ast.literal_eval(embedding_str)
            embedding_array = np.array(embedding_list, dtype=float)
            cache[node_id] = embedding_array  # Store in cache
            return embedding_array
        except IndexError:
            print(f"Node ID {node_id} not found in DataFrame.")
            cache[node_id] = None  # Cache the miss
            return None
        except (ValueError, SyntaxError):
            print(f"Failed to parse embedding for Node ID {node_id}.")
            cache[node_id] = None  # Cache the miss
            return None

    def _loadEmbeddingsInternal(self):
        """
        Internal method to load the KB and network embeddings into DataFrames.
        """
        print(f"Loading embeddings from {self.kb_embedding_path} and {self.net_embedding_path}...")
        df_kb = pd.read_csv(self.kb_embedding_path, header=0)
        df_kb.rename(columns={df_kb.columns[0]: 'Node', df_kb.columns[1]: 'Embedding'}, inplace=True)

        df_net = pd.read_csv(self.net_embedding_path, header=0)
        df_net.rename(columns={df_net.columns[0]: 'Node', df_net.columns[1]: 'Embedding'}, inplace=True)

        print(f"Loaded {len(df_kb)} KB embeddings and {len(df_net)} network embeddings.")
        return df_kb, df_net

    def computeEmbedding(self, df_obs, df_kb, df_net, include_spatial=True, include_temporal=True):
        """
        Compute embeddings for observations by summing or averaging KB components and optionally including
        network embeddings and temporal features.
        """
        kb_cache = {}
        net_cache = {}
    
        # Determine embedding size
        sample_embedding = None
        for node_id in df_kb['Node']:
            sample_embedding = self.get_embedding(node_id, df_kb, kb_cache)
            if sample_embedding is not None:
                embedding_size = sample_embedding.shape[0]
                break
        if sample_embedding is None:
            raise ValueError("No valid embeddings found in the KB embeddings file.")
    
        # Adjust matrix size based on parameters
        additional_columns = 0
        if include_temporal:
            additional_columns += 1  # Add temporal feature
        obs_matrix = np.zeros((len(df_obs), embedding_size + additional_columns))
    
        for obs_idx, row in df_obs.iterrows():
            kb_embeddings = [
                self.get_embedding(kb_id, df_kb, kb_cache)
                for kb_id in obsToList(row['KB_IDs'])
                if self.get_embedding(kb_id, df_kb, kb_cache) is not None
            ]
    
            if kb_embeddings:
                if self.aggregation_method == "sum":
                    kb_aggregated = np.sum(kb_embeddings, axis=0)
                    if include_spatial:
                        combined_embedding = kb_aggregated + self.get_embedding(row['System_ID'], df_net, net_cache)
                    else:
                        combined_embedding = kb_aggregated
                elif self.aggregation_method == "average":
                    if include_spatial:
                        kb_aggregated = np.sum(kb_embeddings, axis=0) + self.get_embedding(row['System_ID'], df_net, net_cache)
                        combined_embedding = kb_aggregated / (len(kb_embeddings) + 1)
                    else:
                        kb_aggregated = np.sum(kb_embeddings, axis=0)
                        combined_embedding = kb_aggregated / len(kb_embeddings)
            else:
                combined_embedding = np.zeros(embedding_size)
    
            obs_matrix[obs_idx, :embedding_size] = combined_embedding
    
            # Include temporal (average_date) if applicable
            if include_temporal:
                obs_matrix[obs_idx, -1] = row['average_date']
    
        return obs_matrix

        
class ObservationHomogeneousEmbedding(ObservationKGEmbedding):
    pass

class ObservationHeterogeneousEmbedding(ObservationKGEmbedding):
    pass