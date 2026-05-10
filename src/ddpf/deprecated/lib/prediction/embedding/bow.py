from src.ddpf.deprecated.lib.prediction.embedding.embeddingsubmodule import ObservationEmbeddingSubModule
from src.ddpf.deprecated.lib.prediction.utils import obsToList, update_dict_values_by_column_order
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

class ObservationBoWEmbedding(ObservationEmbeddingSubModule):
    
    def __init__(self, predictionModule):
        super().__init__(self.bowEmbedding, predictionModule, self._loadEmbeddingsInternal)

    def _loadEmbeddingsInternal(self):
        """
        Internal method to load the KB and network embeddings into DataFrames.
        """
        print("Loading BoW embeddings...")
        df_kb = pd.read_csv(self.predictionModule.path_to_kb_gazetteer)
        df_kb.reset_index(drop=True, inplace=True)

        df_net = pd.read_csv(self.predictionModule.path_to_netwoork_gazetteer)
        df_net.reset_index(drop=True, inplace=True)

        print(f"Loaded {len(df_kb)} KB gazetteer entries and {len(df_net)} network gazetteer entries.")
        return df_kb, df_net

    def bowEmbedding(self, df_obs, df_kb, df_net, include_spatial=True, include_temporal=True):
        """
        Compute a Bag-of-Words (BoW) style embedding for observations based on KB components and system names,
        optionally including spatial (network) and temporal (average date) features.
        """
        # Setting up components
        obs_set = set([])
        for _, row in df_obs.iterrows():
            obs = obsToList(row['KB_Components'])
            obs_set = obs_set.union(set(obs))
        self.obs_set = obs_set
        self.dict_kb = dict(zip(list(obs_set), range(len(obs_set))))
        self.dict_net = dict(zip(list(df_obs['System_Name']), range(len(df_obs['System_Name'])))) if include_spatial else {}
        self.dict_obs = dict(zip(list(df_obs['Observation_Name']), range(len(df_obs['Observation_Name']))))
    
        # Determine the number of columns based on parameters
        nb_columns = len(self.dict_kb)
        if include_spatial:
            nb_columns += len(np.unique(df_obs['System_Name']))
        if include_temporal:
            nb_columns += 1
    
        # Setting up the observation matrix
        obs_matrix = np.zeros((len(df_obs['Observation_Name']), nb_columns))
    
        # Rearranging KB orders
        self.dict_kb = update_dict_values_by_column_order(self.dict_kb, df_kb, 'Name')
    
        # Rearranging network orders (if spatial is included)
        if include_spatial:
            self.dict_net = update_dict_values_by_column_order(self.dict_net, df_net, 'Name')
    
        # Populate the observation matrix
        for _, row in tqdm(df_obs.iterrows()):
            system_kb_items = obsToList(row['KB_Components'])
            for kb_item in system_kb_items:
                if kb_item in self.dict_kb:
                    obs_matrix[self.dict_obs[row['Observation_Name']], self.dict_kb[kb_item]] = 1
    
            if include_spatial and row['System_Name'] in self.dict_net:
                obs_matrix[self.dict_obs[row['Observation_Name']], self.dict_net[row['System_Name']] + len(self.dict_kb)] = 1
    
            if include_temporal:
                temporal_index = -1 if include_spatial else len(self.dict_kb)
                obs_matrix[self.dict_obs[row['Observation_Name']], temporal_index] = row['average_date']
    
        return obs_matrix