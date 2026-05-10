import pandas as pd
import numpy as np
from prediction.embedding.embeddingsubmodule import ObservationEmbeddingSubModule
from tqdm import tqdm
from prediction.utils import obsToList
import ast
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import classification_report
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm


class ObservationWangPipeline:
    def __init__(self, predictionModule, kb_concept_path, net_concept_path, kb_sa_path, net_sa_path, kb_sva_path, net_sva_path):
        """
        Initialize the ObservationWangPipeline.
        """
        self.predictionModule = predictionModule
        self.kb_concept_path = kb_concept_path
        self.net_concept_path = net_concept_path
        self.kb_sa_path = kb_sa_path
        self.net_sa_path = net_sa_path
        self.kb_sva_path = kb_sva_path
        self.net_sva_path = net_sva_path

    def _strip_quotes(self, value):
        """
        Remove all double and single quotes from a string, a pandas Series, or elements in a list.
        """
        if isinstance(value, str):
            return value.replace('"', '').replace("'", "")
        elif isinstance(value, list):
            # Recursively strip quotes from each element in the list
            return [self._strip_quotes(v) for v in value]
        elif isinstance(value, pd.Series):
            # Apply stripping to each element in the Series
            return value.apply(lambda x: self._strip_quotes(x))
        else:
            return value

    def _get_unique_concepts(self, df):
        """
        Extract and clean unique concepts from the 'concept' column in a DataFrame.
        """
        raw_concepts = df['concept'].apply(ast.literal_eval)  # Parse the list of concepts
        flattened_concepts = raw_concepts.explode()  # Flatten the list of lists
        cleaned_concepts = flattened_concepts.dropna().apply(self._strip_quotes)  # Remove quotes and drop NaN values
        return sorted(cleaned_concepts.unique())

    def _loadEmbeddingsInternal(self):
        """
        Load all required DataFrames for the Wang pipeline.
        """
        print(f"Loading concept mappings from {self.kb_concept_path} and {self.net_concept_path}...")
        # Load KB concept mapping
        df_kb = pd.read_csv(self.kb_concept_path, header=0)
        df_kb.rename(columns={df_kb.columns[0]: 'component', df_kb.columns[1]: 'concept'}, inplace=True)

        # Load Network concept mapping
        df_net = pd.read_csv(self.net_concept_path, header=0)
        df_net.rename(columns={df_net.columns[0]: 'component', df_net.columns[1]: 'concept'}, inplace=True)

        print(f"Loading semantic annotations (SA) and semantic value annotations (SVA)...")
        df_kb_sa = pd.read_csv(self.kb_sa_path, header=0)
        df_kb_sa.rename(columns={df_kb_sa.columns[0]: 'concept1', df_kb_sa.columns[1]: 'concept2', df_kb_sa.columns[2]: 'distance'}, inplace=True)

        df_net_sa = pd.read_csv(self.net_sa_path, header=0)
        df_net_sa.rename(columns={df_net_sa.columns[0]: 'concept1', df_net_sa.columns[1]: 'concept2', df_net_sa.columns[2]: 'distance'}, inplace=True)

        df_kb_sva = pd.read_csv(self.kb_sva_path, header=0)
        df_kb_sva.rename(columns={df_kb_sva.columns[0]: 'concept', df_kb_sva.columns[1]: 'weight'}, inplace=True)

        df_net_sva = pd.read_csv(self.net_sva_path, header=0)
        df_net_sva.rename(columns={df_net_sva.columns[0]: 'concept', df_net_sva.columns[1]: 'weight'}, inplace=True)

        print("DataFrames loaded successfully.")
        return df_kb, df_net, df_kb_sa, df_net_sa, df_kb_sva, df_net_sva

    def computeEmbedding(self, df_obs, df_kb, df_net, df_kb_sa, df_net_sa, include_spatial=True, include_temporal=True):
        """
        Compute embeddings for observations using the Wang pipeline approach.
        """
        print("Processing KB concepts...")
        unique_kb_concepts = self._get_unique_concepts(df_kb)
        kb_embedding_size = len(unique_kb_concepts)
        
        if include_spatial:
            print("Processing Network concepts...")
            unique_net_concepts = self._get_unique_concepts(df_net)
            net_embedding_size = len(unique_net_concepts)
    
        print("Processing complete. Computing embeddings for observations...")
        aggregated_embeddings = []
    
        # Iterate through each observation
        for _, row in tqdm(df_obs.iterrows(), total=len(df_obs)):
    
            # Process KB components
            kb_components = obsToList(row['KB_IDs'])
            list_of_concepts = []
            for component in kb_components:
                matching_kb_rows = df_kb[df_kb['component'] == component]
                if not matching_kb_rows.empty:
                    concepts = ast.literal_eval(matching_kb_rows.iloc[0]['concept'])
                    list_of_concepts.extend(concepts)
    
            # Create the KB embedding
            kb_value = np.zeros(kb_embedding_size)
            for i, target_concept in enumerate(unique_kb_concepts):
                max_distance = 0
                for concept in list_of_concepts:
                    max_row = df_kb_sa[
                        ((df_kb_sa['concept1'] == concept) & (df_kb_sa['concept2'] == target_concept)) |
                        ((df_kb_sa['concept2'] == concept) & (df_kb_sa['concept1'] == target_concept))
                    ]
                    if not max_row.empty:
                        max_distance = max(max_distance, max_row['distance'].max())
                kb_value[i] = max_distance
    
            # Initialize the final embedding with only the KB value
            final_embedding = kb_value
    
            # Process System ID if spatial embedding is included
            if include_spatial:
                system_id = row['System_ID']
                matching_net_rows = df_net[df_net['component'] == system_id]
                list_of_net_concepts = []
                if not matching_net_rows.empty:
                    net_concepts = ast.literal_eval(matching_net_rows.iloc[0]['concept'])
                    list_of_net_concepts.extend(net_concepts)
    
                net_value = np.zeros(net_embedding_size)
                for i, target_concept in enumerate(unique_net_concepts):
                    max_distance = 0
                    for concept in list_of_net_concepts:
                        max_row = df_net_sa[
                            ((df_net_sa['concept1'] == concept) & (df_net_sa['concept2'] == target_concept)) |
                            ((df_net_sa['concept2'] == concept) & (df_net_sa['concept1'] == target_concept))
                        ]
                        if not max_row.empty:
                            max_distance = max(max_distance, max_row['distance'].max())
                    net_value[i] = max_distance
    
                # Concatenate the net embedding if spatial embedding is included
                final_embedding = np.concatenate([final_embedding, net_value])
    
            # Add the date if temporal embedding is included
            if include_temporal:
                final_embedding = np.append(final_embedding, row['average_date'])
    
            aggregated_embeddings.append(final_embedding)
    
        print("Observation embeddings computed successfully.")
        return np.array(aggregated_embeddings)



    def _calculate_distance(self, o1, o2):
        """
        Calculate the distance between two observations.
        :param o1: Embedding of observation 1.
        :param o2: Embedding of observation 2.
        :return: Distance value.
        """
        numerator = np.dot(o1, o2)
        denominator = np.sqrt(np.sum(o1 ** 2)) * np.sqrt(np.sum(o2 ** 2))
        similarity = numerator / denominator if denominator != 0 else 0
        return 1 - similarity
        
    def clusterObservations(self, embedding_matrix, df_obs, k=2, random_seed=42, include_spatial=True, include_temporal=True):
        """
        Cluster observations using k-medoids and evaluate the clusters.
        
        :param embedding_matrix: The matrix of embeddings for observations.
        :param df_obs: DataFrame containing observation data.
        :param k: Number of clusters for k-medoids.
        :param random_seed: Random seed for reproducibility.
        :param include_temporal: If True, includes date in the distance calculation.
        :return: Cluster labels, the best type list, and the best F1 score.
        """
        m = embedding_matrix.shape[0]
        distance_matrix = np.zeros((m, m))
        
        # Rescale the last column if include_temporal is True
        if include_temporal:
            dates = embedding_matrix[:, -1]
            max_date = np.max(dates)
            if max_date > 0:
                rescaled_dates = dates / max_date
            else:
                rescaled_dates = dates.copy()
    
        # Compute the distance matrix
        for i in range(m):
            for j in range(m):
                if include_temporal:
                    distance_vector = self._calculate_distance(embedding_matrix[i, :-1], embedding_matrix[j, :-1])
                    date_difference = abs(rescaled_dates[i] - rescaled_dates[j])
                    distance_matrix[i, j] = distance_vector + date_difference
                else:
                    distance_matrix[i, j] = self._calculate_distance(embedding_matrix[i], embedding_matrix[j])
    
        # Rescale the entire distance matrix to the range [0, 1]
        max_distance = np.max(distance_matrix)
        min_distance = np.min(distance_matrix)
        if max_distance > min_distance:  # Avoid division by zero
            distance_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
        
        print("Clustering observations with k-medoids...")
        kmedoids = KMedoids(n_clusters=k, metric="precomputed", random_state=random_seed)
        kmedoids_labels = kmedoids.fit_predict(distance_matrix)
        
        print("Evaluating clustering...")
        list_type_0 = []
        list_type_1 = []
        for _, row in df_obs.iterrows():
            if "COVID" in row["Observation_Name"]:
                list_type_1.append(0)
                list_type_0.append(1)
            else:
                list_type_1.append(1)
                list_type_0.append(0)
        
        report_1 = classification_report(list_type_1, kmedoids_labels, output_dict=True)
        score_1 = report_1['weighted avg']['f1-score']
        report_0 = classification_report(list_type_0, kmedoids_labels, output_dict=True)
        score_0 = report_0['weighted avg']['f1-score']
        
        if score_1 >= score_0:
            best_score = score_1
            best_type = list_type_1
        else:
            best_score = score_0
            best_type = list_type_0
        
        print(f"Clustering completed. Best F1 Score: {best_score:.4f}")
        return kmedoids_labels, best_type, best_score





class ObservationSemanticEmbedding(ObservationEmbeddingSubModule):
    def __init__(self, predictionModule, kb_concept_path, net_concept_path,
                 computeEmbedding, aggregation_method="sum"):
        """
        Initialize the ObservationSemanticEmbedding module.
        :param predictionModule: The prediction module containing configurations.
        :param entity_concept_path: Path to the CSV file containing entity-concept mappings.
        :param aggregation_method: Aggregation method for combining embeddings ('sum' or 'average').
        """
        super().__init__(computeEmbedding, predictionModule, self._loadEmbeddingsInternal)
        self.kb_concept_path = kb_concept_path
        self.net_concept_path = net_concept_path
        if aggregation_method not in {"sum", "average"}:
            raise ValueError("aggregation_method must be either 'sum' or 'average'")
        self.aggregation_method = aggregation_method
        
    def _loadEmbeddingsInternal(self):
        """
        Internal method to load the KB and network concept mappings into DataFrames.
        """
        print(f"Loading concept mappings from {self.kb_concept_path} and {self.net_concept_path}...")
    
        # Load KB concept mapping
        df_kb = pd.read_csv(self.kb_concept_path, header=0)
        df_kb.rename(columns={df_kb.columns[0]: 'component', df_kb.columns[1]: 'concept'}, inplace=True)
    
        # Load Network concept mapping
        df_net = pd.read_csv(self.net_concept_path, header=0)
        df_net.rename(columns={df_net.columns[0]: 'component', df_net.columns[1]: 'concept'}, inplace=True)
    
        print(f"Loaded {len(df_kb)} KB concept mappings and {len(df_net)} Network concept mappings.")
        return df_kb, df_net
    
    def _getDistanceFromComponent(self, component, df_component_concept, df_concept_distance, cache, embedding_size):
        """
        Retrieve the distance vector for a component, using a cache to speed up repeated lookups.
        """
        # Use the cache if available
        if component in cache:
            return cache[component]
    
        # Check if the component exists in the component-to-concept DataFrame
        matching_rows = df_component_concept[df_component_concept['component'] == component]
        if matching_rows.empty:
            # Cache and return a zero vector if the component is not found
            cache[component] = np.zeros(embedding_size)
            return cache[component]
    
        # Parse and process the concepts
        try:
            concepts = ast.literal_eval(matching_rows.iloc[0]['concept'])
        except (ValueError, SyntaxError):
            # Cache and return a zero vector if parsing fails
            cache[component] = np.zeros(embedding_size)
            return cache[component]
    
        # Retrieve the distance vectors for all concepts
        concept_vectors = []
        for concept in concepts:
            concept_stripped = self._strip_quotes(concept)
            concept_row = df_concept_distance[df_concept_distance['concept'].str.contains(concept_stripped, na=False)]
            if concept_row.empty:
                # Append a zero vector if the concept is not found
                concept_vectors.append(np.zeros(embedding_size))
                continue
            concept_vector = np.array(concept_row.iloc[0]['distance'], dtype=float)
            concept_vectors.append(concept_vector)
    
        # Average the vectors for all concepts related to the component
        if concept_vectors:
            result_vector = np.mean(concept_vectors, axis=0)
        else:
            result_vector = np.zeros(embedding_size)
    
        # Cache the result
        cache[component] = result_vector
        return result_vector

    def _create_concept_distance_dataframe(self, path):
        """
        Create a DataFrame where the first column is 'concept' (a unique concept from the source/target list),
        and the second column is 'distance' (a list of distances to all other concepts).
        
        :param path: Path to the CSV file containing source, target, and distance columns.
        :return: A DataFrame with 'concept' as a list of strings and 'distance' as a list of distances.
        """
        print(f"Loading data from {path}...")
        # Load the CSV into a DataFrame and rename columns
        df = pd.read_csv(path, header=0)
        df.rename(columns={df.columns[0]: 'source', df.columns[1]: 'target', df.columns[2]: 'distance'}, inplace=True)
        
        # Extract all unique concepts from source and target columns
        unique_concepts = sorted(set(df['source'].unique()).union(set(df['target'].unique())))
        
        # Create a dictionary to store the distances for faster lookup
        distance_dict = {}
        for _, row in df.iterrows():
            pair = (row['source'], row['target'])
            distance_dict[pair] = row['distance']
        
        # Create the new DataFrame with the required structure
        concept_data = []
        for concept in unique_concepts:
            # Create a list of distances for the current concept to all other concepts
            distances = [
                distance_dict.get((concept, target), 0.0)  # Default to 0.0 if no distance is found
                for target in unique_concepts
            ]
            # Enclose the concept in a list of strings
            concept_data.append({'concept': concept.replace('[', '').replace(']', ''), 'distance': distances})
        
        # Convert the list of dictionaries to a DataFrame
        result_df = pd.DataFrame(concept_data)
        print(f"Created DataFrame with {len(result_df)} concepts.")
        
        return result_df

    def _strip_quotes(self, value):
        """
        Remove all double and single quotes from a string, a pandas Series, or elements in a list.
        """
        if isinstance(value, str):
            return value.replace('"', '').replace("'", "")
        elif isinstance(value, list):
            # Recursively strip quotes from each element in the list
            return [self._strip_quotes(v) for v in value]
        elif isinstance(value, pd.Series):
            # Apply stripping to each element in the Series
            return value.apply(lambda x: self._strip_quotes(x))
        else:
            return value

class ObservationLCEmbedding(ObservationSemanticEmbedding):
    def __init__(self, predictionModule, kb_concept_path, net_concept_path,
                 kb_distance_path, net_distance_path,
                 aggregation_method="sum"):
        super().__init__(predictionModule, kb_concept_path, net_concept_path,
                 self.computeEmbedding, aggregation_method=aggregation_method)
        self.kb_distance_path = kb_distance_path
        self.net_distance_path = net_distance_path

    def computeEmbedding(self, df_obs, df_kb, df_net, include_spatial=True, include_temporal=True):
        """
        Compute embeddings for observations by aggregating KB and Network distances.
        """
        # Step 1: Use the new method to process KB and Network distances
        print(f"Processing KB distances from {self.kb_distance_path}...")
        df_kb_distance = self._create_concept_distance_dataframe(self.kb_distance_path)
        
        if include_spatial:
            print(f"Processing Network distances from {self.net_distance_path}...")
            df_net_distance = self._create_concept_distance_dataframe(self.net_distance_path)
    
        print("Processing complete. Computing aggregated embeddings for observations...")
    
        # Initialize embedding results
        aggregated_embeddings = []
        embedding_size = len(df_kb_distance.iloc[0]['distance']) if not df_kb_distance.empty else 1  # Default size
    
        # Initialize caches
        kb_cache = {}
        net_cache = {}
    
        # Iterate through observations
        for _, row in tqdm(df_obs.iterrows(), total=len(df_obs)):
            kb_embeddings = []
    
            # Process KB components
            kb_components = obsToList(row['KB_IDs'])
            for component in kb_components:
                kb_embedding = self._getDistanceFromComponent(
                    component,
                    df_component_concept=df_kb,
                    df_concept_distance=df_kb_distance,
                    cache=kb_cache,
                    embedding_size=embedding_size
                )
                kb_embeddings.append(kb_embedding)
    
            # Process system ID if spatial embedding is included
            if include_spatial:
                system_id = self._strip_quotes(row['System_ID'])
                system_embedding = self._getDistanceFromComponent(
                    system_id,
                    df_component_concept=df_net,
                    df_concept_distance=df_net_distance,
                    cache=net_cache,
                    embedding_size=embedding_size
                )
            else:
                system_embedding = np.array([])
    
            # Aggregate embeddings
            if kb_embeddings:
                if self.aggregation_method == "sum":
                    kb_aggregated = np.sum(kb_embeddings, axis=0)
                elif self.aggregation_method == "average":
                    kb_aggregated = np.sum(kb_embeddings, axis=0) / len(kb_embeddings)
    
                # Concatenate the KB aggregated result with the system embedding if applicable
                aggregated_array = np.concatenate([kb_aggregated, system_embedding])
            else:
                # If no KB embeddings, use only the system embedding
                aggregated_array = system_embedding
    
            # Include temporal (average_date) if applicable
            if include_temporal:
                aggregated_array = np.append(aggregated_array, row['average_date'])
    
            aggregated_embeddings.append(aggregated_array)
    
        print("Observation embeddings computed successfully.")
        return np.array(aggregated_embeddings)

    
    

class ObservationHSEmbedding(ObservationSemanticEmbedding):
    def __init__(self, predictionModule, kb_concept_path, net_concept_path,
                 kb_distance_path, net_distance_path,
                 kb_turn_path, net_turn_path,
                 C=8,k=1,
                 aggregation_method="sum"):
        super().__init__(predictionModule, kb_concept_path, net_concept_path,
                         self.computeEmbedding, aggregation_method=aggregation_method)
        self.kb_distance_path = kb_distance_path
        self.net_distance_path = net_distance_path
        self.kb_turn_path = kb_turn_path
        self.net_turn_path = net_turn_path
        self.C = C
        self.k = k


    def computeEmbedding(self, df_obs, df_kb, df_net, include_spatial=True, include_temporal=True):
        """
        Compute embeddings for observations by aggregating KB and Network distances, adjusted by turns.
        """
        # Step 1: Use the new method to process KB and Network distances and turns
        print(f"Processing KB distances from {self.kb_distance_path}...")
        df_kb_distance = self._create_concept_distance_dataframe(self.kb_distance_path)
        print(f"Processing KB turns from {self.kb_turn_path}...")
        df_kb_turns = self._create_concept_distance_dataframe(self.kb_turn_path)
        if include_spatial:
            print(f"Processing Network distances from {self.net_distance_path}...")
            df_net_distance = self._create_concept_distance_dataframe(self.net_distance_path)
            print(f"Processing Network turns from {self.net_turn_path}...")
            df_net_turns = self._create_concept_distance_dataframe(self.net_turn_path)
            
        print("Processing complete. Computing aggregated embeddings for observations...")
    
        # Initialize embedding results
        aggregated_embeddings = []
        embedding_size = len(df_kb_distance.iloc[0]['distance']) if not df_kb_distance.empty else 1  # Default size
    
        # Initialize caches
        kb_cache = {}
        net_cache = {}
    
        # Iterate through observations
        for _, row in tqdm(df_obs.iterrows(), total=len(df_obs)):
            kb_embeddings = []
    
            # Process KB components
            kb_components = obsToList(row['KB_IDs'])
            for component in kb_components:
                kb_distance = self._getDistanceFromComponent(
                    component,
                    df_component_concept=df_kb,
                    df_concept_distance=df_kb_distance,
                    cache=kb_cache,
                    embedding_size=embedding_size
                )
                kb_turn = self._getDistanceFromComponent(
                    component,
                    df_component_concept=df_kb,
                    df_concept_distance=df_kb_turns,
                    cache=kb_cache,
                    embedding_size=embedding_size
                )
                kb_embeddings.append(self.C - kb_distance - self.k * kb_turn)
    
            # Process system ID if spatial embedding is included
            if include_spatial:
                system_id = self._strip_quotes(row['System_ID'])
                system_distance = self._getDistanceFromComponent(
                    system_id,
                    df_component_concept=df_net,
                    df_concept_distance=df_net_distance,
                    cache=net_cache,
                    embedding_size=embedding_size
                )
                system_turn = self._getDistanceFromComponent(
                    system_id,
                    df_component_concept=df_net,
                    df_concept_distance=df_net_turns,
                    cache=net_cache,
                    embedding_size=embedding_size
                )
                system_embedding = self.C - system_distance - self.k * system_turn
            else:
                system_embedding = np.array([])
    
            # Aggregate embeddings
            if kb_embeddings:
                if self.aggregation_method == "sum":
                    kb_aggregated = np.sum(kb_embeddings, axis=0)
                elif self.aggregation_method == "average":
                    kb_aggregated = np.sum(kb_embeddings, axis=0) / len(kb_embeddings)
    
                # Concatenate KB aggregated result with the system embedding if applicable
                aggregated_array = np.concatenate([kb_aggregated, system_embedding])
            else:
                aggregated_array = system_embedding
    
            # Include temporal (average_date) if applicable
            if include_temporal:
                aggregated_array = np.append(aggregated_array, row['average_date'])
    
            aggregated_embeddings.append(aggregated_array)
    
        print("Observation embeddings computed successfully.")
        return np.array(aggregated_embeddings)



class ObservationRelTopicEmbedding(ObservationSemanticEmbedding):
    def __init__(self, predictionModule, kb_concept_path, net_concept_path,
                 kb_distance_path, net_distance_path,
                 kb_neighborhood_path, net_neighborhood_path,
                 kb_specialdegree_path, net_specialdegree_path,
                 aggregation_method="sum"):
        super().__init__(predictionModule, kb_concept_path, net_concept_path,
                         self.computeEmbedding, aggregation_method=aggregation_method)
        self.kb_distance_path = kb_distance_path
        self.net_distance_path = net_distance_path
        self.kb_neighborhood_path = kb_neighborhood_path
        self.net_neighborhood_path = net_neighborhood_path
        self.kb_specialdegree_path = kb_specialdegree_path
        self.net_specialdegree_path = net_specialdegree_path

    def _get_unique_sorted_concepts(self, df):
        """
        Extract and clean unique concepts from the 'concept' column in a DataFrame.
        """
        raw_concepts = df['concept'].apply(ast.literal_eval)  # Parse the list of concepts
        flattened_concepts = raw_concepts.explode()  # Flatten the list of lists
        cleaned_concepts = flattened_concepts.dropna().apply(self._strip_quotes)  # Remove quotes and drop NaN values
        return sorted(cleaned_concepts.unique())


    def _strip_quotes(self, value):
        """
        Remove all double and single quotes from a string, a pandas Series, or elements in a list.
        """
        if isinstance(value, str):
            return value.replace('"', '').replace("'", "")
        elif isinstance(value, list):
            # Recursively strip quotes from each element in the list
            return [self._strip_quotes(v) for v in value]
        elif isinstance(value, pd.Series):
            # Apply stripping to each element in the Series
            return value.apply(lambda x: self._strip_quotes(x))
        else:
            return value
            
    def computeEmbedding(self, df_obs, df_kb, df_net, include_spatial=True, include_temporal=True):
        """
        Compute embeddings for observations using KB and Network distances, adjusted by relationships and topics.
        """
        # Step 1: Load supporting DataFrames
        print(f"Loading KB distances from {self.kb_distance_path}...")
        df_kb_distance = self._create_concept_distance_dataframe(self.kb_distance_path)
        print(f"Loading KB neighborhood data from {self.kb_neighborhood_path}...")
        kb_neighborhood_df = pd.read_csv(self.kb_neighborhood_path, usecols=['semanticType', 'weight'])
        print(f"Loading KB special degree data from {self.kb_specialdegree_path}...")
        kb_specialdegree_df = pd.read_csv(self.kb_specialdegree_path, usecols=['semanticType', 'specialDegree'])
    
        if include_spatial:
            print(f"Loading Network distances from {self.net_distance_path}...")
            df_net_distance = self._create_concept_distance_dataframe(self.net_distance_path)
            print(f"Loading Network neighborhood data from {self.net_neighborhood_path}...")
            net_neighborhood_df = pd.read_csv(self.net_neighborhood_path, usecols=['semanticType', 'weight'])
            print(f"Loading Network special degree data from {self.net_specialdegree_path}...")
            net_specialdegree_df = pd.read_csv(self.net_specialdegree_path, usecols=['semanticType', 'specialDegree'])
    
        print("Processing complete. Computing aggregated embeddings for observations...")
    
        # Get unique sorted concepts for KB and Network
        unique_kb_concepts = self._get_unique_sorted_concepts(df_kb)
        kb_embedding_size = len(unique_kb_concepts)
        net_embedding_size = 0
        if include_spatial:
            unique_net_concepts = self._get_unique_sorted_concepts(df_net)
            net_embedding_size = len(unique_net_concepts)
    
        # Initialize embedding results
        aggregated_embeddings = []
    
        # Iterate through observations
        for _, row in tqdm(df_obs.iterrows(), total=len(df_obs)):
            kb_embeddings = []
    
            # Process KB components
            kb_components = obsToList(row['KB_IDs'])
            for component in kb_components:
                component = self._strip_quotes(component)
                matching_kb_rows = df_kb[df_kb['component'] == component]
                if matching_kb_rows.empty:
                    kb_embeddings.append(np.zeros(kb_embedding_size))
                    continue
    
                concepts = [self._strip_quotes(x) for x in ast.literal_eval(matching_kb_rows.iloc[0]['concept'])]
                kb_weights = np.zeros(kb_embedding_size)
                kb_degrees = np.zeros(kb_embedding_size)
                kb_distances = np.zeros(kb_embedding_size)
                concept_counts = np.zeros(kb_embedding_size)
    
                for concept in concepts:
                    concept = self._strip_quotes(concept)
                    for i, target_concept in enumerate(unique_kb_concepts):
                        target_concept = self._strip_quotes(target_concept)
    
                        # Neighborhood weights
                        weight_kb = (
                            kb_neighborhood_df.loc[
                                self._strip_quotes(kb_neighborhood_df['semanticType']) == concept, 'weight'
                            ].sum()
                            + kb_neighborhood_df.loc[
                                self._strip_quotes(kb_neighborhood_df['semanticType']) == target_concept, 'weight'
                            ].sum()
                        )
                        kb_weights[i] += weight_kb
    
                        # Special degrees
                        degree_kb = (
                            kb_specialdegree_df.loc[
                                self._strip_quotes(kb_specialdegree_df['semanticType']) == concept, 'specialDegree'
                            ].sum()
                            + kb_specialdegree_df.loc[
                                self._strip_quotes(kb_specialdegree_df['semanticType']) == target_concept, 'specialDegree'
                            ].sum()
                        )
                        kb_degrees[i] += degree_kb
    
                        # Distance adjustments
                        distance_row = df_kb_distance.loc[
                            self._strip_quotes(df_kb_distance['concept']) == concept
                        ]
                        if not distance_row.empty:
                            distance_vector = np.array(distance_row.iloc[0]['distance'], dtype=float)
                            kb_distances[i] += 1 / (1 + distance_vector[i])
                            concept_counts[i] += 1
    
                # Compute averages
                kb_weights /= np.maximum(concept_counts, 1)
                kb_degrees /= np.maximum(concept_counts, 1)
                kb_distances /= np.maximum(concept_counts, 1)
    
                # Compute the adjusted embedding
                with np.errstate(divide='ignore', invalid='ignore'):
                    kb_value = kb_distances + np.log(kb_weights + 1) / (kb_degrees + 1e-6)
    
                kb_embeddings.append(kb_value)
    
            # Process system ID if spatial embedding is included
            if include_spatial:
                system_id = self._strip_quotes(row['System_ID'])
                net_embeddings = []
    
                matching_net_rows = df_net[df_net['component'] == system_id]
                if matching_net_rows.empty:
                    net_embeddings.append(np.zeros(net_embedding_size))
                else:
                    system_concepts = [self._strip_quotes(x) for x in ast.literal_eval(matching_net_rows.iloc[0]['concept'])]
                    net_weights = np.zeros(net_embedding_size)
                    net_degrees = np.zeros(net_embedding_size)
                    net_distances = np.zeros(net_embedding_size)
                    concept_counts_net = np.zeros(net_embedding_size)
    
                    for system_concept in system_concepts:
                        system_concept = self._strip_quotes(system_concept)
                        for i, target_concept in enumerate(unique_net_concepts):
                            target_concept = self._strip_quotes(target_concept)
    
                            # Neighborhood weights
                            weight_net = (
                                net_neighborhood_df.loc[
                                    self._strip_quotes(net_neighborhood_df['semanticType']) == system_concept, 'weight'
                                ].sum()
                                + net_neighborhood_df.loc[
                                    self._strip_quotes(net_neighborhood_df['semanticType']) == target_concept, 'weight'
                                ].sum()
                            )
                            net_weights[i] += weight_net
    
                            # Special degrees
                            degree_net = (
                                net_specialdegree_df.loc[
                                    self._strip_quotes(net_specialdegree_df['semanticType']) == system_concept, 'specialDegree'
                                ].sum()
                                + net_specialdegree_df.loc[
                                    self._strip_quotes(net_specialdegree_df['semanticType']) == target_concept, 'specialDegree'
                                ].sum()
                            )
                            net_degrees[i] += degree_net
    
                            # Distance adjustments
                            distance_row = df_net_distance.loc[
                                self._strip_quotes(df_net_distance['concept']) == system_concept
                            ]
                            if not distance_row.empty:
                                distance_vector = np.array(distance_row.iloc[0]['distance'], dtype=float)
                                net_distances[i] += 1 / (1 + distance_vector[i])
                                concept_counts_net[i] += 1
    
                    # Compute averages
                    net_weights /= np.maximum(concept_counts_net, 1)
                    net_degrees /= np.maximum(concept_counts_net, 1)
                    net_distances /= np.maximum(concept_counts_net, 1)
    
                    # Compute the adjusted embedding
                    with np.errstate(divide='ignore', invalid='ignore'):
                        net_value = net_distances + np.log(net_weights + 1) / (net_degrees + 1e-6)
                    net_embeddings.append(net_value)
    
            # Aggregate embeddings
            if kb_embeddings:
                if self.aggregation_method == "sum":
                    kb_aggregated = np.sum(kb_embeddings, axis=0)
                elif self.aggregation_method == "average":
                    kb_aggregated = np.mean(kb_embeddings, axis=0)
    
                if include_spatial:
                    aggregated_array = np.concatenate([kb_aggregated, np.sum(net_embeddings, axis=0)])
                else:
                    aggregated_array = kb_aggregated
            else:
                aggregated_array = np.sum(net_embeddings, axis=0) if include_spatial else np.zeros(kb_embedding_size)
    
            # Include temporal (average_date) if applicable
            if include_temporal:
                aggregated_array = np.append(aggregated_array, row['average_date'])
    
            aggregated_embeddings.append(aggregated_array)
    
        print("Observation embeddings computed successfully.")
        return np.array(aggregated_embeddings)


