class ObservationEmbeddingSubModule:
    def __init__(self, embeddingFunction, predictionModule, embeddingLoader):
        self._applyEmbedding = embeddingFunction
        self._loadEmbeddings = embeddingLoader
        self.predictionModule = predictionModule

    def loadEmbeddings(self):
        """
        Public method to load KB and network embeddings. Returns DataFrames.
        """
        return self._loadEmbeddings()

    def applyEmbedding(self, df_obs, df_kb, df_net):
        """
        Public method to compute embeddings given the observation and embedding DataFrames.
        """
        return self._applyEmbedding(df_obs, df_kb, df_net)