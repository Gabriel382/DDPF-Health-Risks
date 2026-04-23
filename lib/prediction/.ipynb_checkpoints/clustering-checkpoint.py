# Custom library
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

class ObservationClusteringSubModule():
    
    def __init__(self, clusteringFunction, predictionModule):
        self._applyClustering = clusteringFunction
        self.predictionModule = predictionModule

    def applyClustering(self, embeddingObsMatrix, df_obs,
                        nb_clusters=2,
                        random_state=0):
        return self._applyClustering(embeddingObsMatrix,df_obs,nb_clusters,random_state)

class KMeansClustering(ObservationClusteringSubModule):

    def __init__(self, predictionModule,
                 dimensionReduction=True,
                 tsne_random_state=9,
                 tsne_n_components=2):
        super().__init__(self._KMeansEmbedding, predictionModule)
        self.tsne_random_state=tsne_random_state
        self.tsne_n_components = tsne_n_components
        if dimensionReduction:
            self.tsne = TSNE(n_components=tsne_n_components, random_state=tsne_random_state)

    def _KMeansEmbedding(self,embeddingObsMatrix, df_obs,
                        nb_clusters=2,
                        random_state=0):
        obs_set = set([])
        list_indexes = list(range(embeddingObsMatrix.shape[0]))# ALL LIST
        
        n_init="auto"
        
        n_samples = embeddingObsMatrix.shape[0]
        perplexity = max(1, min(30, n_samples // 3))
        if n_samples<=30.0:
            self.tsne = TSNE(n_components=self.tsne_n_components, random_state=self.tsne_random_state,perplexity=perplexity)
        else:
            self.tsne = TSNE(n_components=self.tsne_n_components, random_state=self.tsne_random_state)
        matrix = self.tsne.fit_transform(embeddingObsMatrix)
        kmeans = KMeans(n_clusters=nb_clusters, random_state=random_state, n_init=n_init).fit(matrix)
        list_type_0 = []
        list_type_1 = []
        for index, row in df_obs.iterrows():
            if "COVID" in row["Observation_Name"]:
                list_type_1.append(0)
                list_type_0.append(1)
            else:
                list_type_1.append(1)
                list_type_0.append(0)
        list_type = list_type_1
        report = classification_report(list_type_1, kmeans.labels_, output_dict=True)
        score = report['weighted avg']['f1-score']
        report = classification_report(list_type_0, kmeans.labels_, output_dict=True)
        if(score < report['weighted avg']['f1-score']):
            score = report['weighted avg']['f1-score']
            list_type = list_type_0
        return score,kmeans.labels_, list_type