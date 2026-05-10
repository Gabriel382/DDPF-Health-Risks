from sklearn.metrics import f1_score
from tqdm import tqdm
from src.ddpf.deprecated.lib.prediction.preprocessing import PredictionModulePreprocessingHandler
from src.ddpf.deprecated.lib.prediction.embedding.semantic import ObservationWangPipeline
from src.ddpf.deprecated.lib.prediction.utils import *
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

# Global variables

PHENOMENON_DATASETS = {
        "Monkeypox" : [("Journal","2022-05"),("Medical","2022-06"),("Social","2022-05")],
        "COVID" : [("Journal","2019-11"),("Medical","2019-12"),("Social","2020-02")]
}
PHENOMENON_IDS = {
        "Monkeypox" : "C0276180",
        "COVID" : "C5203670"
}

# Classes
class PredictionModule():
    
    def __init__(self,
                path_to_kb_gazetteer,
                path_to_netwoork_gazetteer,
                path_to_observationcsv,
                dict_embedders={},
                dict_clusteres={}):
        # assigning paths
        self.path_to_kb_gazetteer = path_to_kb_gazetteer
        self.path_to_netwoork_gazetteer = path_to_netwoork_gazetteer
        self.path_to_observationcsv = path_to_observationcsv
        
        # Getting all observations from all pairs of (source, RE method) into a dictionary
        preprocessingHandler = PredictionModulePreprocessingHandler()
        self.observation_dict = preprocessingHandler.getObservationDictionary(path_to_observationcsv)
        self.df_observations, self.re_dict = preprocessingHandler.preprocessListOfREClusters(
            self,
            ["DocumentMatching","ParagraphMatching", "SentenceMatching"],
            25)
        self.dict_embedders = dict_embedders
        self.dict_clusteres = dict_clusteres
        self.bench = BenchMark()

    def _getTrueIndexes(self,phenomenonSourceName, date,df_observations,df_result):
        jm = df_observations[
        df_observations['Observation_Name'].str.contains(phenomenonSourceName)]
        jm = jm[jm["System_ID"].isin(df_result[df_result["Date"]==date]['Country'].tolist())]
        return jm.index.tolist()

    def getFilteredDF(self,neowrapper,re_method,typePerSourceDateDict, sourceIDDict):
        index = []
        phenomenonIndexDict = {}
        for key in typePerSourceDateDict:
            strQuery = """MATCH (b:CUI)-[r:isReported]->(c:Country) WHERE b.id = "{0}" return c.id as Country, r.date as Date;""".format(
                sourceIDDict[key])
            result = neowrapper.sendQuery([strQuery])
            df_result_source = result[0]
            df_result_source["Date"] = df_result_source["Date"].apply(lambda x: x[:7].lower())
            df_result_source = df_result_source.drop_duplicates()
            typeindex = []
            for sourceDateTuple in typePerSourceDateDict[key]:
                typeindex += self._getTrueIndexes(sourceDateTuple[0]+"_"+key, sourceDateTuple[1],
                                                     self.re_dict[re_method],df_result_source)
            phenomenonIndexDict[key] = typeindex
            index += typeindex
        return index,phenomenonIndexDict

    def getAllObservationsFromMethod(self, method):
        df_observations = None
        for key in self.observation_dict.keys():
            if method in key:
                if df_observations is None:
                    df_observations = self.observation_dict[key]
                else:
                    df_observations = pd.concat([df_observations,self.observation_dict[key]], ignore_index=True)
        return df_observations

    def RunBenchMark(self, filtered=False,neowrapper=None,include_spatial=True, include_temporal=True):
        for embedders in self.dict_embedders: # for each embedding
            print("Setting up for " + embedders)
            df_kb, df_net = self.dict_embedders[embedders].loadEmbeddings()
            for re in self.re_dict.keys(): # for each re
                score = 0
                studiedRE = self.re_dict[re]
                if filtered: # if filter
                    studiedRE = studiedRE.loc[self.getFilteredDF(neowrapper, re, PHENOMENON_DATASETS, PHENOMENON_IDS)[0]].reset_index(drop=True)
                embeddingMatrix = self.dict_embedders[embedders].applyEmbedding(studiedRE,df_kb, df_net,
                                                                                          include_spatial=include_spatial,
                                                                                           include_temporal=include_temporal)
                for cluter_method in self.dict_clusteres: # for each clustering method
                    print("Benchmark for " + str((re, embedders, cluter_method)))
                    clusteringResults = self.dict_clusteres[cluter_method].applyClustering(embeddingMatrix, studiedRE)
                    score = clusteringResults[0]
                    self.bench.dict_bench[(re, embedders, cluter_method)] = score
            del df_kb
            del df_net
            
    def RunWangBenchMark(self, kb_concept_path,net_concept_path,
                        kb_sa_path, net_sa_path,
                        kb_sva_path, net_sva_path,
                         filtered=False,neowrapper=None,include_spatial=True, include_temporal=True):
        wang = ObservationWangPipeline(self,kb_concept_path,net_concept_path,
                        kb_sa_path, net_sa_path,
                        kb_sva_path, net_sva_path)
        df_kb, df_net, df_kb_sa, df_net_sa, df_kb_sva, df_net_sva = wang._loadEmbeddingsInternal()
        for re in self.re_dict.keys(): # for each re
            studiedRE = self.re_dict[re]
            if filtered: # if filter
                studiedRE = studiedRE.loc[self.getFilteredDF(neowrapper, re, PHENOMENON_DATASETS, PHENOMENON_IDS)[0]].reset_index(drop=True)
            pipeline_methods = (re, "Wang", "K-Medoids")
            print("Benchmark for " + str(pipeline_methods))
            embeddingMatrix = wang.computeEmbedding(studiedRE, df_kb, 
                                        df_net, df_kb_sa,
                                        df_net_sa,
                                      include_spatial=include_spatial,
                                       include_temporal=include_temporal)
            kmedoids_labels, best_type, best_score = wang.clusterObservations(embeddingMatrix, studiedRE, k=2, 
                                                                  random_seed=42, include_temporal=include_temporal)
            self.bench.dict_bench[pipeline_methods] = best_score
        
class BenchMark():

    def __init__(self):
        self.dict_bench = {}
        return
    
    def plot(self, embeddingMatrix,legend_list, labels, apply_pca=False,apply_tsne=True, pca_random_state=1):
        if apply_pca:
            pca = PCA(2, random_state=pca_random_state)
            matrix = pca.fit_transform(embeddingMatrix)
        elif apply_tsne:
            perplexity = max(5, min(30, embeddingMatrix.shape[0] // 3))
            tsne = TSNE(n_components=2, random_state=pca_random_state)
            matrix = tsne.fit_transform(embeddingMatrix)
        else:
            matrix = embeddingMatrix
        u_labels = np.unique(labels)
        for i in u_labels:
            plt.scatter(matrix[labels == i , 0] , matrix[labels == i , 1] , label = i)
        # Label points
        for (i,j, legend) in zip(matrix[:,0], matrix[:,1],legend_list):
            plt.text(i, j, legend)
        plt.legend()
        plt.show()
    
    def smartPrint(self, np_obslist, labels):
        u_labels = np.unique(labels)
        print('Clusters:')
        for i in range(len(u_labels)):
            print("clusters : " + str(u_labels[i]))
            print(np_obslist[labels==u_labels[i]])