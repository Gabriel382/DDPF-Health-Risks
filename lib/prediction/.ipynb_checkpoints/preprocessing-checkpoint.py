from prediction.utils import  getTopMax
import pandas as pd
import glob

class PredictionModulePreprocessingHandler():

    def __init__(self):
        return

    def getObservationDictionary(self, path_to_observations):
        observation_dict = {}
        for file in glob.glob(path_to_observations + "*.csv"):
            source = file.split("/")[-1].replace("_Observation.csv", "")
            source_prefix = '_'.join(source.split("_")[:2])
            df_observation = pd.read_csv(file)
            # Create a new column 'Observation_Name' without modifying 'System_Name'
            df_observation['Observation_Name'] = source_prefix + '_' + df_observation['System_Name'].astype(
                str).str.replace('"', '', regex=False)
            observation_dict[source] = df_observation
        return observation_dict

    def preprocessListOfREClusters(self, predictionModule,listOfRE, topNB=None):
        list_of_df = []
        for re in listOfRE:
            list_of_df.append(predictionModule.getAllObservationsFromMethod(re))
        if topNB is not None:
            for df in list_of_df:
                df = getTopMax(df,top=topNB)
        # Getting return types
        df_observations = list_of_df[0]
        dict_re = {}
        dict_re[listOfRE[0]] = list_of_df[0]
        for i in range(1,len(list_of_df)):
            df_observations = pd.concat([df_observations,list_of_df[i]])
            dict_re[listOfRE[i]] = list_of_df[i]
        return df_observations,dict_re