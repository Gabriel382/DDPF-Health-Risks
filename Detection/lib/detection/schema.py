import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
import glob

class Term:
    """A Term is a singleword or a multiword 
    string that refers to a single unit of knowledge.
    They represent the words of interest in the corpus.
    """
    def __init__(self, label : str) -> None:
        self.label = label
        self.termRepresentation = None
        
    def termRepresentationFunction(self, representationFunction) -> None:
        """Updates de termRepresentation variable

        Parameters
        ----------
        representationFunction : Function
            Function that extracts the representation of the term
        """
        self.termRepresentation = representationFunction(self.label)
        
    def similarityValue(self,similarityFunction, otherTerm) -> float:
        """Retrieves the similarity value out of two terms

        Parameters
        ----------
        similarityFunction : Function
            Function for similarity retrieval
        otherTerm : Term
            Second term for the similarity function
        Returns
        ----------
        Value of similarity between terms
        """
        assert self.termRepresentation is not None
        assert otherTerm.termRepresentation is not None
        return similarityFunction(self.termRepresentation,otherTerm.termRepresentation)
    
class Concept:
    """It is a conceptualization of an unit of 
    knowledge that may have multiple Terms associated with.
    """
    def __init__(self, list_of_terms=None,list_of_ids=None):
        if list_of_terms is None:
            self.list_of_terms = []
        else:
            self.list_of_terms = list_of_terms
        if list_of_ids is None:
            self.list_of_ids = []
        else:
            self.list_of_ids = list_of_ids
    
    def setOfTermStrings(self,cleaningFunction=None):
        """Returns a clean list of all term's strings

        Parameters
        ----------
        cleaningFunction : Function
            Function for normalizing and cleaning every string if necessary
        Returns
        ----------
        Cleanned string list
        """
        termList = list(set([term.label for term in self.list_of_terms]))
        if cleaningFunction is not None:
            for i in range(len(termList)):
                termList[i] = cleaningFunction(termList[i])
        return termList
    
def df_to_concepts(df):
    dict_id = {}
    dict_concept = {}
    print("Finding Terms")
    for index, row in tqdm(df.iterrows()):
        if row["ID"] in dict_id:
            inDict = False
            # Check for duplicatas
            for t in dict_id[row["ID"]].list_of_terms:
                if row["Name"] == t.label:
                    inDict = True
                    break
            # If no duplicatas
            if inDict == False:
                newTerm = Term(row["Name"])
                dict_id[row["ID"]].list_of_terms.append(newTerm)
                dict_concept[row["Name"]] = dict_id[row["ID"]]
        elif row["Name"] in dict_concept:
            dict_concept[row["Name"]].list_of_ids.append(row["ID"])
            dict_id[row["ID"]] = dict_concept[row["Name"]]
        else:
            newterm = Term(row["Name"])
            newconcept = Concept([newterm],[row["ID"]])
            dict_concept[row["Name"]] = newconcept
            dict_id[row["ID"]] = newconcept
    print("Creating Term list")
    listset = set()
    for key in dict_id:
        listset.add(dict_id[key])
    for key in dict_concept:
        listset.add(dict_concept[key])
    return list(listset)

def cleaningPlaceStr(string):
    return str(string).replace('"','')

def capPlaceStr(string):
    return cleaningPlaceStr(string).upper()

def lowerPlaceStr(string):
    return cleaningPlaceStr(string).lower()

def conceptsToGazetteer(concept_list,path_to_list, cleanningFunction=None):
    """Save the list of concetps into a gazetteer, while it returns the dict{term:concept}

        Parameters
        ----------
        concept_list : List of Concepts
            List of all concepts that are going to make part of the gazetteer
        cleaningFunction : Function
            Function for normalizing and cleaning every string if necessary
        Returns
        ----------
        Saves file into path and returns dictionary of terms from the concepts having their equivalent concept as value
    """
    list_str = ""
    dict_term_concept = {}
    for concept in tqdm(concept_list):
        for term in concept.setOfTermStrings(cleanningFunction):
            list_str += term + "\n"
            dict_term_concept[term] = concept
    with open(path_to_list, "w") as text_file:
        text_file.write(list_str)
    return dict_term_concept

def cleanKeys(dictionary, clean_list):
    for c in clean_list:
        if c in dictionary:
            del dictionary[c]
    return dictionary