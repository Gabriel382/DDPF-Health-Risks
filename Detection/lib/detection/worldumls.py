from detection.schema import *

def umlsConceptCleanner(concept : Concept):
    terms_to_remove = []
    terms_to_add = []
    for term in concept.list_of_terms:
        label = str(term.label)
        if len(label) == 0:
            terms_to_remove.append(term)
        if len(label.split(':')) > 1:
            terms_to_remove.append(term)
            for l in label.split(':'):
                newterm = Term(l)
                terms_to_add.append(newterm)
    for term in terms_to_remove:
        concept.list_of_terms.remove(term)
    concept.list_of_terms += terms_to_add
    return concept

def isEnglish(s):
    return s.isascii()

def worldConceptCleanner(concept : Concept):
    terms_to_remove = []
    terms_to_add = []
    for term in concept.list_of_terms:
        label = str(term.label)
        if not isEnglish(label):
            terms_to_remove.append(term)
            continue
        if len(label.split(':')) > 1:
            terms_to_remove.append(term)
            for l in label.split(':'):
                newterm = Term(l)
                terms_to_add.append(newterm)
    for term in terms_to_remove:
        concept.list_of_terms.remove(term)
    concept.list_of_terms += terms_to_add
    return concept

def ClearnWorldKGGazetteer(df_network,clear_net_list):
    df_network = df_network.drop(df_network[(df_network["Name"]=='"China"') & (df_network["ID"]!="wkg:424313582")].index)
    df_network = df_network[~df_network['Name'].isin(clear_net_list)]
    return df_network

