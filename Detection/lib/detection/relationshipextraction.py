import nltk
from nltk.corpus import stopwords
import string
import os
from gatenlp import Document
from gatenlp.processing.gazetteer import TokenGazetteer, StringGazetteer
# create a tokenizer based on the NLTK WordPunctTokenizer. 
from gatenlp.processing.tokenizer import NLTKTokenizer
from nltk.tokenize.regexp import WordPunctTokenizer
from tqdm import tqdm
import pandas as pd

def cleanKeys(dictionary, clean_list):
    for c in clean_list:
        if c in dictionary:
            del dictionary[c]
    return dictionary

def CleanDicts(netdict,kbdict):
    nltk.download('stopwords')
    stopwords_list = stopwords.words('english')
    punctuation = [i for i in string.punctuation  ]
    clean_list = stopwords_list + punctuation
    netdict = cleanKeys(netdict,clean_list)
    kbdict = cleanKeys(kbdict,clean_list+list(netdict.keys()))
    return netdict, kbdict

class RelationMatrix():
    
    def __init__(self, matrix_id):
        self.matrix_dict = {}
        self.matrix_id = matrix_id
        
    def getValue(self,i,j):
        if (i,j) in self.matrix_dict.keys():
            return self.matrix_dict[(i,j)]
        else:
            return None
        
    def setValue(self,i,j,v):
        self.matrix_dict[(i,j)] = v
        
    def increaseBy(self,i,j,v):
        v0 = self.getValue(i,j)
        if v0 is None:
            v0 = 0
        self.setValue(i,j,v+v0)
        
class RMGenerator():
    
    def __init__(self, corpus,gateExtractor, gs):
        self.corpus = corpus
        self.gateExtractor = gateExtractor
        self.gs = gs
    
    def directTermMatching(self, matrix_id):
        rm = RelationMatrix(matrix_id)
        # Per document
        for doc in tqdm(self.corpus):
            pdoc = self.gs.gdoc2pdoc(doc)
            pdoc = self.gateExtractor.tokenizer(pdoc)
            pdoc = self.gateExtractor.tok_gaz(pdoc)
            # Making the rm links
            for kb_annotation in pdoc.annset().with_type("kb"):
                for network_annoation in pdoc.annset().with_type("network"):
                    rm.increaseBy(self.gateExtractor.dict_kb[kb_annotation.features['key']], 
                                self.gateExtractor.dict_network[network_annoation.features['key']],1)
            self.gs.del_resource(doc)
        return rm
    
    def paragraphTermMatching(self, matrix_id):
        assert 'annie' in self.gateExtractor.extra_pr.keys()
        rm = RelationMatrix(matrix_id)
        # Per document
        for doc in tqdm(self.corpus):
            # Run annie
            if len(self.gs.gdoc2pdoc(doc).text) <= 0:
                self.gs.del_resource(doc)
                continue
            self.gs.worker.run4Document(self.gateExtractor.extra_pr['annie'], doc)
            pdoc = self.gs.gdoc2pdoc(doc)            
            # Get network and kb
            pdoc = self.gateExtractor.tok_gaz(pdoc)
            # Get paragraph
            praragraphann = pdoc.annset('Original markups').with_type("paragraph")
            # For each paragraph
            for ann in praragraphann:
                # Making the rm links
                for kb_annotation in pdoc.annset().within(ann).with_type('kb'):
                    for network_annoation in pdoc.annset().within(ann).with_type('network'):
                        rm.increaseBy(self.gateExtractor.dict_kb[kb_annotation.features['key']], 
                                self.gateExtractor.dict_network[network_annoation.features['key']],1)
            self.gs.del_resource(doc)
        return rm
    
    def sentenceTermMatching(self, matrix_id):
        assert 'annie' in self.gateExtractor.extra_pr.keys()
        rm = RelationMatrix(matrix_id)
        # Per document
        for doc in tqdm(self.corpus):
            # Run annie
            if len(self.gs.gdoc2pdoc(doc).text) <= 0:
                self.gs.del_resource(doc)
                continue
            self.gs.worker.run4Document(self.gateExtractor.extra_pr['annie'], doc)
            pdoc = self.gs.gdoc2pdoc(doc)            
            # Get network and kb
            pdoc = self.gateExtractor.tok_gaz(pdoc)
            # Get paragraph
            sentenceann = pdoc.annset('').with_type("Sentence")
            # For each paragraph
            for ann in sentenceann:
                # Making the rm links
                for kb_annotation in pdoc.annset().within(ann).with_type('kb'):
                    for network_annoation in pdoc.annset().within(ann).with_type('network'):
                        rm.increaseBy(self.gateExtractor.dict_kb[kb_annotation.features['key']], 
                                self.gateExtractor.dict_network[network_annoation.features['key']],1)
            self.gs.del_resource(doc)
        return rm
    
class RelationshipDiscovery():
    
    def __init__(self,corpus, gateExtractor, gs, rmGen=None):
        self.corpus = corpus
        self.gateExtractor = gateExtractor
        if rmGen is not None:
            self.rmGen = rmGen
            assert self.rmGen.corpus == self.corpus
            assert self.rmGen.gateExtractor == self.gateExtractor
        else:
            self.rmGen = RMGenerator(self.corpus, self.gateExtractor, gs)
            
class GateExtractor():
    
    def __init__(self, dict_kb, dict_network, extra_pr=None):
        self.tokenizer = NLTKTokenizer(
            nltk_tokenizer=WordPunctTokenizer(), 
            token_type="Token", outset_name="")
        self.dict_kb = dict_kb
        self.dict_network = dict_network
        print('Creating KB gazetteer...')
        self.kb_gazetteer = self.gazetteer_creator(self.dict_kb.keys())
        print('Creating Network gazetteer...')
        self.network_gazetteer = self.gazetteer_creator(self.dict_network.keys())
        print('Creating Merging gazetteer...')
        self.tok_gaz = TokenGazetteer(longest_only=False,
                          skip_longest=False, outset_name="", ann_type="Lookup",
                          annset_name="", token_type="Token")
        self.tok_gaz.append(source=self.kb_gazetteer, source_fmt="gazlist", list_type="kb")
        self.tok_gaz.append(source=self.network_gazetteer, source_fmt="gazlist", list_type="network")
        if extra_pr is None:
            self.extra_pr = {}
        else:
            self.extra_pr = extra_pr
        self.extra_pr['tokenizer'] = self.tokenizer
        self.extra_pr['tok_gaz'] = self.tok_gaz
        
    def _text2tokenstrings(self, text):
        tmpdoc = Document(text)
        self.tokenizer(tmpdoc)
        tokens = list(tmpdoc.annset().with_type("Token"))
        return [tmpdoc[tok] for tok in tokens]
    
    def gazetteer_creator(self, list_of_entries):
        return [(self._text2tokenstrings(txt),
                            {'key' : txt}) for txt in tqdm(list_of_entries)]
    
def rmToRelationCSV(rm, source_value, trustworthiness_value, typeof_value, cluster_date=None):
    list_start = []
    list_end = []
    date = []
    source = []
    trustworthiness = []
    typeof = []
    intensity = []
    if cluster_date is None:
        cluster_date = '-'.join(rm.matrix_id.split('-')[1:])
    for key in list(rm.matrix_dict.keys()):
        c1 = key[0]
        c2 = key[1]
        intentisyValue = rm.matrix_dict[key]
        for c1id in c1.list_of_ids:
            for c2id in c2.list_of_ids:
                list_start.append(c1id)
                list_end.append(c2id)
                intensity.append(intentisyValue)
    date = [cluster_date] * len(list_start)
    source = [source_value] * len(list_start)
    trustworthiness = [trustworthiness_value] * len(list_start)
    typeof = [typeof_value] * len(list_start)
    df = pd.DataFrame(data={
        ":START_ID" : list_start,
        ":END_ID" : list_end,
        ":TYPE" : typeof,
        "date" : date,
        "source" : source,
        "trustworthiness" : trustworthiness,
        "intensity" : intensity
    })
    return df