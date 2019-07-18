# ScriptName : DocumentRetrievalModel.py
# Description : Script preprocesses article and question to computer TFIDF.
#               Additionally, helps in answer processing 
# Arguments : 
#       Input :
#           question(list)        : List of question
#           useStemmer(boolean)     : Indicate to use stemmer for word tokens
#           removeStopWord(boolean) : Indicate to remove stop words from 
#                                     question in order to keep relevant words
#       Output :
#           Instance of DocumentRetrievalModel with following structure
#               query(function) : Take instance of processedQuestion and return
#                                 answer based on IR and Answer Processing
#                                 techniques

# Importing Library
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tree import Tree
from nltk import pos_tag,ne_chunk
# from DateExtractor import extractDate
from models import GetAnswer
import json
import math
import re

class AnswerRetrievalModel:
    def __init__(self,question, removeStopWord = False,useStemmer = False):
        self.idf = {}               # dict to store IDF for words in question
        self.questionInfo = {}     # structure to store questionVector
        self.question = question
        self.totalQuestions = len(question)
        self.stopwords = stopwords.words('english')
        self.removeStopWord = removeStopWord
        self.useStemmer = useStemmer
        self.vData = None
        self.stem = lambda k:k.lower()
        if(useStemmer):
            ps = PorterStemmer()
            self.stem = ps.stem
            
        # Initialize
        self.computeTFIDF()
        
    # Return term frequency for question
    # Input:
    #       question(str): question as a whole in string format
    # Output:
    #       wordFrequence(dict) : Dictionary of word and term frequency
    def getTermFrequencyCount(self,question):
        sentences = sent_tokenize(  question)
        wordFrequency = {}
        for sent in sentences:
            for word in word_tokenize(sent):
                if self.removeStopWord == True:
                    if word.lower() in self.stopwords:
                        #Ignore stopwords
                        continue
                    if not re.match(r"[a-zA-Z0-9\-\_\\/\.\']+",word):
                        continue
                #Use of Stemmer
                if self.useStemmer:
                    word = self.stem(word)
                    
                if word in wordFrequency.keys():
                    wordFrequency[word] += 1
                else:
                    wordFrequency[word] = 1
        return wordFrequency
    
    # Computes term-frequency inverse document frequency for every token of each
    # question
    # Output:
    #       questionInfo(dict): Dictionary for every question with following 
    #                            keys
    #                               vector : dictionary of TFIDF for every word
    def computeTFIDF(self):
        # Compute Term Frequency
        self.questionInfo = {}
        for index in range(0,len(self.question)):
            wordFrequency = self.getTermFrequencyCount(self.question[index])
            self.questionInfo[index] = {}
            self.questionInfo[index]['wF'] = wordFrequency
        
        wordQuestionFrequency = {}
        for index in range(0,len(self.questionInfo)):
            for word in self.questionInfo[index]['wF'].keys():
                if word in wordQuestionFrequency.keys():
                    wordQuestionFrequency[word] += 1
                else:
                    wordQuestionFrequency[word] = 1
        
        self.idf = {}
        for word in wordQuestionFrequency:
            # Adding Laplace smoothing by adding 1 to total number of documents
            self.idf[word] = math.log((self.totalQuestions+1)/wordQuestionFrequency[word])
        
        #Compute Question Vector
        for index in range(0,len(self.questionInfo)):
            self.questionInfo[index]['vector'] = {}
            for word in self.questionInfo[index]['wF'].keys():
                self.questionInfo[index]['vector'][word] = self.questionInfo[index]['wF'][word] * self.idf[word]
    

    # To find answer to the question by first finding relevant Question, then
    # by finding relevant sentence and then by procssing sentence to get answer
    # based on expected answer type
    # Input:
    #           pQ(ProcessedQuestion) : Instance of ProcessedQuestion
    # Output:
    #           answer(str) : Response of QA System
    def query(self,pQ):
        
        # Get relevant Question
        relevantQuestion = self.getSimilarQuestion(pQ.qVector)

        # Get All sentences
        sentences = []
        for tup in relevantQuestion:
            if tup != None:
                p2 = self.question[tup[0]]
                sentences.extend(sent_tokenize(p2))
        
        # Get Relevant Sentences
        if len(sentences) == 0:
            return "Oops! Unable to find answer"

        # Get most relevant sentence using unigram similarity
        relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)

        # AnswerType
        aType = pQ.aType
        
        # Default question matched
        question_matched = relevantSentences[0][0]
        id = self.question.index(question_matched) + 1
        get_ans = GetAnswer()
        ans = get_ans.add_row(id = id)

        return ans[0]
        
    # Get top 3 relevant Question based on cosine similarity between question 
    # vector and Question vector
    # Input :
    #       queryVector(dict) : Dictionary of words in question with their 
    #                           frequency
    # Output:
    #       pRanking(list) : List of tuple with top 3 Question with its
    #                        similarity coefficient
    def getSimilarQuestion(self,queryVector):    
        queryVectorDistance = 0
        for word in queryVector.keys():
            if word in self.idf.keys():
                queryVectorDistance += math.pow(queryVector[word]*self.idf[word],2)
        queryVectorDistance = math.pow(queryVectorDistance,0.5)
        if queryVectorDistance == 0:
            return [None]
        pRanking = []
        for index in range(0,len(self.questionInfo)):
            sim = self.computeSimilarity(self.questionInfo[index], queryVector, queryVectorDistance)
            pRanking.append((index,sim))
        
        return sorted(pRanking,key=lambda tup: (tup[1],tup[0]), reverse=True)[:3]
    
    # Compute cosine similarity betweent queryVector and questionVector
    # Input:
    #       pInfo(dict)         : Dictionary containing wordFrequency and 
    #                             question Vector
    #       queryVector(dict)   : Query vector for question
    #       queryDistance(float): Distance of queryVector from origin
    # Output:
    #       sim(float)          : Cosine similarity coefficient
    def computeSimilarity(self, pInfo, queryVector, queryDistance):
        # Computing pVectorDistance
        pVectorDistance = 0
        for word in pInfo['wF'].keys():
            pVectorDistance += math.pow(pInfo['wF'][word]*self.idf[word],2)
        pVectorDistance = math.pow(pVectorDistance,0.5)
        if(pVectorDistance == 0):
            return 0

        # Computing dot product
        dotProduct = 0
        for word in queryVector.keys():
            if word in pInfo['wF']:
                q = queryVector[word]
                w = pInfo['wF'][word]
                idf = self.idf[word]
                dotProduct += q*w*idf*idf
        
        sim = dotProduct / (pVectorDistance * queryDistance)
        return sim
    
    # Get most relevant sentences using unigram similarity between question
    # sentence and sentence in question containing potential answer
    # Input:
    #       sentences(list)      : List of sentences in order of occurance as in
    #                              question
    #       pQ(ProcessedQuestion): Instance of processedQuestion
    #       nGram(int)           : Value of nGram (default 3)
    # Output:
    #       relevantSentences(list) : List of tuple with sentence and their
    #                                 similarity coefficient
    def getMostRelevantSentences(self, sentences, pQ, nGram=3):
        relevantSentences = []
        for sent in sentences:
            sim = 0
            if(len(word_tokenize(pQ.question))>nGram+1):
                sim = self.sim_ngram_sentence(pQ.question,sent,nGram)
            else:
                sim = self.sim_sentence(pQ.qVector, sent)
            relevantSentences.append((sent,sim))
        
        return sorted(relevantSentences,key=lambda tup:(tup[1],tup[0]),reverse=True)
    
    # Compute ngram similarity between a sentence and question
    # Input:
    #       question(str)   : Question string
    #       sentence(str)   : Sentence string
    #       nGram(int)      : Value of n in nGram
    # Output:
    #       sim(float)      : Ngram Similarity Coefficient
    def sim_ngram_sentence(self, question, sentence,nGram):
        #considering stop words as well
        ps = PorterStemmer()
        getToken = lambda question:[ ps.stem(w.lower()) for w in word_tokenize(question) ]
        getNGram = lambda tokens,n:[ " ".join([tokens[index+i] for i in range(0,n)]) for index in range(0,len(tokens)-n+1)]
        qToken = getToken(question)
        sToken = getToken(sentence)

        if(len(qToken) > nGram):
            q3gram = set(getNGram(qToken,nGram))
            s3gram = set(getNGram(sToken,nGram))
            if(len(s3gram) < nGram):
                return 0
            qLen = len(q3gram)
            sLen = len(s3gram)
            sim = len(q3gram.intersection(s3gram)) / len(q3gram.union(s3gram))
            return sim
        else:
            return 0
    
    # Compute similarity between sentence and queryVector based on number of 
    # common words in both sentence. It doesn't consider occurance of words
    # Input:
    #       queryVector(dict)   : Dictionary of words in question
    #       sentence(str)       : Sentence string
    # Ouput:
    #       sim(float)          : Similarity Coefficient    
    def sim_sentence(self, queryVector, sentence):
        sentToken = word_tokenize(sentence)
        ps = PorterStemmer()
        for index in range(0,len(sentToken)):
            sentToken[index] = ps.stem(sentToken[index])
        sim = 0
        for word in queryVector.keys():
            w = ps.stem(word)
            if w in sentToken:
                sim += 1
        return sim/(len(sentToken)*len(queryVector.keys()))
    
    # Get Named Entity from the sentence in form of PERSON, GPE, & ORGANIZATION
    # Input:
    #       answers(list)       : List of potential sentence containing answer
    # Output:
    #       chunks(list)        : List of tuple with entity and name in ranked 
    #                             order
    def getNamedEntity(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            nc = ne_chunk(pos_tag(answerToken))
            entity = {"label":None,"chunk":[]}
            for c_node in nc:
                if(type(c_node) == Tree):
                    if(entity["label"] == None):
                        entity["label"] = c_node.label()
                    entity["chunk"].extend([ token for (token,pos) in c_node.leaves()])
                else:
                    (token,pos) = c_node
                    if pos == "NNP":
                        entity["chunk"].append(token)
                    else:
                        if not len(entity["chunk"]) == 0:
                            chunks.append((entity["label"]," ".join(entity["chunk"])))
                            entity = {"label":None,"chunk":[]}
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["label"]," ".join(entity["chunk"])))
        return chunks
    
    # To get continuous chunk of similar POS tags.
    # E.g.  If two NN tags are consequetive, this method will merge and return
    #       single NN with combined value.
    #       It is helpful in detecting name of single person like John Cena, 
    #       Steve Jobs
    # Input:
    #       answers(list) : list of potential sentence string
    # Output:
    #       chunks(list)  : list of tuple with entity and name in ranked order
    def getContinuousChunk(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            if(len(answerToken)==0):
                continue
            nc = pos_tag(answerToken)
            
            prevPos = nc[0][1]
            entity = {"pos":prevPos,"chunk":[]}
            for c_node in nc:
                (token,pos) = c_node
                if pos == prevPos:
                    prevPos = pos       
                    entity["chunk"].append(token)
                elif prevPos in ["DT","JJ"]:
                    prevPos = pos
                    entity["pos"] = pos
                    entity["chunk"].append(token)
                else:
                    if not len(entity["chunk"]) == 0:
                        chunks.append((entity["pos"]," ".join(entity["chunk"])))
                        entity = {"pos":pos,"chunk":[token]}
                        prevPos = pos
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks
    
    def getqRev(self, pq):
        if self.vData == None:
            # For testing purpose
            self.vData = json.loads(open("validatedata.py","r").readline())
        revMatrix = []
        for t in self.vData:
            sent = t["q"]
            revMatrix.append((t["a"],self.sim_sentence(pq.qVector,sent)))
        return sorted(revMatrix,key=lambda tup:(tup[1],tup[0]),reverse=True)[0][0]
        
    def __repr__(self):
        msg = "Total Questions " + str(self.totalQuestions) + "\n"
        msg += "Total Unique Word " + str(len(self.idf)) + "\n"
        msg += str(self.getMostSignificantWords())
        return msg