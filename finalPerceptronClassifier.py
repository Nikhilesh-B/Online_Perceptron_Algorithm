from ctypes import sizeof
from sys import breakpointhook
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time
from dask import dataframe as df1 
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack
from nltk.corpus import stopwords
import csv


class classifier():
    def __init__(self, trainfile='./restaurant_reviews_data/mediumDataTrain.csv',testfile='./restaurant_reviews_data/minisculeDataTest.csv'):
        self.vocabulary = dict()
        self.totalWords = 0 
        self.UnigramMatrix = None
        self.docCount = 0
       

        self.totalTrials = 0
        self.correctlyClassifiedUnigram = 0
        self.correctlyClassifiedTFIDF = 0
        self.correctlyClassifiedBigram = 0
        self.stopwords = set(stopwords.words('english'))
        self.wordCounts = dict()
        self.trainingFile = trainfile
        self.testingFile = testfile
        self.bigramCounts = dict()
        self.bigramVocab = dict()
        self.bigramMatrix = None



        self.classificationUnigram = dict()
        self.classificationBigram = dict()

    def computeVocab(self):
        counts  = dict()
        file = open(self.trainingFile)
        csvreader = csv.reader(file)
        header = next(csvreader)

        i = 0 
        start_time = time()

        wordSet = set()
        bigramSet = set()
        wordList = []
        bigramList = []
        for row in csvreader: 
            document = row[1]
            sentenceList = document.split(" ")
            words = set(sentenceList)

            for word in words:
                if word not in wordSet and word not in self.stopwords: 
                    self.wordCounts[word] = 1
                    wordList.append(word)
                    wordSet.add(word)

                if word in wordSet:
                    self.wordCounts[word] += 1

            for i in range(0, len(sentenceList)-1):
                if(i+1<=len(sentenceList)-1):
                    word1 = sentenceList[i]
                    word2 = sentenceList[i+1]
                    bigram = (word1, word2)

                if (word1 not in self.stopwords) and (word2 not in self.stopwords) and bigram not in bigramSet:
                    self.bigramCounts[bigram] = 1
                    bigramList.append(bigram) 
                    bigramSet.add(bigram)

                if bigram in bigramSet:
                    self.bigramCounts[bigram] += 1 


            self.docCount = self.docCount+1
        
        importantWords = []
        importantBigrams = []

        for i in range(len(wordList)):
            word  = wordList[i]
            count = self.wordCounts[word]

            if(count>0.01*self.docCount):
                importantWords.append(word)


        for i in range(len(bigramList)):
            bigram = bigramList[i]
            count = self.bigramCounts[bigram]

            if(count>0.01*self.docCount):
                importantBigrams.append(bigram)

        for j in range(len(importantWords)):
            self.vocabulary[importantWords[j]] = j

        for j in range(len(importantBigrams)):
            self.bigramVocab[importantBigrams[j]] = j

        self.totalWords = len(importantWords)
        self.totalBigrams = len(importantBigrams)

        file.close()
        end_time = time()

        time_elapsed = end_time-start_time
        print("first loop")
        print(time_elapsed)



    def populateBigramMatrix(self):
        t1 = time()
        file = open(self.trainingFile)
        csvreader = csv.reader(file)
        header = next(csvreader)

        self.bigramMatrix = csr_matrix((1,self.totalBigrams+1))
        
        self.classificationBigram[tuple(np.zeros(self.totalBigrams+1))]=1
        for r in csvreader:
            document = r[1]
            classification  = int(r[0])
            sentence = document.split(" ")
            indexes = [self.totalBigrams]
            for i in range(0, len(sentence)-1):
                if(i+1<=len(sentence)-1):
                    word1 = sentence[i]
                    word2 = sentence[i+1]
                    bigram = (word1, word2)
                    if (bigram in self.bigramVocab):
                        bigramId = self.bigramVocab[bigram]
                        indexes.append(bigramId)

            indexes = np.array(indexes)
            b1 = np.bincount(indexes, minlength=self.totalBigrams+1)


            self.classificationBigram[tuple(b1)] = classification
            self.bigramMatrix = vstack([self.bigramMatrix, b1])
   
        file.close()

        t2 = time()
        
        print("Bigram Matrix population time")
        print(t2-t1)



    def populateMatrix(self):
        t1 = time()
        file = open(self.trainingFile)
        csvreader = csv.reader(file)
        header = next(csvreader)

        self.UnigramMatrix = csr_matrix((1,self.totalWords+1))
        self.classificationUnigram[tuple(np.zeros(self.totalWords+1))]=1

        for r in csvreader: 
            document = r[1]
            classification = int(r[0])
            sentence = document.split(" ")
            indexes = [self.totalWords]
            for word in sentence:
                if(word in self.vocabulary):
                    wordId = self.vocabulary[word]
                    indexes.append(wordId)

            indexes = np.array(indexes)
            b1 = np.bincount(indexes, minlength=self.totalWords+1)
            self.classificationUnigram[tuple(b1)] = classification
            self.UnigramMatrix = vstack([self.UnigramMatrix, b1])
            
   
        file.close()

        t2 = time()
        self.TFIDFhashmap()
        
        print("Matrix population time")
        print(t2-t1)
    

    def TFIDFhashmap(self):
        self.TFIDFhashMAP = dict()
    
        unigramColumns = self.UnigramMatrix.tocoo()

        for i in range(0,self.totalWords+1):
            wordArr = unigramColumns.getcol(i).toarray()[:,0]
            documentsWithTCount = np.count_nonzero(wordArr)
            if(documentsWithTCount == 0):
                TFIDF = 0
            else:
                TFIDF =  (self.docCount)/documentsWithTCount

            self.TFIDFhashMAP[i] = TFIDF

        
        self.UnigramMatrix =self.UnigramMatrix.tocsr()


    def initializeWeightVector(self):
        self.unigramWeightVector = np.zeros(self.totalWords+1)
        self.unigramWeightVectorHistory = [self.unigramWeightVector]
        self.unigramWeightDictHistory = {tuple(self.unigramWeightVector):0}

        self.TFIDFWeightVector = np.zeros(self.totalWords+1)
        self.TFIDFWeightVectorHistory  = [self.TFIDFWeightVector]
        self.TFIDFWeightDictHistory = {tuple(self.TFIDFWeightVector):0}

        self.bigramWeightVector = np.zeros(self.totalBigrams+1)
        self.bigramWeightVectorHistory  = [self.bigramWeightVector]
        self.bigramWeightDictHistory = {tuple(self.bigramWeightVector):0}


    def trainLinearBigramClassifier(self):
        trainingBigram = time()
        
        for docNo in range(0,self.docCount):
            documentFeatures = self.bigramMatrix.getrow(docNo+1).toarray()[0]
            groundTruthLabel = self.classificationBigram[tuple(documentFeatures)]

            if (groundTruthLabel==0):
                groundTruthLabel = -1

            if(np.sign(np.dot(documentFeatures,self.bigramWeightVector)) != groundTruthLabel):
                self.bigramWeightVector = self.bigramWeightVector+groundTruthLabel*documentFeatures
                self.bigramWeightVectorHistory.append(self.bigramWeightVector)
                self.bigramWeightDictHistory[tuple(self.bigramWeightVector)] = docNo
            

        finishedTrainingBigram = time()


        elapsedTimeForTrainingLinearClassifier = finishedTrainingBigram-trainingBigram

        print("Elapsed training bigram linear classifier", elapsedTimeForTrainingLinearClassifier)



   
    def trainLinearClassifier(self):
            trainingLinear = time()
        
            for docNo in range(0,self.docCount):
                documentFeatures = self.UnigramMatrix.getrow(docNo+1).toarray()[0]

                groundTruthLabel = self.classificationUnigram[tuple(documentFeatures)]


                if (groundTruthLabel==0):
                    groundTruthLabel = -1

                if(np.sign(np.dot(documentFeatures,self.unigramWeightVector)) != groundTruthLabel):
                    self.unigramWeightVector = self.unigramWeightVector+groundTruthLabel*documentFeatures
                    self.unigramWeightVectorHistory.append(self.unigramWeightVector)
                    self.unigramWeightDictHistory[tuple(self.unigramWeightVector)] = docNo


                tfidfFeatureVect = documentFeatures.copy()

                for i in range(len(tfidfFeatureVect)):
                    if documentFeatures[i] == 0:
                        continue
                    else:
                        TFIDF = self.TFIDFhashMAP[i]
                        logTFIDF = np.log10(TFIDF)
                        tfidfFeatureVect[i] = tfidfFeatureVect[i]*logTFIDF

                if(np.sign(np.dot(tfidfFeatureVect,self.TFIDFWeightVector)) != groundTruthLabel):
                    self.TFIDFWeightVector = self.unigramWeightVector+groundTruthLabel*tfidfFeatureVect
                    self.TFIDFWeightVectorHistory.append(self.TFIDFWeightVector)
                    self.TFIDFWeightDictHistory[tuple(self.TFIDFWeightVector)] = docNo

                

            finishedTrainingLinear = time()



            elapsedTimeForTrainingLinearClassifier = finishedTrainingLinear-trainingLinear

            print("Elapsed training linear classifier", elapsedTimeForTrainingLinearClassifier)

    

    def classifyBigram(self):
        file = open(self.testingFile)
        csvreader = csv.reader(file)
        header = next(csvreader)

        t1 = time()
        for r in csvreader:
            document = r[1]
            groundTruthLabel = int(r[0])
            sentence = document.split(" ")
            currentDocument = set()

            if(groundTruthLabel==0):
                groundTruthLabel = -1

            indexes = [self.totalBigrams]
            

            for i in range(0, len(sentence)-1):
                if(i+1<=len(sentence)-1):
                    word1 = sentence[i]
                    word2 = sentence[i+1]
                    bigram = (word1, word2)

                if (bigram in self.bigramVocab):
                    bigramId = self.bigramVocab[bigram]
                    indexes.append(bigramId)
            indexes = np.array(indexes)
            bigramFeatureVect = np.bincount(indexes, minlength=self.totalBigrams+1)

            if(np.sign(np.dot(bigramFeatureVect,self.bigramWeightVector)) == groundTruthLabel):
               self.correctlyClassifiedBigram+=1

            self.totalTrials+=1
         
        t2 = time()
        print("Bigram classification time")
        print(t2-t1)



    def onlineToBatchConversionUnigrams(self):
        threshold = self.docCount//2

        
    
        documentCounts = []
        finalUnigramWeightVector = np.zeros(self.totalWords+1)
        for vector in self.unigramWeightVectorHistory:
            documentCount = self.unigramWeightDictHistory[tuple(vector)]
            documentCounts.append(documentCount)


        previousDocumentCount = None
        j = 0 
        for i in range(len(self.unigramWeightVectorHistory)):
            documentCount = documentCounts[i]
            if documentCount > threshold and j==0:
                multiplier = documentCount - threshold
                previousDocumentCount = documentCount
                finalUnigramWeightVector = finalUnigramWeightVector+multiplier*self.unigramWeightVectorHistory[i]

                j+=1
            if documentCount>threshold and j!=0:
                multiplier = documentCount-previousDocumentCount
                finalUnigramWeightVector = finalUnigramWeightVector+multiplier*self.unigramWeightVectorHistory[i]

   
        self.unigramWeightVector = (1/threshold)*finalUnigramWeightVector
        self.unigramWeightVector2 = self.unigramWeightVector.copy()
        
        top_10_index_list = []
        for i in range(10):
            top_10_index_list.append(np.argmax(self.unigramWeightVector))
            self.unigramWeightVector[top_10_index_list[-1]] = -float('inf')
        
        print("top 10")
        for j in top_10_index_list:
            for key in self.vocabulary:
                if self.vocabulary[key] == j:
                    print(key)
                    
                    
        bottom_10_index_list = []
        
        print("bottom 10")
        for i in range(10):
            bottom_10_index_list.append(np.argmin(self.unigramWeightVector2))
            self.unigramWeightVector2[bottom_10_index_list[-1]] = float('inf')

        for j in bottom_10_index_list:
            for key in self.vocabulary:
                if self.vocabulary[key] == j:
                    print(key)
                    

    def onlineToBatchConversionBigrams(self):
        threshold = self.docCount//2

    
        documentCounts = []
        finalBigramWeightVector = np.zeros(self.totalBigrams+1)
        for vector in self.bigramWeightVectorHistory:
            documentCount = self.bigramWeightDictHistory[tuple(vector)]
            documentCounts.append(documentCount)


        previousDocumentCount = None
        j = 0 

        for i in range(len(self.bigramWeightVectorHistory)):
            documentCount = documentCounts[i]
            if documentCount > threshold and j==0:
                multiplier = documentCount - threshold
                previousDocumentCount = documentCount
                finalBigramWeightVector = finalBigramWeightVector+multiplier*self.bigramWeightVectorHistory[i]
                j+=1
            if documentCount>threshold and j!=0:
                multiplier = documentCount-previousDocumentCount
                finalBigramWeightVector = finalBigramWeightVector+multiplier*self.bigramWeightVectorHistory[i]


        self.bigramWeightVector = (1/threshold)*finalBigramWeightVector




    def onlineToBatchConversionTFIDF(self):
        threshold = self.docCount//2
    
        documentCounts = []
        finalTFIDFweightVector = np.zeros(self.totalWords+1)
        for vector in self.TFIDFWeightVectorHistory:
            documentCount = self.TFIDFWeightDictHistory[tuple(vector)]
            documentCounts.append(documentCount)

        previousDocumentCount = None
        j = 0 

        for i in range(len(self.TFIDFWeightVectorHistory)):
            documentCount = documentCounts[i]
            if documentCount > threshold and j==0:
                multiplier = documentCount - threshold
                previousDocumentCount = documentCount
                finalTFIDFweightVector = finalTFIDFweightVector+multiplier*self.TFIDFWeightVectorHistory[i]
                j+=1

            if documentCount>threshold and j!=0:
                multiplier = documentCount-previousDocumentCount
                finalTFIDFweightVector = finalTFIDFweightVector+multiplier*self.TFIDFWeightVectorHistory[i]

        
        self.TFIDFWeightVector = (1/threshold)*finalTFIDFweightVector

    
        
    


    def classify(self):

        classifierStartTime = time()

        self.TFIDFtestSet = dict()

        file = open(self.testingFile)

        csvreader = csv.reader(file)
        header = next(csvreader)
        self.totalTestDocuments = 0
        
        for r in csvreader:
            document = r[1]
            groundTruthLabel = int(r[0])
            sentence = document.split(" ")
            currentDocument = set()

            for word in sentence:
                if(word in self.vocabulary):
                    if(word not in currentDocument):
                        wordId = self.vocabulary[word]

                        if wordId in self.TFIDFtestSet:
                            self.TFIDFtestSet[wordId] += 1
                        else: 
                            self.TFIDFtestSet[wordId] = 1
                        currentDocument.add(word)
                    
            self.totalTestDocuments += 1
        file.close()



        file = open(self.testingFile)
        csvreader = csv.reader(file)
        header = next(csvreader)
        for r in csvreader:
            document = r[1]
            groundTruthLabel = int(r[0])
            sentence = document.split(" ")
            currentDocument = set()

            if(groundTruthLabel==0):
                groundTruthLabel = -1

            indexes = [self.totalWords]
            
            for word in sentence:
                if(word in self.vocabulary):
                    wordId = self.vocabulary[word]
                    indexes.append(wordId)
        

            indexes = np.array(indexes)
            featureVect = np.bincount(indexes, minlength=self.totalWords+1)

            tfidfFeatureVect = featureVect.copy()

            for i in range(len(tfidfFeatureVect)-1):
                    if featureVect[i] == 0:
                        continue
                    else:
                        totalDocumentswithWordT = self.TFIDFtestSet[i]
                        idf = (self.totalTestDocuments)/(totalDocumentswithWordT)
                        logTFIDF = np.log10(idf)
                        tfidfFeatureVect[i] = tfidfFeatureVect[i]*logTFIDF


            if(np.sign(np.dot(featureVect,self.unigramWeightVector)) == groundTruthLabel):
               self.correctlyClassifiedUnigram+=1

            if(np.sign(np.dot(tfidfFeatureVect,self.TFIDFWeightVector)) == groundTruthLabel):
               self.correctlyClassifiedTFIDF+=1

        file.close()
     
        classifierEndTime = time()

        classificationtime = classifierEndTime-classifierStartTime
        print("Classification time:",classificationtime)

    def computeUnigramAccuracy(self):
        print("Unigram Accuracy:")
        print(self.correctlyClassifiedUnigram/self.totalTrials)
        return self.correctlyClassifiedUnigram/self.totalTrials

    
    def computeTFIDFAccuracy(self):
        print("TFIDF accuracy:")
        print(self.correctlyClassifiedTFIDF/self.totalTrials)
        return self.correctlyClassifiedTFIDF/self.totalTrials


    def computeBigramAccuracy(self):
        print("Bigram Accuracy:")
        print(self.correctlyClassifiedBigram/self.totalTrials)
        return self.correctlyClassifiedBigram/self.totalTrials

    

    def shuffleMatrixUnigramandBigram(self):
        index1 = np.arange(np.shape(self.UnigramMatrix)[0])
        np.random.shuffle(index1)
        self.UnigramMatrix = self.UnigramMatrix[index1,:]

        index2 = np.arange(np.shape(self.bigramMatrix)[0])
        np.random.shuffle(index2)
        self.UnigramMatrix = self.UnigramMatrix[index2,:]
