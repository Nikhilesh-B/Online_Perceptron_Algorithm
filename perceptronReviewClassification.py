import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time
from dask import dataframe as df1 
import scipy as sp
import csv


#hashtag use as many dictionaries as possible.


class proccessData():
    def __init__(self):
        self.unigramIndexDict = dict()
        self.unigramEnumeration = 0 
        self.unigramMatrix = np.array([[]])
        self.currentWordId = 0
        self.wordIndex = dict()
        self.WVectors = np.array([])
        self.Windexes = dict()
        self.classifications = np.array([])
        self.unigramClassifierWeight = None
        

    def workingThroughDictionary(self):
        counts  = dict()
        file = open('./restaurant_reviews_data/toyData.csv')
        csvreader = csv.reader(file)
        header = next(csvreader)

        i = 0 
        start_time = time()
        for row in csvreader: 
            classification = row[0]
            document = row[1]
            sentenceList = document.split(" ")
            words = set(sentenceList)
            for word in words:
                if word not in counts:
                    counts[word]=1
                else:
                    counts[word]=counts[word]+1

        end_time = time()

        time_elapsed = end_time-start_time
        print(time_elapsed)
        print(counts)
    
    def populateUnigramMatrix(self, sentenceList):
        dataVect  = np.array(np.zeros(self.currentWordId))

        
        for word in sentenceList:
            if word not in self.wordIndex:
                self.wordIndex[word] = self.currentWordId
                self.currentWordId = self.currentWordId+1 
                dataVect = np.append(dataVect,1)

            else:
                index = self.wordIndex[word]
                dataVect[index] = dataVect[index]+1


        len_difference = self.currentWordId-len(self.unigramMatrix[0])
        current_height = len(self.unigramMatrix)


        if(len_difference>0 and current_height>=1):
            zeroVect = np.zeros((current_height, len_difference))
            self.unigramMatrix = np.hstack((self.unigramMatrix, zeroVect))

        self.unigramMatrix  = np.vstack((self.unigramMatrix, dataVect))


    def initializeList(self, sentenceList):
        for word in sentenceList:
            self.wordIndex[word] = self.currentWordId
            self.currentWordId = self.currentWordId+1 
        
        self.unigramMatrix = [np.ones(len(sentenceList))]
        self.unigramMatrix = np.array(self.unigramMatrix)


    def resetUnigramMatrix(self): 
        self.unigramMatrix = np.array([[]])



    def computeWVectors(self, batch):
        docNo =  batch*30
        for i in range(len(self.unigramMatrix)):
            features = self.unigramMatrix[i]
            #the above are the features for the unigram matrix 
            classification = self.classifications[docNo+i]
            #now given the features we need the classificaiton

            if(not np.sign(np.dot(features,self.unigramClassifierWeight))==classification):
                self.unigramClassifierWeight = self.unigramClassifierWeight+classification*features
                np.vstack((self.WVectors,self.unigramClassifierWeight))
            
            wVectorTuple = tuple(self.unigramClassifierWeight)
            self.Windexes[wVectorTuple] = docNo




    


    def inputData(self):
        #self.training_file = open("./restaurant_reviews_data/reviews_tr.csv")
        #self.testing_file =  open("./restaurant_reviews_data/reviews_te.csv")
        file = open('./restaurant_reviews_data/toyData.csv')
        csvreader = csv.reader(file)
        header = next(csvreader)

        start_time = time()
        docNo = 0 
        for row in csvreader: 
            classification = row[0]
            np.vstack((self.classifications, classification))
            document = row[1]
            sentenceList = document.split(" ")

            if(docNo==0):
                self.initializeList(sentenceList)

            else:
                self.populateUnigramMatrix(sentenceList)

            docNo = docNo+1
            if docNo % 29 == 0:
                batch =  docNo/29

                if(batch == 1):
                    self.unigramClassifierWeight = np.zeros(len(self.unigramMatrix[0]))
                    np.vstack((self.WVectors,self.unigramClassifierWeight))
                

                else:
                    correctLength = len(self.unigramMatrix[0])
                    supposedLength = len(self.unigramClassifierWeight)
                    diff = correctLength - supposedLength
                    if(diff>0):
                        np.concatenate((self.unigramClassifierWeight, np.zeros(diff)),axis=0)

                self.computeWVectors(batch)
                self.resetUnigramMatrix()

        end_time = time()
        elapsed_time = end_time-start_time
        print(elapsed_time)
        print(self.unigramMatrix)




class perceptronClassifier():
    def __init__(self):
        self.trainingData =  pd.read_csv("./restaurant_reviews_data/reviews_tr.csv")
        self.holdoutTestData = pd.read_csv("./restaurant_reviews_data/reviews_te.csv")

        self.unigramCount = dict()
        self.tfidf = dict()
        self.bigramRepresentation = dict()
        
        self.countVectorizerUnigram = CountVectorizer()
        self.TFIDFVectorizer = TfidfVectorizer(sublinear_tf=True)
        self.countVectorizerBigram = CountVectorizer(ngram_range=(2,2))

        self.unigramClassifierWeights = np.array([[]])
        self.TFIDFClassifierWeights = np.array([[]])
        self.bigramClassiferWeights = np.array([[]])

    def splitTraningData(self, split):
        self.trainingData = self.trainingData.sample(frac=split)


    def processData(self):
        A = self.countVectorizerUnigram.fit_transform(self.trainingData["text"].tolist())
        B = self.TFIDFVectorizer.fit_transform(self.trainingData["text"].tolist())
        C = self.countVectorizerBigram.fit_transform(self.trainingData["text"].tolist())

        widthUnigram = len(A[0])
        widthBigram  = len(B[0])
        widthTFIDF = len(C[0])

        self.unigramClassifierWeight = np.zeros(widthUnigram+1)
        self.bigramClassifierWeight =  np.zeros(widthBigram+1)
        self.TFIDFClassifierWeight = np.zeros(widthTFIDF+1)

        self.unigramClassifierWeights = np.vstack(self.unigramClassifierWeights, self.unigramClassifierWeight)
        self.bigramClassifierWeights = np.vstack(self.bigramClassifierWeights, self.bigramClassifierWeight)
        self.TFIDFClassifierWeights = np.vstack(self.TFIDFClassifierWeights, self.TFIDFClassifierWeight)



    def trainLinearClassifier(self):
        for j in range(2):
            for index, row in self.trainingData.iterrows():
                document = row["text"]
                groundTruthLabel = row["label"]

                if (groundTruthLabel==0):
                    groundTruthLabel = -1

                #process the document using the above functions as required 
                #unigram classifier upfirst where we will go through and process the data as required 


                #returns a matrix i think, so convertinf to a numpy array regular 
                featureUnigram =  self.countVectorizerUnigram.transform([document])[0]
                featureUnigram =  np.hstack(featureUnigram, np.array([1]))


                if(not np.sign(np.dot(featureUnigram,self.unigramClassifierWeight))==groundTruthLabel):
                    self.unigramClassifierWeight = self.unigramClassifierWeight+groundTruthLabel*featureUnigram

                self.unigramClassifierWeights = np.vstack(self.unigramClassifierWeights, self.unigramClassifierWeight)
                


                #bigram classifier next 

                featureBigram =  self.countVectorizerBigram.transform([document])[0]
                featureBigram =  np.hstack(featureBigram, np.array([1]))


                if(np.sign(np.dot(featureBigram,self.bigramClassifierWeight)) != groundTruthLabel):
                    self.bigramClassifierWeight = self.bigramClassifierWeight+groundTruthLabel*featureBigram

                self.bigramClassifierWeights = np.vstack(self.bigramClassifierWeights, self.bigramClassifierWeight)
                
                #tfidf classifier 

                featureTFIDF =  self.TFIDFVectorizer.transform([document])[0]
                featureTFIDF =  np.hstack(featureTFIDF, np.array([1]))


                if(not np.sign(np.dot(featureTFIDF,self.TFIDFClassifierWeight))==groundTruthLabel):
                    self.TFIDFClassifierWeight = self.TFIDFClassifierWeight+groundTruthLabel*featureTFIDF

                self.TFIDFClassifierWeights = np.vstack(self.TFIDFClassifierWeights, self.TFIDFClassifierWeight)



    def computeAverageOfLastHalf(self, matrix):
        length = len(matrix)//2
        matrix = matrix[length:len(matrix)]

        final_w = np.average(matrix, axis=1)
        return final_w


    def classify(self):
        self.finalWeightVectorUnigram = self.computeAverageOfLastHalf(self.unigramClassifierWeights)
        self.finalWeightVectorBigram = self.computeAverageOfLastHalf(self.bigramClassifierWeights)
        self.finalWeightVectorTFIDF  = self.computeAverageOfLastHalf(self.TFIDFClassifierWeights)


        self.totalExamples = len(self.holdoutTestData)



        self.unigramCorrectlyClassified = 0 
        self.bigramCorrectlyClassified = 0 
        self.TFIDFCorrectlyClassified = 0


        for index, row in self.holdoutTestData.iterrows():
                document = row["text"]
                groundTruthLabel = row["label"]

                featureUnigram =  self.countVectorizerUnigram.transform([document])[0]
                featureUnigram =  np.hstack(featureUnigram, np.array([1]))


                featureBigram =  self.countVectorizerBigram.transform([document])[0]
                featureBigram =  np.hstack(featureBigram, np.array([1]))
                
                
                featureTFIDF =  self.TFIDFVectorizer.transform([document])[0]
                featureTFIDF =  np.hstack(featureTFIDF, np.array([1]))


                if (groundTruthLabel==0):
                    groundTruthLabel = -1

                if(np.sign(np.dot(self.finalWeightVectorUnigram,featureUnigram)) == groundTruthLabel):
                    self.unigramCorrectlyClassified = self.unigramCorrectlyClassified+1
                
                if(np.sign(np.dot(self.finalWeightVectorBigram,featureBigram)) == groundTruthLabel):
                    self.bigramCorrectlyClassified = self.bigramCorrectlyClassified+1
                    
                if(np.sign(np.dot(self.finalWeightVectorTFIDF,featureTFIDF)) == groundTruthLabel):
                    self.TFIDFCorrectlyClassified = self.TFIDFCorrectlyClassified+1
        
               
    def computeAccuracy(self):

        self.unigramAcc = self.unigramCorrectlyClassified/self.totalExamples 

        self.bigramAcc = self.bigramCorrectlyClassified/self.totalExamples

        self.TFIDFAcc = self.TFIDFCorrectlyClassified/self.totalExamples

        print("UNIGRAM accuracy:")
        print(self.unigramAcc)

        print("BIGRAM accuracy:")
        print(self.bigramAcc)
        
        print("TFIDF accuracy:")
        print(self.TFIDFAcc)