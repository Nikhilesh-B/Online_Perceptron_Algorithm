#from perceptronReviewClassification import perceptronClassifier
import numpy as np
from finalPerceptronClassifier import classifier
import matplotlib.pyplot as plt





files = [('./restaurant_reviews_data/mediumTrain.csv','./restaurant_reviews_data/reviews_te.csv')]


trainingSplit = [0.08,1] 

unigramAccuracy = []
tfidfAccuracy = []
bigramAccuracy = []



for file in files:
    trainFile = file[0]
    testFile = file[1]
    
    algo = classifier(trainfile=trainFile, testfile=testFile)

    algo.computeVocab()
    algo.populateMatrix()
    #algo.populateBigramMatrix()
    algo.initializeWeightVector()


    algo.trainLinearClassifier()
    #algo.trainLinearBigramClassifier()
    ##algo.shuffleMatrixUnigramandBigram()
    #algo.trainLinearClassifier()
    #algo.trainLinearBigramClassifier()


    #algo.onlineToBatchConversionBigrams()
    #algo.onlineToBatchConversionTFIDF()
    algo.onlineToBatchConversionUnigrams()


    #algo.classify()
    #algo.classifyBigram()

    #unigramAcc = algo.computeUnigramAccuracy()
    #bigramAcc = algo.computeBigramAccuracy()
    #tfidfAcc = algo.computeTFIDFAccuracy()

    #unigramAccuracy.append(unigramAcc)
    #tfidfAccuracy.append(tfidfAcc)
    #bigramAccuracy.append(bigramAcc)

#plt.plot(trainingSplit, unigramAccuracy)
#plt.plot(trainingSplit, bigramAccuracy)
#plt.plot(trainingSplit, tfidfAccuracy)

#plt.legend(["unigram","bigram","Tfidf"]) 
#plt.xlabel("Training split")
#plt.ylabel("Accuracy on holdout test")
#plt.show()


































































