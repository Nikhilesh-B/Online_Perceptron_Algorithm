README for nb2953 HW2


We iterate through different files and then perform the classification as required.There are two files that we have for this coding assignment. finalPerceptronClassifier and perceptronReviewClassificationTesting. perceptronReviewClassificationTesting is where we run all the code to actually test our models, whereas final perceptron classifier is simply were we build the classifier. 




The code should be executed in the following steps exactly:


//Files is a list of files that you input into the data (where I had smaller files for convenience including reviews_tr, and reviews_te. If you wanted to run those two you would just input them in as above.



files = [('./restaurant_reviews_data/mediumTrain.csv','./restaurant_reviews_data/reviews_te.csv')]

//training split is the % of training data a particular training file represents. Indexes must match up for files and training split. I.e. the training file in index 0 of "files" has a training split of value equal to index 0 of "training split's value"


trainingSplit = [1] 

unigramAccuracy = []
tfidfAccuracy = []
bigramAccuracy = []


for file in files:
    trainFile = file[0]
    testFile = file[1]
    
    algo = classifier(trainfile=trainFile, testfile=testFile)

    //preprocesing and feature selection is done through these lines.
    algo.computeVocab()
    algo.populateMatrix()
    algo.populateBigramMatrix()
    algo.initializeWeightVector()

    //training the classifiers
    algo.trainLinearClassifier()
    algo.trainLinearBigramClassifier()
    algo.shuffleMatrixUnigramandBigram()
    algo.trainLinearClassifier()
    algo.trainLinearBigramClassifier()

    //averaging weights
    algo.onlineToBatchConversionBigrams()
    algo.onlineToBatchConversionTFIDF()
    algo.onlineToBatchConversionUnigrams()

    //classifying for the three models, algo.classify() does both TFIDF and unigram
    algo.classify()
    algo.classifyBigram()

    unigramAcc = algo.computeUnigramAccuracy()
    bigramAcc = algo.computeBigramAccuracy()
    tfidfAcc = algo.computeTFIDFAccuracy()

    unigramAccuracy.append(unigramAcc)
    tfidfAccuracy.append(tfidfAcc)
    bigramAccuracy.append(bigramAcc)

//plotting the accuracies below.
plt.plot(trainingSplit, unigramAccuracy)
plt.plot(trainingSplit, bigramAccuracy)
plt.plot(trainingSplit, tfidfAccuracy)

plt.legend(["unigram","bigram","Tfidf"]) 
plt.xlabel("Training split")
plt.ylabel("Accuracy on holdout test")
plt.show()

