# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import print_function
import tensorflow as tf
import edgeml.utils as utils
import numpy as np
import os
import sys


class BonsaiTrainer:

    def __init__(self, bonsaiObj, lW, lT, lV, lZ, sW, sT, sV, sZ,
                 learningRate, X, Y, useMCHLoss=False, outFile=None):
        '''
        bonsaiObj - Initialised Bonsai Object and Graph
        lW, lT, lV and lZ are regularisers to Bonsai Params
        sW, sT, sV and sZ are sparsity factors to Bonsai Params
        learningRate - learningRate fro optimizer
        X is the Data Placeholder - Dims [_, dataDimension]
        Y - Label placeholder for loss computation
        useMCHLoss - For choice between HingeLoss vs CrossEntropy
        useMCHLoss - True - MultiClass - multiClassHingeLoss
        useMCHLoss - False - MultiClass - crossEntropyLoss
        '''

        self.bonsaiObj = bonsaiObj

        self.lW = lW
        self.lV = lV
        self.lT = lT
        self.lZ = lZ

        self.sW = sW
        self.sV = sV
        self.sT = sT
        self.sZ = sZ

        self.Y = Y
        self.X = X

        self.useMCHLoss = useMCHLoss

        if outFile is not None:
            self.outFile = open(outFile, 'w')
        else:
            self.outFile = sys.stdout

        self.learningRate = learningRate

        self.assertInit()

        self.sigmaI = tf.placeholder(tf.float32, name='sigmaI')

        self.score, self.X_ = self.bonsaiObj(self.X, self.sigmaI)

        self.loss, self.marginLoss, self.regLoss = self.lossGraph()

        self.trainStep = self.trainGraph()
        self.accuracy = self.accuracyGraph()
        self.prediction = self.bonsaiObj.getPrediction()

        if self.sW > 0.99 and self.sV > 0.99 and self.sZ > 0.99 and self.sT > 0.99:
            self.isDenseTraining = True
        else:
            self.isDenseTraining = False

        self.hardThrsd()
        self.sparseTraining()

    def lossGraph(self):
        '''
        Loss Graph for given Bonsai Obj
        '''
        self.regLoss = 0.5 * (self.lZ * tf.square(tf.norm(self.bonsaiObj.Z)) +
                              self.lW * tf.square(tf.norm(self.bonsaiObj.W)) +
                              self.lV * tf.square(tf.norm(self.bonsaiObj.V)) +
                              self.lT * tf.square(tf.norm(self.bonsaiObj.T)))

        if (self.bonsaiObj.numClasses > 2):
            if self.useMCHLoss is True:
                self.batch_th = tf.placeholder(tf.int64, name='batch_th')
                self.marginLoss = utils.multiClassHingeLoss(
                    tf.transpose(self.score), self.Y,
                    self.batch_th)
            else:
                self.marginLoss = utils.crossEntropyLoss(
                    tf.transpose(self.score), self.Y)
            self.loss = self.marginLoss + self.regLoss
        else:
            self.marginLoss = tf.reduce_mean(tf.nn.relu(
                1.0 - (2 * self.Y - 1) * tf.transpose(self.score)))
            self.loss = self.marginLoss + self.regLoss

        return self.loss, self.marginLoss, self.regLoss

    def trainGraph(self):
        '''
        Train Graph for the loss generated by Bonsai
        '''
        self.bonsaiObj.TrainStep = tf.train.AdamOptimizer(
            self.learningRate).minimize(self.loss)

        return self.bonsaiObj.TrainStep

    def accuracyGraph(self):
        '''
        Accuracy Graph to evaluate accuracy when needed
        '''
        if (self.bonsaiObj.numClasses > 2):
            correctPrediction = tf.equal(
                tf.argmax(tf.transpose(self.score), 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correctPrediction, tf.float32))
        else:
            y_ = self.Y * 2 - 1
            correctPrediction = tf.multiply(tf.transpose(self.score), y_)
            correctPrediction = tf.nn.relu(correctPrediction)
            correctPrediction = tf.ceil(tf.tanh(correctPrediction))
            self.accuracy = tf.reduce_mean(
                tf.cast(correctPrediction, tf.float32))

        return self.accuracy

    def hardThrsd(self):
        '''
        Set up for hard Thresholding Functionality
        '''
        self.__Wth = tf.placeholder(tf.float32, name='Wth')
        self.__Vth = tf.placeholder(tf.float32, name='Vth')
        self.__Zth = tf.placeholder(tf.float32, name='Zth')
        self.__Tth = tf.placeholder(tf.float32, name='Tth')

        self.__Woph = self.bonsaiObj.W.assign(self.__Wth)
        self.__Voph = self.bonsaiObj.V.assign(self.__Vth)
        self.__Toph = self.bonsaiObj.T.assign(self.__Tth)
        self.__Zoph = self.bonsaiObj.Z.assign(self.__Zth)

        self.hardThresholdGroup = tf.group(
            self.__Woph, self.__Voph, self.__Toph, self.__Zoph)

    def sparseTraining(self):
        '''
        Set up for Sparse Retraining Functionality
        '''
        self.__Wops = self.bonsaiObj.W.assign(self.__Wth)
        self.__Vops = self.bonsaiObj.V.assign(self.__Vth)
        self.__Zops = self.bonsaiObj.Z.assign(self.__Zth)
        self.__Tops = self.bonsaiObj.T.assign(self.__Tth)

        self.sparseRetrainGroup = tf.group(
            self.__Wops, self.__Vops, self.__Tops, self.__Zops)

    def runHardThrsd(self, sess):
        '''
        Function to run the IHT routine on Bonsai Obj
        '''
        currW = self.bonsaiObj.W.eval()
        currV = self.bonsaiObj.V.eval()
        currZ = self.bonsaiObj.Z.eval()
        currT = self.bonsaiObj.T.eval()

        self.__thrsdW = utils.hardThreshold(currW, self.sW)
        self.__thrsdV = utils.hardThreshold(currV, self.sV)
        self.__thrsdZ = utils.hardThreshold(currZ, self.sZ)
        self.__thrsdT = utils.hardThreshold(currT, self.sT)

        fd_thrsd = {self.__Wth: self.__thrsdW, self.__Vth: self.__thrsdV,
                    self.__Zth: self.__thrsdZ, self.__Tth: self.__thrsdT}
        sess.run(self.hardThresholdGroup, feed_dict=fd_thrsd)

    def runSparseTraining(self, sess):
        '''
        Function to run the Sparse Retraining routine on Bonsai Obj
        '''
        currW = self.bonsaiObj.W.eval()
        currV = self.bonsaiObj.V.eval()
        currZ = self.bonsaiObj.Z.eval()
        currT = self.bonsaiObj.T.eval()

        newW = utils.copySupport(self.__thrsdW, currW)
        newV = utils.copySupport(self.__thrsdV, currV)
        newZ = utils.copySupport(self.__thrsdZ, currZ)
        newT = utils.copySupport(self.__thrsdT, currT)

        fd_st = {self.__Wth: newW, self.__Vth: newV,
                 self.__Zth: newZ, self.__Tth: newT}
        sess.run(self.sparseRetrainGroup, feed_dict=fd_st)

    def assertInit(self):
        err = "sparsity must be between 0 and 1"
        assert self.sW >= 0 and self.sW <= 1, "W " + err
        assert self.sV >= 0 and self.sV <= 1, "V " + err
        assert self.sZ >= 0 and self.sZ <= 1, "Z " + err
        assert self.sT >= 0 and self.sT <= 1, "T " + err
        errMsg = "Dimension Mismatch, Y has to be [_, " + \
            str(self.bonsaiObj.numClasses) + "]"
        errCont = " numClasses are 1 in case of Binary case by design"
        assert (len(self.Y.shape) == 2 and
                self.Y.shape[1] == self.bonsaiObj.numClasses), errMsg + errCont

    def saveParams(self, currDir):
        '''
        Function to save Parameter matrices
        '''
        self.bonsaiObj.saveModel(currDir)

    def getModelSize(self):
        '''
        Function to get aimed model size
        '''
        nnzZ, sizeZ, sparseZ = utils.countnnZ(self.bonsaiObj.Z, self.sZ)
        nnzW, sizeW, sparseW = utils.countnnZ(self.bonsaiObj.W, self.sW)
        nnzV, sizeV, sparseV = utils.countnnZ(self.bonsaiObj.V, self.sV)
        nnzT, sizeT, sparseT = utils.countnnZ(self.bonsaiObj.T, self.sT)

        totalnnZ = (nnzZ + nnzT + nnzV + nnzW)
        totalSize = (sizeZ + sizeW + sizeV + sizeT)
        hasSparse = (sparseW or sparseV or sparseT or sparseZ)
        return totalnnZ, totalSize, hasSparse

    def train(self, batchSize, totalEpochs, sess,
              Xtrain, Xtest, Ytrain, Ytest, dataDir, currDir):
        '''
        The Dense - IHT - Sparse Retrain Routine for Bonsai Training
        '''
        resultFile = open(dataDir + '/TFBonsaiResults.txt', 'a+')
        numIters = Xtrain.shape[0] / batchSize

        totalBatches = numIters * totalEpochs

        bonsaiObjSigmaI = 1

        counter = 0
        if self.bonsaiObj.numClasses > 2:
            trimlevel = 15
        else:
            trimlevel = 5
        ihtDone = 0
        maxTestAcc = -10000

        if self.isDenseTraining is True:
            ihtDone = 1
            bonsaiObjSigmaI = 1
            itersInPhase = 0

        header = '*' * 20
        for i in range(totalEpochs):
            print("\nEpoch Number: " + str(i), file=self.outFile)

            trainAcc = 0.0
            trainLoss = 0.0

            numIters = int(numIters)
            for j in range(numIters):

                if counter == 0:
                    msg = " Dense Training Phase Started "
                    print("\n%s%s%s\n" %
                          (header, msg, header), file=self.outFile)

                # Updating the indicator sigma
                if ((counter == 0) or (counter == int(totalBatches / 3.0)) or
                        (counter == int(2 * totalBatches / 3.0))) and (self.isDenseTraining is False):
                    bonsaiObjSigmaI = 1
                    itersInPhase = 0

                elif (itersInPhase % 100 == 0):
                    indices = np.random.choice(Xtrain.shape[0], 100)
                    batchX = Xtrain[indices, :]
                    batchY = Ytrain[indices, :]
                    batchY = np.reshape(
                        batchY, [-1, self.bonsaiObj.numClasses])

                    _feed_dict = {self.X: batchX}
                    Xcapeval = self.X_.eval(feed_dict=_feed_dict)
                    Teval = self.bonsaiObj.T.eval()

                    sum_tr = 0.0
                    for k in range(0, self.bonsaiObj.internalNodes):
                        sum_tr += (np.sum(np.abs(np.dot(Teval[k], Xcapeval))))

                    if(self.bonsaiObj.internalNodes > 0):
                        sum_tr /= (100 * self.bonsaiObj.internalNodes)
                        sum_tr = 0.1 / sum_tr
                    else:
                        sum_tr = 0.1
                    sum_tr = min(
                        1000, sum_tr * (2**(float(itersInPhase) /
                                            (float(totalBatches) / 30.0))))

                    bonsaiObjSigmaI = sum_tr

                itersInPhase += 1
                batchX = Xtrain[j * batchSize:(j + 1) * batchSize]
                batchY = Ytrain[j * batchSize:(j + 1) * batchSize]
                batchY = np.reshape(
                    batchY, [-1, self.bonsaiObj.numClasses])

                if self.bonsaiObj.numClasses > 2:
                    if self.useMCHLoss is True:
                        _feed_dict = {self.X: batchX, self.Y: batchY,
                                      self.batch_th: batchY.shape[0],
                                      self.sigmaI: bonsaiObjSigmaI}
                    else:
                        _feed_dict = {self.X: batchX, self.Y: batchY,
                                      self.sigmaI: bonsaiObjSigmaI}
                else:
                    _feed_dict = {self.X: batchX, self.Y: batchY,
                                  self.sigmaI: bonsaiObjSigmaI}

                # Mini-batch training
                _, batchLoss, batchAcc = sess.run(
                    [self.trainStep, self.loss, self.accuracy],
                    feed_dict=_feed_dict)

                trainAcc += batchAcc
                trainLoss += batchLoss

                # Training routine involving IHT and sparse retraining
                if (counter >= int(totalBatches / 3.0) and
                    (counter < int(2 * totalBatches / 3.0)) and
                    counter % trimlevel == 0 and
                        self.isDenseTraining is False):
                    self.runHardThrsd(sess)
                    if ihtDone == 0:
                        msg = " IHT Phase Started "
                        print("\n%s%s%s\n" %
                              (header, msg, header), file=self.outFile)
                    ihtDone = 1
                elif ((ihtDone == 1 and counter >= int(totalBatches / 3.0) and
                       (counter < int(2 * totalBatches / 3.0)) and
                       counter % trimlevel != 0 and
                       self.isDenseTraining is False) or
                        (counter >= int(2 * totalBatches / 3.0) and
                            self.isDenseTraining is False)):
                    self.runSparseTraining(sess)
                    if counter == int(2 * totalBatches / 3.0):
                        msg = " Sprase Retraining Phase Started "
                        print("\n%s%s%s\n" %
                              (header, msg, header), file=self.outFile)
                counter += 1

            print("Train Loss: " + str(trainLoss / numIters) +
                  " Train accuracy: " + str(trainAcc / numIters),
                  file=self.outFile)

            oldSigmaI = bonsaiObjSigmaI
            bonsaiObjSigmaI = 1e9

            if self.bonsaiObj.numClasses > 2:
                if self.useMCHLoss is True:
                    _feed_dict = {self.X: Xtest, self.Y: Ytest,
                                  self.batch_th: Ytest.shape[0],
                                  self.sigmaI: bonsaiObjSigmaI}
                else:
                    _feed_dict = {self.X: Xtest, self.Y: Ytest,
                                  self.sigmaI: bonsaiObjSigmaI}
            else:
                _feed_dict = {self.X: Xtest, self.Y: Ytest,
                              self.sigmaI: bonsaiObjSigmaI}

            # This helps in direct testing instead of extracting the model out

            testAcc, testLoss, regTestLoss = sess.run(
                [self.accuracy, self.loss, self.regLoss], feed_dict=_feed_dict)
            if ihtDone == 0:
                maxTestAcc = -10000
                maxTestAccEpoch = i
            else:
                if maxTestAcc <= testAcc:
                    maxTestAccEpoch = i
                    maxTestAcc = testAcc

            print("Test accuracy %g" % testAcc, file=self.outFile)
            print("MarginLoss + RegLoss: " + str(testLoss - regTestLoss) +
                  " + " + str(regTestLoss) + " = " + str(testLoss) + "\n",
                  file=self.outFile)
            self.outFile.flush()

            bonsaiObjSigmaI = oldSigmaI

        # sigmaI has to be set to infinity to ensure
        # only a single path is used in inference
        bonsaiObjSigmaI = 1e9
        print("\nMaximum Test accuracy at compressed" +
              " model size(including early stopping): " +
              str(maxTestAcc) + " at Epoch: " +
              str(maxTestAccEpoch + 1) + "\nFinal Test" +
              " Accuracy: " + str(testAcc), file=self.outFile)
        print("\nNon-Zeros: " + str(self.getModelSize()[0]) + " Model Size: " +
              str(float(self.getModelSize()[1]) / 1024.0) + " KB hasSparse: " +
              str(self.getModelSize()[2]) + "\n", file=self.outFile)

        resultFile.write("MaxTestAcc: " + str(maxTestAcc) +
                         " at Epoch(totalEpochs): " +
                         str(maxTestAccEpoch + 1) +
                         "(" + str(totalEpochs) + ")" + " ModelSize: " +
                         str(float(self.getModelSize()[1]) / 1024.0) +
                         " KB hasSparse: " + str(self.getModelSize()[2]) +
                         " Param Directory: " +
                         str(os.path.abspath(currDir)) + "\n")
        self.saveParams(currDir)
        print("The Model Directory: " + currDir + "\n")

        resultFile.close()
        self.outFile.flush()
        self.outFile.close()
