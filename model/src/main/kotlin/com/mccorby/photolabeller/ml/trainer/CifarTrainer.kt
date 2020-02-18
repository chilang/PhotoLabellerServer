package com.mccorby.photolabeller.ml.trainer

import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.api.TrainingListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import org.nd4j.linalg.learning.config.Adam

class CifarTrainer(private val config: SharedConfig) {

    fun createModel(seed: Int, iterations: Int, numLabels: Int): MultiLayerNetwork {
        val modelConf = NeuralNetConfiguration.Builder()
                .seed(seed.toLong())
                .updater(Updater.ADAM)
//                .iterations(iterations)//removed in later version, default to 1 anyway
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-4)
//                .regularization(true)//removed in later version, default to true
                .l2(5 * 1e-4)
                .list()
                .layer(0, ConvolutionLayer.Builder(intArrayOf(4, 4), intArrayOf(1, 1), intArrayOf(0, 0))
                        .name("cnn1")
                        .convolutionMode(ConvolutionMode.Same)
                        .nIn(3)
                        .nOut(32)
                        .weightInit(WeightInit.XAVIER_UNIFORM)
                        .activation(Activation.RELU)
//                        .learningRate(1e-2)
                        .updater(Adam(1e-2))
                        .biasInit(1e-2)
//                        .biasLearningRate(1e-2 * 2)
                        .biasUpdater(Adam(1e-2 * 2))
                        .build())
                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(3, 3))
                        .name("pool1")
                        .build())
                .layer(2, LocalResponseNormalization.Builder(3.0, 5e-05, 0.75)
                        .build())
                .layer(3, DenseLayer.Builder()
                        .name("ffn1")
                        .nOut(64)
                        .dropOut(0.5)
                        .build())
                .layer(4, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
//                .backprop(true)
//                .pretrain(false)
                .setInputType(InputType.convolutional(config.imageSize.toLong(), config.imageSize.toLong(), config.channels.toLong()))
                .build()

        return MultiLayerNetwork(modelConf).also { it.init() }
    }

    fun train(model: MultiLayerNetwork, numSamples: Int, epochs: Int, scoreListener: TrainingListener): MultiLayerNetwork {
        model.setListeners(scoreListener)
        val cifar = Cifar10DataSetIterator(
                config.batchSize,
                intArrayOf(config.imageSize, config.imageSize),
                DataSetType.TRAIN)

        for (i in 0 until epochs) {
            println("Epoch=====================$i")
            model.fit(cifar)
        }

        return model
    }

    fun eval(model: MultiLayerNetwork, numSamples: Int): Evaluation {
        val cifarEval = Cifar10DataSetIterator(config.batchSize,
                intArrayOf(config.imageSize, config.imageSize),
                DataSetType.VALIDATION)

        println("=====eval model========")
        val eval = Evaluation(cifarEval.labels)
        while (cifarEval.hasNext()) {
            val testDS = cifarEval.next(config.batchSize)
            val output = model.output(testDS.features)
            eval.eval(testDS.labels, output)
        }
        return eval
    }

    fun saveModel(model: MultiLayerNetwork, location: String) {
        ModelSerializer.writeModel(model, File(location), true)
    }
}
