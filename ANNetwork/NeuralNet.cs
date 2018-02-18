using System;
using System.Collections.Generic;

namespace ai.net.neuralnet
{

	using Backpropagation = ai.net.neuralnet.learn.Backpropagation;
	using ActivationFncENUM = ai.net.neuralnet.learn.Training.ActivationFncENUM;
	using TrainingTypesENUM = ai.net.neuralnet.learn.Training.TrainingTypesENUM;

	public class NeuralNet
	{

		private InputLayer inputLayer;
		private HiddenLayer hiddenLayer;
		private List<HiddenLayer> listOfHiddenLayer;
		private OutputLayer outputLayer;
		private int numberOfHiddenLayers;

		private double[][] trainSet;
		private double[][] validationSet;
		private double[] realOutputSet;
		private double[][] realMatrixOutputSet;
		private int maxEpochs;
		private double learningRate;
		private double targetError;
		private double trainingError;
		private double errorMean;

		private List<double> listOfMSE = new List<double>();
		private ActivationFncENUM activationFnc;
		private ActivationFncENUM activationFncOutputLayer;
		private TrainingTypesENUM trainType;

		public virtual NeuralNet initNet(int numberOfInputNeurons, int numberOfHiddenLayers, int numberOfNeuronsInHiddenLayer, int numberOfOutputNeurons)
		{
			inputLayer = new InputLayer();
			inputLayer.NumberOfNeuronsInLayer = numberOfInputNeurons;

			listOfHiddenLayer = new List<HiddenLayer>();
			for (int i = 0; i < numberOfHiddenLayers; i++)
			{
				hiddenLayer = new HiddenLayer();
				hiddenLayer.NumberOfNeuronsInLayer = numberOfNeuronsInHiddenLayer;
				listOfHiddenLayer.Add(hiddenLayer);
			}

			outputLayer = new OutputLayer();
			outputLayer.NumberOfNeuronsInLayer = numberOfOutputNeurons;

			inputLayer = inputLayer.initLayer(inputLayer);

			if (numberOfHiddenLayers > 0)
			{
				listOfHiddenLayer = hiddenLayer.initLayer(hiddenLayer, listOfHiddenLayer, inputLayer, outputLayer);
			}

			outputLayer = outputLayer.initLayer(outputLayer);

			NeuralNet newNet = new NeuralNet();
			newNet.InputLayer = inputLayer;
			newNet.HiddenLayer = hiddenLayer;
			newNet.ListOfHiddenLayer = listOfHiddenLayer;
			newNet.NumberOfHiddenLayers = numberOfHiddenLayers;
			newNet.OutputLayer = outputLayer;

			return newNet;
		}

		public virtual void printNet(NeuralNet n)
		{
			inputLayer.printLayer(n.InputLayer);
			Console.WriteLine();
			if (n.HiddenLayer != null)
			{
				hiddenLayer.printLayer(n.ListOfHiddenLayer);
				Console.WriteLine();
			}
			outputLayer.printLayer(n.OutputLayer);
		}

		public virtual NeuralNet trainNet(NeuralNet n)
		{

			NeuralNet trainedNet = new NeuralNet();

			switch (n.trainType)
			{
			case TrainingTypesENUM.BACKPROPAGATION:
				Backpropagation b = new Backpropagation();
				trainedNet = b.train(n);
				return trainedNet;
			default:
				throw new System.ArgumentException(n.trainType + " does not exist in TrainingTypesENUM");
			}

		}

		public virtual void printTrainedNetResult(NeuralNet n)
		{
			switch (n.trainType)
			{
			case TrainingTypesENUM.BACKPROPAGATION:
				Backpropagation b = new Backpropagation();
				b.printTrainedNetResult(n);
				break;
			default:
				throw new System.ArgumentException(n.trainType + " does not exist in TrainingTypesENUM");
			}
		}

		public virtual double[][] getNetOutputValues(NeuralNet trainedNet)
		{

			int rows = trainedNet.TrainSet.Length;

			int cols = trainedNet.OutputLayer.NumberOfNeuronsInLayer;

			double[][] matrixOutputValues = RectangularArrays.ReturnRectangularDoubleArray(rows, cols);

			switch (trainedNet.trainType)
			{
				case TrainingTypesENUM.BACKPROPAGATION:
					Backpropagation b = new Backpropagation();

					for (int rows_i = 0; rows_i < rows; rows_i++)
					{
						for (int cols_i = 0; cols_i < cols; cols_i++)
						{

							matrixOutputValues[rows_i][cols_i] = b.forward(trainedNet, rows_i).OutputLayer.ListOfNeurons[cols_i].OutputValue;

						}
					}

					break;
				default:
					throw new System.ArgumentException(trainedNet.trainType + " does not exist in TrainingTypesENUM");
			}

			return matrixOutputValues;

		}

		public virtual InputLayer InputLayer
		{
			get
			{
				return inputLayer;
			}
			set
			{
				this.inputLayer = value;
			}
		}


		public virtual HiddenLayer HiddenLayer
		{
			get
			{
				return hiddenLayer;
			}
			set
			{
				this.hiddenLayer = value;
			}
		}


		public virtual List<HiddenLayer> ListOfHiddenLayer
		{
			get
			{
				return listOfHiddenLayer;
			}
			set
			{
				this.listOfHiddenLayer = value;
			}
		}


		public virtual OutputLayer OutputLayer
		{
			get
			{
				return outputLayer;
			}
			set
			{
				this.outputLayer = value;
			}
		}


		public virtual int NumberOfHiddenLayers
		{
			get
			{
				return numberOfHiddenLayers;
			}
			set
			{
				this.numberOfHiddenLayers = value;
			}
		}


		public virtual double[][] TrainSet
		{
			get
			{
				return trainSet;
			}
			set
			{
				this.trainSet = value;
			}
		}


		public virtual double[] RealOutputSet
		{
			get
			{
				return realOutputSet;
			}
			set
			{
				this.realOutputSet = value;
			}
		}


		public virtual int MaxEpochs
		{
			get
			{
				return maxEpochs;
			}
			set
			{
				this.maxEpochs = value;
			}
		}


		public virtual double TargetError
		{
			get
			{
				return targetError;
			}
			set
			{
				this.targetError = value;
			}
		}


		public virtual double LearningRate
		{
			get
			{
				return learningRate;
			}
			set
			{
				this.learningRate = value;
			}
		}


		public virtual double TrainingError
		{
			get
			{
				return trainingError;
			}
			set
			{
				this.trainingError = value;
			}
		}


		public virtual ActivationFncENUM ActivationFnc
		{
			get
			{
				return activationFnc;
			}
			set
			{
				this.activationFnc = value;
			}
		}


		public virtual TrainingTypesENUM TrainType
		{
			get
			{
				return trainType;
			}
			set
			{
				this.trainType = value;
			}
		}


		public virtual List<double> ListOfMSE
		{
			get
			{
				return listOfMSE;
			}
			set
			{
				this.listOfMSE = value;
			}
		}


		public virtual double[][] RealMatrixOutputSet
		{
			get
			{
				return realMatrixOutputSet;
			}
			set
			{
				this.realMatrixOutputSet = value;
			}
		}


		public virtual double ErrorMean
		{
			get
			{
				return errorMean;
			}
			set
			{
				this.errorMean = value;
			}
		}


		public virtual ActivationFncENUM ActivationFncOutputLayer
		{
			get
			{
				return activationFncOutputLayer;
			}
			set
			{
				this.activationFncOutputLayer = value;
			}
		}


		public virtual double[][] ValidationSet
		{
			get
			{
				return validationSet;
			}
			set
			{
				this.validationSet = value;
			}
		}




	}

}