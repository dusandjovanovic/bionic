using System;
using System.Collections.Generic;

namespace ai.net.neuralnet.learn
{


	public abstract class Training
	{

		private int epochs;
		private double error;
		private double mse;

		public enum TrainingTypesENUM
		{
			BACKPROPAGATION,
			//KOHONEN
		}

		public virtual NeuralNet train(NeuralNet n)
		{

			List<double> inputWeightIn = new List<double>();

			int rows = n.TrainSet.Length;
			int cols = n.TrainSet[0].Length;

			while (this.Epochs < n.MaxEpochs)
			{

				double estimatedOutput = 0.0;
				double realOutput = 0.0;

				for (int i = 0; i < rows; i++)
				{

					double netValue = 0.0;

					for (int j = 0; j < cols; j++)
					{
						inputWeightIn = n.InputLayer.ListOfNeurons[j].ListOfWeightIn;
						double inputWeight = inputWeightIn[0];
						netValue = netValue + inputWeight * n.TrainSet[i][j];
					}

					estimatedOutput = this.activationFnc(n.ActivationFnc, netValue);
					realOutput = n.RealOutputSet[i];

					this.Error = realOutput - estimatedOutput;

					if (Math.Abs(this.Error) > n.TargetError)
					{
						// fix weights
						InputLayer inputLayer = new InputLayer();
						inputLayer.ListOfNeurons = this.teachNeuronsOfLayer(cols, i, n, netValue);
						n.InputLayer = inputLayer;
					}

				}

				this.Mse = Math.Pow(realOutput - estimatedOutput, 2.0);
				n.ListOfMSE.Add(this.Mse);

				this.Epochs = this.Epochs + 1;

			}

			n.TrainingError = this.Error;

			return n;
		}

		private List<Neuron> teachNeuronsOfLayer(int numberOfInputNeurons, int line, NeuralNet n, double netValue)
		{
			List<Neuron> listOfNeurons = new List<Neuron>();
			List<double> inputWeightsInNew = new List<double>();
			List<double> inputWeightsInOld = new List<double>();

			for (int j = 0; j < numberOfInputNeurons; j++)
			{
				inputWeightsInOld = n.InputLayer.ListOfNeurons[j].ListOfWeightIn;
				double inputWeightOld = inputWeightsInOld[0];

				inputWeightsInNew.Add(this.calcNewWeight(n.TrainType, inputWeightOld, n, this.Error, n.TrainSet[line][j], netValue));

				Neuron neuron = new Neuron();
				neuron.ListOfWeightIn = inputWeightsInNew;
				listOfNeurons.Add(neuron);
				inputWeightsInNew = new List<double>();
			}

			return listOfNeurons;

		}

		private double calcNewWeight(TrainingTypesENUM trainType, double inputWeightOld, NeuralNet n, double error, double trainSample, double netValue)
		{
		    return inputWeightOld + n.LearningRate * error * trainSample;
		}

		public enum ActivationFncENUM
		{
			STEP,
			LINEAR,
			SIGLOG,
			HYPERTAN
		}

		protected internal virtual double activationFnc(ActivationFncENUM fnc, double value)
		{
			switch (fnc)
			{
			case ai.net.neuralnet.learn.Training.ActivationFncENUM.STEP:
				return fncStep(value);
			case ai.net.neuralnet.learn.Training.ActivationFncENUM.LINEAR:
				return fncLinear(value);
			case ai.net.neuralnet.learn.Training.ActivationFncENUM.SIGLOG:
				return fncSigLog(value);
			case ai.net.neuralnet.learn.Training.ActivationFncENUM.HYPERTAN:
				return fncHyperTan(value);
			default:
				throw new System.ArgumentException(fnc + " does not exist in ActivationFncENUM");
			}
		}

		public virtual double derivativeActivationFnc(ActivationFncENUM fnc, double value)
		{
			switch (fnc)
			{
			case ai.net.neuralnet.learn.Training.ActivationFncENUM.LINEAR:
				return derivativeFncLinear(value);
			case ai.net.neuralnet.learn.Training.ActivationFncENUM.SIGLOG:
				return derivativeFncSigLog(value);
			case ai.net.neuralnet.learn.Training.ActivationFncENUM.HYPERTAN:
				return derivativeFncHyperTan(value);
			default:
				throw new System.ArgumentException(fnc + " does not exist in ActivationFncENUM");
			}
		}

		private double fncStep(double v)
		{
			if (v >= 0)
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		}
		private double fncLinear(double v)
		{
			return v;
		}
		private double fncSigLog(double v)
		{
			return (1.0 / (1.0 + Math.Exp(-v)));
		}
		private double fncHyperTan(double v)
		{
			return Math.Tanh(v);
		}

		private double derivativeFncLinear(double v)
		{
			return 1.0;
		}
		private double derivativeFncSigLog(double v)
		{
			return (v * (1.0 - v));
		}
		private double derivativeFncHyperTan(double v)
		{
			return (1.0 / Math.Pow(Math.Cosh(v), 2.0));
		}

		public virtual void printTrainedNetResult(NeuralNet trainedNet)
		{

			int rows = trainedNet.TrainSet.Length;
			int cols = trainedNet.TrainSet[0].Length;

			List<double> inputWeightIn = new List<double>();

			for (int i = 0; i < rows; i++)
			{
				double netValue = 0.0;
				for (int j = 0; j < cols; j++)
				{
					inputWeightIn = trainedNet.InputLayer.ListOfNeurons[j].ListOfWeightIn;
					double inputWeight = inputWeightIn[0];
					netValue = netValue + inputWeight * trainedNet.TrainSet[i][j];

					Console.Write(trainedNet.TrainSet[i][j] + "\t");
				}

				double estimatedOutput = this.activationFnc(trainedNet.ActivationFnc, netValue);

				int colsOutput = trainedNet.RealMatrixOutputSet[0].Length;

				double realOutput = 0.0;
				for (int k = 0; k < colsOutput; k++)
				{
					realOutput = realOutput + trainedNet.RealMatrixOutputSet[i][k];
				}

				Console.Write(" NET OUTPUT: " + estimatedOutput + "\t");
				Console.Write(" REAL OUTPUT: " + realOutput + "\t");
				double error = estimatedOutput - realOutput;
				Console.Write(" ERROR: " + error + "\n");

			}

		}

		public virtual int Epochs
		{
			get
			{
				return epochs;
			}
			set
			{
				this.epochs = value;
			}
		}


		public virtual double Error
		{
			get
			{
				return error;
			}
			set
			{
				this.error = value;
			}
		}


		public virtual double Mse
		{
			get
			{
				return mse;
			}
			set
			{
				this.mse = value;
			}
		}


	}

}