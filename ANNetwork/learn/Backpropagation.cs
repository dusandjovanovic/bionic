using System;
using System.Collections.Generic;

namespace ai.net.neuralnet.learn
{
	public class Backpropagation : Training
	{

        /* algritam ucenja :: povratna propagacija */

		internal int epoch = 0;

		public override NeuralNet train(NeuralNet n)
		{

			Mse = 1.0;

			while (Mse > n.TargetError)
			{

				if (epoch >= n.MaxEpochs)
				{
					break;
				}

				int rows = n.TrainSet.Length;
				double sumErrors = 0.0;

				for (int rows_i = 0; rows_i < rows; rows_i++)
				{

					n = forward(n, rows_i);

					n = backpropagation(n, rows_i);

					sumErrors = sumErrors + n.ErrorMean;

				}

				Mse = sumErrors / rows;

				n.ListOfMSE.Add(Mse);

				epoch++;

			}

			return n;

		}

		public virtual NeuralNet forward(NeuralNet n, int row)
		{

			List<HiddenLayer> listOfHiddenLayer = new List<HiddenLayer>();

			listOfHiddenLayer = n.ListOfHiddenLayer;

			double estimatedOutput = 0.0;
			double realOutput = 0.0;
			double sumError = 0.0;

			if (listOfHiddenLayer.Count > 0)
			{

				int hiddenLayer_i = 0;

				foreach (HiddenLayer hiddenLayer in listOfHiddenLayer)
				{

					int numberOfNeuronsInLayer = hiddenLayer.NumberOfNeuronsInLayer;

					foreach (Neuron neuron in hiddenLayer.ListOfNeurons)
					{

						double netValueOutN = 0.0;

						if (neuron.ListOfWeightIn.Count > 0)
						{
							double netValueN = 0.0;

							for (int layer_j = 0; layer_j < numberOfNeuronsInLayer - 1; layer_j++)
							{
								double hiddenWeightIn = neuron.ListOfWeightIn[layer_j];
								netValueN = netValueN + hiddenWeightIn * n.TrainSet[row][layer_j];
							}

							// hidden layer (1)
							netValueOutN = base.activationFnc(n.ActivationFnc, netValueN);
							neuron.OutputValue = netValueOutN;
						}
						else
						{
							neuron.OutputValue = 1.0;
						}

					}


					// hidden layer (2)
					double netValue = 0.0;
					double netValueOut = 0.0;
					for (int outLayer_i = 0; outLayer_i < n.OutputLayer.NumberOfNeuronsInLayer; outLayer_i++)
					{

						foreach (Neuron neuron in hiddenLayer.ListOfNeurons)
						{
							double hiddenWeightOut = neuron.ListOfWeightOut[outLayer_i];
							netValue = netValue + hiddenWeightOut * neuron.OutputValue;
						}

						netValueOut = activationFnc(n.ActivationFncOutputLayer, netValue);

						n.OutputLayer.ListOfNeurons[outLayer_i].OutputValue = netValueOut;

						// error
						estimatedOutput = netValueOut;
						realOutput = n.RealMatrixOutputSet[row][outLayer_i];
						double error = realOutput - estimatedOutput;
						n.OutputLayer.ListOfNeurons[outLayer_i].Error = error;
						sumError = sumError + Math.Pow(error, 2.0);

					}

					// mse
					double errorMean = sumError / n.OutputLayer.NumberOfNeuronsInLayer;
					n.ErrorMean = errorMean;

					n.ListOfHiddenLayer[hiddenLayer_i].ListOfNeurons = hiddenLayer.ListOfNeurons;

					hiddenLayer_i++;
				}
			}
			return n;
		}

		private NeuralNet backpropagation(NeuralNet n, int row)
		{

			List<Neuron> outputLayer = new List<Neuron>();
			outputLayer = n.OutputLayer.ListOfNeurons;

			List<Neuron> hiddenLayer = new List<Neuron>();
			hiddenLayer = n.ListOfHiddenLayer[0].ListOfNeurons;

			double error = 0.0;
			double netValue = 0.0;
			double sensibility = 0.0;

			foreach (Neuron neuron in outputLayer)
			{
				error = neuron.Error;
				netValue = neuron.OutputValue;
				sensibility = derivativeActivationFnc(n.ActivationFncOutputLayer, netValue) * error;

				neuron.Sensibility = sensibility;
			}

			n.OutputLayer.ListOfNeurons = outputLayer;

			foreach (Neuron neuron in hiddenLayer)
			{

				sensibility = 0.0;

				if (neuron.ListOfWeightIn.Count > 0)
				{
					List<double> listOfWeightsOut = new List<double>();

					listOfWeightsOut = neuron.ListOfWeightOut;

					double tempSensibility = 0.0;

					int weight_i = 0;
					foreach (double? weight in listOfWeightsOut)
					{
						tempSensibility = tempSensibility + ((double)weight * outputLayer[weight_i].Sensibility);
						weight_i++;
					}

					sensibility = derivativeActivationFnc(n.ActivationFnc, neuron.OutputValue) * tempSensibility;

					neuron.Sensibility = sensibility;

				}

			}

			// popravljanje grana (ucenje)
			for (int outLayer_i = 0; outLayer_i < n.OutputLayer.NumberOfNeuronsInLayer; outLayer_i++)
			{

				foreach (Neuron neuron in hiddenLayer)
				{

					double newWeight = neuron.ListOfWeightOut[outLayer_i] + (n.LearningRate * outputLayer[outLayer_i].Sensibility * neuron.OutputValue);

					neuron.ListOfWeightOut[outLayer_i] = newWeight;
				}

			}

            // popravljanje grana (ucenje)
            foreach (Neuron neuron in hiddenLayer)
			{

				List<double> hiddenLayerInputWeights = new List<double>();
				hiddenLayerInputWeights = neuron.ListOfWeightIn;

				if (hiddenLayerInputWeights.Count > 0)
				{

					int hidden_i = 0;
					double newWeight = 0.0;
					for (int i = 0; i < n.InputLayer.NumberOfNeuronsInLayer; i++)
					{

						newWeight = hiddenLayerInputWeights[hidden_i] + (n.LearningRate * neuron.Sensibility * n.TrainSet[row][i]);

						neuron.ListOfWeightIn[hidden_i] = newWeight;

						hidden_i++;
					}

				}

			}

			n.ListOfHiddenLayer[0].ListOfNeurons = hiddenLayer;

			return n;

		}
	}

}