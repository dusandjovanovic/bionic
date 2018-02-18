using System;
using System.Collections.Generic;

namespace ai.net.neuralnet
{

	public class OutputLayer : Layer
	{

		public virtual OutputLayer initLayer(OutputLayer outputLayer)
		{
			List<double> listOfWeightOutTemp = new List<double>();
			List<Neuron> listOfNeurons = new List<Neuron>();

			for (int i = 0; i < outputLayer.NumberOfNeuronsInLayer; i++)
			{
				Neuron neuron = new Neuron();

				listOfWeightOutTemp.Add(neuron.initNeuron());

				neuron.ListOfWeightOut = listOfWeightOutTemp;
				listOfNeurons.Add(neuron);

				listOfWeightOutTemp = new List<double>();
			}

			outputLayer.ListOfNeurons = listOfNeurons;

			return outputLayer;

		}

		public virtual void printLayer(OutputLayer outputLayer)
		{
			Console.WriteLine("### OUTPUT LAYER ###");
			int n = 1;
			foreach (Neuron neuron in outputLayer.ListOfNeurons)
			{
				Console.WriteLine("Neuron #" + n + ":");
				n++;
			}
		}

	}

}