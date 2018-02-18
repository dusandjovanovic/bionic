using System;
using System.Collections.Generic;

namespace ai.net.neuralnet
{

	public class InputLayer : Layer
	{

		public virtual InputLayer initLayer(InputLayer inputLayer)
		{

			List<double> listOfWeightInTemp = new List<double>();
			List<Neuron> listOfNeurons = new List<Neuron>();

			for (int i = 0; i < inputLayer.NumberOfNeuronsInLayer; i++)
			{
				Neuron neuron = new Neuron();

				listOfWeightInTemp.Add(neuron.initNeuron());
				//listOfWeightInTemp.add( neuron.initNeuron( i ) );

				neuron.ListOfWeightIn = listOfWeightInTemp;
				listOfNeurons.Add(neuron);

				listOfWeightInTemp = new List<double>();
			}

			inputLayer.ListOfNeurons = listOfNeurons;

			return inputLayer;
		}

		public virtual void printLayer(InputLayer inputLayer)
		{
			Console.WriteLine("### INPUT LAYER ###");
			int n = 1;
			foreach (Neuron neuron in inputLayer.ListOfNeurons)
			{
				Console.WriteLine("Neuron #" + n + ":");
				n++;
			}
		}


		public override int NumberOfNeuronsInLayer
		{
			set
			{
				this.numberOfNeuronsInLayer = value + 1; // bias
			}
		}


	}

}