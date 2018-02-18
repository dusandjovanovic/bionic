using System;
using System.Collections.Generic;

namespace ai.net.neuralnet
{

	public class HiddenLayer : Layer
	{

		public virtual List<HiddenLayer> initLayer(HiddenLayer hiddenLayer, List<HiddenLayer> listOfHiddenLayer, InputLayer inputLayer, OutputLayer outputLayer)
		{

			List<double> listOfWeightIn = new List<double>();
			List<double> listOfWeightOut = new List<double>();
			List<Neuron> listOfNeurons = new List<Neuron>();

			int numberOfHiddenLayers = listOfHiddenLayer.Count;

			for (int hdn_i = 0; hdn_i < numberOfHiddenLayers; hdn_i++)
			{
				for (int neuron_i = 0; neuron_i < hiddenLayer.NumberOfNeuronsInLayer; neuron_i++)
				{
					Neuron neuron = new Neuron();

					int limitIn = 0;
					int limitOut = 0;

					if (hdn_i == 0)
					{ // first
						limitIn = inputLayer.NumberOfNeuronsInLayer;
						if (numberOfHiddenLayers > 1)
						{
							limitOut = listOfHiddenLayer[hdn_i + 1].NumberOfNeuronsInLayer;
						}
						else if (numberOfHiddenLayers == 1)
						{
							limitOut = outputLayer.NumberOfNeuronsInLayer;
						}
					}
					else if (hdn_i == numberOfHiddenLayers - 1)
					{ // last
						limitIn = listOfHiddenLayer[hdn_i - 1].NumberOfNeuronsInLayer;
						limitOut = outputLayer.NumberOfNeuronsInLayer;
					}
					else
					{ // middle
						limitIn = listOfHiddenLayer[hdn_i - 1].NumberOfNeuronsInLayer;
						limitOut = listOfHiddenLayer[hdn_i + 1].NumberOfNeuronsInLayer;
					}

					limitIn = limitIn - 1; // bias no
					limitOut = limitOut - 1;

					if (neuron_i >= 1)
					{
						for (int k = 0; k <= limitIn; k++)
						{
							listOfWeightIn.Add(neuron.initNeuron());
							//listOfWeightIn.add(neuron.initNeuron(k, neuron_i, 1));
						}
					}
					for (int k = 0; k <= limitOut; k++)
					{
						listOfWeightOut.Add(neuron.initNeuron());
						//listOfWeightOut.add(neuron.initNeuron(k, neuron_i, 2));
					}

					neuron.ListOfWeightIn = listOfWeightIn;
					neuron.ListOfWeightOut = listOfWeightOut;
					listOfNeurons.Add(neuron);

					listOfWeightIn = new List<double>();
					listOfWeightOut = new List<double>();

				}

				listOfHiddenLayer[hdn_i].ListOfNeurons = listOfNeurons;

				listOfNeurons = new List<Neuron>();

			}

			return listOfHiddenLayer;

		}

		public virtual void printLayer(List<HiddenLayer> listOfHiddenLayer)
		{
			if (listOfHiddenLayer.Count > 0)
			{
				Console.WriteLine("### HIDDEN LAYER ###");
				int h = 1;
				foreach (HiddenLayer hiddenLayer in listOfHiddenLayer)
				{
					Console.WriteLine("Hidden Layer #" + h);
					int n = 1;
					foreach (Neuron neuron in hiddenLayer.ListOfNeurons)
					{
						Console.WriteLine("Neuron #" + n);
						n++;
					}
					h++;
				}
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