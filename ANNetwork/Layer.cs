using System.Collections.Generic;

namespace ai.net.neuralnet
{

	public abstract class Layer
	{

		private List<Neuron> listOfNeurons;
		protected internal int numberOfNeuronsInLayer;

		public virtual void printLayer()
		{
		}

		public virtual List<Neuron> ListOfNeurons
		{
			get
			{
				return listOfNeurons;
			}
			set
			{
				this.listOfNeurons = value;
			}
		}


		public virtual int NumberOfNeuronsInLayer
		{
			get
			{
				return numberOfNeuronsInLayer;
			}
			set
			{
				this.numberOfNeuronsInLayer = value;
			}
		}





	}

}