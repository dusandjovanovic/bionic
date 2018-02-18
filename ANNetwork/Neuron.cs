using System;
using System.Collections.Generic;

namespace ai.net.neuralnet
{

	public class Neuron
	{

		private List<double> listOfWeightIn;
		private List<double> listOfWeightOut;
		private double outputValue;
		private double error;
		private double sensibility;

		public virtual double initNeuron()
		{
			Random r = new Random();
			return r.NextDouble();
		}

		public virtual List<double> ListOfWeightIn
		{
			get
			{
				return listOfWeightIn;
			}
			set
			{
				this.listOfWeightIn = value;
			}
		}


		public virtual List<double> ListOfWeightOut
		{
			get
			{
				return listOfWeightOut;
			}
			set
			{
				this.listOfWeightOut = value;
			}
		}


		public virtual double OutputValue
		{
			get
			{
				return outputValue;
			}
			set
			{
				this.outputValue = value;
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


		public virtual double Sensibility
		{
			get
			{
				return sensibility;
			}
			set
			{
				this.sensibility = value;
			}
		}


	}

}