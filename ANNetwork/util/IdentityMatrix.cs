namespace ai.net.neuralnet.util
{
	public class IdentityMatrix : Matrix
	{

		public IdentityMatrix(int order) : base(order,order)
		{
			for (int i = 0;i < order;i++)
			{
				for (int j = 0;j < order;j++)
				{
					setValue(i,j,(i == j)?1:0);
				}
			}
		}

		public virtual void setValue(int row, int column)
		{

		}

	}

}