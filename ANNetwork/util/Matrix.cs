namespace ai.net.neuralnet.util
{

	public class Matrix
	{
		private double[][] content;
		private int numberOfRows;
		private int numberOfColumns;


		private double? determinant_Renamed;

		public Matrix(int nRows, int nColumns)
		{
			numberOfRows = nRows;
			numberOfColumns = nColumns;

			content = RectangularArrays.ReturnRectangularDoubleArray(numberOfRows, numberOfColumns);
		}

		public Matrix(double[][] matrix)
		{
			numberOfRows = matrix.Length;
			numberOfColumns = matrix[0].Length;
			content = matrix;
		}

		public Matrix(double[] vector)
		{
			numberOfRows = 1;
			numberOfColumns = vector.Length;
			content = RectangularArrays.ReturnRectangularDoubleArray(numberOfRows, numberOfColumns);
			content[0] = vector;
		}

		public Matrix(Matrix a)
		{
			numberOfRows = a.NumberOfRows;
			numberOfColumns = a.NumberOfColumns;
			content = RectangularArrays.ReturnRectangularDoubleArray(numberOfRows, numberOfColumns);
			for (int i = 0;i < numberOfRows;i++)
			{
				for (int j = 0;j < numberOfColumns;j++)
				{
					setValue(i,j,a.getValue(i,j));
				}
			}
		}

		public virtual Matrix add(Matrix a)
		{
			int nRows = a.NumberOfRows;
			int nColumns = a.NumberOfColumns;

			if (numberOfRows != a.NumberOfRows)
			{
				throw new System.ArgumentException("Number of rows of both matrices must match");
			}

			if (numberOfColumns != a.NumberOfColumns)
			{
				throw new System.ArgumentException("Number of colmns of both matrices must match");
			}

			Matrix result = new Matrix(nRows,nColumns);

			for (int i = 0;i < nRows;i++)
			{
				for (int j = 0;j < nColumns;j++)
				{
					result.setValue(i, j, getValue(i,j) + a.getValue(i, j));
				}
			}

			return result;
		}

		public static Matrix add(Matrix a, Matrix b)
		{
			int nRows = a.NumberOfRows;
			int nColumns = a.NumberOfColumns;

			if (a.numberOfRows != b.NumberOfRows)
			{
				throw new System.ArgumentException("Number of rows of both matrices must match");
			}

			if (a.numberOfColumns != b.NumberOfColumns)
			{
				throw new System.ArgumentException("Number of colmns of both matrices must match");
			}

			Matrix result = new Matrix(nRows,nColumns);

			for (int i = 0;i < nRows;i++)
			{
				for (int j = 0;j < nColumns;j++)
				{
					result.setValue(i, j, a.getValue(i,j) + b.getValue(i, j));
				}
			}

			return result;
		}

		public virtual Matrix subtract(Matrix a)
		{
			int nRows = a.NumberOfRows;
			int nColumns = a.NumberOfColumns;

			if (numberOfRows != a.NumberOfRows)
			{
				throw new System.ArgumentException("Number of rows of both matrices must match");
			}

			if (numberOfColumns != a.NumberOfColumns)
			{
				throw new System.ArgumentException("Number of colmns of both matrices must match");
			}

			Matrix result = new Matrix(nRows,nColumns);

			for (int i = 0;i < nRows;i++)
			{
				for (int j = 0;j < nColumns;j++)
				{
					result.setValue(i, j, getValue(i,j) - a.getValue(i, j));
				}
			}

			return result;
		}

		public static Matrix subtract(Matrix a, Matrix b)
		{
			int nRows = a.NumberOfRows;
			int nColumns = a.NumberOfColumns;

			if (a.numberOfRows != b.NumberOfRows)
			{
				throw new System.ArgumentException("Number of rows of both matrices must match");
			}

			if (a.numberOfColumns != b.NumberOfColumns)
			{
				throw new System.ArgumentException("Number of colmns of both matrices must match");
			}

			Matrix result = new Matrix(nRows,nColumns);

			for (int i = 0;i < nRows;i++)
			{
				for (int j = 0;j < nColumns;j++)
				{
					result.setValue(i, j, a.getValue(i,j) - b.getValue(i, j));
				}
			}
			return result;
		}

		public virtual Matrix transpose()
		{
			Matrix result = new Matrix(numberOfColumns,numberOfRows);
			for (int i = 0;i < numberOfRows;i++)
			{
				for (int j = 0;j < numberOfColumns;j++)
				{
					result.setValue(j, i, getValue(i,j));
				}
			}
			return result;
		}

		public static Matrix transpose(Matrix a)
		{
			Matrix result = new Matrix(a.NumberOfColumns,a.NumberOfRows);
			for (int i = 0;i < a.NumberOfRows;i++)
			{
				for (int j = 0;j < a.NumberOfColumns;j++)
				{
					result.setValue(j, i, a.getValue(i,j));
				}
			}
			return result;
		}

		public virtual Matrix multiply(Matrix a)
		{
			Matrix result = new Matrix(NumberOfRows,a.NumberOfColumns);
			if (NumberOfColumns != a.NumberOfRows)
			{
				throw new System.ArgumentException("Number of Columns of first Matrix must match the number of Rows of second Matrix");
			}

			for (int i = 0;i < NumberOfRows;i++)
			{
				for (int j = 0;j < a.NumberOfColumns;j++)
				{
					double value = 0;
					for (int k = 0;k < a.NumberOfRows;k++)
					{
						value += getValue(i,k) * a.getValue(k,j);
					}
					result.setValue(i, j, value);
				}
			}
			return result;
		}

		public virtual Matrix multiply(double a)
		{
			Matrix result = new Matrix(NumberOfRows,NumberOfColumns);

			for (int i = 0;i < NumberOfRows;i++)
			{
				for (int j = 0;j < NumberOfColumns;j++)
				{
					result.setValue(i, j, getValue(i,j) * a);
				}
			}

			return result;
		}

		public static Matrix multiply(Matrix a, Matrix b)
		{
			Matrix result = new Matrix(a.NumberOfRows,b.NumberOfColumns);
			if (a.NumberOfColumns != b.NumberOfRows)
			{
				throw new System.ArgumentException("Number of Columns of first Matrix must match the number of Rows of second Matrix");
			}

			for (int i = 0;i < a.NumberOfRows;i++)
			{
				for (int j = 0;j < b.NumberOfColumns;j++)
				{
					double value = 0;
					for (int k = 0;k < b.NumberOfRows;k++)
					{
						value += a.getValue(i,k) * b.getValue(k,j);
					}
					result.setValue(i, j, value);
				}
			}
			return result;
		}

		public static Matrix multiply(Matrix a, double b)
		{
			Matrix result = new Matrix(a.NumberOfRows,a.NumberOfColumns);

			for (int i = 0;i < a.NumberOfRows;i++)
			{
				for (int j = 0;j < a.NumberOfColumns;j++)
				{
					result.setValue(i, j, a.getValue(i,j) * b);
				}
			}

			return result;
		}

		public virtual Matrix[] LUdecomposition()
		{
			Matrix[] result = new Matrix[2];
			Matrix LU = new Matrix(this);
			Matrix L = new Matrix(LU.NumberOfRows,LU.NumberOfColumns);
			L.setZeros();
			L.setValue(0,0,1.0);
			for (int i = 1;i < LU.NumberOfRows;i++)
			{
				L.setValue(i,i,1.0);
				for (int j = 0;j < i;j++)
				{
					double multiplier = -LU.getValue(i, j) / LU.getValue(j, j);
					LU.sumRowByRow(i, j, multiplier);
					L.setValue(i, j, -multiplier);
				}
			}
			Matrix U = new Matrix(LU);
			result[0] = L;
			result[1] = U;
			return result;
		}

		public virtual void multiplyRow(int row, double multiplier)
		{
			if (row > NumberOfRows)
			{
				throw new System.ArgumentException("Row index must be lower than the number of rows");
			}
			sumRowByRow(row,row,multiplier);
		}

		public virtual void sumRowByRow(int row, int rowSum, double multiplier)
		{
			if (row > NumberOfRows)
			{
				throw new System.ArgumentException("Row index must be lower than the number of rows");
			}
			if (rowSum > NumberOfRows)
			{
				throw new System.ArgumentException("Row index must be lower than the number of rows");
			}
			for (int j = 0;j < NumberOfColumns;j++)
			{
				setValue(row,j,getValue(row,j) + getValue(rowSum,j) * multiplier);
			}
		}

		public virtual double determinant()
		{
			if (determinant_Renamed != null)
			{
				return determinant_Renamed.Value;
			}

			double result = 0;
			if (NumberOfRows != NumberOfColumns)
			{
				throw new System.ArgumentException("Only square matrices can have determinant");
			}

			if (NumberOfColumns == 1)
			{
				return content[0][0];
			}
			else if (NumberOfColumns == 2)
			{
				return (content[0][0] * content[1][1]) - (content[1][0] * content[0][1]);
			}
			else
			{
				Matrix[] LU = LUdecomposition();
				return LU[1].multiplyDiagonal();
			}
	//        else{
	//            for(int k=0;k<getNumberOfColumns();k++){
	//                Matrix minorMatrix = subMatrix(0,k);
	//                result+= ((k%2==0)? getValue(0,k): -getValue(0,k)) * minorMatrix.determinant();
	//            }
	//            setDeterminant(result);
	//            return result;
	//        }
		}

		private double Determinant
		{
			set
			{
				determinant_Renamed = value;
			}
			get
			{
				if (determinant_Renamed != null)
				{
					return determinant_Renamed.Value;
				}
				else
				{
					return determinant();
				}
			}
		}

		public static double determinant(Matrix a)
		{
			if (a.determinant_Renamed != null)
			{
				return a.Determinant;
			}

			if (a.NumberOfRows != a.NumberOfColumns)
			{
				throw new System.ArgumentException("Only square matrices can have determinant");
			}

			if (a.NumberOfColumns == 1)
			{
				return a.getValue(0, 0);
			}
			else if (a.NumberOfColumns == 2)
			{
				return (a.getValue(0, 0) * a.getValue(1, 1)) - (a.getValue(1, 0) * a.getValue(0, 1));
			}
			else
			{
				Matrix[] LU = a.LUdecomposition();
				return LU[1].multiplyDiagonal();
			}
	//        for(int k=0;k<a.getNumberOfColumns();k++){
	//            Matrix minorMatrix = a.subMatrix(0, k);
	//            result+= ((k%2==0)? a.getValue(0,k): -a.getValue(0,k)) * minorMatrix.determinant();
	//        }
	//        a.setDeterminant(result);

		}

		public virtual double multiplyDiagonal()
		{
			double result = 1;
			for (int i = 0;i < NumberOfColumns;i++)
			{
				result *= getValue(i,i);
			}
			return result;
		}

		public virtual Matrix subMatrix(int row, int column)
		{
			if (row > NumberOfRows)
			{
				throw new System.ArgumentException("Row index out of matrix`s limits");
			}
			if (column > NumberOfColumns)
			{
				throw new System.ArgumentException("Column index out of matrix`s limits");
			}

			Matrix result = new Matrix(NumberOfRows - 1,NumberOfColumns - 1);
			for (int i = 0;i < NumberOfRows;i++)
			{
				if (i == row)
				{
					continue;
				}
				for (int j = 0;j < NumberOfRows;j++)
				{
					if (j == column)
					{
						continue;
					}
					result.setValue((i < row?i:i - 1), (j < column?j:j - 1), getValue(i,j));
				}
			}
			return result;
		}

		public static Matrix subMatrix(Matrix a, int row, int column)
		{
			if (row > a.NumberOfRows)
			{
				throw new System.ArgumentException("Row index out of matrix`s limits");
			}
			if (column > a.NumberOfColumns)
			{
				throw new System.ArgumentException("Column index out of matrix`s limits");
			}

			Matrix result = new Matrix(a.NumberOfRows - 1,a.NumberOfColumns - 1);
			for (int i = 0;i < a.NumberOfRows;i++)
			{
				if (i == row)
				{
					continue;
				}
				for (int j = 0;j < a.NumberOfRows;j++)
				{
					if (j == column)
					{
						continue;
					}
					result.setValue((i < row?i:i - 1), (j < column?j:j - 1), a.getValue(i,j));
				}
			}
			return result;
		}

		public virtual Matrix coFactors()
		{
			Matrix result = new Matrix(NumberOfRows,NumberOfColumns);
			for (int i = 0;i < NumberOfRows;i++)
			{
				for (int j = 0;j < NumberOfColumns;j++)
				{
					result.setValue(i, j, subMatrix(i,j).determinant());
				}
			}
			return result;
		}

		public static Matrix coFactors(Matrix a)
		{
			Matrix result = new Matrix(a.NumberOfRows,a.NumberOfColumns);
			for (int i = 0;i < a.NumberOfRows;i++)
			{
				for (int j = 0;j < a.NumberOfColumns;j++)
				{
					result.setValue(i, j, a.subMatrix(i,j).determinant());
				}
			}
			return result;
		}

		public virtual Matrix inverse()
		{
			Matrix result = coFactors().transpose().multiply((1 / determinant()));
			return result;
		}

		public static Matrix inverse(Matrix a)
		{
			if (a.Determinant == 0)
			{
				throw new System.ArgumentException("This matrix is not inversible");
			}
			Matrix result = a.coFactors().transpose().multiply((1 / a.determinant()));
			return result;
		}

		public virtual double getValue(int i, int j)
		{
			if (i >= numberOfRows)
			{
				throw new System.ArgumentException("Number of Row outside the matrix`s limits");
			}
			if (j >= numberOfColumns)
			{
				throw new System.ArgumentException("Number of Column outside the matrix`s limits");
			}

			return content[i][j];
		}

		public virtual void setValue(int i, int j, double value)
		{
			if (i >= numberOfRows)
			{
				throw new System.ArgumentException("Number of Row outside the matrix`s limits");
			}
			if (j >= numberOfColumns)
			{
				throw new System.ArgumentException("Number of Column outside the matrix`s limits");
			}

			content[i][j] = value;
			determinant_Renamed = null;
		}

		public virtual void setZeros()
		{
			for (int i = 0;i < NumberOfRows;i++)
			{
				for (int j = 0;j < NumberOfColumns;j++)
				{
					setValue(i,j,0.0);
				}
			}
		}

		public virtual void setOnes()
		{
			for (int i = 0;i < NumberOfRows;i++)
			{
				for (int j = 0;j < NumberOfColumns;j++)
				{
					setValue(i,j,1.0);
				}
			}
		}

		public virtual int NumberOfRows
		{
			get
			{
				return numberOfRows;
			}
		}

		public virtual int NumberOfColumns
		{
			get
			{
				return numberOfColumns;
			}
		}


	}

}