using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace ai.net.neuralnet.util
{

	public class Data
	{

		private string path;
		private string fileName;
		public enum NormalizationTypesENUM
		{
			MAX_MIN,
			MAX_MIN_EQUALIZED
		}

		public Data(string path, string fileName)
		{
			this.path = path;
			this.fileName = fileName;
		}

		public Data()
		{

		}

		public virtual double[][] joinArrays(List<double[][]> listOfArraysToJoin)
		{

			int rows = listOfArraysToJoin[0].Length;
			int cols = listOfArraysToJoin.Count;

			double[][] matrix = RectangularArrays.ReturnRectangularDoubleArray(rows, cols);

			for (int cols_i = 0; cols_i < cols; cols_i++)
			{

				double[][] a = listOfArraysToJoin[cols_i];

				for (int rows_i = 0; rows_i < rows; rows_i++)
				{

					matrix[rows_i][cols_i] = a[rows_i][0];

				}

			}

			return matrix;

		}

		public virtual double[][] normalize(double[][] rawMatrix, NormalizationTypesENUM normType)
		{

			int rows = rawMatrix.Length;
			int cols = rawMatrix[0].Length;

			double[][] matrixNorm = RectangularArrays.ReturnRectangularDoubleArray(rows, cols);

			for (int cols_i = 0; cols_i < cols; cols_i++)
			{

				List<double> listColumn = new List<double>();

				for (int rows_j = 0; rows_j < rows; rows_j++)
				{
					listColumn.Add(rawMatrix[rows_j][cols_i]);
				}

				double minColValue = listColumn.Min();
				double maxColValue = listColumn.Max();


				for (int rows_j = 0; rows_j < rows; rows_j++)
				{
					switch (normType)
					{
					case ai.net.neuralnet.util.Data.NormalizationTypesENUM.MAX_MIN:
						matrixNorm[rows_j][cols_i] = rawMatrix[rows_j][cols_i] / Math.Abs(maxColValue);
						break;
					case ai.net.neuralnet.util.Data.NormalizationTypesENUM.MAX_MIN_EQUALIZED:
						if (cols_i > 0)
						{
							matrixNorm[rows_j][cols_i] = (rawMatrix[rows_j][cols_i] - minColValue) / (maxColValue - minColValue);
						}
						else
						{
							matrixNorm[rows_j][cols_i] = rawMatrix[rows_j][cols_i];
						}
						break;
					default:
						throw new System.ArgumentException(normType + " does not exist in NormalizationTypesENUM");
					}

				}

			}

			return matrixNorm;

		}

		public virtual double[][] denormalize(double[][] rawMatrix, double[][] matrixNorm, NormalizationTypesENUM normType)
		{

			int rows = matrixNorm.Length;
			int cols = matrixNorm[0].Length;

			double[][] matrixDenorm = RectangularArrays.ReturnRectangularDoubleArray(rows, cols);

			for (int cols_i = 0; cols_i < cols; cols_i++)
			{

				List<double> listColumn = new List<double>();

				for (int rows_j = 0; rows_j < rows; rows_j++)
				{
					listColumn.Add(rawMatrix[rows_j][cols_i]);
				}

				double minColValue = listColumn.Min();
				double maxColValue = listColumn.Max();

				for (int rows_j = 0; rows_j < rows; rows_j++)
				{
					switch (normType)
					{
					case ai.net.neuralnet.util.Data.NormalizationTypesENUM.MAX_MIN:
						matrixDenorm[rows_j][cols_i] = matrixNorm[rows_j][cols_i] * Math.Abs(maxColValue);
						break;
					case ai.net.neuralnet.util.Data.NormalizationTypesENUM.MAX_MIN_EQUALIZED:
						if (cols_i > 0)
						{
							matrixDenorm[rows_j][cols_i] = (matrixNorm[rows_j][cols_i] * (maxColValue - minColValue)) + minColValue;
						}
						else
						{
							matrixDenorm[rows_j][cols_i] = matrixNorm[rows_j][cols_i];
						}
						break;
					default:
						throw new System.ArgumentException(normType + " does not exist in NormalizationTypesENUM");
					}

				}

			}

			return matrixDenorm;

		}

		public virtual double[][] rawData2Matrix(Data r)
		{

			string fullPath = defineAbsoluteFilePath(r);

			System.IO.StreamReader buffer = new System.IO.StreamReader(fullPath);

			try
			{
				StringBuilder builder = new StringBuilder();

				string line = buffer.ReadLine();

				int columns = line.Split(',').Length;
				int rows = 0;
				while (!string.ReferenceEquals(line, null))
				{
					builder.Append(line);
					builder.Append(Environment.NewLine);
					line = buffer.ReadLine();
					rows++;
				}

				double[][] matrix = RectangularArrays.ReturnRectangularDoubleArray(rows, columns);
				string everything = builder.ToString();

                try
                {
                    rows = 0;
                    using (StringReader sr = new StringReader(everything))
                    {
                        while ((line = sr.ReadLine()) != null)
                        {
                            string[] strVector = line.Split(',');
                            for (int i = 0; i < strVector.Length; i++)
                            {
                                matrix[rows][i] = double.Parse(strVector[i]);
                            }
                            rows++;
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("The file could not be read:");
                    Console.WriteLine(e.Message);
                }

                return matrix;
			}
			finally
			{
				buffer.Close();
			}

		}

		private string defineAbsoluteFilePath(Data r)
		{

			string absoluteFilePath = "";

			string workingDir = Directory.GetCurrentDirectory();

			absoluteFilePath = workingDir + "\\" + r.Path + "\\" + r.FileName;

            if (File.Exists(absoluteFilePath))
			{
				Console.WriteLine("File found!");
				Console.WriteLine(absoluteFilePath);
			}
			else
			{
				Console.Error.WriteLine("File did not find...");
			}

			return absoluteFilePath;

		}

		public virtual string Path
		{
			get
			{
				return path;
			}
			set
			{
				this.path = value;
			}
		}


		public virtual string FileName
		{
			get
			{
				return fileName;
			}
			set
			{
				this.fileName = value;
			}
		}


	}

    class Scanner : System.IO.StringReader
    {
        string currentWord;

        public Scanner(string source) : base(source)
        {
            readNextWord();
        }

        private void readNextWord()
        {
            System.Text.StringBuilder sb = new StringBuilder();
            char nextChar;
            int next;
            do
            {
                next = this.Read();
                if (next < 0)
                    break;
                nextChar = (char)next;
                if (char.IsWhiteSpace(nextChar))
                    break;
                sb.Append(nextChar);
            } while (true);
            while ((this.Peek() >= 0) && (char.IsWhiteSpace((char)this.Peek())))
                this.Read();
            if (sb.Length > 0)
                currentWord = sb.ToString();
            else
                currentWord = null;
        }

        public bool hasNextInt()
        {
            if (currentWord == null)
                return false;
            int dummy;
            return int.TryParse(currentWord, out dummy);
        }

        public int nextInt()
        {
            try
            {
                return int.Parse(currentWord);
            }
            finally
            {
                readNextWord();
            }
        }

        public bool hasNextDouble()
        {
            if (currentWord == null)
                return false;
            double dummy;
            return double.TryParse(currentWord, out dummy);
        }

        public double nextDouble()
        {
            try
            {
                return double.Parse(currentWord);
            }
            finally
            {
                readNextWord();
            }
        }

        public bool hasNext()
        {
            return currentWord != null;
        }
    }
}