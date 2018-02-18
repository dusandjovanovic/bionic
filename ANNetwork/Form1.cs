using ai.net.neuralnet;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace ANNetwork
{

    using ActivationFncENUM = ai.net.neuralnet.learn.Training.ActivationFncENUM;
    using TrainingTypesENUM = ai.net.neuralnet.learn.Training.TrainingTypesENUM;
    using Data = ai.net.neuralnet.util.Data;
    using NormalizationTypesENUM = ai.net.neuralnet.util.Data.NormalizationTypesENUM;

    public partial class Form1 : Form
    {
        private NeuralNet neuralnet;

        public Form1()
        {
            InitializeComponent();
        }

        public Form1(NeuralNet output)
        {
            InitializeComponent();
            neuralnet = output;
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        public void NeuralExe()
        {
            Data weatherDataInput = new Data("data", "input.csv");
            Data weatherDataOutput = new Data("data", "output.csv");

            Data weatherDataInputTestRNA = new Data("data", "input_test.csv");
            Data weatherDataOutputTestRNA = new Data("data", "output_test.csv");

            Data.NormalizationTypesENUM NORMALIZATION_TYPE = Data.NormalizationTypesENUM.MAX_MIN_EQUALIZED;

            try
            {
                double[][] matrixInput = weatherDataInput.rawData2Matrix(weatherDataInput);
                double[][] matrixOutput = weatherDataOutput.rawData2Matrix(weatherDataOutput);

                double[][] matrixInputTestRNA = weatherDataOutput.rawData2Matrix(weatherDataInputTestRNA);
                double[][] matrixOutputTestRNA = weatherDataOutput.rawData2Matrix(weatherDataOutputTestRNA);

                double[][] matrixInputNorm = weatherDataInput.normalize(matrixInput, NORMALIZATION_TYPE);
                double[][] matrixOutputNorm = weatherDataOutput.normalize(matrixOutput, NORMALIZATION_TYPE);

                double[][] matrixInputTestRNANorm = weatherDataOutput.normalize(matrixInputTestRNA, NORMALIZATION_TYPE);
                double[][] matrixOutputTestRNANorm = weatherDataOutput.normalize(matrixOutputTestRNA, NORMALIZATION_TYPE);

                NeuralNet n1 = new NeuralNet();
                n1 = n1.initNet(4, 1, 4, 1);

                n1.TrainSet = matrixInputNorm;
                n1.RealMatrixOutputSet = matrixOutputNorm;

                n1.MaxEpochs = 1000;
                n1.TargetError = 0.00001;
                n1.LearningRate = 0.5;
                n1.TrainType = TrainingTypesENUM.BACKPROPAGATION;
                n1.ActivationFnc = ActivationFncENUM.SIGLOG;
                n1.ActivationFncOutputLayer = ActivationFncENUM.LINEAR;

                NeuralNet n1Trained = new NeuralNet();

                n1Trained = n1.trainNet(n1);
                
                for (int i = 0; i < n1.ListOfMSE.Count; i++)
                {
                    chart1.Series["mse"].Points.AddXY(i + 1, n1.ListOfMSE[i]);
                }

                double[][] matrixOutputRNA = n1Trained.getNetOutputValues(n1Trained);
                double[][] matrixOutputRNADenorm = (new Data()).denormalize(matrixOutput, matrixOutputRNA, NORMALIZATION_TYPE);

                for (int i = 0; i < matrixOutput.Length; i++)
                {
                    chart2.Series["stvarni"].Points.AddXY(i + 1, matrixOutput[i][0]);
                    chart2.Series["predvidjeni"].Points.AddXY(i + 1, matrixOutputRNADenorm[i][0]);
                }


                n1Trained.TrainSet = matrixInputTestRNANorm;
                n1Trained.RealMatrixOutputSet = matrixOutputTestRNANorm;

                double[][] matrixOutputRNATest = n1Trained.getNetOutputValues(n1Trained);
                double[][] matrixOutputRNADenormTest = (new Data()).denormalize(matrixOutputTestRNA, matrixOutputRNATest, NORMALIZATION_TYPE);

                for (int i = 0; i < matrixOutputTestRNA.Length; i++)
                {
                    chart3.Series["stvarni"].Points.AddXY(i + 1, matrixOutputTestRNA[i][0]);
                    chart3.Series["predvidjeni"].Points.AddXY(i + 1, matrixOutputRNADenormTest[i][0]);
                }

            }
            catch (IOException e)
            {
                Console.WriteLine(e.ToString());
                Console.Write(e.StackTrace);
            }

        }

        private void executeToolStripMenuItem_Click(object sender, EventArgs e)
        {
            toolStripStatusLabel1.Text = "Neural network :: training";
            toolStripStatusLabel1.ForeColor = Color.Red;
            statusStrip1.Refresh();

            NeuralExe();

            toolStripStatusLabel1.Text = "Neural network :: done";
            toolStripStatusLabel1.ForeColor = Color.Blue;
            statusStrip1.Refresh();
        }
    }
}
