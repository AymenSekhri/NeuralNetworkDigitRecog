using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace Handwritten_Digits_Neural_Network
{

    public partial class Form1 : Form
    {

        public Form1()
        {
            InitializeComponent();
        }
        byte[][] DataSetInputs;
        byte[] DataSetLables;
        byte[][] TestSetInputs;
        byte[] TestSetLables;
        
        void LoadDataSet()
        {
            byte[] file = System.IO.File.ReadAllBytes("train-images.idx3-ubyte");
            byte[] file2 = System.IO.File.ReadAllBytes("train-labels.idx1-ubyte");
            byte[] sizear = new byte[4];
            Array.Copy(file, 4, sizear, 0, 4);
            Array.Reverse(sizear);
            int DataSetSize = BitConverter.ToInt32(sizear, 0);
            DataSetInputs = new byte[DataSetSize][];
            DataSetLables = new byte[DataSetSize];
            Array.Copy(file2, 8, DataSetLables, 0, DataSetSize);
            for (int y = 0; y < DataSetSize; y++)
            {
                DataSetInputs[y] = new byte[784];
                Array.Copy(file, 16 + 784 * y, DataSetInputs[y], 0, 784);
            }
        }
        void LoadTestSet()
        {
            byte[] file = System.IO.File.ReadAllBytes("t10k-images.idx3-ubyte");
            byte[] file2 = System.IO.File.ReadAllBytes("t10k-labels.idx1-ubyte");
            byte[] sizear = new byte[4];
            Array.Copy(file, 4, sizear, 0, 4);
            Array.Reverse(sizear);
            int DataSetSize = BitConverter.ToInt32(sizear, 0);
            TestSetInputs = new byte[DataSetInputs.Length][];
            TestSetLables = new byte[DataSetSize];
            Array.Copy(file2, 8, TestSetLables, 0, DataSetSize);
            for (int y = 0; y < DataSetSize; y++)
            {
                TestSetInputs[y] = new byte[784];
                Array.Copy(file, 16 + 784 * y, TestSetInputs[y], 0, 784);
            }
        }
        NeuralNetwork N;
        private void Form1_Load(object sender, EventArgs e)
        {
            LoadDataSet();
            LoadTestSet();
            N = new NeuralNetwork(new int[] { 784, 50, 10 });
        }
        void ValidationTests()
        {
            //Validation Set Test
            double percent = 0;
            for (int i = 0; i < 10000; i++)
            {
                Vector<double> Inputs = Vector<double>.Build.DenseOfArray(TestSetInputs[i].Select(x => Convert.ToDouble(x) / 255).ToArray());
                if (N.FeedForward(Inputs).AbsoluteMaximumIndex() == TestSetLables[i])
                {
                    percent++;
                }
            }
            percent = 100 * percent / 10000;
            label2.Text = "Precision : " + percent.ToString() + "%";

        }
        int num = 0;
        private void button1_Click(object sender, EventArgs e)
        {
            Bitmap pic = new Bitmap(28, 28);
            int startofpic = num * 784 + 16;
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    pic.SetPixel(x, y, GetColor(TestSetInputs[num][x + 28 * y]));
                }
            }
            pictureBox1.Image = pic;
            Vector<double> Inputs = Vector<double>.Build.DenseOfArray(TestSetInputs[num].Select(x => Convert.ToDouble(x) / 255).ToArray());
            label1.Text = "Recognized Digit : " + N.FeedForward(Inputs).AbsoluteMaximumIndex().ToString();
            num++;
        }
        Color GetColor(byte Gray)
        {
            return Color.FromArgb(255 - Gray, 255 - Gray, 255 - Gray);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog fd = new OpenFileDialog();
            if (fd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                Image img;
                using (var bmpTemp = new Bitmap(fd.FileName))
                {
                    img = new Bitmap(bmpTemp);
                }

                Bitmap pic = new Bitmap(img, 28, 28);

                byte[] pixelarry = new byte[28 * 28];
                int startofpic = num * 784 + 16;
                Color p;
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        p = pic.GetPixel(x, y);
                        int a = p.A;
                        int r = p.R;
                        int g = p.G;
                        int b = p.B;
                        int avg = (r + g + b) / 3;
                        pixelarry[x + 28 * y] = Convert.ToByte(255 - avg);
                    }
                }
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        pic.SetPixel(x, y, GetColor(pixelarry[x + 28 * y]));
                    }
                }
                pictureBox1.Image = pic;
                Vector<double> Inputs = Vector<double>.Build.DenseOfArray(pixelarry.Select(x => Convert.ToDouble(x) / 255).ToArray());
                label1.Text = "Recognized Digit : " + N.FeedForward(Inputs).AbsoluteMaximumIndex().ToString();

            }




        }

        private void button3_Click(object sender, EventArgs e)
        {
            OpenFileDialog fd = new OpenFileDialog();
            if (fd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                N.LoadNetwork(fd.FileName);
            }

        }

        private void button4_Click(object sender, EventArgs e)
        {
            long milliseconds = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond; ;


            Vector<double>[] Inputs = new Vector<double>[DataSetInputs.Length];
            Vector<double>[] Outputs = new Vector<double>[DataSetInputs.Length]; ;
            for (int i = 0; i < DataSetInputs.Length; i++)
            {
                Inputs[i] = Vector<double>.Build.DenseOfArray(DataSetInputs[i].Select(x => Convert.ToDouble(x) / 255).ToArray());
                Outputs[i] = Vector<double>.Build.Dense(10);
                Outputs[i][DataSetLables[i]] = 1;
            }
            N.Train(Inputs, Outputs, 0.5);
            ValidationTests();
            long mm = ((DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond) - milliseconds)/1000;
            if (MessageBox.Show("Training Completed!\nDuration : " + mm.ToString() + " s" + "\nSave Network ?", "", MessageBoxButtons.YesNo) == System.Windows.Forms.DialogResult.Yes)
            {
                SaveFileDialog fd = new SaveFileDialog();
                if (fd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    N.SaveNetwork(fd.FileName);
                }
            }
            ;
            

        }


    }

}
