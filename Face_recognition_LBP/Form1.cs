using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.GPU;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.Util;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
//using System.Threading.Tasks;
using System.Windows.Forms;

namespace Face_recognition_LBP
{
    public partial class Form1 : Form
    {
        #region Variables
        const string haarcascade = "haarcascade_frontalface_default.xml";
        const string haarcascade_cuda = "haarcascade_frontalface_default_cuda.xml";
        bool GPU_classifier_selector = false;
        int counter, fps;
        GpuCascadeClassifier GPU_classifier;
        CascadeClassifier classifier;
        Capture capture;
        FaceRecognizer recognizer;
        Image<Gray, Byte> frame_gray;
        Image<Bgr, Byte> frame_BGR;
        Rectangle[] detected_faces;
        MCvFont font;
        List<Image<Gray, Byte>> imagesList = new List<Image<Gray, byte>>();
        List<int> imagesLabels_indices = new List<int>();
        List<string> imagesLabels = new List<string>();
        Image<Gray, Byte> temp;

        #endregion

        public Form1()
        {
            InitializeComponent();
            recognizer = new LBPHFaceRecognizer(1, 8, 8, 9, 65);

            classifier = new CascadeClassifier(haarcascade);
            GPU_classifier = new GpuCascadeClassifier(haarcascade_cuda);

            font = new MCvFont(Emgu.CV.CvEnum.FONT.CV_FONT_HERSHEY_TRIPLEX, 0.5, 0.5);
            if (File.Exists(@"traningdata.xml"))
            {
                recognizer.Load(@"traningdata.xml");
            }
            else
            {

                foreach (var file in Directory.GetFiles(Application.StartupPath + @"\Traning Faces\"))
                {
                    try { temp = new Image<Gray, Byte>(file); }
                    catch { continue; }
                    temp._EqualizeHist();

                    var detectedFaces = classifier.DetectMultiScale(temp, 1.1, 15, new Size(24, 24), Size.Empty);
                    if (detectedFaces.Length == 0)
                    {
                        continue;
                    }

                    temp.ROI = detectedFaces[0];
                    temp = temp.Copy();
                    temp = temp.Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
                    imagesList.Add(temp);
                    imagesLabels.Add(Path.GetFileNameWithoutExtension(file));
                }
                for (int i = 0; i < imagesList.Count; i++)
                {
                    imagesLabels_indices.Add(i);
                }

                try { recognizer.Train(imagesList.ToArray(), imagesLabels_indices.ToArray()); }
                catch (Exception ex)
                {
                    MessageBox.Show(ex.Message);
                    Environment.Exit(0);
                }
            }
        }

        private void Video_Click(object sender, EventArgs e)
        {
            try
            {
                capture = new Capture(CaptureType.ANY);
                Video.Enabled = false;
                timer1.Enabled = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message);
            }

            Application.Idle += frame_parser;
        }

        void frame_parser(object sender, EventArgs e)
        {
            frame_BGR = capture.QueryFrame();
            frame_gray = frame_BGR.Convert<Gray, Byte>();
            imageBox1.Image = LBP_recognizer(frame_BGR, frame_gray);
            counter++;
        }

        Image<Bgr, Byte> LBP_recognizer(Image<Bgr, Byte> image_BGR, Image<Gray, Byte> image_gray)
        {
            image_BGR = image_BGR.Copy();
            image_gray = image_gray.Copy();
            image_gray._EqualizeHist();


            if (GPU_classifier_selector)
            {
                detected_faces = GPU_classifier.DetectMultiScale(new GpuImage<Gray, Byte>(image_gray), 1.1, 15, new Size(24, 24));
            }
            else
            {
                detected_faces = classifier.DetectMultiScale(image_gray, 1.1, 15, new Size(24, 24), Size.Empty);
            }

            foreach (var face in detected_faces)
            {
                image_gray.ROI = face;
                var result = recognizer.Predict(image_gray.Copy().Resize(100, 100, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC));
                if (result.Label == -1)
                {
                    image_BGR.Draw(new Rectangle(face.Location, face.Size), new Bgr(0, 0, 255), 3);

                    image_BGR.Draw("Unknown", ref font, new Point(face.X - 20, face.Y - 20), new Bgr(0, 0, 255));
                }
                else
                {
                    image_BGR.Draw(new Rectangle(face.Location, face.Size), new Bgr(0, 255, 0), 3);

                    image_BGR.Draw(imagesLabels[result.Label], ref font, new Point(face.X - 20, face.Y - 20), new Bgr(0, 0, 255));
                }
                //image_BGR.ROI = image_gray.ROI;
                image_gray.ROI = Rectangle.Empty;
            }
            return image_BGR;
        }

        private void saveToolStripMenuItem_Click(object sender, EventArgs e)
        {
            recognizer.Save("traningdata.xml");
        }

        private void loadToolStripMenuItem_Click(object sender, EventArgs e)
        {
            recognizer.Load("traningdata.xml");
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Environment.Exit(0);
        }

        private void GPGPU_CheckedChanged(object sender, EventArgs e)
        {
            GPU_classifier_selector = GPGPU.Checked;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            fps = (counter * 1000) / timer1.Interval;
            label1.Text = "FPS=" + fps.ToString();
            counter = 0;
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            MessageBox.Show("This program licenced under GPL V3","About",MessageBoxButtons.OK,MessageBoxIcon.Information,MessageBoxDefaultButton.Button1);
        }
    }
}
