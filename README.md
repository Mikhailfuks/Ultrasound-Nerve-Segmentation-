using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace UltrasoundNerveSegmentation
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load the ultrasound image
            Image<Gray, byte> ultrasoundImage = new Image<Gray, byte>("ultrasound_image.png");

            // 1. Preprocessing (Enhance Contrast)
            Image<Gray, byte> enhancedImage = Preprocess(ultrasoundImage);

            // 2. Segmentation (Active Contours)
            Image<Gray, byte> segmentedImage = SegmentNerve(enhancedImage);

            // 3. Postprocessing (Smoothing and Refinement)
            Image<Gray, byte> finalImage = Postprocess(segmentedImage);

            // Display the results
            CvInvoke.ImShow("Original Image", ultrasoundImage);
            CvInvoke.ImShow("Enhanced Image", enhancedImage);
            CvInvoke.ImShow("Segmented Image", segmentedImage);
            CvInvoke.ImShow("Final Image", finalImage);

            CvInvoke.WaitKey(0);
        }

        // Preprocessing: Enhance contrast
        static Image<Gray, byte> Preprocess(Image<Gray, byte> image)
        {
            // Adaptive thresholding for better contrast
            Image<Gray, byte> thresholdedImage = new Image<Gray, byte>(image.Size);
            CvInvoke.AdaptiveThreshold(image, thresholdedImage, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 15, 2);
            return thresholdedImage;
        }

        // Segmentation: Active Contours (Snake Algorithm)
        static Image<Gray, byte> SegmentNerve(Image<Gray, byte> image)
        {
            // Create a mask for the nerve region
            Image<Gray, byte> mask = new Image<Gray, byte>(image.Size);
            mask.SetZero();

            // Define the initial contour (approximate shape of the nerve)
            Point[] initialContour = {
                new Point(50, 100), 
                new Point(100, 150),
                new Point(150, 100)
            };

            // Initialize the snake algorithm
            CvInvoke.SnakeImage(image, mask, initialContour, new MCvSnakeParams(1.0, 0.001, 0.001, 0.001, 5, 5, 1, 10, 1.0, 0.0, MCvSnakeType.Fast), null);

            return mask;
        }

        // Postprocessing: Smoothing and Refinement
        static Image<Gray, byte> Postprocess(Image<Gray, byte> image)
        {
            // Apply morphological operations for smoothing and filling holes
            CvInvoke.MorphologyEx(image, image, MorphOp.Close, new Mat(), new Point(-1, -1), 3, BorderType.Default, new MCvScalar(0));
            return image;
        }
    }
}
