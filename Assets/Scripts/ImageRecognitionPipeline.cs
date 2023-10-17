using System;
using System.Collections.Generic;
using System.Drawing;
using UnityEngine;

public class ImageRecognitionPipeline
{

    //veriables for debugging
    String debugPicturePath = "debugPictures/";
    Boolean saveContours = true;

    Boolean saveHsvImage = true;
    Boolean saveDecodedImage = true;
    // find contours of all red, blue obstacles and boundery in image  

    /*                         h
    public List<List<Rectangle>> getObstaclePosition(byte[] image)
    {
        var imageDecoded = new Emgu.CV.Mat();
        var imageHsv = new Emgu.CV.Mat();
        var inRange = new Emgu.CV.Mat();
        Image<Gray, Byte> boundImage = new Mat().ToImage<Gray, Byte>();
        var redRange = new Emgu.CV.Mat();
        Image<Gray, Byte> redImage = new Mat().ToImage<Gray, Byte>();
        var blueRange = new Emgu.CV.Mat();
        Image<Gray, Byte> blueImage = new Mat().ToImage<Gray, Byte>();


        CvInvoke.Imdecode(image, Emgu.CV.CvEnum.ImreadModes.Unchanged, imageDecoded);
        if (saveDecodedImage)
        {
            imageDecoded.Save(debugPicturePath + "cameraImage.png");
        }
        var height = imageDecoded.Height;
        var width = imageDecoded.Width;
        CvInvoke.CvtColor(imageDecoded, imageHsv, Emgu.CV.CvEnum.ColorConversion.Bgr2Hsv);
        //detect bounderies
        ScalarArray lowerBound = new ScalarArray(new MCvScalar(20, 100, 100));
        ScalarArray upperBound = new ScalarArray(new MCvScalar(40, 255, 255));

        CvInvoke.InRange(imageHsv, lowerBound, upperBound, inRange);
        CvInvoke.Threshold(inRange, boundImage, 70, 255, Emgu.CV.CvEnum.ThresholdType.Binary);

        //detect red obstacles
        ScalarArray lowerRed = new ScalarArray(new MCvScalar(0, 50, 50));
        ScalarArray upperRed = new ScalarArray(new MCvScalar(10, 255, 255));

        CvInvoke.InRange(imageHsv, lowerRed, upperRed, redRange);
        CvInvoke.Threshold(redRange, redImage, 70, 255, Emgu.CV.CvEnum.ThresholdType.Binary);

        //detect blue obstacles
        ScalarArray lowerBlue = new ScalarArray(new MCvScalar(110, 0, 99));
        ScalarArray upperBlue = new ScalarArray(new MCvScalar(114, 231, 138));
        if (saveHsvImage)
        {
            imageHsv.Save(debugPicturePath + "hsv.png");
        }
        CvInvoke.InRange(imageHsv, lowerBlue, upperBlue, blueRange);
        CvInvoke.Threshold(blueRange, blueImage, 70, 255, Emgu.CV.CvEnum.ThresholdType.Binary);
        //var hsvRed =Emgu.CV.CvInvoke.CvtColor(greenBGR)
        if (saveContours)
        {
            boundImage.Save(debugPicturePath + "boundImageContours.png");
            redImage.Save(debugPicturePath + "red_test.png");
            blueImage.Save(debugPicturePath + "blue_test.png");

        }
        // find contours boundaries
        VectorOfVectorOfPoint boundObstacleContours = new VectorOfVectorOfPoint();
        Mat boundObstaclesHierarchy = new Emgu.CV.Mat();

        CvInvoke.FindContours(boundImage, boundObstacleContours, boundObstaclesHierarchy,
            Emgu.CV.CvEnum.RetrType.Tree, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);


        // find contours red obstacles
        VectorOfVectorOfPoint redObstacleContours = new VectorOfVectorOfPoint();
        Mat redObstaclesHierarchy = new Emgu.CV.Mat();

        CvInvoke.FindContours(redImage, redObstacleContours, redObstaclesHierarchy,
            Emgu.CV.CvEnum.RetrType.Tree, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);

       
        // find contours blue obstacles
        VectorOfVectorOfPoint blueObstacleContours = new VectorOfVectorOfPoint();
        Mat blueObstaclesHierarchy = new Emgu.CV.Mat();

        CvInvoke.FindContours(blueImage, blueObstacleContours, blueObstaclesHierarchy,
            Emgu.CV.CvEnum.RetrType.Tree, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxSimple);


        // calculate area and remove small elements (potential wrong detected areas)

        // bounderies
        Point[][] boundObstacleContoursArray = boundObstacleContours.ToArrayOfArray();

        List<Rectangle> bounderies = new List<Rectangle>();

        foreach (var contour in boundObstacleContoursArray) {
           double area = CvInvoke.ContourArea(new VectorOfPoint(contour));
           if (area > 10)
            {
                Rectangle boundingBox = CvInvoke.BoundingRectangle(contour);
                bounderies.Add(boundingBox);
            } 
        }

        // blue Obstacle rectangles

        Point[][] blueObstacleContoursArray = blueObstacleContours.ToArrayOfArray();

        List<Rectangle> blueObstacles = new List<Rectangle>();

        foreach (var contour in blueObstacleContoursArray)
        {
            double area = CvInvoke.ContourArea(new VectorOfPoint(contour));
            if (area > 10)
            {
                Rectangle boundingBox = CvInvoke.BoundingRectangle(contour);
                blueObstacles.Add(boundingBox);
            }
        }

        //red obstacles rectangles
        Point[][] redObstacleContoursArray = redObstacleContours.ToArrayOfArray();

        List<Rectangle> redObstacles = new List<Rectangle>();

        foreach (var contour in redObstacleContoursArray)
        {
            double area = CvInvoke.ContourArea(new VectorOfPoint(contour));
            if (area > 10)
            {
                Rectangle boundingBox = CvInvoke.BoundingRectangle(contour);
                redObstacles.Add(boundingBox);
            }
        }

        // order important: [redObstacles, blueObstacles, bounderies]
        List<List<Rectangle>> allContours = new List<List<Rectangle>>();
        allContours.Add(redObstacles);
        allContours.Add(blueObstacles);
        allContours.Add(bounderies);

        return allContours;

    }*/

    public List<List<Vector4>> GetCooridnatesNClosestObstacles(Vector3 position, byte[] image, int n = 4)
    {
        // versuche die 5 nächsten Roten, Gelben und Blauen auf y achse zu return (was er auf bild sieht geradeaus müsste y entfernung sein)
        //überlegung: nach height objeckte sortieren, weil je größer desto näher dran
        var allContours = new List<List<Rectangle>>();//this.getObstaclePosition(image);
        List<Vector4> obstaclePositions = new List<Vector4>();
        List<List<Vector4>> obstaclesSorted = new List<List<Vector4>>();

        // sorts by Y ascending (y is distance to vehicle)
        foreach (List<Rectangle> obs in allContours)
        {
            obs.Sort((x, y) => x.Y.CompareTo(y.Y));
            List<Vector4> vectorList = new List<Vector4>();
            // create Vector from Rectangle object
            foreach (Rectangle rect in obs)
            {
                Vector4 vec = new Vector4();
                vec.x = rect.X;
                vec.y = rect.Y;
                vec.w = rect.Width;
                vec.z = rect.Height;
                vectorList.Add(vec);
            }

            for (int i = 0; i < n; i++)
            {
                if (vectorList.Count < n)
                {
                    Vector4 defaultVec = new Vector4();
                    vectorList.Add(defaultVec);
                }
            }
            //fill until list is length n with default argument

            obstaclesSorted.Add(vectorList.GetRange(0, n));
        }

        //fill until list is length n with default argument



        return obstaclesSorted;


    }

    public List<List<List<Vector4>>> TraceObstcalePosition(List<List<Vector4>> position, List<List<List<Vector4>>> nPositions)
    {
        //move all elements foreward
        for (int i = nPositions.Count - 1; i > 0; i--)
        {
            nPositions[i] = nPositions[i - 1];
        }
        //set new Last element as first
        nPositions[0] = position;

        return nPositions;
    }

    public void SaveImageToPath(byte[] image, String filepath = "test.png")
    {
        //var image1 = new Emgu.CV.Mat();
        //CvInvoke.Imdecode(image, Emgu.CV.CvEnum.ImreadModes.Unchanged, image1);
        //image1.Save(filepath);
        //print("SaveImageToPath called but not implemented");
    }

}


// contours python script: bluecontours =  list 10 of shape [n,1,2] arrays i.e. [2,1,2] -> [[[276  47]],, [[276  51]]]