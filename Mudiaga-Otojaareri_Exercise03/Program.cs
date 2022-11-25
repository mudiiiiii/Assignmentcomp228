using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mudiaga_Otojaareri_Exercise03
{
    class Program
    {
        static readonly string _studentDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Student.csv");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _studentDataPath);

            Evaluate(mlContext, model);

            TestSinglePrediction(mlContext, model);
        }
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // The IDataView object holds the training dataset 
            IDataView dataView = mlContext.Data.LoadFromTextFile<Student>(dataPath, hasHeader: true, separatorChar: ',');

            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Categorical.OneHotEncoding(outputColumnName: "UNSEncoded", inputColumnName: "UNS")
                .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "UNSEncoded"))
                .Append(mlContext.Transforms.Concatenate(featuresColumnName, "STG", "SCG", "STR", "LPR", "PEG"))
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 4));
                
             

            //Create the model
            var model = pipeline.Fit(dataView);

            //Return the trained model
            return model;
        }
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<Student>(_studentDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics output         ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"*       R-Squared Score:      {metrics.RSquared:0.###}");

            Console.WriteLine($"*       Root-Mean-Squared Error:      {metrics.RootMeanSquaredError:#.###}");
            Console.WriteLine("Press Enter to continue...");
            Console.ReadLine();
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<Student, KnowdledgePrediction>(model);

            //Create a single Student object to be used for prediction
            var StudentSample = new Student()
            {
                STG = 0.15F,
                SCG = 0.02F,
                STR = 0.34f,
                LPR = 0.4F,
                PEG = 0.01F,
                UNS = "very_low",
            };
            //Make a prediction
            var prediction = predictionFunction.Predict(StudentSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted of student's knowdledge is: ${prediction.UNS}");
            Console.WriteLine($"**********************************************************************");
            Console.ReadLine();
        }
    }
}
