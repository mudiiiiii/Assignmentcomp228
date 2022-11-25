using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MudiagaOtojareri_Exercise02
{
    class Program
    {
        static readonly string _insuranceDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "insurance.csv");
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            var model = Train(mlContext, _insuranceDataPath);

            TestSinglePrediction(mlContext, model);
        }
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            // The IDataView object holds the training dataset 
            IDataView dataView = mlContext.Data.LoadFromTextFile<Insurance>(dataPath, hasHeader: true, separatorChar: ',');

            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Charges")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AgeEncoded", inputColumnName: "Age"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SexEncoded", inputColumnName: "Sex"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ChildrenEncoded", inputColumnName: "Children"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SmokerEncoded", inputColumnName: "Smoker"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RegionEncoded", inputColumnName: "Region"))
                .Append(mlContext.Transforms.Concatenate("Features", "AgeEncoded", "SexEncoded", "BMI", "ChildrenEncoded", "SmokerEncoded", "RegionEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());

            //Create the model
            var model = pipeline.Fit(dataView);

            //Return the trained model
            return model;
        }
        
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<Insurance, Medicalcostprediction>(model);

            //Create a single Insurance object to be used for prediction
            var InsuranceSample = new Insurance()
            {
                Age = "23",
                Sex = "male",
                BMI = 34.4f,
                Children = "0",
                Smoker = "no",
                Region = "southwest",
                Charges = 0
            };
            //Make a prediction
            var prediction = predictionFunction.Predict(InsuranceSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted cost is: ${prediction.Charges:0.####}");
            Console.WriteLine($"**********************************************************************");
            Console.ReadLine();
        }
    }
}
