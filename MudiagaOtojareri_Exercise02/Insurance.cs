using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MudiagaOtojareri_Exercise02
{
    class Insurance
    {
        [LoadColumn(0)]
        public string Age;

        [LoadColumn(1)]
        public string Sex;

        [LoadColumn(2)]
        public float BMI;

        [LoadColumn(3)]
        public string Children;

        [LoadColumn(4)]
        public string Smoker;

        [LoadColumn(5)]
        public string Region;

        [LoadColumn(6)]
        public float Charges;
    }
}
