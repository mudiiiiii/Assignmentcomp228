using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mudiaga_Otojaareri_Exercise03
{
    class KnowdledgePrediction
    {
        [ColumnName("PredictedLabel")]
        public string UNS;

        public float[] Score;
    }
}
