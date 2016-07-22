using Microsoft.MSR.CNTK.Extensibility.Managed;

namespace RNN_Test
{
    public class TrainedRnnLoader
    {
        private const string ModelOutputs = "E0:bo0:WHO0:WCO0:WXF0:bf0:WHF0:WCF0:WXI0:WHI0:WCI0:WXC0:WXO0:WHC0:bc0:bi0:W2";

        private readonly IEvaluateModelManagedF model;
        private readonly Vocabulary vocabulary;

        private readonly RnnEvaluator rnnEvaluator;

        public TrainedRnnLoader(string modelPath, string vocabularyPath)
        {
            var config = $"outputNodeNames=\"{ModelOutputs}\"\n";
            config += $"modelPath ={modelPath}";

            using (model = new IEvaluateModelManagedF())
            {
                model.CreateNetwork(config, deviceId: -1);
                vocabulary = new Vocabulary(vocabularyPath);

                rnnEvaluator = new RnnEvaluator(model, vocabulary);
            }                
            
        }

        public RnnEvaluator GetEvaluator()
        {
            return rnnEvaluator;
        }
    }
}
