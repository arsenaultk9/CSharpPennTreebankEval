using System;
using System.Linq;

namespace RNN_Test
{
    /*
     * Useful links:
     * https://esciencegroup.com/2016/03/04/fun-with-recurrent-neural-nets-one-more-dive-into-cntk-and-tensorflow/
     * https://github.com/dbgannon/rnn/blob/master/rnn-lstm-for-gethub.ipynb
     * https://github.com/Microsoft/CNTK/wiki/CNTK-Evaluate-Hidden-Layers
     * */
    class Program
    {
        private const string ModelPath = @"..\..\..\..\PennTreebank\Output\Models\rnn.dnn";
        private const string VocabularyPath = @"..\..\..\..\PennTreebank\Output\Models\vocab.txt";

        static void Main(string[] args)
        {
            var rnnLoader = new TrainedRnnLoader(ModelPath, VocabularyPath);
            var evaluator = rnnLoader.GetEvaluator();

            var word = "the";
            var random = new Random((int)DateTime.Now.Ticks);

            for (var currentWordNum = 0; currentWordNum < 50; currentWordNum++)
            {
                var lastWord = word;
                Console.Write(word + " ");

                var result = evaluator.GetNextFollowingOutput(word);
                while (lastWord == word)
                {
                    word = result.ElementAt(random.Next(0, 5));
                }
            }

            Console.WriteLine("");
            Console.WriteLine("Press any key to exit.");
            Console.ReadKey();
        }
    }
}
