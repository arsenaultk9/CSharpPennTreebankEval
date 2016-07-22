using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics;
using Microsoft.MSR.CNTK.Extensibility.Managed;

namespace RNN_Test
{
    public class RnnEvaluator
    {
        private readonly Tensor E;
        private readonly Tensor bO;
        private readonly Tensor WHO;
        private readonly Tensor WCO;
        private readonly Tensor WXF;
        private readonly Tensor bF;
        private readonly Tensor WHF;
        private readonly Tensor WCF;
        private readonly Tensor WXI;
        private readonly Tensor WHI;
        private readonly Tensor WCI;
        private readonly Tensor WXC;
        private readonly Tensor WXO;
        private readonly Tensor WHC;
        private readonly Tensor bC;
        private readonly Tensor bI;
        private readonly Tensor W2;

        private readonly Vocabulary vocabulary;

        private Tensor previousH;
        private Tensor previousC;

        public RnnEvaluator(IEvaluateModelManagedF model, Vocabulary vocabulary)
        {
            // NOTE: Loading of data is to get full column then increment to next column.
            /* Ex: 1, 2, 3, 4 for 2x2 is:
            *{1, 3}
            *{2, 4}
            **/
            E = Tensor.FromItemsToMatrix(model.Evaluate("E0"), 150, 10000);     // 1 500 000    [150,10000]
            bO = Tensor.FromItemsToMatrix(model.Evaluate("bo0"), 200, 1);       // 200          [200,1]
            WHO = Tensor.FromItemsToMatrix(model.Evaluate("WHO0"), 200, 200);   // 40 000       [200,200] 
            WCO = Tensor.FromItemsToMatrix(model.Evaluate("WCO0"), 200, 1);     // 200          [200,1]
            WXF = Tensor.FromItemsToMatrix(model.Evaluate("WXF0"), 200, 150);   // 30 000       [200,150]
            bF = Tensor.FromItemsToMatrix(model.Evaluate("bf0"), 200, 1);       // 200          [200,1]
            WHF = Tensor.FromItemsToMatrix(model.Evaluate("WHF0"), 200, 200);   // 40 000       [200,200]
            WCF = Tensor.FromItemsToMatrix(model.Evaluate("WCF0"), 200, 1);     // 200          [200,1]
            WXI = Tensor.FromItemsToMatrix(model.Evaluate("WXI0"), 200, 150);   // 30 000       [200,150]  
            WHI = Tensor.FromItemsToMatrix(model.Evaluate("WHI0"), 200, 200);   // 40 000       [200,200]
            WCI = Tensor.FromItemsToMatrix(model.Evaluate("WCI0"), 200, 1);     // 200          [200,1]
            WXC = Tensor.FromItemsToMatrix(model.Evaluate("WXC0"), 200, 150);   // 30 000       [200,150]
            WXO = Tensor.FromItemsToMatrix(model.Evaluate("WXO0"), 200, 150);   // 30 000       [200,150]
            WHC = Tensor.FromItemsToMatrix(model.Evaluate("WHC0"), 200, 200);   // 40 000       [200,200]
            bC = Tensor.FromItemsToMatrix(model.Evaluate("bc0"), 200, 1);       // 200          [200,1]
            bI = Tensor.FromItemsToMatrix(model.Evaluate("bi0"), 200, 1);       // 200          [200,1]
            W2 = Tensor.FromItemsToMatrix(model.Evaluate("W2"), 200, 10000);    // 2 000 000    [200,10000]

            model.GetNodeDimensions(NodeGroup.nodeOutput);

            this.vocabulary = vocabulary;

            previousH = Tensor.FromItemsToMatrix(new float[200], 200, 1);
            previousC = Tensor.FromItemsToMatrix(new float[200], 200, 1);
        }

        public IEnumerable<string> GetNextFollowingOutput(string word)
        {
            var xVec = GetWordVector(word);

            var i = Sigmoid(WXI*xVec + WHI*previousH + WCI*previousC + bI);
            var f = Sigmoid(WXF*xVec + WHF*previousH + WCF*previousC + bF);

            var c = f*previousC + i*Tanh(WXC*xVec + WHC*previousH + bC);

            var o = Sigmoid(WXO*xVec + WHO*previousH + WCO*c + bO);

            var h = o*Tanh(c);

            var q = h.Invert();
            var output = q*W2;

            previousC = c;
            previousH = h;

            return GetWordsFromOutput(output);
        }

        #region Private methods

        private Tensor GetWordVector(string word)
        {
            var wordIndex = vocabulary.GetWordIndex(word);

            return E.GetColumnTensor(wordIndex);
        }

        private Tensor Sigmoid(Tensor tensor)
        {
            return tensor.ApplyFunctionOnData(SpecialFunctions.Logistic);
        }

        private Tensor Tanh(Tensor tensor)
        {
            return tensor.ApplyFunctionOnData(Trig.Tanh);
        }

        private IEnumerable<string> GetWordsFromOutput(Tensor output)
        {
            var outputRow = output.GetColumnList(0);
            var wordsAndPossibilities = new Dictionary<string, float>();

            var wordIndex = 0;
            foreach (var outputItem in outputRow)
            {
                wordsAndPossibilities.Add(vocabulary.GetWord(wordIndex), outputItem);
                wordIndex++;
            }

            return wordsAndPossibilities.
                OrderBy(wordAndIndex => wordAndIndex.Value).
                Select(wordAndIndex => wordAndIndex.Key);
        }

        #endregion
    }
}
