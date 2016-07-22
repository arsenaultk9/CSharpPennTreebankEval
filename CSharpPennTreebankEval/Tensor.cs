using System;
using System.Collections.Generic;

namespace RNN_Test
{
    public class Tensor
    {
        public readonly float[,] Data;


        private Tensor(float[,] data)
        {
            this.Data = data;
        }

        #region Factory Methods

        public static Tensor FromItemsToMatrix(IEnumerable<float> items, int rowCount, int columnCount)
        {
            var matrixData = new float[rowCount, columnCount];

            var currentRowIndex = 0;
            var currentColumnIndex = 0;

            foreach (var item in items)
            {
                if (currentRowIndex == rowCount)
                {
                    currentColumnIndex++;
                    currentRowIndex = 0;
                }

                matrixData[currentRowIndex, currentColumnIndex] = item;
                currentRowIndex++;
            }

            return new Tensor(matrixData);
        }

        #endregion


        #region Methods

        public Tensor ApplyFunctionOnData(Func<double, double> funcToApply)
        {
            var dataAppliedToFunction = new float[Data.GetLength(0), Data.GetLength(1)];

            for (var rowIndex = 0; rowIndex < Data.GetLength(0); rowIndex++)
            {
                for (var columnIndex = 0; columnIndex < Data.GetLength(1); columnIndex++)
                {
                    var item = Data[rowIndex, columnIndex];
                    var itemAppliedToFunction = (float)funcToApply(item);
                    dataAppliedToFunction[rowIndex, columnIndex] = itemAppliedToFunction;
                }
            }

            return new Tensor(dataAppliedToFunction);
        }

        public Tensor GetColumnTensor(int columnIndex)
        {
            var columnData = new float[Data.GetLength(0), 1];

            for (var rowIndex = 0; rowIndex < Data.GetLength(0); rowIndex++)
            {
                columnData[rowIndex, 0] = Data[rowIndex, columnIndex];
            }

            return new Tensor(columnData);
        }

        public List<float> GetColumnList(int columnIndex)
        {
            var rowData = new List<float>();
            for (var currentColumnIndex = 0; currentColumnIndex < Data.GetLength(1); currentColumnIndex++)
            {
                rowData.Add(Data[columnIndex, currentColumnIndex]);
            }

            return rowData;
        }

        public Tensor Invert()
        {
            var invertedData = new float[Data.GetLength(1), Data.GetLength(0)];

            for (var rowIndex = 0; rowIndex < Data.GetLength(0); rowIndex++)
            {
                for (var columnIndex = 0; columnIndex < Data.GetLength(1); columnIndex++)
                {
                    invertedData[columnIndex, rowIndex] = Data[rowIndex, columnIndex];
                }
            }

            return new Tensor(invertedData);
        }

        #endregion


        #region Operators

        public static Tensor operator +(Tensor tensorA, Tensor tensorB)
        {
            var rowCount = tensorA.Data.GetLength(0);
            var columnCount = tensorA.Data.GetLength(1);

            var summedTensor = new float[rowCount, columnCount];

            for (var currentRow = 0; currentRow < rowCount; currentRow++)
            {
                for (var currentColumn = 0; currentColumn < columnCount; currentColumn++)
                {
                    var summedItem = tensorA.Data[currentRow, currentColumn] + tensorB.Data[currentRow, currentColumn];
                    summedTensor[currentRow, currentColumn] = summedItem;
                }
            }

            return new Tensor(summedTensor);
        }


        public static Tensor operator *(Tensor tensorA, Tensor tensorB)
        {
            var rowCount = tensorA.Data.GetLength(0);
            var columnCount = tensorB.Data.GetLength(1);

            var multipliedData = new float[rowCount, columnCount];

            for (var rowIndex = 0; rowIndex < rowCount; rowIndex++)
            {
                for (var columnIndex = 0; columnIndex < columnCount; columnIndex++)
                {
                    float productSum = 0;
                    for (var sumIndex = 0; sumIndex < tensorA.Data.GetLength(1); sumIndex++)
                    {
                        productSum += tensorA.Data[rowIndex, sumIndex] * tensorB.Data[sumIndex, columnIndex];
                    }

                    multipliedData[rowIndex, columnIndex] = productSum;
                }
            }

            return new Tensor(multipliedData);
        }

        #endregion
    }
}
