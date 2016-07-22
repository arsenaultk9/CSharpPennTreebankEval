using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RNN_Test
{
    public class Vocabulary
    {
        private readonly Dictionary<int, string> wordsIndexed;

        public Vocabulary(string vocabularyPath)
        {
            wordsIndexed = new Dictionary<int, string>();

            PopulateWordsFromFile(vocabularyPath);
        }

        public string GetWord(int wordIndex)
        {
            return wordsIndexed[wordIndex];
        }

        public int GetWordIndex(string word)
        {
            return wordsIndexed.Single(kv => kv.Value == word).Key;
        }

        #region Private methods

        private void PopulateWordsFromFile(string vocabularyPath)
        {
            using (var vocabularyFileStream = new StreamReader(vocabularyPath))
            {
                var currentLine = vocabularyFileStream.ReadLine();
                var currentWordIndex = 0;

                while (currentLine != null)
                {
                    var columns = currentLine.Split('\t');
                    var word = columns[2];

                    wordsIndexed.Add(currentWordIndex, word);

                    currentLine = vocabularyFileStream.ReadLine();
                    currentWordIndex++;
                }
            }
        }

        #endregion

    }
}
