# KNN-Text-Classification
KNN Text Classification using Apache Spark

Using the <a href="http://qwone.com/~jason/20Newsgroups/">20 NewsGroup dataset</a> and Apache Spark, I built a k-nearest neighbors classifier that classifies text data.
This code first computes a TF-IDF (Term Frequency - Inverse Document Frequency) Matrix for the top 20k words of the corpus.
The TF-IDF matrix is then used to compute similarity distances between a given query text and each of the documents in the corpus.

It will for instance predict that the string "How many goals did Vancouver score last year?" belongs to the class "/rec.sport.hockey".
