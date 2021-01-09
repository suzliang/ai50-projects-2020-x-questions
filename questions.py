import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    d = dict()
    for file in os.listdir(directory):
        # Open and read file
        f = open(os.path.join(directory, file), "r", encoding="utf8")
        # Add to dictionary
        d[file] = f.read()

    return d

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # List lowercased words without punctuation
    words = nltk.word_tokenize(document.lower().translate(str.maketrans('', '', string.punctuation)))
    # Remove stopwords
    words = [word for word in words if word not in nltk.corpus.stopwords.words("english")]
    
    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    total_docs = len(documents)
    idfs = dict()
    for d in documents:
        for w in documents[d]:
            count = 0
            for d in documents:
                if w in documents[d]:
                    count += 1
                    
            # Calculate IDF value
            idfs[w] = math.log(total_docs / count)
    
    return idfs

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = []
    for f in files:
        for q in query:
            # Reset tf-idf value
            tf_idf = 0
            # tf_idf = term frequency * idf
            tf_idf += files[f].count(q) * idfs[q]
        tf_idfs.append((f, tf_idf))

    # Sort tf-idf's from high to low
    tf_idfs = sorted(tf_idfs, key=lambda i: i[1], reverse=True)

    # Return n top files
    top_f = []
    for i in range(n):
        top_f.append(tf_idfs[i][0])
    
    return top_f


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # (s, idf, qtf)
    s_idfs = []
    for s in sentences:
        # IDF
        idf = 0
        # QTF
        count = 0
        for q in query:
            if q in sentences[s]:
                idf += idfs[q]
                count += s.count(q)
        # Query term density/qtd = # query words / # sentence words
        qtd = count / len(s)
        s_idfs.append((s, idf, qtd))

    # Sort idf's then qtd's from high to low
    s_idfs = sorted(s_idfs, key=lambda i: (i[1], i[2]), reverse=True)
    #print(s_idfs)

    # Return n top sentences
    top_sent = []
    for i in range(n):
        top_sent.append(s_idfs[i][0])
    
    return top_sent


if __name__ == "__main__":
    main()
