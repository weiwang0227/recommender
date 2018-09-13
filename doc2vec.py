import sys
import re
import string
import os
import numpy as np
import codecs
import math

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def load_glove(filename):
    """
    Read all lines from the indicated file and return a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    So split each line on spaces into a list; the first element is the word
    and the remaining elements represent factor components. The length of the vector
    should not matter; read vectors of any length.
    """
    word_vector_map = {}
    with open(filename, "r") as file:
        lines = file.readlines()
    for line in lines:
        # find the index of the first space ' '
        index_of_1st_space = line.find(' ')
        first_part, second_part = line[:index_of_1st_space], line[index_of_1st_space:]
        # Remove the leading or trailing spaces
        word = first_part.strip()
        number_list = second_part.strip().split(' ')
        vector = np.array(number_list, dtype=np.float64)
        word_vector_map[word] = vector
    return word_vector_map



def filelist(root):
    """Return a fully-qualified list of filenames under root directory"""
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return allfiles


def get_text(filename):
    """
    Load and return the text of a text file, assuming latin-1 encoding as that
    is what the BBC corpus uses.  Use codecs.open() function not open().
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text):
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    Remove English stop words
    """
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)  # delete stuff but leave at least a space to avoid clumping together
    words = nopunct.split(" ")
    words = [w for w in words if len(w) > 2]  # ignore a, an, to, at, be, ...
    words = [w.lower() for w in words]

    processed_words = ''
    for w in words:
        if w not in ENGLISH_STOP_WORDS:
            processed_words += w
            processed_words += ' '

    processed_words.rstrip()  # strip the trailing space.
    return processed_words




def load_articles(articles_dirname, gloves):
    """
    Load all .txt files under articles_dirname and return a table (list of lists/tuples)
    where each record is a list of:

      [filename, title, article-text-minus-title, wordvec-centroid-for-article-text]

    We use gloves parameter to compute the word vectors and centroid.

    The filename is stripped of the prefix of the articles_dirname pulled in as
    script parameter sys.argv[2]. E.g., filename will be "business/223.txt"
    """

    articles_dirname = articles_dirname.rstrip(os.sep)

    if len(gloves) <= 0 or not os.path.isdir(articles_dirname):
        return None

    ret_table =[]
    for file in filelist(articles_dirname):
        content = get_text(file)

        # get the title
        title = content.split('\n\n')[0]
        article_text_minus_title = words(content.lstrip(title))
        centroid = doc2vec(article_text_minus_title, gloves)

        parts = file.split('/')
        if file.find('COPYRIGHT') == -1:
            ret_table.append((
                parts[-2] + os.sep + parts[-1],
                title,
                article_text_minus_title,
                centroid
            ))
    return ret_table


def doc2vec(text, gloves):
    """
    Return the word vector centroid for the text. Sum the word vectors
    for each word and then divide by the number of words. Ignore words
    not in gloves.
    """
    if len(gloves) <= 0 or len(text) <= 0:
        return None
    vector_sum = np.zeros(300, dtype=np.float64)
    counter = 0
    for word in text.split():
        if gloves.get(word) is not None:
            vector_sum = np.add(vector_sum, gloves[word])
            counter += 1
    return np.divide(vector_sum, counter)

def distances(article, articles):
    """
    Compute the euclidean distance from article to every other article and return
    a list of (distance, a) tuples for all a in articles. The article is one
    of the elements (tuple) from the articles list.
    """
    if article == None or articles == None or len(articles) == 0:
        return None

    centroid_of_article = article[3]
    ret_list = []
    for each_article in articles:
        centroid_of_each = each_article[3]
        distance = euclidean_dist(centroid_of_article, centroid_of_each)
        ret_list.append((distance, each_article))

    ret_list.sort()     # sort the list according to the distance
    return ret_list

def euclidean_dist(vector1, vector2):
    """
    Calculate the Euclidean distance of two vectors.
    If any error or exception happens, -1 is returned.
    """
    if type(vector1) != np.ndarray:
        print "ERROR: vector1 is not of type numpy.ndarray: ", type(vector1)
        return -1
    if type(vector2) != np.ndarray:
        print "ERROR: vector2 is not of type numpy.ndarray: ", type(vector2)
        return -1
    len_vec_1 = len(vector1)
    len_vec_2 = len(vector2)
    if len_vec_1 != len_vec_2:
        print "ERROR: lengths of two vectors are different: ", len_vec_1, " vs ", len_vec_2
        return -1

    difference = vector2 - vector1
    squareDistance = np.dot(difference.T, difference)
    return math.sqrt(squareDistance)

def recommended(article, articles, n):
    """
    Return a list of the n articles (records with filename, title, etc...)
    closest to article's word vector centroid. The article is one of the elements
    (tuple) from the articles list.
    """
    dist_list = distances(article, articles)
    return dist_list[1:n+1] # don't return the 0th element which is article itself.

