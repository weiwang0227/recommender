# Launch with
#
# gunicorn -D --threads 4 -b 0.0.0.0:5000 --access-logfile server.log --timeout 60 server:app glove.6B.300d.txt bbc

from flask import Flask, render_template
from doc2vec import *
import sys
import os

rcmd_num = 5
app = Flask(__name__, template_folder='templates')

@app.route("/")
def articles():
    """Show a list of article titles"""
    return render_template('articles.html', html_article_list=the_articles)

@app.route("/article/<topic>/<filename>")
def article(topic,filename):
    """
    Show an article with relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    title = None
    content = None
    centroid = None
    the_article = None

    for element in the_articles:
        if element[0] == topic + os.sep + filename:
            title = element[1]
            filename = articles_dirname + os.sep + element[0]
            f = codecs.open(filename, encoding='latin-1', mode='r')
            content = f.read()
            f.close()
            content = content.replace(title, '')
            the_article = element
            break

    article_paragraph_list = content.split('\n')

    if title is None:
        return render_template('article.html', html_title = 'Error: file not found!')

    rcmd_list = recommended(the_article, the_articles, rcmd_num)

    rcmd_articles = []
    for i in range(rcmd_num):
        rcmd_articles.append(rcmd_list[i][1])

    return render_template('article.html',
                            html_title = title,
                            html_article_paragraph_list = article_paragraph_list,
                            html_similar_articles = rcmd_articles)


# initialization
i = sys.argv.index('server:app')
glove_filename = sys.argv[i+1]
articles_dirname = sys.argv[i+2]

gloves = load_glove(glove_filename)
the_articles = load_articles(articles_dirname, gloves)
