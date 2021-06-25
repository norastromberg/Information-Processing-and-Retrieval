#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Processing and Retrieval @IST 2019/2020
Course Project Part2
Exercise 4 - A practical application
Group 23


4 A practical application

This exercise concerns with the development of a program for illustrating keyphrase extraction
in a practical application. Your program should parse one of the XML/RSS feeds from The New
York Times4
, extract the titles and descriptions for the news articles in the feed, and show the
most relevant keyphrases currently in the news.
The keyphrase extraction method can be based on either of the programs developed in this
second part of the course project. As a result, your program should generate an HTML page
illustrating the most relevant key-phrases (e.g., using either a list, a chart, word clouds, etc.). The
evaluation for this exercise will value creative solutions for presenting the results (e.g., you can
consider clustering the news articles according to keyphrases, use different types of graphical
presentations, etc.)

"""
import os
import webbrowser
from xml.dom.minidom import parse
import re
import feedparser
#Get the real phrases.
from proj.part2.exercise1 import e1, find_candidate_word_occurences
from proj.part2.exercise2 import get_top_candidates

def get_most_relevant_key_phrases():
    document = newsfeed_to_one_document("https://rss.nytimes.com/services/xml/rss/nyt/World.xml")
    documents = newsfeed_to_documents("https://rss.nytimes.com/services/xml/rss/nyt/World.xml")
    doc_candidate_dict = {}
    doc_dict ={}
    for i,doc in enumerate(documents):
        doc_dict[i]=doc
        candidate_dict, word_pos_dict = find_candidate_word_occurences(doc, (1, 4))
        candidates = get_top_candidates("word_similarity", 5, candidate_dict, "", word_pos_dict)
        #candidates = get_top_candidates("co-occurrences", 5, candidate_dict, "", word_pos_dict)
        doc_candidate_dict[i] = candidates
    candidate_dict, word_pos_dict = find_candidate_word_occurences(document,(1,4))
    candidates = get_top_candidates("word_similarity", 20 ,candidate_dict,"",word_pos_dict)
    #candidates = get_top_candidates("co-occurrences", 20 ,candidate_dict,"",word_pos_dict)
    #candidates = get_top_candidates("n_gram_length", 20 ,candidate_dict,"",word_pos_dict)
    return candidates, doc_candidate_dict, doc_dict


#"https://rss.nytimes.com/services/xml/rss/nyt/World.xml" for New York Times World news
#Makes the all the newsarticles into one document, with one line for each article. The articles have two-three sentences each
def newsfeed_to_one_document(source):
    NewsFeed = feedparser.parse(source)
    entries = NewsFeed.entries
    document = ""
    for entry in entries:
        title = entry.title
        description = entry.summary
        document += title + ". "+description
    return document

def newsfeed_to_documents(source):
    NewsFeed = feedparser.parse(source)
    entries = NewsFeed.entries
    documents = []
    for entry in entries:
        title = entry.title
        description = entry.summary
        documents.append(title + ". " + description)
    return documents

def get_category(keyphrase):
    try :
        words = keyphrase.split(" ")
        if len(words) == 1:
            return "Unigram"
        if len(words) == 2:
            return "Bigram"
        if len(words) == 3:
            return "Trigram"
    except :
        return "None"

def all_articles_word_cloud():
    key_phrases,doc_candidate_dict,doc_dict = get_most_relevant_key_phrases()
    data = []
    for (key,val) in key_phrases:
        if isinstance(key,tuple): #Multigram
            phrase = ""
            for k in key:
                phrase += k + " "
        else :
            phrase = key
        data.append({"x": phrase, "value": str(val),"category": get_category(phrase.strip())})
    static_content = """
     <head>
      <title>News Keyphrases Cloud Chart</title>
      <script src="https://cdn.anychart.com/releases/v8/js/anychart-base.min.js"></script>
      <script src="https://cdn.anychart.com/releases/v8/js/anychart-tag-cloud.min.js"></script>
     </head>
       <div id="container" style = "height : 550px;">
        <script>
        anychart.onDocumentReady(function() {
        var data = """ + str(data) + """
        var chart = anychart.tagCloud(data);
        chart.tooltip().useHtml(true);
        //configure tooltips
        chart.tooltip().titleFormat("Keyphrase : {%x}")
        chart.tooltip().format("PageRank: <b>{%value}</b> <br> <br> Percentage of total : <b>{%yPercentOfTotal}%</b> <br>  Type: <b>{%category}</b>")
        chart.title('Most relevant keyphrases in the world news right now')
        chart.angles([0]);
        chart.textSpacing(8);
        chart.mode("spiral");
        chart.colorRange(true);
        chart.colorRange().length('60%');
        chart.container("container");
        chart.draw();
    });
    </script>
    </div>
    """
    return static_content

def each_document_keyphrase():
    candidates, doc_candidate_dict,doc_dict = get_most_relevant_key_phrases()
    tab_contents = []
    for key in doc_candidate_dict.keys():
        tab_content = """<div id="""+str(key)+""" class="tabcontent"><h3>Article : """+str(key)+"""</h3>"""
        tab_content += """<h3>The Article : </h3> <p>"""+ doc_dict[key]+"""</p>"""
        tab_content +="""<h3>Keyphrases</h3> <ul>"""
        list = ""
        for (candidate,val) in doc_candidate_dict[key]:
            list +="<li>" + str(candidate) + ", Value : " + str(val)+"</li>"
        tab_content+= list
        tab_content+= """</ul></div>"""
        tab_content+="\n"
        tab_contents.append(tab_content)
    return tab_contents

def make_html(tab_contents,filename):
    word_cloud = all_articles_word_cloud()
    html_file = open(filename, 'w')
    static_content = """<!DOCTYPE html>
        <html>"""
    tab_head = "<div class=\"tab\">"
    tabs = ""
    for i,tab_content in enumerate(tab_contents):
        tab_head += "<button class=\"tablinks\" onclick=\"openTab(event, "+str(i) +")\">Article: "+str(i)+"</button> \n"
        tabs += tab_content
    tab_head += "<button class=\"tablinks\" id = \"defaultOpen\" onclick=\"openTab(event,"+str(100)+")\">Word cloud all articles</button>"
    tab_head +="</div>"
    static_content += tab_head
    tabs += """<div id=100 class="tabcontent">"""+word_cloud+"""</div>"""
    static_content += tabs

    static_content += """<script>function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    // Hide all tabs 
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
      tabcontent[i].style.display = "none";
    }

    // remove all "active" from every element with class = tablinks
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
     tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show and make the open tab active
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
    }
    // Click on default tab = word cloud 
    document.getElementById("defaultOpen").click();
    console.log("clicked");
    </script>
    
    </html>"""
    html_file.write(static_content)


def open_file_in_browser(filename):
    file_part = "file://"
    first_part = os.getcwd()
    path = file_part+first_part+"/"+filename
    webbrowser.open_new_tab(path)


def e4():
    tab_contents = each_document_keyphrase()
    make_html(tab_contents,"representation.html")
    open_file_in_browser("representation.html")


if __name__ == '__main__':
    e4()
