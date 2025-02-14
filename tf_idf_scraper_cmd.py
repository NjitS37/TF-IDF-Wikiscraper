# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:01:47 2025

@author: tijna
"""

import argparse
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.request import urlopen
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def contentscraper(link):
    """Scrapes content from a Wikipedia article and removes HTML elements."""
    html = urlopen(link).read()
    soup = BeautifulSoup(html, features="html.parser")

    # Remove all script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    content_area = soup.find(id="mw-content-text")
    references_header = content_area.find("h2", string=re.compile(r"References", re.I))

    if references_header:
        content = []
        for element in content_area.find_all():
            if element == references_header:
                break
            if element.name in ["p", "ul", "ol"]:
                content.append(element.text)
        text = "\n".join(content)
    else:
        text = content_area.get_text()

    text = re.sub(r"\[\d+\]", "", text)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)


def linklist(url):
    """Extracts all Wikipedia article links from a given page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    titles_written = [url]
    allLinks = soup.find(id="bodyContent").find_all("a")

    for link in allLinks:
        if 'href' not in link.attrs:
            continue
        if link['href'].find("/wiki/") == -1 or ":" in link['href']:
            continue
        full_url = "https://en.wikipedia.org" + link['href']
        if full_url not in titles_written:
            titles_written.append(full_url)

    return titles_written


def custom_tokenizer(text):
    """Tokenizes words and filters out common Wikipedia-specific terms."""
    words = re.findall(r'\b[a-zA-Z\'-]{3,}\b', text.lower())
    return [word for word in words if word not in ["citation", "citation needed"]]


def wikiscraper(input_file, output_file, N, ngram):
    """Scrapes Wikipedia articles, applies TF-IDF, and outputs the top N words."""
    print("Start met scrapen.")
    t0 = time.time()
    with open(input_file, 'r') as f:
        urls = [line.strip() for line in f]

    vectorizer = TfidfVectorizer(ngram_range=(1, ngram), stop_words='english', tokenizer=custom_tokenizer, token_pattern=None)
    all_results = []


    for url in urls:
        content = []
        for i in linklist(url):
            content.append(contentscraper(i))

        if not content:
            continue

        X = vectorizer.fit_transform(content)
        tfidf_tokens = vectorizer.get_feature_names_out()
        result = pd.DataFrame(data=X.toarray(), columns=tfidf_tokens)

        data = result.T
        data["gemiddelde"] = data.mean(axis=1)
        data = data.sort_values(by="gemiddelde", ascending=False)

        all_results.append(data[["gemiddelde"]].reset_index())

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.columns = ['word', 'score']
        final_df = final_df.groupby("word").mean().sort_values(by="score", ascending=False)
        top_terms = final_df.head(N).index.tolist()
    else:
        top_terms = []

    with open(output_file, 'w') as f:
        for term in top_terms:
            f.write(term + '\n')

    t1 = time.time()
    print(f"Generating the wordlist took {t1 - t0:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Wikipedia scraper for word ranking.")
    parser.add_argument("--input", required=True, help="Path to input file with URLs")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--N", type=int, default=100, help="Number of words to output (default: 100)")
    parser.add_argument("--ngram", type=int, default=1, help="Maximum n-gram size (default: 1)")

    args = parser.parse_args()

    wikiscraper(args.input, args.output, args.N, args.ngram)


if __name__ == "__main__":
    main()
