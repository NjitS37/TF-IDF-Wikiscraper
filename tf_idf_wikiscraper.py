# Import the useful libraries

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.request import urlopen
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def contentscraper(link):
  """Deze functie neemt een url als string in en stript alle HMTL elementen om zo alleen de text inhoud te returen als een string."""
  html = urlopen(link).read()
  soup = BeautifulSoup(html, features="html.parser")

  # Remove all script and style elements
  for script in soup(["script", "style"]):
      script.extract()

  # Find the main html area for the article
  content_area = soup.find(id="mw-content-text")

  # Find the "References" header using regex
  references_header = content_area.find("h2", string=re.compile(r"References", re.I))

  # If "References" header is found, extract content before it
  if references_header:
      content = []
      for element in content_area.find_all():  # Iterate through all elements
          if element == references_header:
              break  # Stop when we reach the "References" header
          if element.name in ["p", "ul", "ol"]:  # Select relevant elements
              content.append(element.text)

      text = "\n".join(content)  # Join the content with newlines

  else:
      # If "References" header is not found, use all content
      text = content_area.get_text()

  # Remove citations and edits within brackets []
  text = re.sub(r"\[\d+\]", "", text)

  # Clean up the text

  lines = (line.strip() for line in text.splitlines())
  chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
  text = '\n'.join(chunk for chunk in chunks if chunk)

  return(text)


def linklist(url):
  """Deze functie neemt een url als string en levert een lijst van alle urls waarnaar gelinked wordt in het artikel, samen met de link van het artikel zelf."""
  response = requests.get(url)

  soup = BeautifulSoup(response.content, 'html.parser')

  # Here we will get all the pages our starting page links to.
  # In HTML, <a> </a> is used for creating hyperlinks

  titles_written = [url]

  allLinks = soup.find(id="bodyContent").find_all("a")

  for link in allLinks:
    # Check if the link has the 'href' attribute before accessing it
    if 'href' not in link.attrs:  # Check for 'href' attribute first
        continue
    if link['href'].find("/wiki/") == -1 or ":" in link['href']:
        continue  # Skip if it's not a wikipedia article or it is a special page.

    # Construct the full URL
    full_url = "https://en.wikipedia.org" + link['href']

    if full_url not in titles_written:
        titles_written.append(full_url)

  return (titles_written)


def custom_tokenizer(text):
  """Tokenizer die woorden filtert op basis van lengte en speciale tekens."""
  words = re.findall(r'\b[a-zA-Z\'-]{3,}\b', text.lower())  # Filter out words with at least 3 characters

  # Filter out "citation" and "citation needed", dit zijn notities die in artikelen staan
  filtered_words = [word for word in words if word not in ["citation", "citation needed"]]

  return filtered_words  # Return the filtered words


def wikiscraper(input_file, output_file, N, ngram):
    """Deze scraper neemt een file met urls als string, een aantal woorden die de functie output in een woordenlijst (N), en hoeveel woorden er maximaal gepakt worden tegelijk (ngram).
    De functie maakt een ranking van de woorden uit Wikipedia-artikelen met TF-IDF en returnt een woordenlijst."""

    with open(input_file, 'r') as f:
        urls = [line.strip() for line in f]

    vectorizer = TfidfVectorizer(ngram_range=(1, ngram), stop_words='english', tokenizer=custom_tokenizer, token_pattern=None)
    all_results = []  # Store results per main link

    t0 = time.time()

    for url in urls:
        content = []  # Reset content for each main link

        for i in linklist(url):  # Get linked articles
            content.append(contentscraper(i))  # Scrape article content

        if not content:  # Skip if no content was retrieved
            continue

        # Apply TF-IDF for this main link
        X = vectorizer.fit_transform(content)
        tfidf_tokens = vectorizer.get_feature_names_out()

        # Create a DataFrame for this main link
        result = pd.DataFrame(data=X.toarray(), columns=tfidf_tokens)

        # Aggregate scores and rank words **for this main link only**
        data = result.T
        data["gemiddelde"] = data.mean(axis=1)
        data = data.sort_values(by="gemiddelde", ascending=False)

        # Take top N words for this main link
        # Bewaar alle woorden met hun TF-IDF scores (zonder vroegtijdige selectie)
        all_results.append(data[["gemiddelde"]].reset_index())  # Voeg alles toe

    # Combine all results and compute global ranking
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # Rename the columns to 'word' and 'score'
        final_df.columns = ['word', 'score']  # Rename columns here
        final_df = final_df.groupby("word").mean().sort_values(by="score", ascending=False)  # Gemiddelde per woord
        top_terms = final_df.head(N).index.tolist()  # Pas nu pas de selectie toe
    else:
        top_terms = []

    # Write the top N terms to a file
    with open(output_file, 'w') as f:
        for term in top_terms:
            f.write(term + '\n')

    t1 = time.time()
    print(f"Generating the wordlist took {t1 - t0:.2f} seconds.")