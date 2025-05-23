# TF-IDF-Wikiscraper
This Python script scrapes English Wikipedia articles based on a list of main article URLs provided in a `.txt` file. This code needs an internet connection.
The script collects all hyperlinks within the main articles, then analyzes the content of those articles to extract the most relevant terms. 
The output consists of the terms with the highest average TF-IDF (Term Frequency-Inverse Document Frequency) across all articles. The length of the wordlist and a range of the word `n-grams` can be specified.

The code, `tf_idf_wikiscraper.py` can be ran from the command line.

The following Python packages are used in this code:
- `argparse` (for running the code from command line)
- `requests`
- `beautifulsoup4`
- `numpy`
- `pandas`
- `urllib.request`
- `re` (Regular Expressions, part of Python's standard library)
- `sklearn` (for TF-IDF analysis)
- `time` (for time-related functionality, part of Python's standard library)

The necessary packages (those not included in Python's standard library), can be installed using pip:
```bash
pip install requests beautifulsoup4 numpy pandas scikit-learn
```


The input is a `.txt` file where each line contains the URL of a relevant Wikipedia article. For example:

https://en.wikipedia.org/wiki/Python_(programming_language)  
https://en.wikipedia.org/wiki/Artificial_intelligence  
https://en.wikipedia.org/wiki/Machine_learning  

Functions of the code:
- Article Scraping: The code scrapes the main Wikipedia articles provided in the input `.txt` file.  
- Hyperlink Extraction: For each main article, the script extracts all hyperlinks pointing to other relevant Wikipedia articles.  
- TF-IDF Analysis: The script computes the Term Frequency-Inverse Document Frequency (TF-IDF) for words found across all the scraped articles.  
- Output: The code outputs the top N terms with the highest average TF-IDF scores, where N is the number of main articles initially provided in the input.

Running the Code:
To run the script, ensure that the `.txt` file with the list of Wikipedia article URLs is prepared. Then, execute the Python script. this can be done in an IDE or from, for example, Anaconda prompt.

Example Command:
```bash
python tf_idf_wikiscraper.py --input input.txt --output wordlist.txt --N 50000 --ngram_max 2
```

Where `input.txt` is the `.txt` file containing the list of URLs and `wordlist.txt` is the output file containing the scraped wordlist. `N` is the length of the returned wordlist, and `ngram_min` is the minimum and `ngram_max` is the maximum of how many words a term in the wordlist can exist of. When not specified, the standard value for `N` is 100, and the standard values for `ngram_min` and `ngram_max` are 1. An option `include_weights` is also included, where weights are added before the word separated by a space, so that PCFG can be applied to this wordlist.

The script will output the terms with the highest average TF-IDF values for all scraped articles. These terms can be considered the most relevant across the provided Wikipedia articles.
