# TF-IDF-Wikiscraper
This Python script scrapes Wikipedia articles based on a list of main article URLs provided in a `.txt` file. This code needs an internet connection.
The script collects all hyperlinks within the main articles, then analyzes the content of those articles to extract the most relevant terms. 
The output consists of the terms with the highest average TF-IDF (Term Frequency-Inverse Document Frequency) across all articles.

The following Python packages are used in this code:
- `requests`
- `beautifulsoup4`
- `numpy`
- `pandas`
- `urllib.request`
- `re` (Regular Expressions, part of Python's standard library)
- `sklearn` (for TF-IDF analysis)
- `time` (for time-related functionality, part of Python's standard library)

To install the necessary packages (those not included in Python's standard library), run the following command:
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
To run the script, ensure that the `.txt` file with the list of Wikipedia article URLs is prepared. Then, execute the Python script.

Example Command:
```bash
python wikiscraper.py --input input.txt --output wordlist.txt
```

Where `wikiscraper.py` is the name of your Python script, `input.txt` is the `.txt` file containing the list of URLs and `wordlist.txt` is the output file containing the scraped wordlist.

The script will output the terms with the highest average TF-IDF values for all scraped articles. These terms can be considered the most relevant across the provided Wikipedia articles.
