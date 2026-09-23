from os import link
from unittest import result

import feedparser
from numpy import number
from sympy import re
RSS_SOURCES = ["https://www.iltalehti.fi/rss/uutiset.xml",
               "https://www.yle.fi/rss/uutiset.xml",
               "https://www.hs.fi/rss/tuoreimmat.xml",
               "https://www.mtv.fi/rss/uutiset.xml",
               "https://www.is.fi/rss/tuoreimmat.xml",]
def collect_articles():
    articles = []
    for url in RSS_SOURCES:
        print(f"[+] haetaan: {url}")
        feed = feedparser.parse(url)
        for number, article in enumerate(feed.entries, 1):
            title=article.get("title", "")
            summary=article.get("summary", "")
            text=f"{title}\n{summary}"
            print(f"\n{'='*50}")
            print(f"UUTINEN {number}/{len(articles)}")
            if text.strip():
                articles.append({"text": text, "title": title, "link": link})
    return articles