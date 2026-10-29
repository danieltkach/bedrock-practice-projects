import re
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "data" / "transcripts"
UA = {"User-Agent": "Mozilla/5.0"}
TERMS = ["2024", "2023"]  # Supreme Court terms

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()

for term in TERMS:
    page = fetch(f"https://www.supremecourt.gov/oral_arguments/argument_transcript/{term}").decode("utf-8", "ignore")
    links = sorted(set(re.findall(rf"argument_transcripts/{term}/[^\"']+\.pdf", page)))
    print(f"term {term}: {len(links)} transcripts")
    for link in links:
        dest = OUT / f"{term}_{Path(link).name}"
        if dest.exists():
            continue
        dest.write_bytes(fetch("https://www.supremecourt.gov/oral_arguments/" + link))
        time.sleep(1)  # be polite to the server
        print("  saved", dest.name, file=sys.stderr)
