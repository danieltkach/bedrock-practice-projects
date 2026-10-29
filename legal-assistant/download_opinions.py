import re
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "data" / "opinions"
UA = {"User-Agent": "Mozilla/5.0"}
TERMS = ["24", "23"]  # Supreme Court terms (OT2024, OT2023)

def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()

for term in TERMS:
    page = fetch(f"https://www.supremecourt.gov/opinions/slipopinion/{term}").decode("utf-8", "ignore")
    links = sorted(set(re.findall(rf"/opinions/{term}pdf/[^\"']+\.pdf", page)))
    print(f"term {term}: {len(links)} opinions")
    for link in links:
        dest = OUT / Path(link).name
        if dest.exists():
            continue
        dest.write_bytes(fetch("https://www.supremecourt.gov" + link))
        time.sleep(1)  # be polite to the server
        print("  saved", dest.name, file=sys.stderr)
