# Wikipedia dump file URL
def make_dump_url(lang: str) -> str:
  return f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles-multistream-index.txt.bz2"
