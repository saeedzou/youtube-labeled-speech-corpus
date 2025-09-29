# Wikipedia dump file URL
def make_dump_url(lang: str) -> str:
  return f"https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles-multistream-index.txt.bz2"

# YouTube Search URL
def make_query_url(query: str, cc: bool=False) -> str:
  q = query.rstrip("\n").strip(" ").replace(" ", "+")
  if cc:
    return f"https://www.youtube.com/results?search_query={q}&sp=EgYQASgBMAE%253D"
  return f"https://www.youtube.com/results?search_query={q}&sp=EgQQASgB"

# YouTube video URL
def make_video_url(videoid: str) -> str:
  return f"https://www.youtube.com/watch?v={videoid}"