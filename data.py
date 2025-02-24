import pandas as pd

def load_verizon_queries():
    df = pd.read_csv("data/verizon_clustered.csv")
    queries = list(df['text'])

    return queries

def _parse_airbnb_queries(user_question):
  if "[customer]" in user_question:
    lines = user_question.split("\n")
    for l in lines:
      if "[customer]" in l:
        return l.split("[customer]")[1].strip()
      elif "[supportbot]" in l:
        continue
  else:
    return user_question
  
def load_airbnb_queries():
    df = pd.read_csv("data/airbnb_truthfulness_sample.csv")

    df["query"] = df.apply(lambda row: _parse_airbnb_queries(row["user_question"]), axis=1)

    return list(df['query'])