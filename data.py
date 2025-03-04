import pandas as pd

def load_verizon_queries():
    df = pd.read_csv("data/verizon_clustered.csv")
    queries = list(df['text'])

    return queries

def load_python_code_queries():
    df = pd.read_csv("data/python_code_instructions.csv")
    queries = list(df['query'])

    return queries

def load_mental_health_queries():
    df = pd.read_csv("data/mental_health.csv")
    queries = list(df['query'])

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

def load_ood_queries():
    df = pd.read_csv("data/clinic_ood_queries.csv")
    queries = list(df['query'])

    return queries