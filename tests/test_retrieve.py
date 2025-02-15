import pickle
from knowledge_storm.interface import Retriever, Information
from knowledge_storm.base import get_rm
from typing import List

rm = get_rm("bing")
retriever = Retriever(rm=rm, max_thread=10)


def test_retrieve():
    queries = [
        "OpenRouter LLM routing platform benefits",
        "Why developers prefer OpenRouter over direct LLM integration",
        "OpenRouter vs individual LLM providers",
    ]
    searched_results: List[Information] = retriever.retrieve(list(set(queries)))
    with open("searched_results.pkl", "wb") as f:
        pickle.dump(searched_results, f)
    info = ""
    for n, r in enumerate(searched_results):
        info += "\n".join(f"[{n + 1}]: {s}" for s in r.snippets[:1])
        info += "\n\n"
    print(info)


if __name__ == "__main__":
    test_retrieve()
