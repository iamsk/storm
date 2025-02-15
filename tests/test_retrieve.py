import os
from knowledge_storm.rm import YouRM
from knowledge_storm.interface import Retriever, Information
from typing import List

ydc_api_key = os.getenv("YDC_API_KEY")
print(ydc_api_key)

rm = YouRM(ydc_api_key=ydc_api_key, k=3)
retriever = Retriever(rm=rm, max_thread=10)

def test_retrieve():
    queries = []
    searched_results: List[Information] = retriever.retrieve(list(set(queries)))
