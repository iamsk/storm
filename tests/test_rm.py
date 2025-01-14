import os
from knowledge_storm.rm import YouRM


ydc_api_key = os.getenv("YDC_API_KEY")
rm = YouRM(ydc_api_key=ydc_api_key, k=3)

def test_query():
    q = 'MicroStrategy public statements about Bitcoin investment'
    retrieved_data_list = rm(query_or_queries=[q])
    print(retrieved_data_list)
    assert isinstance(retrieved_data_list[0]['snippets'], list)
