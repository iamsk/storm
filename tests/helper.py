import os
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    VectorRM,
    StanfordOvalArxivRM,
    SerperRM,
    BraveRM,
    SearXNG,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    GoogleSearch,
    AzureAISearch,
)
from knowledge_storm.utils import load_api_key

load_api_key(toml_file_path="secrets.toml")


def get_engine(model, max_tokens=10000):
    openai_kwargs = {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "temperature": 1.0,
        "top_p": 0.9,
    }
    engine = LitellmModel(
        model=model,
        max_tokens=max_tokens,
        **openai_kwargs,
    )
    return engine


def get_rm(retriever):
    search_top_k = 3
    match retriever:
        case "you":
            rm = YouRM(ydc_api_key=os.getenv("YDC_API_KEY"), k=search_top_k)
        case "bing":
            rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=search_top_k,
            )
        case "serper":
            rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=search_top_k,
            )
        case "searxng":
            rm = SearXNG(
                searxng_api_url="https://searx.be/",
                searxng_api_key=os.getenv("SEARXNG_API_KEY"),
                k=search_top_k,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(k=search_top_k, safe_search="On", region="us-en")
        case "tavily":
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=search_top_k,
                include_raw_content=True,
            )
        case "azure":
            rm = AzureAISearch(
                azure_ai_search_api_key=os.getenv("AZURE_AI_SEARCH_API_KEY"),
                k=search_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", or "searxng"'
            )
    return rm
