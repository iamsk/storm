from dotenv import load_dotenv
from argparse import ArgumentParser
from waybackpy import WaybackMachineCDXServerAPI
from knowledge_storm.base import get_rm

load_dotenv()


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"],
        help="The search engine API to use for retrieving information.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="MicroStrategy Bitcoin purchase date and amount",
        help="The search query to excute.",
    )

    args = parser.parse_args()
    query = args.query
    retriever = args.retriever
    execute(query, retriever)


def execute(query, retriever):
    print(f"{retriever} query: {query}")
    rm = get_rm(retriever)
    retrieved_data_list = rm(query_or_queries=query, exclude_urls=[])
    # import pdb;pdb.set_trace()
    for retrieved_data in retrieved_data_list:
        url = retrieved_data["url"]
        try:
            user_agent = "gbot"
            cdx_api = WaybackMachineCDXServerAPI(url, user_agent)
            _date = cdx_api.oldest().timestamp[:8]
        except:
            _date = None
        print(f"{_date}: {retrieved_data['title']} - {url}")


def run():
    query = "MicroStrategy Bitcoin purchase date and amount"
    for retriever in ["bing", "you", "brave", "serper", "duckduckgo", "tavily"]:
        execute(query, retriever)


if __name__ == "__main__":
    # main()
    run()
