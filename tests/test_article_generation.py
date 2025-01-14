import os
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.storm_wiki.modules.article_generation import StormArticleGenerationModule, StormArticle
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformationTable

key = os.getenv("OPENAI_API_KEY")
openai_kwargs = {
    "api_key": key,
    "temperature": 1.0,
    "top_p": 0.9,
}
engine = OpenAIModel(model="gpt-3.5-turbo", max_tokens=500, **openai_kwargs)
topic = "Deep Research of MSTR's Bitcoin Investment"


def test_generate_section():
    storm_article_generation = StormArticleGenerationModule(article_gen_lm=engine)
    information_table = StormInformationTable.from_conversation_log_file(
            "results/default/Deep_Research_of_MSTR's_Bitcoin_Investment/conversation_log.json")
    article_with_outline = StormArticle.from_outline_file(
        topic=topic,
        file_path="results/default/Deep_Research_of_MSTR's_Bitcoin_Investment/storm_gen_outline.txt",
    )
    section_title = "Initial Bitcoin Acquisition"
    section_query = article_with_outline.get_outline_as_list(
        root_section_name=section_title, add_hashtags=False
    )
    print(f"section_query: {section_query}")
    queries_with_hashtags = article_with_outline.get_outline_as_list(
        root_section_name=section_title, add_hashtags=True
    )
    print(f"queries_with_hashtags: {queries_with_hashtags}")
    section_outline = "\n".join(queries_with_hashtags)
    print(f"section_outline: {section_outline}")
    information_table.prepare_table_for_retrieval()
    section_output_dict = storm_article_generation.generate_section(
                topic=topic,
                section_name=section_title,
                information_table=information_table,
                section_outline=section_outline,
                section_query=section_query,
            )
    print(section_output_dict)
    assert "pivotal decision" in section_output_dict['section_content']
