import os
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.storm_wiki.modules.article_polish import PolishPageModule

key = os.getenv("OPENAI_API_KEY")
openai_kwargs = {
    "api_key": key,
    "temperature": 1.0,
    "top_p": 0.9,
}
engine = OpenAIModel(model="gpt-3.5-turbo", max_tokens=500, **openai_kwargs)
topic = "Deep Research of MSTR's Bitcoin Investment"


def test_polish_page():
    polish_page = PolishPageModule(engine, engine)
    article_text = ""
    polish_result = polish_page(
        topic=topic, draft_page=article_text, polish_whole_page=True
    )
    print(polish_result)
    assert "pivotal decision" in polish_result.lead_section
    assert "pivotal decision" in polish_result.page
