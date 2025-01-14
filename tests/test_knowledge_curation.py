import os
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.storm_wiki.modules.knowledge_curation import (
    TopicExpert,
    WikiWriter,
)
from knowledge_storm.rm import YouRM
from knowledge_storm.interface import Retriever

key = os.getenv("OPENAI_API_KEY")
openai_kwargs = {
    "api_key": key,
    "temperature": 1.0,
    "top_p": 0.9,
}
engine = OpenAIModel(model="gpt-3.5-turbo", max_tokens=500, **openai_kwargs)
topic = "Deep Research of MSTR's Bitcoin Investment"
ydc_api_key = os.getenv("YDC_API_KEY")
rm = YouRM(ydc_api_key=ydc_api_key, k=3)
retriever = Retriever(rm=rm, max_thread=10)


def test_ask_question():
    wiki_writer = WikiWriter(engine=engine)
    persona = "Bitcoin Historian: This editor will focus on the history section of the article, providing details on the creation of Bitcoin, key milestones in its growth, regulatory actions, and its current status."
    question = wiki_writer(topic=topic, persona=persona, dialogue_turns=[])
    print(question)
    # Can you provide details on when MicroStrategy (MSTR) first started investing in Bitcoin and what motivated their decision to do so?
    assert "motivated" in question.question


def test_question_answering():
    topic_expert = TopicExpert(
        engine=engine,
        max_search_queries=3,
        search_top_k=3,
        retriever=retriever,
    )
    question = "Can you provide information on MicroStrategy's initial Bitcoin investment, including the date, amount of Bitcoin purchased, and any public statements they made regarding the investment at that time?"
    answer = topic_expert(topic, question, "")
    print(answer)
    # MicroStrategy made its initial Bitcoin investment in August 2020, purchasing 21,454 BTC for approximately $250 million. This move...
    assert "initial" in ";".join(answer.queries)
    assert "2020" in answer.answer
