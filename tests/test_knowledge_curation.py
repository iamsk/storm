import dspy
from knowledge_storm.storm_wiki.modules.knowledge_curation import (
    TopicExpert,
    WikiWriter,
    ConvSimulator,
    QuestionToQuery,
)
from knowledge_storm.interface import Retriever
from knowledge_storm.base import get_rm
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from helper import get_engine


def test_ask_question():
    def _ask_question(topic, persona, engine):
        wiki_writer = WikiWriter(engine=engine)
        question = wiki_writer(topic=topic, persona=persona, dialogue_turns=[])
        print(f"# {engine.model}")
        print(question.question)
        # Can you provide details on when MicroStrategy (MSTR) first started investing in Bitcoin and what motivated their decision to do so?
        # assert "motivated" in question.question

    topic = "deep research on OpenRouter as a LLM routing platform, focusing on the key reasons why users choose it over alternatives"
    persona = '**Developer Experience Advocate**: Details OpenRouter’s unified API, SDKs, real-time analytics, and simplified billing—contrasting fragmented alternatives. Mirrors "Hugging Face Hub" developer-centric design principles.'
    models = [
        "openrouter/deepseek/deepseek-r1",
        "openrouter/openai/gpt-4o-2024-11-20",
        "openrouter/openai/o3-mini",
        "openrouter/openai/o3-mini-high",
        "openrouter/google/gemini-2.0-flash-001",
        "openrouter/google/gemini-2.0-pro-exp-02-05:free",
        "openrouter/google/gemini-2.0-flash-thinking-exp:free",
    ]
    for model in models:
        engine = get_engine(model)
        _ask_question(topic, persona, engine)


def test_generate_queries():
    def _generate_queries(question, topic, engine):
        with dspy.settings.context(lm=engine, show_guidelines=False):
            generate_queries = dspy.Predict(QuestionToQuery)
            queries = generate_queries(topic=topic, question=question).queries
            print(f"# {engine.model}")
            print(queries)

    topic = "deep research on OpenRouter as a LLM routing platform, focusing on the key reasons why users choose it over alternatives"
    question = "As a developer experience advocate, I'm keen to understand what are the primary reasons developers choose OpenRouter over directly integrating with individual LLM providers?"
    models = [
        "openrouter/deepseek/deepseek-r1",
        "openrouter/openai/gpt-4o-2024-11-20",
        "openrouter/openai/o3-mini",
        "openrouter/openai/o3-mini-high",
        "openrouter/google/gemini-2.0-flash-001",
        "openrouter/google/gemini-2.0-flash-thinking-exp:free",
        "openrouter/anthropic/claude-3.5-sonnet",
    ]
    for model in models:
        engine = get_engine(model)
        _generate_queries(question, topic, engine)


def test_question_answering(question, topic, retriever, engine):
    def _question_answering():
        topic_expert = TopicExpert(
            engine=engine,
            max_search_queries=3,
            search_top_k=3,
            retriever=retriever,
        )
        answer = topic_expert(topic, question, "")
        print(f"# {engine.model}")
        print(answer.queries)
        print(answer.answer)
        # MicroStrategy made its initial Bitcoin investment in August 2020, purchasing 21,454 BTC for approximately $250 million. This move...
        # assert "initial" in ";".join(answer.queries)
        # assert "2020" in answer.answer

    topic = "deep research on OpenRouter as a LLM routing platform, focusing on the key reasons why users choose it over alternatives"
    question = "As a developer experience advocate, I'm keen to understand what are the primary reasons developers choose OpenRouter over directly integrating with individual LLM providers?"
    models = [
        # "openrouter/deepseek/deepseek-r1",
        "openrouter/openai/gpt-4o-2024-11-20",
        "openrouter/openai/o3-mini",
        "openrouter/openai/o3-mini-high",
        "openrouter/google/gemini-2.0-flash-001",
        "openrouter/google/gemini-2.0-flash-thinking-exp:free",
        "openrouter/anthropic/claude-3.5-sonnet",
    ]
    rm = get_rm("bing")
    retriever = Retriever(rm=rm, max_thread=10)
    for model in models:
        engine = get_engine(model)
        _question_answering(question, topic, retriever, engine)


def test_conv_simulator():
    model = "openrouter/openai/o1-preview"
    engine = get_engine(model)
    rm = get_rm("you")
    retriever = Retriever(rm=rm, max_thread=10)
    topic = "Deep Research of MSTR's Bitcoin Investment"
    persona = "Bitcoin Historian: This editor will focus on the history section of the article, providing details on the creation of Bitcoin, key milestones in its growth, regulatory actions, and its current status."
    conv_simulator = ConvSimulator(
        topic_expert_engine=engine,
        question_asker_engine=engine,
        retriever=retriever,
        max_search_queries_per_turn=3,
        search_top_k=3,
        max_turn=3,
    )
    conversations = conv_simulator(
        topic=topic,
        ground_truth_url="",
        persona=persona,
        callback_handler=BaseCallbackHandler(),
    )
    print(conversations)


if __name__ == "__main__":
    # test_ask_question()
    # test_generate_queries()
    test_question_answering()
    # test_conv_simulator()
