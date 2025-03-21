import dspy

from helper import get_engine
from knowledge_storm.storm_wiki.modules.knowledge_curation import AskQuestionWithPersona
from knowledge_storm.storm_wiki.modules.persona_generator import venture_capitalist, data_analyst


def run(engine):
    with dspy.settings.context(lm=engine):
        # topic = "Deep Research of MSTR's Bitcoin Investment"
        topic = "deep research on OpenRouter as a LLM routing platform, focusing on the key reasons why users choose it over alternatives"
        # topic = "conduct an in-depth analysis of Docker as developer tool"
        ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        # persona = "Bitcoin Historian: This editor will focus on the history section of the article, providing details on the creation of Bitcoin, key milestones in its growth, regulatory actions, and its current status."
        question = ask_question_with_persona(topic=topic, persona=data_analyst, conv="").question
        print(f"{engine.model}: {question}")


if __name__ == "__main__":
    # _engine = get_engine('deepseek/deepseek-reasoner', max_tokens=8192, platform_key="DEEPSEEK_API_KEY")
    # run(_engine)
    # exit()
    models = [
        "openrouter/anthropic/claude-3.7-sonnet",
        "openrouter/anthropic/claude-3.7-sonnet:thinking",
        # "openrouter/google/gemini-2.0-flash-001",
        # "openrouter/google/gemini-2.0-pro-exp-02-05:free",
        # "openrouter/openai/gpt-4.5-preview",
        # "openrouter/deepseek/deepseek-r1",
        # "openrouter/openai/o3-mini-high",
        # "openrouter/openai/o1",
    ]
    for _model in models:
        _engine = get_engine(_model)
        run(_engine)
