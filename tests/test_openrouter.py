import dspy

from helper import get_engine
from knowledge_storm.storm_wiki.modules.knowledge_curation import AskQuestionWithPersona


def run(engine):
    with dspy.settings.context(lm=engine):
        topic = "Deep Research of MSTR's Bitcoin Investment"
        ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        persona = "Bitcoin Historian: This editor will focus on the history section of the article, providing details on the creation of Bitcoin, key milestones in its growth, regulatory actions, and its current status."
        question = ask_question_with_persona(
            topic=topic, persona=persona, conv=""
        ).question
        print(f"{engine.model}: {question}")


if __name__ == "__main__":
    # _engine = get_engine('deepseek/deepseek-reasoner', max_tokens=8192, platform_key="DEEPSEEK_API_KEY")
    # run(_engine)
    # exit()
    models = [
        # "openrouter/openai/o1-preview",
        # "openrouter/openai/o1-mini",
        # "openrouter/google/gemini-2.0-flash-001",
        # "openrouter/google/gemini-2.0-pro-exp-02-05:free",
        "openrouter/deepseek/deepseek-r1",
    ]
    for _model in models:
        _engine = get_engine(_model)
        run(_engine)
