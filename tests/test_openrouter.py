import dspy
from knowledge_storm.storm_wiki.modules.knowledge_curation import AskQuestionWithPersona
from helper import get_engine


def run(model):
    engine = get_engine(model)

    with dspy.settings.context(lm=engine):
        topic = "Deep Research of MSTR's Bitcoin Investment"
        ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        persona = "Bitcoin Historian: This editor will focus on the history section of the article, providing details on the creation of Bitcoin, key milestones in its growth, regulatory actions, and its current status."
        question = ask_question_with_persona(
            topic=topic, persona=persona, conv=""
        ).question
        print(f"{model}: {question}")


if __name__ == "__main__":
    models = [
        "openrouter/openai/o1-preview",
        "openrouter/openai/o1-mini",
        "openrouter/google/gemini-2.0-flash-001",
        "openrouter/google/gemini-2.0-pro-exp-02-05:free",
        "openrouter/deepseek/deepseek-r1",
    ]
    for _model in models:
        run(_model)
