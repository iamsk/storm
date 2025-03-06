import dspy

from helper import get_engine
from knowledge_storm.storm_wiki.modules.article_polish import Translator


def test_translate_page(engine):
    with dspy.settings.context(lm=engine, show_guidelines=False):
        translator = dspy.Predict(Translator)
        text = "Deep Research of MSTR's Bitcoin Investment"
        translated_article = translator(page=text).translated_page
        print(translated_article)


if __name__ == "__main__":
    models = [
        # "openrouter/openai/gpt-4o-2024-11-20",
        # "openrouter/openai/o3-mini",
        "openrouter/google/gemini-2.0-flash-001",
        # "openrouter/google/gemini-2.0-pro-exp-02-05:free",
        # "openrouter/deepseek/deepseek-r1",
    ]
    for _model in models:
        _engine = get_engine(_model)
        test_translate_page(_engine)
        _engine.inspect_history(n=1)
