import dspy
from knowledge_storm.storm_wiki.modules.outline_generation import (
    WritePageOutline,
    StormOutlineGenerationModule,
)
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformationTable
from helper import get_engine


def test_write_draft_page_outline():
    engine = get_engine("openrouter/openai/gpt-4o-2024-11-20")
    topic = "Deep Research of MSTR's Bitcoin Investment"
    with dspy.settings.context(lm=engine):
        write_outline = dspy.Predict(WritePageOutline)
        result = write_outline(
            topic=topic,
        )
        print(result)  # old_outline
        assert "MSTR" in result.outline


old_outline = """
# Introduction
## Background information on MicroStrategy (MSTR)
## Overview of Bitcoin investment by MSTR
# History
## Timeline of MSTR's involvement with Bitcoin
## Impact of Bitcoin investment on MSTR's stock price
# Investment Strategy
## Reasons for MSTR's decision to invest in Bitcoin
## Analysis of risks and benefits of Bitcoin investment
# Reception
## Reaction from investors and analysts
## Comparison to other companies investing in Bitcoin
# Future Outlook
## Potential impact of Bitcoin investment on MSTR's future performance
## Speculation on MSTR's future Bitcoin investment strategies
# References
"""


def test_generate_outline():
    def _generate_outline(topic, engine):
        storm_outline_generation_module = StormOutlineGenerationModule(
            outline_gen_lm=engine
        )
        information_table = StormInformationTable.from_conversation_log_file(
            "results/default/deep_research_on_OpenRouter_as_a_LLM_routing_platform,_focusing_on_the_key_reasons_why_users_choose_it_over_alternatives/conversation_log.json"
        )
        outline, draft_outline = storm_outline_generation_module.generate_outline(
            topic=topic,
            information_table=information_table,
            return_draft_outline=True,
        )
        print(f"# {engine.model}")
        print(outline.to_string())
        # print(draft_outline)
        # assert 'Bitcoin' in information_table.conversations[0][1][0].agent_utterance

    topic = "deep research on OpenRouter as a LLM routing platform, focusing on the key reasons why users choose it over alternatives"
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
        _generate_outline(topic, engine)


if __name__ == "__main__":
    test_generate_outline()
