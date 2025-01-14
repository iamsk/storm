import dspy
import os
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.storm_wiki.modules.outline_generation import (
    WritePageOutline,
    StormOutlineGenerationModule
)
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformationTable

key = os.getenv("OPENAI_API_KEY")
openai_kwargs = {
    "api_key": key,
    "temperature": 1.0,
    "top_p": 0.9,
}
engine = OpenAIModel(model="gpt-3.5-turbo", max_tokens=500, **openai_kwargs)
topic = "Deep Research of MSTR's Bitcoin Investment"


def test_write_draft_page_outline():
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
    storm_outline_generation_module = StormOutlineGenerationModule(outline_gen_lm=engine)
    information_table = StormInformationTable.from_conversation_log_file(
            "results/default/Deep_Research_of_MSTR's_Bitcoin_Investment/conversation_log.json")
    outline, draft_outline = storm_outline_generation_module.generate_outline(
            topic=topic,
            information_table=information_table,
            return_draft_outline=True,
        )
    print(outline)
    print(draft_outline)
    assert 'Bitcoin' in information_table.conversations[0][1][0].agent_utterance
