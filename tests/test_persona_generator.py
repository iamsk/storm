import dspy
import os
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.storm_wiki.modules.persona_generator import FindRelatedTopic, get_wiki_page_title_and_toc, CreateWriterWithPersona

key = os.getenv("OPENAI_API_KEY")
openai_kwargs = {
    "api_key": key,
    "temperature": 1.0,
    "top_p": 0.9,
}
engine = OpenAIModel(model="gpt-3.5-turbo", max_tokens=500, **openai_kwargs)
topic = "Deep Research of MSTR's Bitcoin Investment"


def test_find_related_topics():
    with dspy.settings.context(lm=engine):
        find_related_topic = dspy.ChainOfThought(FindRelatedTopic)
        response = find_related_topic(topic=topic)
        print(response.related_topics)
        # 1. https://en.wikipedia.org/wiki/Bitcoin
        # 2. https://en.wikipedia.org/wiki/Cryptocurrency
        # 3. https://en.wikipedia.org/wiki/Investment_strategy
        # 4. https://en.wikipedia.org/wiki/Case_study
        assert "https://en.wikipedia.org/wiki/Bitcoin" in response.related_topics


def test_get_wiki_page_title_and_toc():
    title, toc = get_wiki_page_title_and_toc('https://en.wikipedia.org/wiki/Bitcoin')
    print(title)
    print(toc)
    # History
    #   Background
    #   2008–2009: Creation
    #   2010–2012: Early growth
    #   2013–2014: First regulatory actions
    #   2015–2019
    #   2020–present
    # Design
    #   Units and divisibility
    #   Blockchain
    #   Addresses and transactions
    #   Mining
    #   Privacy and fungibility
    #   Wallets
    #   Scalability and decentralization challenges
    # Economics and usage
    #   Bitcoin's theoretical roots and ideology
    #   Recognition as a currency and legal status
    #   Use for payments
    #   Use for investment and status as an economic bubble
    # Further reading
    assert 'Economics' in toc
    
    
def test_generate_persona():
    create_writer_with_persona = CreateWriterWithPersona(engine=engine)
    personas = create_writer_with_persona(topic=topic)
    print(personas)
    # ["Bitcoin Historian: This editor will focus on the history section of the article, providing details on the creation of Bitcoin, key milestones in its growth, regulatory actions, and its current status."]
    assert 'Historian' in personas.personas[0]
