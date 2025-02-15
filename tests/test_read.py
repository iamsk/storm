from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformationTable

information_table_local_path = (
    "results/default/deep_research_on_OpenRouter_as_LLM_routing_platform,_focusing_on_the_key_reasons_why_users_choose_it_over_alternatives/conversation_log.json"
)

information_table = StormInformationTable.from_conversation_log_file(
    information_table_local_path
)

for conversation in information_table.conversations:
    print(f"# {conversation[0].split(':')[0].strip()}")
    print(f"{conversation[0].split(':')[1].strip()}")
    for num, dialogue in enumerate(conversation[1], start=1):
        print(f"## Question {num}:")
        print(f"{dialogue.user_utterance}")
        print("### Queries:")
        for index, query in enumerate(dialogue.search_queries, start=1):
            print(f"{index}. {query}")
        print("### Answer:")
        print(f"{dialogue.agent_utterance}")
