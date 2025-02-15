import os
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.rm import YouRM, SerperRM
from knowledge_storm.utils import load_api_key
from tests.helper import get_engine, get_rm
from knowledge_storm.lm import GoogleModel

load_api_key(toml_file_path="secrets.toml")

lm_configs = STORMWikiLMConfigs()

llm = get_engine("openrouter/openai/o3-mini-high")
# llm = get_engine("openrouter/deepseek/deepseek-r1")
# gemini_kwargs = {
#     "api_key": os.getenv("GOOGLE_API_KEY"),
#     "temperature": 1.0,
#     "top_p": 0.9,
# }
# llm = GoogleModel(model="gemini-2.0-flash-thinking-exp-01-21", max_tokens=1000, **gemini_kwargs)
llm2 = get_engine("openrouter/openai/o1-mini", 20000)
lm_configs.set_conv_simulator_lm(llm)
lm_configs.set_question_asker_lm(get_engine("openrouter/deepseek/deepseek-r1"))
lm_configs.set_outline_gen_lm(llm2)
lm_configs.set_article_gen_lm(llm2)
lm_configs.set_article_polish_lm(llm2)

engine_args = STORMWikiRunnerArguments(output_dir="./results/default",
                                    #    max_conv_turn=5,
                                    #    max_perspective=5,
                                    #    max_search_queries_per_turn=3,
                                    #    search_top_k=5,
                                    #    retrieve_top_k=5,
                                    #    max_thread_num=20
                                       )

# rm = get_rm('you')
rm = get_rm('duckduckgo')
# rm = SerperRM(serper_search_api_key=os.getenv("SERPER_API_KEY"), query_params={"autocorrect": True, "num": 10, "page": 1, "tbs": "qdr:y"},)
runner = STORMWikiRunner(engine_args, lm_configs, rm)

topic = "deep research on OpenRouter as a LLM routing platform, focusing on the key reasons why users choose it over alternatives"
runner.run(
    topic=topic,
    # do_research=False,
    # do_generate_outline=False,
    # do_generate_article=False,
    # do_polish_article=True,
    remove_duplicate=True
)
runner.post_run()
runner.summary()
