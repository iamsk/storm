from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.utils import load_api_key
from tests.helper import get_engine, get_rm

load_api_key(toml_file_path="secrets.toml")

lm_configs = STORMWikiLMConfigs()

lm_configs.set_conv_simulator_lm(get_engine("openrouter/anthropic/claude-3.5-sonnet"))
# lm_configs.set_question_asker_lm(get_engine("deepseek/deepseek-reasoner", max_tokens=8192, platform_key="DEEPSEEK_API_KEY"))
lm_configs.set_question_asker_lm(get_engine("openrouter/anthropic/claude-3.7-sonnet:thinking"))
lm_configs.set_outline_gen_lm(get_engine("openrouter/openai/gpt-4o-2024-11-20"))
lm_configs.set_article_gen_lm(get_engine("openrouter/openai/gpt-4o-2024-11-20"))
lm_configs.set_article_polish_lm(get_engine("openrouter/google/gemini-2.0-flash-001"))

engine_args = STORMWikiRunnerArguments(output_dir="./results/default",
                                       #    max_conv_turn=5,
                                       #    max_perspective=5,
                                       #    max_search_queries_per_turn=3,
                                       #    search_top_k=5,
                                       #    retrieve_top_k=5,
                                       #    max_thread_num=20
                                       )

rm = get_rm('serper')
runner = STORMWikiRunner(engine_args, lm_configs, rm)

topic = "give me a deep study on Coinbase's revenue and profit break down by following segments, exchange vs others (further break into wallet, L2 - base, custody, etc. ), and within exchanges, futher break down into individual vs institution, spot  vs derivatives etc. then analysis on the growth drivers."
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
