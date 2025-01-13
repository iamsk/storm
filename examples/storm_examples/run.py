import os
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import YouRM
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

lm_configs = STORMWikiLMConfigs()
key = os.getenv("OPENAI_API_KEY")
print(key)
ydc_api_key = os.getenv("YDC_API_KEY")
print(ydc_api_key)
openai_kwargs = {
    'api_key': key,
    'temperature': 1.0,
    'top_p': 0.9,
}
# STORM is a LM system so different components can be powered by different models to reach a good balance between cost and quality.
# For a good practice, choose a cheaper/faster model for `conv_simulator_lm` which is used to split queries, synthesize answers in the conversation.
# Choose a more powerful model for `article_gen_lm` to generate verifiable text with citations.
gpt_35 = OpenAIModel(model='gpt-3.5-turbo', max_tokens=500, **openai_kwargs)
gpt_4 = OpenAIModel(model='gpt-4o', max_tokens=3000, **openai_kwargs)
lm_configs.set_conv_simulator_lm(gpt_35)
lm_configs.set_question_asker_lm(gpt_35)
lm_configs.set_outline_gen_lm(gpt_4)
lm_configs.set_article_gen_lm(gpt_4)
lm_configs.set_article_polish_lm(gpt_4)
# Check out the STORMWikiRunnerArguments class for more configurations.
engine_args = STORMWikiRunnerArguments(output_dir="./results/default")

rm = YouRM(ydc_api_key=ydc_api_key, k=engine_args.search_top_k)
runner = STORMWikiRunner(engine_args, lm_configs, rm)

topic = "Deep Research of whatnot which is a live streaming e-commerce platform"
graphviz = GraphvizOutput()
graphviz.output_type = 'dot'
graphviz.output_file = 'pycallgraph.dot'
with PyCallGraph(output=graphviz):
    runner.run(
        topic=topic,
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=True,
    )
    runner.post_run()
    runner.summary()
