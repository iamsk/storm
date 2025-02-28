from utils.file import get_demo_dir
from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
from knowledge_storm.rm import YouRM
import streamlit as st
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from tests.helper import get_engine


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, status_container):
        self.status_container = status_container

    def on_identify_perspective_start(self, **kwargs):
        self.status_container.info(
            "Start identifying different perspectives for researching the topic."
        )

    def on_identify_perspective_end(self, perspectives: list[str], **kwargs):
        perspective_list = "\n- ".join(perspectives)
        self.status_container.success(
            f"Finish identifying perspectives. Will now start gathering information"
            f" from the following perspectives:\n- {perspective_list}"
        )

    def on_information_gathering_start(self, **kwargs):
        self.status_container.info("Start browsing the Internet.")

    def on_dialogue_turn_end(self, dlg_turn, **kwargs):
        urls = list(set([r.url for r in dlg_turn.search_results]))
        for url in urls:
            self.status_container.markdown(
                f"""
                    <style>
                    .small-font {{
                        font-size: 14px;
                        margin: 0px;
                        padding: 0px;
                    }}
                    </style>
                    <div class="small-font">Finish browsing <a href="{url}" class="small-font" target="_blank">{url}</a>.</div>
                    """,
                unsafe_allow_html=True,
            )

    def on_information_gathering_end(self, **kwargs):
        self.status_container.success("Finish collecting information.")

    def on_information_organization_start(self, **kwargs):
        self.status_container.info(
            "Start organizing information into a hierarchical outline."
        )

    def on_direct_outline_generation_end(self, outline: str, **kwargs):
        self.status_container.success(
            f"Finish leveraging the internal knowledge of the large language model."
        )

    def on_outline_refinement_end(self, outline: str, **kwargs):
        self.status_container.success(f"Finish leveraging the collected information.")


def set_storm_runner():
    current_working_dir = os.path.join(get_demo_dir(), "DEMO_WORKING_DIR")
    if not os.path.exists(current_working_dir):
        os.makedirs(current_working_dir)

    lm_configs = STORMWikiLMConfigs()

    lm_configs.set_conv_simulator_lm(get_engine("openrouter/anthropic/claude-3.5-sonnet"))
    lm_configs.set_question_asker_lm(get_engine("openrouter/deepseek/deepseek-r1"))
    lm_configs.set_outline_gen_lm(get_engine("openrouter/openai/gpt-4o-2024-11-20"))
    lm_configs.set_article_gen_lm(get_engine("openrouter/openai/gpt-4o-2024-11-20"))
    lm_configs.set_article_polish_lm(get_engine("openrouter/openai/o3-mini"))
    engine_args = STORMWikiRunnerArguments(
        output_dir=current_working_dir,
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=3,
        retrieve_top_k=5,
    )

    rm = YouRM(ydc_api_key=st.secrets["YDC_API_KEY"], k=engine_args.search_top_k)

    runner = STORMWikiRunner(engine_args, lm_configs, rm)
    st.session_state["runner"] = runner