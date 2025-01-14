from knowledge_storm.storm_wiki.modules.storm_dataclass import ArticleTextProcessing


def test_parse_article_into_dict():
    article_text = ""
    article_dict = ArticleTextProcessing.parse_article_into_dict(article_text)
    print(article_dict)
    assert False


def test_insert_or_create_section():
    assert False


def test_post_processing():
    assert False
