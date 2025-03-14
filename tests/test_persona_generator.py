import dspy

from knowledge_storm.storm_wiki.modules.persona_generator import (
    FindRelatedTopic,
    GenPersona,
    get_wiki_page_title_and_toc,
    CreateWriterWithPersona,
)
from tests.helper import get_engine


def test_find_related_topics(topic, engine):
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
    title, toc = get_wiki_page_title_and_toc("https://en.wikipedia.org/wiki/Bitcoin")
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
    assert "Economics" in toc


def test_gen_persona(topic, engine):
    with dspy.settings.context(lm=engine):
        gen_persona = dspy.ChainOfThought(GenPersona)
        examples = """
Title: 产品xxx
Table of Contents:

1. APP（移动应用）
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 产品使用指标：
     - 使用频率
     - 平均使用时长
   - 收入相关指标：
     - 平均每用户收入（ARPU）
     - 内购转化率

2. 网站
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 产品使用指标：
     - 页面浏览量（PV）
     - 访问时长
     - 跳出率
     - 站内搜索次数
   - 收入相关指标：
     - 平均每用户收入（ARPU）
     - 广告点击率
     - 广告曝光量

3. 电商（电子商务平台）
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 购买相关指标：
     - 客单价
     - 购物车放弃率
     - 转化率
     - 回购率
   - 收入相关指标：
     - 平均每用户收入（ARPU）
     - 净利润率
     - 每笔订单平均收入

4. SaaS（软件即服务）
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 产品使用指标：
     - 使用次数
     - 每次会话的平均使用时间
     - 功能使用率
   - 收入相关指标：
     - 每月经常性收入（MRR）
     - 年经常性收入（ARR）
     - 平均每用户收入（ARPU）
     - 用户流失率

5. 游戏应用
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 产品使用指标：
     - 平均游戏时长
     - 平均每日会话数
     - 游戏成就完成率
   - 收入相关指标：
     - 平均每用户收入（ARPU）
     - 内购转化率
     - 广告收入

6. 社交媒体平台
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 产品使用指标：
     - 帖子数量
     - 帖子参与度（评论、点赞、分享）
     - 平均会话时长
   - 收入相关指标：
     - 广告收入
     - 付费会员收入

7. 内容平台（如新闻网站、博客）
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 产品使用指标：
     - 页面浏览量（PV）
     - 访问时长
     - 跳出率
     - 文章阅读完成率
   - 收入相关指标：
     - 广告收入
     - 订阅收入

8. 物联网设备（IoT）
   - 用户相关指标：
     - 设备注册数
     - 活跃设备数
     - 日活跃设备（DAU）
     - 月活跃设备（MAU）
     - 设备留存率
   - 产品使用指标：
     - 设备使用频率
     - 平均使用时长
     - 功能使用率
   - 收入相关指标：
     - 设备销售收入
     - 订阅服务收入

9. 健康与健身应用
   - 用户相关指标：
     - 注册用户数
     - 活跃用户数
     - 日活跃用户（DAU）
     - 月活跃用户（MAU）
     - 用户留存率
     - 用户增长率
   - 产品使用指标：
     - 平均每日步数
     - 每次锻炼的平均时长
     - 锻炼频率
   - 收入相关指标：
     - 订阅收入
     - 健身课程收入

10. 教育平台
    - 用户相关指标：
      - 注册用户数
      - 活跃用户数
      - 日活跃用户（DAU）
      - 月活跃用户（MAU）
      - 用户留存率
      - 用户增长率
    - 产品使用指标：
      - 课程完成率
      - 平均学习时长
      - 互动频率（提问、讨论）
    - 收入相关指标：
      - 课程收入
      - 订阅收入
        """
        response = gen_persona(topic=topic, examples=examples)
        print(f"# {engine.model}")
        print(response.personas)
# openrouter/google/gemini-2.0-flash-001
# 1.  **Financial Analyst specializing in Corporate Finance:** Focuses on analyzing MSTR's balance sheets, cash flow statements, and profitability ratios to determine the impact of Bitcoin investments on the company's financial health. Also examines the volatility and risk associated with holding Bitcoin as a corporate asset.
# 2.  **Former MSTR Executive:** Provides insights into the internal discussions and decision-making processes behind the Bitcoin investment strategy, including the rationale, risk assessments, and goals.
# 3.  **Bitcoin Advocate and Investor:** Contributes a perspective on how MSTR's investment validates Bitcoin as a viable asset and store of value, while also covering the impact on Bitcoin's market sentiment and price.
# 4.  **Financial Analyst with a Contrarian View:** Focuses on the potential downsides and risks of MSTR's Bitcoin strategy, including the impact on stock valuation, shareholder concerns, and the company's core business operations. This includes analysis of potential for margin calls on Bitcoin-backed loans.
# 5.  **Securities Law Expert:** Analyzes the legal and regulatory implications of MSTR's Bitcoin investment, including disclosure requirements, potential securities law issues, and interactions with regulatory bodies like the SEC. This includes the impact on reporting and accounting standards.
# 6.  **Data Scientist:** Analyzes market data related to MSTR stock price and Bitcoin price to reveal the correlation and causation in this connection.
# 7.  **Competitor Analyst:** Focuses on how Microstrategy's competitors have reacted to their Bitcoin investment. This includes whether they adopted similar strategies, and the overall effect on the market.
# 8.  **Academic Researcher specializing in Cryptocurrencies:** Provides an objective academic perspective on the theoretical and empirical aspects of corporate Bitcoin adoption, referencing relevant research and academic literature.


def test_create_writer_with_persona(topic, engine):
    create_writer_with_persona = CreateWriterWithPersona(engine=engine)
    personas = create_writer_with_persona(topic=topic)
    print(f"# {engine.model}")
    for persona in personas.personas:
        # ["Bitcoin Historian: This editor will focus on the history section of the article, providing details on the creation of Bitcoin, key milestones in its growth, regulatory actions, and its current status."]
        print(persona)


if __name__ == "__main__":
    _topic = "Deep Research of MSTR's Bitcoin Investment, the research must be based on data throughout the research process and conclusions"
    # _topic = "deep research on OpenRouter as a LLM routing platform, focusing on the key reasons why users choose it over alternatives"
    models = [
        # "openrouter/deepseek/deepseek-r1",
        # "openrouter/openai/gpt-4o-2024-11-20",
        # "openrouter/claude-3.7-sonnet",
        # "openrouter/claude-3.7-sonnet:thinking",
        "openrouter/google/gemini-2.0-flash-001",
    ]
    for model in models:
        _engine = get_engine(model)
        # test_find_related_topics(_topic, _engine)
        test_gen_persona(_topic, _engine)
        # test_create_writer_with_persona(_topic, _engine)
