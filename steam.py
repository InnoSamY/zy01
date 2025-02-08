import time
from datetime import datetime
import re
import pandas as pd
from snownlp import SnowNLP
from selenium import webdriver
from parsel import Selector
import jieba
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt

# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']

# 标准化日期格式函数，将不同格式的日期字符串转换为统一格式
def standardize_date(date_str):
    # 去除日期字符串中不必要的前缀
    date_str = re.sub(r"^(发布于：|Posted: )", "", date_str.strip())
    date_formats = [
        "%m 月 %d 日",
        "%Y 年 %m 月 %d 日",
    ]
    for date_format in date_formats:
        try:
            # 尝试根据不同的格式解析日期字符串
            date = datetime.strptime(date_str, date_format)
            if "%Y" not in date_format:
                # 如果年份不在原始格式中，则使用当前年份
                date = date.replace(year=datetime.now().year)
            return date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"{date_str} format not recognized")

# 启动Selenium WebDriver，用于网页自动化操作
driver = webdriver.Chrome()

# 获取Steam应用评论函数
def get_reviews(app_id, max_scroll, language="schinese"):
    all_reviews = list()
    reviews = list()
    url = f"https://steamcommunity.com/app/{app_id}/reviews/?filterLanguage={language}"
    driver.get(url)
    for i in range(max_scroll):
        # 模拟用户滚动页面以加载更多评论
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)  # 等待页面加载
        r = driver.page_source
        s = Selector(r)
        # 检查是否加载到了新的评论
        if len(s.xpath("//div[@class='apphub_CardTextContent']")) > len(reviews):
            reviews = s.xpath("//div[@class='apphub_CardTextContent']")
        else:
            break  # 如果没有加载到新的评论则停止

    for review in reviews:
        publish_date = review.xpath("./div[@class='date_posted']/text()").get()
        content = review.xpath("./text()").extract()
        content = "".join(content).replace("\t", "").replace("\n", "")
        c = {
            "publish_date": standardize_date(publish_date),
            "content": content
        }
        all_reviews.append(c)
    return all_reviews

# 使用获取评论函数获取评论数据，并关闭WebDriver
app_id = "2358720"  # 黑神话：悟空 Steam 应用 ID
raw_reviews = get_reviews(app_id, max_scroll=50)
driver.quit()

# 动态加载停用词文件，返回一个停用词集合
def load_stopwords(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        stopwords = set(word.strip() for word in file.readlines())
    return stopwords

# 加载停用词
stopwords_path = "CNstopwords.txt"  # 停用词文件路径
custom_stopwords = load_stopwords(stopwords_path)

# 清理文本并进行分词
def clean_and_tokenize(content):
    # 移除非中文字符和多余空格
    content = re.sub(r"[^\u4e00-\u9fff]", "", content)
    words = jieba.lcut(content)  # 使用jieba库进行分词
    # 移除停用词和长度为1的单字
    return [word for word in words if word not in custom_stopwords and len(word) > 1]

# 对获取到的评论数据进行清理、分词等预处理
reviews_df = pd.DataFrame(raw_reviews)
reviews_df.dropna(subset=["content"], inplace=True)
reviews_df["content"] = reviews_df["content"].str.strip()
reviews_df["tokens"] = reviews_df["content"].apply(clean_and_tokenize)

# 统计高频词
all_tokens = [word for tokens in reviews_df["tokens"] for word in tokens]
word_counts = Counter(all_tokens)

# 输出前20个高频词
print("Top 20 most common Chinese words:", word_counts.most_common(20))

# 生成词云图
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    font_path="simhei.ttf"  # 确保提供中文字体路径
).generate_from_frequencies(word_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# 添加情感分析函数，使用SnowNLP进行情感打分
def analyze_sentiment(content):
    s = SnowNLP(content)
    sentiment_score = s.sentiments  # 获得情感值，范围在0-1之间，越接近1表示越积极
    return sentiment_score

# 对评论内容进行情感分析，并分类为正面或负面
reviews_df["sentiment"] = reviews_df["content"].apply(analyze_sentiment)
reviews_df["sentiment_label"] = reviews_df["sentiment"].apply(
    lambda x: "positive" if x >= 0.5 else "negative"
)

# 绘制情感分析结果的图表
plt.figure(figsize=(6, 4))
sentiment_counts = reviews_df["sentiment_label"].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red'])
plt.title("评论的情绪分析")
plt.xlabel("情绪")
plt.ylabel("评论数量")
plt.xticks(rotation=0)
plt.show()

# 按日期分组计算平均情感分数
reviews_df["publish_date"] = pd.to_datetime(reviews_df["publish_date"])  # 确保日期是日期格式
reviews_df["publish_date"] = reviews_df["publish_date"].dt.date  # 仅保留年月日部分

# 按日期分组并计算情感得分的平均值
sentiment_by_date = reviews_df.groupby("publish_date")["sentiment"].mean()

# 绘制随时间变化的情感趋势图
plt.figure(figsize=(10, 6))
sentiment_by_date.plot(kind='line', marker='o', color='blue')
plt.title("随时间变化的情绪趋势")
plt.xlabel("日期")
plt.ylabel("平均情绪得分")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 打印每个日期的平均情感分数
print("随时间变化的情绪分析:")
print(sentiment_by_date)