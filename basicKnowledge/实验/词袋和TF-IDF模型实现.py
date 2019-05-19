
# coding: utf-8

# # 词袋和 TF-IDF 模型介绍及实现

# ---

# ### 实验介绍

# 特征提取是将文本应用于分词前的重要步骤。本次的实验，将会介绍两种不同的特征提取方法，分别是词袋模型以及 TF-IDF 模型。这两种模型十分简单也比较容易实现。

# ### 实验知识点

# - 词袋模型
# - TF-IDF 模型
# - 两种模型的实现

# ### 目录索引

# - <a href="#词袋模型">词袋模型</a>
# - <a href="#TF-IDF-模型">TF-IDF 模型</a>
# - <a href="#实现">实现</a>

# ---

# 做文本分类等问题的时，需要从大量语料中提取特征，并将这些文本特征变换为数值特征。一般而言，特征提取有下面的两种经典方法。

# #### 词袋模型

# 词袋模型是最原始的一类特征集，忽略掉了文本的语法和语序，用一组无序的单词序列来表达一段文字或者一个文档。可以这样理解，把整个文档集的所有出现的词都丢进袋子里面，然后无序的排出来（去掉重复的）。对每一个文档，按照词语出现的次数来表示文档。

# 句子1：我/有/一个/苹果
# 
# 句子2：我/明天/去/一个/地方
# 
# 句子3：你/到/一个/地方
# 
# 句子4：我/有/我/最爱的/你

# 把所有词丢进一个袋子：我，有，一个，苹果，明天，去，地方，你，到，最爱的。这 4 句话中总共出现了这 10 个词。

# 现在我们建立一个无序列表：我，有，一个，苹果，明天，去，地方，你，到，最爱的。并根据每个句子中词语出现的次数来表示每个句子。

# ![此处输入图片的描述](https://doc.shiyanlou.com/document-uid214893labid7506timestamp1542185689722.png)

# - 句子 1 特征: ( 1 , 1 , 1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 )
# - 句子 2 特征: ( 1 , 0 , 1 , 0 , 1 , 1 , 1 , 0 , 0 , 0 )
# - 句子 3 特征: ( 0 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 0 )
# - 句子 4 特征: ( 2 , 1 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 1 )

# 这样的一种特征表示，我们就称之为词袋模型的特征。

# #### TF-IDF 模型

# 这种模型主要是用词汇的统计特征来作为特征集。TF-IDF 由两部分组成：TF（Term frequency，词频），IDF（Inverse document frequency，逆文档频率）两部分组成。

# TF 和 IDF 都很好理解，我们直接来说一下他们的计算公式。

# TF：

# $$tf_{ij} = \frac{n_{ij}}{\sum_{k}n_{kj}}$$

# 其中分子  $n_{ij}$  表示词  $i$ 在文档 $j$ 中出现的频次。分母则是所有词频次的总和，也就是所有词的个数。

# 举个例子：
# 
# 句子1：上帝/是/一个/女孩
# 
# 句子2：桌子/上/有/一个/苹果
# 
# 句子3：小明/是/老师
# 
# 句子4：我/有/我/最喜欢/的/

# 每个句子中词语的 TF ：

# ![此处输入图片的描述](https://doc.shiyanlou.com/document-uid214893labid7506timestamp1542185688511.png)

# IDF:

# $$idf_{i} = log\left ( \frac{\left | D \right |}{1+\left | D_{i} \right |} \right )$$

# 其中 $\left | D \right |$ 代表文档的总数，分母部分  $\left | D_{i} \right |$  则是代表文档集中含有 $i$ 词的文档数。原始公式是分母没有 $+1$ 的，这里 $+1$ 是采用了拉普拉斯平滑，避免了有部分新的词没有在语料库中出现而导致分母为零的情况出现。

# 用 idf 公式计算句子中每个词的 IDF 值：

# ![此处输入图片的描述](https://doc.shiyanlou.com/document-uid214893labid7506timestamp1542185689041.png)

# 最后，把 TF 和 IDF 两个值相乘就可以得到 TF-IDF 的值。即：

# $$tf*idf(i,j)=tf_{ij}*idf_{i}= \frac{n_{ij}}{\sum_{k}n_{kj}} *log\left ( \frac{\left | D \right |}{1+\left | D_{i} \right |} \right )$$

# 每个句子中，词语的 TF-IDF 值：

# ![此处输入图片的描述](https://doc.shiyanlou.com/document-uid214893labid7506timestamp1542185689364.png)

# 把每个句子中每个词的 TF-IDF 值 添加到向量表示出来就是每个句子的 TF-IDF 特征。

# 句子 1 的特征：

# $$(  0.25 * log(2) , 0.25 * log(1.33) , 0.25 * log(1.33) , 0.25 * log(2) , 0 ,0 ,0 ,0 , 0 , 0 , 0 , 0 , 0 )$$

# 同样的方法得到句子 2，3，4 的特征。

# 在 Python 当中，我们可以通过 scikit-learn 来分别实现词袋模型以及 TF-IDF 模型。并且，使用 scikit-learn 库将会非常简单。这里要用到 `CountVectorizer()` 类以及  `TfidfVectorizer()` 类。

# 看一下两个类的参数：

# ```python
# #词袋
# sklearn.featur_extraction.text.CountVectorizer(min_df=1, ngram_range=(1,1))
# ```

# - `min_df` :忽略掉词频严格低于定阈值的词
# - `ngram_range` :将 text 分成 n1,n1+1,……,n2个不同的词组。比如比如'Python is useful'中ngram_range(1,3)之后可得到 'Python' ， 'is' ， 'useful' ， 'Python is' ， 'is useful' ， 'Python is useful'。如果是ngram_range (1,1) 则只能得到单个单词'Python' ， 'is' ， 'useful'。

# ```python
# #Tf-idf
# sklearn.feature_extraction.text.TfidfVectorizer(min_df=1,norm='l2',smooth_idf=True,use_idf=True,ngram_range=(1,1)）
# ```

# - `min_df`： 忽略掉词频严格低于定阈值的词。
# - `norm` ：标准化词条向量所用的规范。
# - `smooth_idf`：添加一个平滑 idf 权重，即 idf 的分母是否使用平滑，防止0权重的出现。
# - `use_idf`： 启用 idf 逆文档频率重新加权。
# - `ngram_range`：同词袋模型

# 接下来我们开始使用这两个类，完成词袋和TF-IDF的计算。

# 加载词袋类：

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# 调整类的参数：

# In[ ]:


vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))


# 建立文本库

# In[ ]:


corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document?']
corpus


# 训练数据 corpus 获得词袋特征:

# In[ ]:


a = vectorizer.fit_transform(corpus)
a


# 根据显示结果，我们得到 4 个9 维向量组成的 4 X 9 特征矩阵。

# 我们可以获取对应的特征名称看看词是怎么排列的：

# In[ ]:


vectorizer.get_feature_names()


# 为了便于显示，我们把 a 转换成 array 类型显示：

# In[ ]:


a.toarray()


# 加载 TF-IDF 类 ：

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# 调整类的参数：

# In[ ]:


vectorizer = TfidfVectorizer(
    min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=(1, 1))


# 用 TF-IDF 类去训练上面同一个 corpus：

# In[ ]:


b = vectorizer.fit_transform(corpus)
b


# 看 b 的显示结果，我们得到由 4 个 9 维的向量组成的 4 X 9 特征矩阵。

# 跟词袋模型调用方法一样，我们也可以通过下面获取 TF-IDF 模型词排列的顺序：

# In[ ]:


vectorizer.get_feature_names()


# 为了便于观察，我们将 b 转换成 array() 类型显示。

# In[ ]:


b.toarray()


# 需要注意的是 `b` 这个特征矩阵是以稀疏矩阵的形式存在的，使用 Compressed Row Storage 格式存储，也就是这个特征矩阵的信息是被压缩了。
# 
# 为什么要这么做呢？因为在实际运用是文本数据量是非常大的，如果完整的存储会消耗大量的内存，因此会将其压缩存储。但是仍然可以像 numpy 数组一样取特征矩阵中的数据。

# 看一下 b 的类型:

# In[ ]:


type(b)


# 当不调用 `toarray()` 函数转化成 numpy 数组时，依然可以像 numpy 数组一样操作，并节省内存。

# In[ ]:


b[0, 0]


# In[ ]:


b[0, :].toarray()


# 上面就是词袋特征和 TF-IDF 特征的训练方法。实现起来很简单。

# ### 实验总结：
# 本次实验，我们介绍了词袋模型和 TF-IDF 模型。这两种模型都可以为我们提取到文本的特征，并将提取的特征用于后续的任务。因此，是文本分类任务当中非常重要的一个环节。
