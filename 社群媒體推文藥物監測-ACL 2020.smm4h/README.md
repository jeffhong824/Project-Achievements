# Medication Mention Detection in Tweets Using ELECTRA Transformers and Decision Trees

參加國際競賽 - Social Media Mining for Health Applications，目標是對極度不平衡之醫藥文本進行分類，過程中主導研究方向，並提供許多想法讓成績得以提升，我們運用DL(ELECTRA模型)結合ML對額外爬蟲資料進行篩選，以及在ML(決策樹模型)上保持(1)資料比例不變(2)改變每次訓練的資料，分別強化了ML和DL分別在不平衡資料上的缺陷，再對兩種模型預測結果進行交集，最終獲得75.78的F1-score，高於各國隊伍平均的66.46的F1-score，也發表一篇國際論文，最終排名第5，論文發表於 In Proceedings of the Fifth Social Media Mining for Health Applications Workshop & Shared Task, pages 131–133, Barcelona, Spain (Online). Association for Computational Linguistics.

提高方法效果主張為分別強化ML與DL模型的特點，並補足各自弱項，並將兩者結果進行投票。
1. DL模型(ELECTRA transformers)在極度不平衡資料上泛用性佳，經實驗後評估主要問題在資料量不足，然而要添加大量標註資料是令人沮喪的，故我們採用半監督式學習（Semi-supervised Learning）的精神，利用ML模型(SVM)訓練個Precision高的標註用模型，將Web crawler的資料做auto-labeling，再由DL模型訓練。
2. ML模型(Decision tree)經實驗在極度不平衡資料上較其他統計學的機器學習模型不受資料比例影響，擁有高度Precision的同時也具有一定程度的泛用性，而主要問題在

This study describes our proposed model design for the SMM4H 2020 Task 1. We fine-tune ELECTRA transformers using our trained SVM filter for data augmentation, along with decision trees to detect medication mentions in tweets. Our best F1-score of 0.7578 exceeded the mean score 0.6646 of all 15 submitting teams.

The Social Media Mining for Health Applications (SMM4H) shared task involves natural language processing challenges for using social media data for health research. We participated in the SMM4H 2020 Task 1, focusing on automatic classification of tweets that mention medications (Klein et al., 2020). This binary classification task involves distinguishing tweets that mention a medication or dietary supplement (annotated as ‘1’) from those that do not (annotated as ‘0’). This task was first organized in 2018 using a data set containing an artificially balanced distribution of the two classes (Weissenbacher et al., 2018). Several approaches have been presented to address this binary classification task (Coltekin and Rama, 2018; Xherija, 2018; Wu et al., 2018). However, this year’s task is more challenging. The data set represents a natural, highly imbalanced distribution of the two classes from tweets posted by 112 women during pregnancy, with only approximately 0.2% of the tweets mentioning a medication (Sarker et al., 2017; Weissenbacher et al., 2019).

This paper describes the NCUEE (National Central University, Dept. of Electrical Engineering) system for the SMM4H 2020 Task 1. To deal with highly imbalanced distribution, the support vector machine trained using the training data is used as a filter to crawl and select more tweets for data augmentation. We then fine-tune the pre-trained ELECTRA transformers (Clark et al., 2020), using our augmented data for medication mention detection. In addition, we train the decision tree as a supplementary classifier. Finally, the integrated set of testing instances are detected as a positive class from ELECTRA and decision trees are regraded as label ‘1’ and the remaining cases as label ‘0’ to form our submissions.

## ACL網站連結
![Medication Mention Detection in Tweets Using ELECTRA Transformers and Decision Trees](https://user-images.githubusercontent.com/61589737/225249542-cb8e0497-b2f4-4388-aa7e-16495150f1e2.png)
* https://aclanthology.org/2020.smm4h-1.23/


## The NCUEE System
In addition to the training data provided by task organizers, we crawl and select highly related tweets for data augmentation. We manually check small positive tweets to pick up textual terms that may refer to medications, and then use these terms as seeds for query expansions. The pre-trained Word2Vec embedding from Twitter data is used to look up word vectors and compare their cosine similarities. The top 10 similar terms of seeds are collected, where expanded terms are kept if the document frequency (DF) of an expanded term in the positive class exceeds that in the negative class. Each expanded term, along with the query term ‘pregnant’ is regarded as an individual query to search for possibly related tweets from Twitter. To automatically label highly positive cases, we train the support vector machine (SVM) using the provided training data and select crawled tweets predicted to be positive cases by the SVM. Finally, we construct an augmented data set including the original training sets for neural computing. ELECTRA (Efficiently Learning as Encoder that Classifiers Token Replacements Accurately) is a new pre-training approach that aims to match or exceed the downstream performance of an MLM (Masked Language Modeling) pre-trained model while using less computational loading (Clark et al., 2020). ELECTRA trains two transformer models: the generator, which replaces the tokens in a sequence for training a masked language model; and the discriminator, which tries to identify which tokens in the sequence were replaced by the generator. We use pre-trained ELECTRA transformers and fine-tune them using our augmented data to detect medication mentions in tweets.

According to our empirical results from the validation set, the decision tree (DT) classifiers usually achieved a high degree of precision if the discriminated features had been extracted and learned, but very low recall if the testing cases were significantly different from the trained ones. Hence, we use the same trained SVM as a filter to select the positive cases (predicted as ‘1’ ) of from the 2018 task training data that may be closely similar to the positive tweets in this task and include these in an augmented set. We then adopt the TF-IDF (Term Frequency-Inverse Document Frequency) weighting method to extract discriminated features of positive tweets from this augmented set and use them with the original training data to train the decision trees.

Finally, based on our error analysis, we found the decision tree classifier was partially complementary to ELECTRA. So, the integrated set of testing instances predicted as the positive class from the ELECTRA transformers and decision trees are labeled ‘1’, otherwise ‘0 ’ in our submissions.

## Evaluation

We picked up 112 seeds from 181 positive tweets to further expand the dataset by 72 unique terms via cosine similarity through the pre-trained Word2Vec embeddings of the tweets. Without using the SVM as a filter, we have 57,678 positive tweets and 105,273 negative tweets. With SVM, we have 32,619 positive tweets and 65,238 negative tweets. The distribution of tweets after data augmentation (DA) was still remained imbalanced, with a positive to negative ratio close to 1:2. The pre-trained ELECTRALarge was downloaded from HuggingFace (Wolf et al., 2019). The hyper-parameters used for fine-tuning ELECTRA are as follows: batch size 16; gradient accumulation steps 16; learning rate 1e-5; and number of training epochs 6.

Table 1 shows the results on the validation and test sets. The evaluation metric is the F1-score for the positive class (i.e. tweets that mention medications). For the test set, compared with submission 1 that does not use SVM as a filter for data augmentation and submission 3 that adds the prediction result of the RoBERTa transformer (Liu et al., 2019), our submission 2 achieved the best F1-score of 0.7578. Their relative ranks were identical to those of the validation set.

![image](https://user-images.githubusercontent.com/61589737/225250868-0ede687b-4271-4c6f-a7b2-2fcccf38cdc1.png)

## 專案故事

* 完成專案的目標與成果/ Achieved project goals and results: 
參與科技部計畫—與臺北榮民總醫院跨域合作開發「中風後癲癇偵測系統」。對於中風而形成癲癇病灶的情況，醫師並不建議用目前已有的癲癇樣波來分辨，原因是中風為腦部後天構造性病變，會造成部分腦區域受損，使腦電圖在中風急性期也可能出現癲癇樣波，造成判斷癲癇用的樣波較不準確，故希望能有其他指標能協助診斷中風後癲癇。故開發該系統主要是使用ML與DL技術，建立出可偵測「異常腦電圖」之模型。過程包含大量的醫學數據分析、訊號處理，及使用多種機器學習模型，其實驗證明該結果可有效改善現今偵測系統使用之狀況。

* 專案貢獻/ Project contributions: 
概念發想與驗證 Proof Of Concept --> 軟體開發 Development --> 軟體驗證 Verfication --> 軟體卻笑 Validation 

* 關鍵成果/ Key outcomes: 
新議題探索性研究-中風後癲癇訊號的大量實驗、臨床決策支援系統

* 完成時間/ Completion time: 
2020

* 專案技能/ Project Skills:
機器學習(Scikit-learn)、深度學習(Tensorflow)、訊號處理(Filter、Common spatial patterns、Wavelet Transform)、腦電圖(EEG)、中風癲癇、臨床決策支援系統(Tkinter)

* 專案角色/ Project Roles:
研究助理、碩士生、軟體開發

* 專案負責人/ Project leader:
李龍豪教授

* 專案協作者/ Project collaborator:
周建成醫師

* 團隊成員/ Team member:
無
