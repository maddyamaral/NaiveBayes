using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

//This program uses the Naive Bayes classifier algorithm to determine if a news article is "fake" or "true".
//The truth value of an article is determined by whether the article contains any objective lies, and perhaps how many lies it contains. An objective lie is something that can
//  be proven to be false. For example, if the article says "Donald Trump is a terrible president", this is not an objective lie or truth because different people may have different
//  opinions about it, and thus it will not contribute to the decision of whether or not the article is fake. However, if the article says "Donald Trump is an invader from Mars", 
//  then this is an objective lie because it can be proven with current knowledge that it is not true. While the classifier itself will not analyze each sentence in an article and 
//  look for lies, this is the basis upon which we will decide if the classifier is properly classifying articles.
//To train this classifier, articles were found that had already been labeled as "fake" or "true" using the same standards explained above. Different features from these articles 
//  are looked at to determine likelihood that an article, without an existing "fake" or "true" label, should be classified as "fake" or "true".

namespace NaiveBayes
{
    class Article
    {
        //This class holds the different features of an article that will be looked at to determine whether it is true or not
        //  URL: This will help determine whether the article comes from a well known, trusted source, or something that is potentially unreliable.
        //  Word Count: This will help reveal a relationship between article length and truth value. Articles that are very short or extremely long could be suspicious.
        //  Headline: This will help reveal a pattern between headline structure or content and the truth value of that article.
        //  Date Published: This will help determine whether articles from a certain time period are more likely to be fake than others (for example, the 2016 election period).
        //  Author: This will help show if some authors are more likely to write fake articles than others.
        //  Keywords: These will be political keywords that appear in the article. This will help reveal if articles containing certain words are more likely to be fake.

        public string url; //input if it was "trusted" or "unreliable"
        public string wordCount; //input if it was "short", "average", or "long"
        public string headline; //input if it was "standard" or "unusual"
        public string datePublished; //input if activity for fake news at that time was "low", "average", or "high"
        public string author; //input if the author was "trusted" or "unreliable"
        public string keywords; //input if keywords were "rare", "unindicative", or "common" for fake news articles

        //constructors
        public Article()
        {
            url = "unknown";
            wordCount = "unknown";
            headline = "unknown";
            datePublished = "unknown";
            author = "unknown";
            keywords = "unknown";
        }
        public Article(string newUrl, string newWC, string newHead, string newDate, string newAuthor, string newKey)
        {
            url = newUrl;
            wordCount = newWC;
            headline = newHead;
            datePublished = newDate;
            author = newAuthor;
            keywords = newKey;
        }
    }

    class TrainingData
    {
        //This class takes an article and adds a label to it for training purposes

        public Article articleInfo;
        public string label; //"fake" or "true"

        //constructors
        public TrainingData()
        {
            articleInfo = null;
            label = "unknown";
        }
        public TrainingData(Article newArticle, string newLabel)
        {
            articleInfo = newArticle;
            label = newLabel;
        }
    }

    class ProcessTraining
    {
        //This class gets proportions needed for Naive Bayes
        //For each feature, and each possible value for each feature, divide number of articles with that feature value by total number of fake articles, and total number of true articles
        //(number of articles with feature label A) / (number of fake articles), (number of articles with feature label A) / (number of true articles)
        //Example:
        //  (num_articles_url=trusted / num_fake_articles), (num_articles_url=trusted / num_true_articles), (num_articles_url=unreliable / num_fake_articles),
        //  (num_articles_url=unreliable / num_true_articles)

        public List<TrainingData> trainingArticles;

        //constructors
        public ProcessTraining()
        {
            trainingArticles = new List<TrainingData>();
        }
        public ProcessTraining(List<TrainingData> newData)
        {
            trainingArticles = newData;
        }

        //Separate data into "fake" and "true"
        public List<TrainingData> getFakeArticles()
        {
            List<TrainingData> fakeArticles = new List<TrainingData>();
            int numArticles = trainingArticles.Count;
            
            for(int i = 0; i < numArticles; i++)
            {
                if (trainingArticles[i].label == "false")
                    fakeArticles.Add(trainingArticles[i]);
            }

            return fakeArticles;
        }
        public List<TrainingData> getTrueArticles()
        {
            List<TrainingData> trueArticles = new List<TrainingData>();
            int numArticles = trainingArticles.Count;

            for (int i = 0; i < numArticles; i++)
            {
                if (trainingArticles[i].label == "true")
                    trueArticles.Add(trainingArticles[i]);
            }

            return trueArticles;
        }
        public double getFakeProp()
        {
            double totalCount = trainingArticles.Count;
            //avoid divide by 0
            if (totalCount == 0)
                return 0.0;

            List<TrainingData> fakeArticles = getFakeArticles();
            double fakeCount = fakeArticles.Count;

            double fakeProp = fakeCount / totalCount;

            return fakeProp;
        }
        public double getTrueProp()
        {
            double totalCount = trainingArticles.Count;
            //avoid divide by 0
            if (totalCount == 0)
                return 0.0;

            List<TrainingData> trueArticles = getTrueArticles();
            double trueCount = trueArticles.Count;

            double trueProp = trueCount / totalCount;

            return trueProp;
        }

        //For each article feature, we want the proportion of fake articles that take on a certain value of that feature, and the number of true articles with that feature value.
        //For now, we will only look at url, wordCount, and keywords

        //url - trusted or unreliable
        public double urlTrustedFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of fake articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numUrlTrustedFalse = 0;

            for(int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.url == "trusted")
                    numUrlTrustedFalse++;
            }

            prop = (double)numUrlTrustedFalse / numFake;

            return prop;
        }
        public double urlTrustedTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numUrlTrustedTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.url == "trusted")
                    numUrlTrustedTrue++;
            }

            prop = (double)numUrlTrustedTrue / numTrue;

            return prop;
        }
        public double urlUnreliableFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of fake articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numUrlUnreliableFalse = 0;

            for (int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.url == "unreliable")
                    numUrlUnreliableFalse++;
            }

            prop = (double)numUrlUnreliableFalse / numFake;

            return prop;
        }
        public double urlUnreliableTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numUrlUnreliableTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.url == "unreliable")
                    numUrlUnreliableTrue++;
            }

            prop = (double)numUrlUnreliableTrue / numTrue;

            return prop;
        }
        //wordCount - short, average, or long
        public double wcShortFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of true articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numWcShortFalse = 0;

            for (int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.wordCount == "short")
                    numWcShortFalse++;
            }

            prop = (double)numWcShortFalse / numFake;

            return prop;
        }
        public double wcShortTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numWcShortTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.wordCount == "short")
                    numWcShortTrue++;
            }

            prop = (double)numWcShortTrue / numTrue;

            return prop;
        }
        public double wcAvgFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of true articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numWcAvgFalse = 0;

            for (int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.wordCount == "average")
                    numWcAvgFalse++;
            }

            prop = (double)numWcAvgFalse / numFake;

            return prop;
        }
        public double wcAvgTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numWcAvgTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.wordCount == "average")
                    numWcAvgTrue++;
            }

            prop = (double)numWcAvgTrue / numTrue;

            return prop;
        }
        public double wcLongFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of true articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numWcLongFalse = 0;

            for (int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.wordCount == "long")
                    numWcLongFalse++;
            }

            prop = (double)numWcLongFalse / numFake;

            return prop;
        }
        public double wcLongTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numWcLongTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.wordCount == "long")
                    numWcLongTrue++;
            }

            prop = (double)numWcLongTrue / numTrue;

            return prop;
        }
        //keywords - rare, unindicative, or common
        public double keyRareFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of true articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numKeyRareFalse = 0;

            for (int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.keywords == "rare")
                    numKeyRareFalse++;
            }

            prop = (double)numKeyRareFalse / numFake;

            return prop;
        }
        public double keyRareTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numKeyRareTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.keywords == "rare")
                    numKeyRareTrue++;
            }

            prop = (double)numKeyRareTrue / numTrue;

            return prop;
        }
        public double keyUnFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of true articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numKeyUnFalse = 0;

            for (int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.keywords == "unindicative")
                    numKeyUnFalse++;
            }

            prop = (double)numKeyUnFalse / numFake;

            return prop;
        }
        public double keyUnTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numKeyUnTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.keywords == "unindicative")
                    numKeyUnTrue++;
            }

            prop = (double)numKeyUnTrue / numTrue;

            return prop;
        }
        public double keyCommonFalseProp(List<TrainingData> fakeArticles)
        {
            double prop = 0.0;
            int numFake = fakeArticles.Count; //number of true articles
            //avoid divide by 0
            if (numFake == 0)
                return 0.0;

            int numKeyCommonFalse = 0;

            for (int i = 0; i < numFake; i++)
            {
                if (fakeArticles[i].articleInfo.keywords == "common")
                    numKeyCommonFalse++;
            }

            prop = (double)numKeyCommonFalse / numFake;

            return prop;
        }
        public double keyCommonTrueProp(List<TrainingData> trueArticles)
        {
            double prop = 0.0;
            int numTrue = trueArticles.Count; //number of true articles
            //avoid divide by 0
            if (numTrue == 0)
                return 0.0;

            int numKeyCommonTrue = 0;

            for (int i = 0; i < numTrue; i++)
            {
                if (trueArticles[i].articleInfo.keywords == "common")
                    numKeyCommonTrue++;
            }

            prop = (double)numKeyCommonTrue / numTrue;

            return prop;
        }
    }

    class BayesProportions
    {
        #region MemberVariables
        //general
        public double falseArticles;
        public double trueArticles;

        //url - trusted or unreliable
        public double urlTrustedFalse;
        public double urlTrustedTrue;
        public double urlUnreliableFalse;
        public double urlUnreliableTrue;

        //wordCount - short, average, long
        public double wcShortFalse;
        public double wcShortTrue;
        public double wcAvgFalse;
        public double wcAvgTrue;
        public double wcLongFalse;
        public double wcLongTrue;

        //headline - skipped for now
        //datePublished - skipped for now
        //author - skipped for now

        //keywords - rare, unindicative, common for fake news articles
        public double keyRareFalse;
        public double keyRareTrue;
        public double keyUnFalse;
        public double keyUnTrue;
        public double keyCommonFalse;
        public double keyCommonTrue;

        public List<double> proportions;
        #endregion

        public BayesProportions()
        {
            falseArticles = 0.0;
            trueArticles = 0.0;

            urlTrustedFalse = 0.0;
            urlTrustedTrue = 0.0;
            urlUnreliableFalse = 0.0;
            urlUnreliableTrue = 0.0;

            wcShortFalse = 0.0;
            wcShortTrue = 0.0;
            wcAvgFalse = 0.0;
            wcAvgTrue = 0.0;
            wcLongFalse = 0.0;
            wcLongTrue = 0.0;

            keyRareFalse = 0.0;
            keyRareTrue = 0.0;
            keyUnFalse = 0.0;
            keyUnTrue = 0.0;
            keyCommonFalse = 0.0;
            keyCommonTrue = 0.0;
        }

        public BayesProportions(double fa, double ta, double utf, double utt, double uuf, double uut, double wsf, double wst, double waf, double wat,
            double wlf, double wlt, double krf, double krt, double kuf, double kut, double kcf, double kct)
        {
            falseArticles = fa;
            trueArticles = ta;

            urlTrustedFalse = utf;
            urlTrustedTrue = utt;
            urlUnreliableFalse = uuf;
            urlUnreliableTrue = uut;

            wcShortFalse = wsf;
            wcShortTrue = wst;
            wcAvgFalse = waf;
            wcAvgTrue = wat;
            wcLongFalse = wlf;
            wcLongTrue = wlt;

            keyRareFalse = krf;
            keyRareTrue = krt;
            keyUnFalse = kuf;
            keyUnTrue = kut;
            keyCommonFalse = kcf;
            keyCommonTrue = kct;
        }

        public double partialProb(double articles, double urlValue, double wordCountValue, double keywordValue)
        {
            double pp = 0.0;

            //partial probability multiplies all the probabilities together that relate to one class
            pp = articles * urlValue * wordCountValue * keywordValue;

            return pp;
        }

        public double classifyArticle(double fakeArticles, double urlValueFake, double wordCountValueFake, double keywordValueFake, 
            double trueArticles, double urlValueTrue, double wordCountValueTrue, double keywordValueTrue)
        {
            double likelihoodFalse = 0.0;
            double ppFalse = partialProb(fakeArticles, urlValueFake, wordCountValueFake, keywordValueFake);
            double ppTrue = partialProb(trueArticles, urlValueTrue, wordCountValueTrue, keywordValueTrue);

            //account for special cases when it's 100% probable that it's fake or 100% probable that it's true
            if (ppFalse == 0)
                return 1.0;
            if (ppTrue == 0)
                return 0.0;

            likelihoodFalse = (double)ppFalse / (ppFalse * ppTrue);

            return likelihoodFalse;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Article article1 = new Article("trusted", "average", "unknown", "unknown", "unknown", "rare");
            Article article2 = new Article("trusted", "long", "unknown", "unknown", "unknown", "common");
            Article article3 = new Article("unreliable", "long", "unknown", "unknown", "unknown", "common");

            TrainingData td1 = new TrainingData(article1, "true");
            TrainingData td2 = new TrainingData(article2, "false");
            TrainingData td3 = new TrainingData(article3, "false");

            List<TrainingData> articles = new List<TrainingData>();
            articles.Add(td1);
            articles.Add(td2);
            articles.Add(td3);
            ProcessTraining info = new ProcessTraining(articles);

            //use ProcessTraining info object to get proportions - percentage of all articles that take on a certain feature value and belong in a certain class
            #region Get Proportions
            List<TrainingData> fakeArticles = info.getFakeArticles();
            List<TrainingData> trueArticles = info.getTrueArticles();
            double percentFake = info.getFakeProp();
            double percentTrue = info.getTrueProp();

            double percentUrlTrustedFalse = info.urlTrustedFalseProp(fakeArticles);
            double percentUrlTrustedTrue = info.urlTrustedTrueProp(trueArticles);
            double percentUrlUnreliableFalse = info.urlUnreliableFalseProp(fakeArticles);
            double percentUrlUnreliableTrue = info.urlUnreliableTrueProp(trueArticles);

            double percentWCShortFalse = info.wcShortFalseProp(fakeArticles);
            double percentWCShortTrue = info.wcShortTrueProp(trueArticles);
            double percentWCAvgFalse = info.wcAvgFalseProp(fakeArticles);
            double percentWCAvgTrue = info.wcAvgTrueProp(trueArticles);
            double percentWCLongFalse = info.wcLongFalseProp(fakeArticles);
            double percentWCLongTrue = info.wcLongTrueProp(trueArticles);

            double percentKeyRareFalse = info.keyRareFalseProp(fakeArticles);
            double percentKeyRareTrue = info.keyRareTrueProp(trueArticles);
            double percentKeyUnFalse = info.keyUnFalseProp(fakeArticles);
            double percentKeyUnTrue = info.keyUnTrueProp(trueArticles);
            double percentKeyCommonFalse = info.keyCommonFalseProp(fakeArticles);
            double percentKeyCommonTrue = info.keyCommonTrueProp(trueArticles);
            #endregion 

            BayesProportions trainingProportions = new BayesProportions(percentFake, percentTrue, percentUrlTrustedFalse, percentUrlTrustedTrue, percentUrlUnreliableFalse,
                percentUrlUnreliableTrue, percentWCShortFalse, percentWCShortTrue, percentWCAvgFalse, percentWCAvgTrue, percentWCLongFalse, percentWCLongTrue, percentKeyRareFalse,
                percentKeyRareTrue, percentKeyUnFalse, percentKeyUnTrue, percentKeyCommonFalse, percentKeyCommonTrue);

            //Now we have established the proportions for our training data
            //Use relevant proportions for a new test article to see the likelihood that the article is false, based on the training data
            //For testing purposes, the new test article will be the same as article 1 above, which was classified "true", so we can expect that
            //  the test article will be classified "true" as well
            Article testArticle = new Article("trusted", "long", "unknown", "unknown", "unknown", "common");
            string testArticleClassification = "unknown";

            //probability testArticle is false given url="trusted", wordCount="average", and keywords="rare" (all other features are ignored for now)
            double prob = trainingProportions.classifyArticle(percentFake, percentUrlTrustedFalse, percentWCAvgFalse, percentKeyRareFalse,
                percentTrue, percentUrlTrustedTrue, percentWCAvgTrue, percentKeyRareTrue);
            Console.WriteLine("Probability testArticle is fake: " + prob);

            if (prob < 0.5)
                testArticleClassification = "true";
            else
                testArticleClassification = "false";

            Console.WriteLine("Classification of new article: " + testArticleClassification);
        }
    }
}
