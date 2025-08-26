import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import time
from urllib.parse import urljoin, urlparse
import openai
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class NewsTracker:
    def __init__(self):
        self.stop_words = set(stopwords.words('english')) if stopwords else set()
        self.sia = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        
    def scrape_google_news(self, query="", country="US", language="en", max_articles=50):
        """Scrape Google News for articles"""
        try:
            # Google News RSS feed URL
            base_url = "https://news.google.com/rss"
            
            # Build URL with parameters
            params = {
                'hl': language,
                'gl': country,
                'ceid': f"{country}:{language}"
            }
            
            if query:
                params['q'] = query
                
            # Make request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse RSS feed
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            articles = []
            for item in items[:max_articles]:
                try:
                    title = item.title.text if item.title else "No Title"
                    link = item.link.text if item.link else ""
                    pub_date = item.pubDate.text if item.pubDate else ""
                    description = item.description.text if item.description else ""
                    source = item.source.text if item.source else "Unknown"
                    
                    # Clean description (remove HTML tags)
                    description = BeautifulSoup(description, 'html.parser').get_text()
                    
                    articles.append({
                        'title': title,
                        'link': link,
                        'pub_date': pub_date,
                        'description': description,
                        'source': source,
                        'scraped_at': datetime.now()
                    })
                except Exception as e:
                    continue
                    
            return articles
            
        except Exception as e:
            st.error(f"Error scraping news: {str(e)}")
            return []
    
    def extract_keywords(self, text, top_n=10):
        """Extract keywords from text"""
        try:
            # Clean text
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            words = word_tokenize(text)
            
            # Filter out stop words and short words
            keywords = [word for word in words 
                       if word not in self.stop_words and len(word) > 3]
            
            # Count frequency
            word_freq = Counter(keywords)
            return word_freq.most_common(top_n)
            
        except Exception as e:
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        try:
            if self.sia:
                scores = self.sia.polarity_scores(text)
                return scores
            else:
                # Fallback to TextBlob
                blob = TextBlob(text)
                return {
                    'compound': blob.sentiment.polarity,
                    'pos': max(0, blob.sentiment.polarity),
                    'neu': 1 - abs(blob.sentiment.polarity),
                    'neg': max(0, -blob.sentiment.polarity)
                }
        except:
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
    
    def summarize_with_openai(self, text, api_key, max_length=150):
        """Summarize text using OpenAI API"""
        try:
            openai.api_key = api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a news summarization expert. Provide concise, factual summaries."},
                    {"role": "user", "content": f"Summarize this news content in {max_length} words or less:\n\n{text}"}
                ],
                max_tokens=max_length * 2,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Summary unavailable: {str(e)}"
    
    def calculate_trending_score(self, articles):
        """Calculate trending score based on article frequency, recency, and sentiment"""
        if not articles:
            return 0
            
        # Factors for trending score
        article_count = len(articles)
        
        # Recency factor (more recent = higher score)
        now = datetime.now()
        recency_scores = []
        
        for article in articles:
            try:
                pub_date = datetime.strptime(article['pub_date'], '%a, %d %b %Y %H:%M:%S %Z')
                hours_ago = (now - pub_date).total_seconds() / 3600
                recency_score = max(0, 100 - hours_ago * 2)  # Decay over time
                recency_scores.append(recency_score)
            except:
                recency_scores.append(50)  # Default score
        
        avg_recency = np.mean(recency_scores) if recency_scores else 50
        
        # Sentiment factor
        sentiments = []
        for article in articles:
            sentiment = self.analyze_sentiment(article['title'] + ' ' + article['description'])
            sentiments.append(abs(sentiment['compound']))  # Absolute sentiment intensity
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.5
        
        # Calculate final trending score
        trending_score = min(100, (article_count * 5) + avg_recency + (avg_sentiment * 20))
        
        return round(trending_score, 1)

def main():
    st.set_page_config(
        page_title="Global News Topic Tracker",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .trending-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .news-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .news-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Global News Topic Tracker</h1>
        <p>AI-Powered Real-Time News Analysis & Trending Topics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize tracker
    tracker = NewsTracker()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Configuration
    st.sidebar.subheader("🤖 AI Configuration")
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key (Optional)", 
        type="password",
        help="Enter your OpenAI API key for advanced summarization"
    )
    
    # News Configuration
    st.sidebar.subheader("📰 News Settings")
    country = st.sidebar.selectbox(
        "Country/Region",
        ["US", "GB", "CA", "AU", "IN", "DE", "FR", "JP", "BR"],
        help="Select country for regional news"
    )
    
    language = st.sidebar.selectbox(
        "Language",
        ["en", "es", "fr", "de", "it", "pt", "ja"],
        help="Select language for news content"
    )
    
    max_articles = st.sidebar.slider(
        "Max Articles per Topic",
        min_value=10,
        max_value=100,
        value=30,
        help="Number of articles to fetch per topic"
    )
    
    # Search topics
    st.sidebar.subheader("🔍 Topics to Track")
    default_topics = [
        "Artificial Intelligence",
        "Climate Change", 
        "Cryptocurrency",
        "Space Exploration",
        "Global Health",
        "Technology Innovation",
        "Economic News",
        "Political Developments"
    ]
    
    selected_topics = st.sidebar.multiselect(
        "Select Topics",
        default_topics,
        default=default_topics[:5],
        help="Choose topics to track and analyze"
    )
    
    custom_topic = st.sidebar.text_input(
        "Custom Topic",
        placeholder="Enter custom search term"
    )
    
    if custom_topic:
        selected_topics.append(custom_topic)
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh option (disabled by default to prevent multiple tabs)
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (5 min)", value=False)
    
    if auto_refresh:
        # Auto-refresh every 5 minutes with session state control
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 300:  # 5 minutes
            st.session_state.last_refresh = current_time
            st.rerun()
    
    # Main content
    if not selected_topics:
        st.warning("Please select at least one topic to track.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔥 Trending Topics", "📰 Latest News", "📈 Analytics"])
    
    with tab1:
        st.header("📊 News Dashboard")
        
        # Fetch data for all topics
        with st.spinner("Fetching latest news data..."):
            all_articles = []
            topic_data = {}
            
            for topic in selected_topics:
                articles = tracker.scrape_google_news(
                    query=topic,
                    country=country,
                    language=language,
                    max_articles=max_articles
                )
                
                if articles:
                    trending_score = tracker.calculate_trending_score(articles)
                    topic_data[topic] = {
                        'articles': articles,
                        'count': len(articles),
                        'trending_score': trending_score
                    }
                    all_articles.extend(articles)
        
        if not all_articles:
            st.error("No articles found. Please check your internet connection or try different topics.")
            return
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📰 Total Articles</h3>
                <h2>{len(all_articles)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📋 Topics Tracked</h3>
                <h2>{len(selected_topics)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_sentiment = np.mean([
                tracker.analyze_sentiment(article['title'])['compound'] 
                for article in all_articles
            ])
            sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😔"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>😊 Avg Sentiment</h3>
                <h2>{sentiment_emoji} {avg_sentiment:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🕒 Last Updated</h3>
                <h2>{datetime.now().strftime('%H:%M')}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.header("🔥 Trending Topics")
        
        if topic_data:
            # Sort topics by trending score
            sorted_topics = sorted(
                topic_data.items(), 
                key=lambda x: x[1]['trending_score'], 
                reverse=True
            )
            
            for i, (topic, data) in enumerate(sorted_topics):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="trending-item">
                        <h3>#{i+1} {topic}</h3>
                        <p><strong>Articles:</strong> {data['count']} | <strong>Trending Score:</strong> {data['trending_score']}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Trending score gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = data['trending_score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Trending"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "orange"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_{topic}_{i}")
                
                # Show sample articles
                if st.expander(f"📰 Sample Articles for {topic}"):
                    for article in data['articles'][:3]:
                        st.markdown(f"""
                        <div class="news-card">
                            <h4>{article['title']}</h4>
                            <p><strong>Source:</strong> {article['source']} | <strong>Date:</strong> {article['pub_date']}</p>
                            <p>{article['description'][:200]}...</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI Summary if API key provided
                        if openai_api_key and st.button(f"🤖 AI Summary", key=f"summary_{article['title'][:20]}"):
                            with st.spinner("Generating AI summary..."):
                                summary = tracker.summarize_with_openai(
                                    article['title'] + '\n' + article['description'],
                                    openai_api_key
                                )
                                st.success(f"**AI Summary:** {summary}")
    
    with tab3:
        st.header("📰 Latest News")
        
        # News filtering
        col1, col2 = st.columns([2, 1])
        with col1:
            search_filter = st.text_input("🔍 Filter news by keyword")
        with col2:
            sort_by = st.selectbox("Sort by", ["Most Recent", "Relevance", "Sentiment"])
        
        # Display news
        filtered_articles = all_articles
        if search_filter:
            filtered_articles = [
                article for article in all_articles 
                if search_filter.lower() in article['title'].lower() or 
                   search_filter.lower() in article['description'].lower()
            ]
        
        st.write(f"Showing {len(filtered_articles)} articles")
        
        for article in filtered_articles[:20]:  # Show top 20
            sentiment = tracker.analyze_sentiment(article['title'] + ' ' + article['description'])
            sentiment_color = "green" if sentiment['compound'] > 0.1 else "red" if sentiment['compound'] < -0.1 else "gray"
            
            st.markdown(f"""
            <div class="news-card">
                <h4>{article['title']}</h4>
                <p><strong>Source:</strong> {article['source']} | 
                   <strong>Date:</strong> {article['pub_date']} | 
                   <span style="color: {sentiment_color}"><strong>Sentiment:</strong> {sentiment['compound']:.2f}</span></p>
                <p>{article['description']}</p>
                <p><a href="{article['link']}" target="_blank">🔗 Read Full Article</a></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("📈 Analytics & Insights")
        
        if all_articles:
            # Sentiment analysis over time
            st.subheader("📊 Sentiment Analysis")
            
            # Prepare sentiment data
            sentiment_data = []
            for article in all_articles:
                sentiment = tracker.analyze_sentiment(article['title'] + ' ' + article['description'])
                sentiment_data.append({
                    'title': article['title'][:50] + '...',
                    'sentiment': sentiment['compound'],
                    'positive': sentiment['pos'],
                    'negative': sentiment['neg'],
                    'neutral': sentiment['neu']
                })
            
            df_sentiment = pd.DataFrame(sentiment_data)
            
            # Sentiment distribution
            fig = px.histogram(
                df_sentiment, 
                x='sentiment', 
                nbins=20,
                title="Sentiment Distribution",
                labels={'sentiment': 'Sentiment Score', 'count': 'Number of Articles'}
            )
            st.plotly_chart(fig, use_container_width=True, key="sentiment_histogram")
            
            # Topic comparison
            st.subheader("📊 Topic Comparison")
            
            if topic_data:
                comparison_data = [
                    {
                        'topic': topic,
                        'articles': data['count'],
                        'trending_score': data['trending_score']
                    }
                    for topic, data in topic_data.items()
                ]
                
                df_comparison = pd.DataFrame(comparison_data)
                
                fig = px.scatter(
                    df_comparison,
                    x='articles',
                    y='trending_score',
                    size='articles',
                    hover_name='topic',
                    title="Topic Performance: Articles vs Trending Score"
                )
                st.plotly_chart(fig, use_container_width=True, key="topic_comparison_scatter")
            
            # Keyword analysis
            st.subheader("🔤 Top Keywords")
            
            all_text = ' '.join([
                article['title'] + ' ' + article['description'] 
                for article in all_articles
            ])
            
            keywords = tracker.extract_keywords(all_text, top_n=20)
            
            if keywords:
                keyword_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
                
                fig = px.bar(
                    keyword_df.head(10),
                    x='Frequency',
                    y='Keyword',
                    orientation='h',
                    title="Top 10 Keywords in News"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key="keywords_bar_chart")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        🌍 Global News Topic Tracker | Powered by AI & Real-time Data | 
        Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()