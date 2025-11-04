# 🔍 Reddit AI Insights Tool

An AI-powered tool to analyze Reddit discussions with sentiment classification, topic analysis, keyword extraction, and entity recognition.

## Features

- 💭 **Sentiment Analysis**: Classify posts as Positive, Neutral, Negative, or Mixed
- 📊 **Topic Classification**: Categorize discussions into predefined SEO topics
- 🔑 **Keyword Extraction**: Identify key phrases using RAKE algorithm
- 🏢 **Entity Recognition**: Extract people, organizations, and products using spaCy
- 📈 **Visualizations**: Interactive word clouds and distribution charts
- 📥 **Export**: Download results as CSV

## Prerequisites

You'll need API keys from:

1. **Reddit API**
   - Go to https://www.reddit.com/prefs/apps/
   - Click "create another app..."
   - Select "script" type
   - Note your `client_id` and `client_secret`

2. **OpenAI API**
   - Go to https://platform.openai.com/api-keys
   - Create a new API key
   - Note your API key (starts with `sk-`)

## Deploy to Streamlit Cloud (Free)

### Step 1: Fork/Clone this Repository

If you haven't already, fork this repository to your GitHub account.

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - **Repository**: Your forked repository
   - **Branch**: `main` (or your branch)
   - **Main file path**: `streamlit_app.py`
5. Click "Deploy"

### Step 3: Configure Secrets

After deployment starts (or before), configure your secrets:

1. In your Streamlit Cloud dashboard, click on your app
2. Go to **Settings** → **Secrets**
3. Add the following in TOML format:

```toml
REDDIT_CLIENT_ID = "your_reddit_client_id"
REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
REDDIT_USER_AGENT = "reddit-insights-streamlit/1.0 by u/yourusername"
OPENAI_API_KEY = "sk-proj-your_openai_api_key"
```

4. Click "Save"

Your app will automatically restart with the secrets configured!

## Run Locally

### Installation

```bash
# Clone the repository
git clone https://github.com/SEOptimize-LLC/Reddit-AI-Insights-Tool.git
cd Reddit-AI-Insights-Tool

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (if not auto-downloaded)
python -m spacy download en_core_web_sm
```

### Configure Local Secrets

Create a `.streamlit/secrets.toml` file:

```toml
REDDIT_CLIENT_ID = "your_reddit_client_id"
REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
REDDIT_USER_AGENT = "reddit-insights-streamlit/1.0"
OPENAI_API_KEY = "your_openai_api_key"
```

**Important**: Never commit `secrets.toml` to Git!

### Run the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Configure Settings** (in sidebar):
   - Enter subreddit name (without "r/")
   - Set number of threads and comments to analyze
   - Choose listing type (hot/new/top)
   - Select OpenAI model and reasoning level
   - Set number of keywords to extract

2. **Run Analysis**:
   - Click "Run Analysis" button
   - Wait for data fetching and AI analysis to complete

3. **View Results**:
   - Sentiment distribution chart
   - Topic classification results
   - Keyword word cloud
   - Entity word cloud
   - Data preview table

4. **Export**:
   - Download full results as CSV

## Configuration Options

### Reddit Settings
- **Subreddit**: Any public subreddit name
- **Number of Threads**: 1-50 threads to analyze
- **Number of Comments**: 5-200 comments across all threads
- **Listing Type**: hot, new, or top posts

### OpenAI Settings
- **Model**: gpt-4o, gpt-4o-mini, gpt-5, gpt-5-mini, gpt-5-nano
- **Reasoning Level**: minimal, low, medium, high (affects token usage)

### Analysis Settings
- **Keywords to Extract**: 20-200 keywords for word cloud

## Topics

The tool classifies discussions into the following SEO topics:

1. **Link Building**: Backlinks, outreach, digital PR
2. **Technical SEO**: Site infrastructure, crawling, indexation
3. **Local SEO**: Google Business Profile, local citations
4. **Tools**: SEO platforms, rank trackers, audit tools
5. **Keyword Research**: Search volume, difficulty, intent
6. **On-page SEO**: Content optimization, meta tags, structure
7. **AI and SEO**: Generative AI impact on SEO workflows

Posts that don't fit any category are labeled as "Other".

## Streamlit Cloud Limitations

**Free Tier Constraints:**
- **Memory**: 1GB RAM limit
- **Recommendation**: Keep analysis to 5-20 threads and 15-50 comments for optimal performance
- Large analyses may timeout or fail due to memory constraints

## Files Structure

```
Reddit-AI-Insights-Tool/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .streamlit/
│   ├── config.toml              # Streamlit UI configuration
│   └── secrets.toml.example     # Example secrets file
├── Lightning_Lesson_...ipynb    # Original Google Colab notebook
└── README.md                    # This file
```

## Troubleshooting

### Reddit 404 Error ("Subreddit not found")
This is the most common error. If you see a 404 error:

**Check your Reddit API app type:**
1. Go to https://www.reddit.com/prefs/apps/
2. Click on your app
3. Make sure it's set to **"script"** type (not "web app" or "installed app")
4. If it's the wrong type, create a new app with the correct type

**Verify your credentials:**
- The **client_id** is the string under your app name (looks like: `AbCdEf123456`)
- The **client_secret** is labeled "secret" (longer string)
- Copy these exactly with NO extra spaces or quotes
- Update your Streamlit secrets with the correct values

**Check the subreddit name:**
- Enter just the name (e.g., "SEO" not "r/SEO")
- Make sure the subreddit exists and is not private
- Try a popular subreddit first (e.g., "python", "AskReddit")

### App won't start
- Check that all secrets are properly configured
- Verify API keys are valid and have proper permissions
- Check Streamlit Cloud logs for specific errors

### spaCy model download fails
- The app automatically downloads the model on first run
- If it fails, redeploy the app or check the logs

### API Rate Limits
- Reddit API: 60 requests per minute
- OpenAI API: Depends on your tier
- Use smaller batch sizes if hitting limits

### Memory errors
- Reduce number of threads/comments
- Use smaller OpenAI models (gpt-4o-mini instead of gpt-5)
- Set reasoning level to "minimal" or "low"

## Cost Considerations

- **Reddit API**: Free
- **OpenAI API**: Costs vary by model and reasoning level
  - gpt-4o-mini: Most cost-effective
  - gpt-5 with high reasoning: Most expensive
  - Typical analysis (5 threads, 15 comments): $0.01-0.10 depending on model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review Streamlit Cloud logs for deployment issues

## Credits

Built with:
- [Streamlit](https://streamlit.io) - Web framework
- [OpenAI API](https://openai.com) - AI analysis
- [PRAW](https://praw.readthedocs.io) - Reddit API wrapper
- [spaCy](https://spacy.io) - Entity recognition
- [RAKE](https://pypi.org/project/rake-nltk/) - Keyword extraction
