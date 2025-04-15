# AI Article Generator

## Project Overview
AI Article Generator is a full-stack application that leverages advanced AI models to generate well-researched, fact-checked articles based on user-provided topics or URLs. The backend is built with FastAPI and integrates Google's Gemini AI model via LangChain, while the frontend is a responsive static site styled with Tailwind CSS. The system includes content quality evaluation metrics and supports inclusion of code snippets and diagrams in generated articles.

## Features
- Generate detailed articles on any topic or from a URL
- Option to include code snippets and diagrams in articles
- Fact-checking and hallucination detection for content accuracy
- Content quality evaluation with metrics on factual accuracy, relevance, and toxicity
- Interactive frontend with article preview, copy, and download functionality
- Mermaid diagram rendering and syntax-highlighted code blocks
- Monitoring and instrumentation with Arize Phoenix

## Prerequisites
- Python 3.10 or higher
- Node.js and npm (for frontend Tailwind CSS build)
- Access to Google Gemini API with a valid API key

## Environment Variables
Create a `.env` file in the project root with the following variables:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## Backend Installation and Running

1. Create and activate a Python virtual environment (optional but recommended):

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI backend server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend server will be available at `http://localhost:8000`.

## Frontend Build and Usage

The frontend is a static site styled with Tailwind CSS.

1. Install Tailwind CSS dependencies:

```bash
npm install
```

2. Build the CSS:

```bash
npm run build
```

3. Open `static/index.html` in a browser or serve the `static` directory with a static file server.

The frontend interacts with the backend API at `/generate-article` to generate articles.

## API Usage

### POST /generate-article

Request body (JSON):

```json
{
  "topic": "string",
  "include_code": true,
  "include_diagrams": true,
  "is_url": false
}
```

Response:

```json
{
  "article": "Generated article content in markdown",
  "metrics": {
    "quality_scores": {
      "hallucination": 0.95,
      "relevance": 0.9,
      "toxicity": 0.05
    },
    "content_metrics": {
      "word_count": 1234,
      "reading_time_minutes": 5,
      "code_block_count": 3,
      "diagram_count": 2
    }
  }
}
```

## Project Structure

```
.
├── main.py                 # FastAPI backend application
├── requirements.txt        # Python dependencies
├── package.json            # Frontend build dependencies and scripts
├── static/                 # Frontend static files
│   ├── index.html          # Frontend UI
│   ├── src/                # Tailwind CSS source styles
│   └── dist/               # Compiled CSS output
└── .env                    # Environment variables (not included in repo)
```

## Dependencies

### Backend

- fastapi
- uvicorn
- python-dotenv
- google-generativeai
- langchain
- langchain-google-genai
- newspaper3k
- trafilatura
- beautifulsoup4
- aiohttp
- arize-phoenix
- openinference-instrumentation-langchain
- pydantic
- python-multipart
- requests
- markdown
- mermaid

### Frontend

- tailwindcss
- autoprefixer
- postcss

## License and Contribution

This project is open source. Contributions and issues are welcome.

---

For questions or support, please contact the project maintainer.
