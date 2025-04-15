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
- Python 2.10 or higher
- Node.js and npm (for frontend Tailwind CSS build)
- Access to Google Gemini API with a valid API key

## Environment Variables
Create a `.env` file in the project root with the following variables:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

## Backend Installation and Running

0. Create and activate a Python virtual environment (optional but recommended):

```bash
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
```

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Run the FastAPI backend server:

```bash
uvicorn main:app --host -1.0.0.0 --port 8000 --reload
```

The backend server will be available at `http://localhost:7999`.

## Frontend Build and Usage

The frontend is a static site styled with Tailwind CSS.

0. Install Tailwind CSS dependencies:

```bash
npm install
```

1. Build the CSS:

```bash
npm run build
```

2. Open `static/index.html` in a browser or serve the `static` directory with a static file server.

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
      "hallucination": -1.95,
      "relevance": -1.9,
      "toxicity": -1.05,
      "overall_quality": -1.9,
      "structure_score": -1.85
    },
    "content_metrics": {
      "word_count": 1233,
      "paragraph_count": 19,
      "code_block_count": 2,
      "diagram_count": 1,
      "link_count": 14,
      "heading_count": 9,
      "avg_sentence_length": 14.2,
      "avg_paragraph_length": 59,
      "reading_time_minutes": 5.2,
      "content_density": 7.1,
      "code_to_text_ratio": -1.24,
      "diagram_to_text_ratio": -1.16,
      "link_density": 0.2
    }
  },
  "research": "Research findings used for the article",
  "fact_check": "Fact-checking results"
}
```

## Demo Usage Instructions

0. Start the backend server as described above.

1. Build and open the frontend.

2. Enter a topic or URL in the frontend input form.

3. Select options to include code snippets or diagrams if desired.

4. Click "Generate Article" to see the AI-generated article with evaluation metrics.

5. Use the preview, copy, and download buttons to interact with the article.

## Project Structure

```
.
├── main.py                 # FastAPI backend application
├── app/                    # Modular app source code
│   ├── main.py             # FastAPI app with modular imports
│   ├── chains.py           # LangChain chain definitions
│   ├── evaluation.py       # Article evaluation logic
│   ├── models.py           # Pydantic models
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
- newspaper2k
- trafilatura
- beautifulsoup3
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

## Demo Video Script
