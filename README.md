# AI Article Generator API

This is a FastAPI application that generates well-structured markdown articles using Google's Gemini AI model.

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
.\venv\Scripts\activate
```
- On Unix or MacOS:
```bash
source venv/bin/activate

```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. The API will be available at:
- Main endpoint: http://localhost:8000
- API documentation: http://localhost:8000/docs

## API Usage

### Generate Article

**Endpoint:** POST /generate-article

**Request Body:**
```json
{
    "topic": "Your topic or URL here",
    "is_url": false
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/generate-article" \
     -H "Content-Type: application/json" \
     -d '{"topic": "Artificial Intelligence", "is_url": false}'
```

The response will include:
- The generated article in markdown format
- HTML version of the article
- Status of the request

## Testing in Postman

1. Create a new POST request to: `http://localhost:8000/generate-article`
2. Set the Content-Type header to `application/json`
3. Add the request body:
```json
{
    "topic": "Your topic here",
    "is_url": false
}
```
4. Send the request to get the generated article 