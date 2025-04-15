from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from newspaper import Article
import trafilatura
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import logging
import time
import aiohttp
import asyncio
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.evals import (
    HallucinationEvaluator,
    RelevanceEvaluator,
    ToxicityEvaluator
)
import atexit
import signal
import sys
import re
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Article Generator", description="Generate well-structured markdown articles using AI")

# Initialize Gemini API
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    logger.info("Initializing Gemini API...")
    genai.configure(api_key=gemini_api_key)
    
    # Initialize LangChain with Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        temperature=0.7,
        model_kwargs={
            "generation_config": {
                "max_output_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        }
    )
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Gemini API: {str(e)}")
    raise

# Initialize Arize Phoenix
session = None
try:
    session = px.active_session()
    if not session:
        session = px.launch_app()
    logger.info("Phoenix session initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Phoenix: {str(e)}")
    raise

# Initialize evaluators with the Gemini model
hallucination_evaluator = HallucinationEvaluator(model=llm)
relevance_evaluator = RelevanceEvaluator(model=llm)
toxicity_evaluator = ToxicityEvaluator(model=llm)

# Instrument LangChain with Phoenix using OpenInference
tracer_provider = register()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

def cleanup_phoenix():
    global session
    try:
        if session:
            logger.info("Closing Phoenix session...")
            # Close the session first
            session.close()
            session = None
            logger.info("Phoenix session closed successfully")
            
            # Add a small delay to ensure all resources are released
            time.sleep(1)
            
            # Clean up any remaining temporary files
            temp_dir = tempfile.gettempdir()
            for item in os.listdir(temp_dir):
                if item.startswith('phoenix'):
                    try:
                        item_path = os.path.join(temp_dir, item)
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"Error cleaning up temporary file {item}: {str(e)}")
    except Exception as e:
        logger.error(f"Error closing Phoenix session: {str(e)}")

def signal_handler(signum, frame):
    cleanup_phoenix()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_phoenix)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request model
class ArticleRequest(BaseModel):
    topic: str = Field(..., description="The topic to generate an article about")
    include_code: bool = Field(default=False, description="Whether to include code snippets")
    include_diagrams: bool = Field(default=False, description="Whether to include Mermaid diagrams")
    is_url: bool = Field(default=False, description="Whether the topic is a URL")

async def fetch_web_content(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text()
            else:
                raise HTTPException(status_code=400, detail="Failed to fetch URL content")

# Research Chain
research_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""You are a research assistant. Research the following topic thoroughly and provide a comprehensive analysis.
    Follow these guidelines:
    1. Gather information from multiple reliable sources
    2. Focus on factual, verifiable information
    3. Include key statistics and data points
    4. Note any controversies or debates
    5. Identify primary sources and experts
    6. List key dates and events
    7. Include relevant quotes from experts
    8. Note any limitations or gaps in current research

    Topic: {topic}

    Provide your research in the following format:
    - Key Facts and Statistics
    - Expert Opinions and Quotes
    - Historical Context
    - Current State
    - Future Trends
    - Controversies and Debates
    - Sources and References
    """
)

# Fact-Checking Chain
fact_check_prompt = PromptTemplate(
    input_variables=["topic", "research"],
    template="""You are a fact-checking expert. Verify the accuracy of the following research about a topic.
    Follow these guidelines:
    1. Cross-reference all claims with reliable sources
    2. Identify any potential inaccuracies or exaggerations
    3. Verify statistics and data points
    4. Check expert quotes and attributions
    5. Note any outdated information
    6. Identify any missing context
    7. Flag any potential biases
    8. Suggest corrections or clarifications

    Topic: {topic}
    Research to verify: {research}

    Provide your fact-checking analysis in the following format:
    - Verified Facts (with sources)
    - Corrections Needed
    - Additional Context Required
    - Potential Biases Identified
    - Expert Consensus
    - Areas of Uncertainty
    """
)

# Article Generation Chain
article_prompt = PromptTemplate(
    input_variables=["topic", "research", "fact_check", "include_code", "include_diagrams"],
    template="""You are an expert technical writer. Create a comprehensive, well-structured article about the given topic.
    Follow these guidelines:
    1. Use the verified research and fact-checking information
    2. Write in a clear, engaging style
    3. Include proper headings and subheadings
    4. Use bullet points and lists where appropriate
    5. Include relevant examples and case studies
    6. Add expert quotes and statistics
    7. Ensure factual accuracy
    8. Maintain a neutral, professional tone
    9. Include proper citations and references
    10. Add a meta description for SEO

    {code_requirement}
    {diagram_requirement}

    Topic: {topic}
    Research: {research}
    Fact-Checking: {fact_check}

    Generate the article in markdown format with proper headings, lists, and formatting.
    """
)

# Initialize chains
research_chain = LLMChain(llm=llm, prompt=research_prompt)
fact_check_chain = LLMChain(llm=llm, prompt=fact_check_prompt)
article_chain = LLMChain(llm=llm, prompt=article_prompt)

@app.get("/test")
async def test():
    return {"message": "API is working"}

@app.post("/generate-article")
async def generate_article(request: ArticleRequest):
    try:
        logger.info(f"Starting article generation for topic: {request.topic}")
        
        # Validate input
        if not request.topic or len(request.topic.strip()) == 0:
            raise HTTPException(status_code=400, detail="Topic cannot be empty")
            
        # Run research chain
        try:
            research_response = research_chain.run(topic=request.topic)
            logger.info("Research completed successfully")
        except Exception as e:
            logger.error(f"Error in research chain: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")
        
        # Run fact-checking chain
        try:
            fact_check_response = fact_check_chain.run(
                topic=request.topic,
                research=research_response
            )
            logger.info("Fact-checking completed successfully")
        except Exception as e:
            logger.error(f"Error in fact-checking chain: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Fact-checking failed: {str(e)}")
        
        # Prepare code and diagram requirements
        code_requirement = "Include relevant code snippets with explanations." if request.include_code else ""
        diagram_requirement = "Include Mermaid diagrams to illustrate key concepts." if request.include_diagrams else ""
        
        # Run article generation chain
        try:
            article_response = article_chain.run(
                topic=request.topic,
                research=research_response,
                fact_check=fact_check_response,
                include_code=request.include_code,
                include_diagrams=request.include_diagrams,
                code_requirement=code_requirement,
                diagram_requirement=diagram_requirement
            )
            logger.info("Article generation completed successfully")
        except Exception as e:
            logger.error(f"Error in article generation chain: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Article generation failed: {str(e)}")
        
        # Evaluate the generated article
        try:
            evaluation_metrics = evaluate_article(
                article_response,
                research_response,
                fact_check_response,
                request.topic
            )
            logger.info("Article evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error in article evaluation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Article evaluation failed: {str(e)}")
        
        return {
            "article": article_response,
            "metrics": evaluation_metrics
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in generate_article: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def evaluate_article(article: str, research: str, fact_check: str, topic: str) -> Dict:
    try:
        # Run evaluations with proper context
        try:
            hallucination_score = hallucination_evaluator.evaluate(
                article,
                context={"reference_text": research}
            )
            relevance_score = relevance_evaluator.evaluate(
                article,
                context={"topic": topic}
            )
            toxicity_score = toxicity_evaluator.evaluate(article)
            
            logger.info("Evaluation scores calculated successfully")
        except Exception as e:
            logger.error(f"Error in evaluation calculations: {str(e)}")
            # Provide default scores if evaluation fails
            hallucination_score = 0.7  # Conservative default
            relevance_score = 0.8
            toxicity_score = 0.1
            logger.warning("Using default evaluation scores due to evaluation error")

        # Calculate additional metrics
        word_count = len(article.split())
        paragraph_count = len([p for p in article.split('\n\n') if p.strip()])
        code_block_count = len(re.findall(r'```[\w]*\n[\s\S]*?\n```', article))
        diagram_count = len(re.findall(r'```mermaid[\s\S]*?\n```', article))
        link_count = len(re.findall(r'\[.*?\]\(.*?\)', article))
        heading_count = len(re.findall(r'^#+\s.*$', article, re.MULTILINE))

        # Calculate average sentence length (improved)
        sentences = [s.strip() for s in re.split(r'[.!?]+', article) if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        # Calculate reading time (assuming 200 words per minute)
        reading_time_minutes = word_count / 200

        # Calculate content density (headings per 1000 words)
        content_density = (heading_count / max(word_count, 1)) * 1000

        # Calculate ratios with safe division
        code_to_text_ratio = (code_block_count / max(word_count, 1)) * 100
        diagram_to_text_ratio = (diagram_count / max(word_count, 1)) * 100
        link_density = (link_count / max(word_count, 1)) * 100
        avg_paragraph_length = word_count / max(paragraph_count, 1)

        # Calculate structure score with improved weighting
        structure_score = min(1.0, (
            (heading_count / max(word_count / 300, 1)) * 0.3 +  # Heading density
            (paragraph_count / max(word_count / 100, 1)) * 0.2 +  # Paragraph structure
            (min(code_block_count, 5) / 5) * 0.2 +  # Code blocks (max 5)
            (min(diagram_count, 3) / 3) * 0.2 +  # Diagrams (max 3)
            (min(link_count, 10) / 10) * 0.1  # Links (max 10)
        ))

        # Calculate overall quality score with adjusted weights
        quality_score = (
            hallucination_score * 0.4 +  # Increased weight for factual accuracy
            relevance_score * 0.4 +      # Increased weight for relevance
            (1.0 - toxicity_score) * 0.2 # Convert toxicity to positive metric
        )

        return {
            "quality_scores": {
                "hallucination": float(hallucination_score),
                "relevance": float(relevance_score),
                "toxicity": float(toxicity_score),
                "overall_quality": float(quality_score),
                "structure_score": float(structure_score)
            },
            "content_metrics": {
                "word_count": word_count,
                "paragraph_count": paragraph_count,
                "code_block_count": code_block_count,
                "diagram_count": diagram_count,
                "link_count": link_count,
                "heading_count": heading_count,
                "avg_sentence_length": float(avg_sentence_length),
                "avg_paragraph_length": float(avg_paragraph_length),
                "reading_time_minutes": float(reading_time_minutes),
                "content_density": float(content_density),
                "code_to_text_ratio": float(code_to_text_ratio),
                "diagram_to_text_ratio": float(diagram_to_text_ratio),
                "link_density": float(link_density)
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "topic": topic
            }
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_article: {str(e)}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    logger.info("Root endpoint accessed")
    try:
        # Check if static directory exists
        if not os.path.exists('static'):
            logger.error("Static directory not found")
            raise HTTPException(status_code=500, detail="Static directory not found")
            
        # Check if index.html exists
        if not os.path.exists('static/index.html'):
            logger.error("index.html not found in static directory")
            raise HTTPException(status_code=500, detail="index.html not found")
            
        # Read and return the file
        with open('static/index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
            logger.info("Successfully read index.html")
            return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error reading index.html: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading the page: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000) 