from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
from typing import List, Optional, Dict
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

# Configure Phoenix with a different port
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
os.environ["PHOENIX_GRPC_PORT"] = "6007"

# Initialize Phoenix session
session = px.Session(
    project_name="ai-article-generator",
    description="AI Article Generator with Phoenix monitoring"
)

# Initialize evaluators
hallucination_evaluator = HallucinationEvaluator(llm=llm)
relevance_evaluator = RelevanceEvaluator(llm=llm)
toxicity_evaluator = ToxicityEvaluator(llm=llm)

# Instrument LangChain with Phoenix using OpenInference
tracer_provider = register()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Register cleanup handlers
def cleanup_phoenix():
    try:
        session.end()
    except Exception as e:
        logger.error(f"Error cleaning up Phoenix: {str(e)}")

def signal_handler(signum, frame):
    cleanup_phoenix()
    sys.exit(0)

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
    topic: str
    is_url: bool = False
    include_code: bool = False
    include_diagrams: bool = False

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

def research_topic(topic: str) -> str:
    try:
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Research the following topic and provide comprehensive information:
            {topic}
            
            Format the response in clear, well-structured markdown with the following sections:
            
            # Research Findings
            
            ## Key Facts and Claims
            - List the most important facts and claims about the topic
            - Use bullet points for clarity
            - Include specific examples where relevant
            
            ## Supporting Evidence
            - For each major claim, provide:
              * The claim itself
              * Supporting evidence or data
              * Source citations (preferably academic or reputable sources)
            
            ## Recent Developments
            - Highlight recent advancements or changes
            - Include dates and specific examples
            - Note any emerging trends
            
            ## Expert Opinions
            - Include relevant quotes from experts
            - Cite the sources of these opinions
            - Note any consensus or disagreements
            
            ## Statistics and Data
            - Include relevant statistics
            - Provide context for the numbers
            - Cite the sources of the data
            
            Format the entire response in clean markdown with proper headings and bullet points.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({"topic": topic})
        if not response or "text" not in response:
            raise ValueError("Invalid response from research chain")
        return response["text"]
    except Exception as e:
        logger.error(f"Error in research_topic: {str(e)}")
        raise

def fact_check_content(content: str) -> str:
    try:
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""
            Fact-check the following content and provide verification:
            {content}
            
            For each claim:
            1. Verify its accuracy
            2. Provide supporting evidence
            3. Note any uncertainties
            4. Suggest reliable sources
            5. Flag any potential misinformation
            
            Format the response as a structured JSON object with verified facts and sources.
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({"content": content})
        if not response or "text" not in response:
            raise ValueError("Invalid response from fact-check chain")
        return response["text"]
    except Exception as e:
        logger.error(f"Error in fact_check_content: {str(e)}")
        raise

def generate_article(topic: str, research: str, fact_check: str, include_code: bool, include_diagrams: bool) -> Dict:
    try:
        # Create the prompt template
        template = """
        Write a comprehensive, SEO-optimized article about: {topic}
        
        Research findings:
        {research}
        
        Fact-checking results:
        {fact_check}
        
        Requirements:
        1. Use Markdown formatting with proper heading hierarchy (#, ##, ###)
        2. Start with an engaging meta description
        3. Include a clear introduction and conclusion
        4. Use bullet points and numbered lists where appropriate
        5. Include relevant statistics and quotes
        """
        
        # Add code requirement if needed
        if include_code:
            template += "\n6. Include relevant code snippets with explanations"
            
        # Add diagram requirement if needed
        if include_diagrams:
            template += """
            7. Include Mermaid.js diagrams for technical concepts. Use the following format:
            ```mermaid
            graph TD
                A[Start] --> B[Process]
                B --> C[End]
            ```
            For flowcharts, use:
            ```mermaid
            flowchart TD
                A[Start] --> B[Process]
                B --> C[End]
            ```
            For sequence diagrams, use:
            ```mermaid
            sequenceDiagram
                Alice->>John: Hello John, how are you?
                John-->>Alice: Great!
            ```
            """
            
        template += """
        
        Structure:
        - H1: SEO-optimized title
        - Meta description
        - Introduction
        - Main sections (H2)
        - Subsections (H3)
        - Conclusion
        - References and sources
        
        Format the response in clean markdown with proper spacing and formatting.
        """
        
        prompt = PromptTemplate(
            input_variables=["topic", "research", "fact_check"],
            template=template
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke({
            "topic": topic,
            "research": research,
            "fact_check": fact_check
        })
        
        if not response or "text" not in response:
            raise ValueError("Invalid response from article generation chain")
        
        article_text = response["text"]
        
        # Evaluate the generated article
        try:
            # Convert research to a dictionary for evaluation
            research_dict = {"text": research}
            hallucination_score = hallucination_evaluator.evaluate(article_text, research_dict)
            relevance_score = relevance_evaluator.evaluate(article_text, {"topic": topic})
            toxicity_score = toxicity_evaluator.evaluate(article_text)
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            hallucination_score = 0.0
            relevance_score = 0.0
            toxicity_score = 0.0
        
        # Log evaluation metrics
        evaluation_metrics = {
            "hallucination_score": hallucination_score,
            "relevance_score": relevance_score,
            "toxicity_score": toxicity_score,
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "article_length": len(article_text),
            "word_count": len(article_text.split()),
            "has_code_snippets": include_code,
            "has_diagrams": include_diagrams
        }
        
        # Log evaluation metrics
        logger.info(f"Article Evaluation Metrics: {json.dumps(evaluation_metrics, indent=2)}")
        
        return {
            "text": article_text,
            "metrics": evaluation_metrics
        }
    except Exception as e:
        logger.error(f"Error in generate_article: {str(e)}")
        raise

@app.get("/test")
async def test():
    return {"message": "API is working"}

@app.post("/generate-article")
async def create_article(request: ArticleRequest):
    logger.info(f"Generating article for topic: {request.topic}")
    start_time = time.time()
    
    try:
        # Extract content if URL is provided
        if request.is_url:
            logger.info("Processing URL content...")
            content = await fetch_web_content(request.topic)
            topic = trafilatura.extract(content)
            logger.info("URL content extracted successfully")
        else:
            topic = request.topic
            logger.info("Processing topic directly")
        
        # Research phase
        logger.info("Starting research phase...")
        try:
            research_result = research_topic(topic)
            logger.info("Research phase completed successfully")
        except Exception as e:
            logger.error(f"Error in research phase: {str(e)}")
            raise
        
        # Fact-checking phase
        logger.info("Starting fact-checking phase...")
        try:
            fact_check_result = fact_check_content(research_result)
            logger.info("Fact-checking phase completed successfully")
        except Exception as e:
            logger.error(f"Error in fact-checking phase: {str(e)}")
            raise
        
        # Article generation phase
        logger.info("Starting article generation phase...")
        try:
            result = generate_article(
                topic=topic,
                research=research_result,
                fact_check=fact_check_result,
                include_code=request.include_code,
                include_diagrams=request.include_diagrams
            )
            logger.info("Article generation completed successfully")
        except Exception as e:
            logger.error(f"Error in article generation phase: {str(e)}")
            raise
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "article": result["text"],
            "metrics": {
                **result["metrics"],
                "processing_time": processing_time
            },
            "research": research_result,
            "fact_check": fact_check_result
        }
    except Exception as e:
        logger.error(f"Error in create_article: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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