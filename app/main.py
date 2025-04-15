from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import logging
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.evals import (
    HallucinationEvaluator,
    RelevanceEvaluator,
    ToxicityEvaluator
)

from .models import ArticleRequest, ArticleResponse
from .chains import create_research_chain, create_fact_check_chain, create_article_chain
from .evaluation import ArticleEvaluator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Article Generator",
    description="Generate well-structured markdown articles using AI",
    version="1.0.0"
)

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

# Initialize evaluators
hallucination_evaluator = HallucinationEvaluator(model=llm)
relevance_evaluator = RelevanceEvaluator(model=llm)
toxicity_evaluator = ToxicityEvaluator(model=llm)

# Initialize article evaluator
article_evaluator = ArticleEvaluator(
    hallucination_evaluator,
    relevance_evaluator,
    toxicity_evaluator
)

# Initialize chains
research_chain = create_research_chain(llm)
fact_check_chain = create_fact_check_chain(llm)
article_chain = create_article_chain(llm)

# Instrument LangChain with Phoenix
tracer_provider = register()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

@app.get("/")
async def root():
    return HTMLResponse(content=open("static/index.html").read())

@app.post("/generate-article", response_model=ArticleResponse)
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
                code_requirement=code_requirement,
                diagram_requirement=diagram_requirement
            )
            logger.info("Article generation completed successfully")
        except Exception as e:
            logger.error(f"Error in article generation chain: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Article generation failed: {str(e)}")
        
        # Evaluate the generated article
        try:
            evaluation_metrics = article_evaluator.evaluate_article(
                article_response,
                research_response,
                fact_check_response,
                request.topic
            )
            logger.info("Article evaluation completed successfully")
        except Exception as e:
            logger.error(f"Error in article evaluation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Article evaluation failed: {str(e)}")
        
        return ArticleResponse(
            article=article_response,
            metrics=evaluation_metrics,
            research=research_response,
            fact_check=fact_check_response
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in generate_article: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 