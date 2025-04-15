from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ArticleRequest(BaseModel):
    topic: str = Field(..., description="The topic to generate an article about")
    include_code: bool = Field(default=False, description="Whether to include code snippets")
    include_diagrams: bool = Field(default=False, description="Whether to include Mermaid diagrams")
    is_url: bool = Field(default=False, description="Whether the topic is a URL")

class ArticleResponse(BaseModel):
    article: str = Field(..., description="The generated article in markdown format")
    metrics: Dict[str, Any] = Field(..., description="Evaluation metrics for the article")
    research: Optional[str] = Field(None, description="Research findings used for the article")
    fact_check: Optional[str] = Field(None, description="Fact-checking results") 