from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any

def create_research_chain(llm: ChatGoogleGenerativeAI) -> LLMChain:
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
    return LLMChain(llm=llm, prompt=research_prompt)

def create_fact_check_chain(llm: ChatGoogleGenerativeAI) -> LLMChain:
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
    return LLMChain(llm=llm, prompt=fact_check_prompt)

def create_article_chain(llm: ChatGoogleGenerativeAI) -> LLMChain:
    article_prompt = PromptTemplate(
        input_variables=["topic", "research", "fact_check", "code_requirement", "diagram_requirement"],
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
    return LLMChain(llm=llm, prompt=article_prompt) 