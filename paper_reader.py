from paperqa import Docs, Settings  
import asyncio  
  
async def main():  
    # Configure settings  
    settings = Settings()  
    settings.llm = "ollama/ministral-3"  
    settings.summary_llm = "ollama/ministral-3"  
    settings.embedding = "ollama/mxbai-embed-large"  
    settings.answer.answer_max_sources = 3  
      
    # Configure Ollama connection  
    local_llm_config = {  
        "model_list": [  
            {  
                "model_name": "ollama/ministral-3",  
                "litellm_params": {  
                    "model": "ollama/ministral-3",  
                    "api_base": "http://localhost:11434",  
                },  
            }  
        ]  
    }  
    settings.llm_config = local_llm_config  
    settings.summary_llm_config = local_llm_config  
      
    # Create document collection  
    docs = Docs()  
      
    # Add PDF papers  
    paper_paths = ["paper1.pdf"]  
    for paper_path in paper_paths:  
        await docs.aadd(paper_path, settings=settings)  
      
    # Query the documents  
    question = "请用中文总结这些论文的主要观点。"  
    session = await docs.aquery(question, settings=settings)  
      
    print(f"Question: {question}")  
    print(f"Answer: {session.formatted_answer}")  
    print(f"Sources used: {len(session.contexts)}")  
  
# Run the async function  
asyncio.run(main())
