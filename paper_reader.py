from paperqa import Docs, Settings  
import asyncio  
  
async def main():  
    # Configure settings  
    settings = Settings()  
    settings.llm = "ollama/ministral-3:14b"  
    settings.summary_llm = "ollama/ministral-3:14b"  
    settings.embedding = "ollama/mxbai-embed-large"  
    settings.answer.answer_max_sources = 3  
      
    # Configure Ollama connection  
    local_llm_config = {  
        "model_list": [  
            {  
                "model_name": "ollama/ministral-3:14b",  
                "litellm_params": {  
                    "model": "ollama/ministral-3:14b",  
                    "api_base": "http://localhost:11434",
                    "timeout": 300,
                },  
            }  
        ],
        "router_kwargs": {"timeout": 300},
    }  
    settings.llm_config = local_llm_config  
    settings.summary_llm_config = local_llm_config
    settings.parsing.use_doc_details = False  # This is the correct way
    settings.answer.max_concurrent_requests = 1

    # Create document collection  
    docs = Docs()  
      
    # Add PDF papers 
    paper_paths = ["paper1.pdf"]  
    for paper_path in paper_paths:  
        await docs.aadd(paper_path, settings=settings)  
      
    # Query the documents  
    question = """帮我详细解释一下这篇文章，包括以下部分：
    1. 论文核心概念
        对论文核心 insight 的简要总结

    2. 论文内容词解释
        对论文中出现多次，或者比较重要的名词的详细解释

    3. 论文方法
        3.1 过去方法的问题（顺便引出方法的 motivation）
        3.2 整体框架（整个论文的方法部分的核心流程的超详细说明，需要保证通过说明可以完整复现出整个方法，包括细节、公式流程、变量说明）
        3.3 核心难点解析（将方法中比较复杂的部分或者比较关键的部分在这里进行更加直白易懂的解释）

    4. 实验结果与分析
        4.1 实验设置（数据集、模型、指标、超参数设置，对比方法等的内容）
        4.2 实验结果（该方法指标提升了多少，以及其他相关效果或正面的评价）

    5. 结论
        5.1 论文的贡献
        5.2 论文的限制（论文在哪些方面有问题）
        5.3 未来的方向（未来可能的发展方向）
    
    注意： 请以排版清晰方便阅读的markdown的格式进行输出。以论文的标题作为标题放在首行。"""

    session = await docs.aquery(question, settings=settings)  
      
    print(f"Question: {question}")  
    print(f"Answer: {session.formatted_answer}")  
    print(f"Sources used: {len(session.contexts)}")  
  
# Run the async function  
asyncio.run(main())
