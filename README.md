# RAG demo

Uses a blog as context data, Mistral model and embedding, and vector store from
langchain-core.

# Examples

```
(.venv) rag-chain# python rag.py --q "Whats the benefit of 'chain of hindsight'?"  
The 'Chain of Hindsight' (CoH) benefits models by encouraging self-improvement. It presents the model with a sequence of past outputs and corresponding feed
back, allowing it to learn from and build upon its own mistakes. This process can lead to incremental improvements in the model's outputs over time. Additio
nally, CoH can help models learn from human feedback, making their outputs more aligned with human preferences.

(.venv) rag-chain# python rag.py --q "Who wrote the paper ReAct?"  
Yao, Y., Chen, M., & Dai, Z. (2023). ReAct: Synergizing Reasoning and Acting for Language Agents. arXiv:2304.12244.
```

# Install

Install from `requirements.txt`

# Run

	python rag.py --q "What is your question?"

# References

Adapted from: [python.langchain.com/docs/tutorials/rag](https://python.langchain.com/docs/tutorials/rag/)

Source document: [lilianweng.github.io/posts/2023-06-23-agent](https://lilianweng.github.io/posts/2023-06-23-agent/)
