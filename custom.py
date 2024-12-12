def load_custom_docs(path, **kwargs):
    """
    Load docs recursively from path

    For kwargs can use like:
    glob="**/*.md"
    exclude="**/*.png"
    """
    from langchain_community.document_loaders import DirectoryLoader
    
    loader = DirectoryLoader(path, **kwargs)
    docs = loader.load()

    return docs
