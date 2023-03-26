
from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs, fetch_archive_from_http, clean_wiki_text
from haystack.nodes import PreProcessor, DensePassageRetriever, Seq2SeqGenerator, RAGenerator
from haystack.pipelines import GenerativeQAPipeline
import os

### Please note: all encoders/generators will be downloaded to C:\Users\<username>.cache\huggingface\transformers on Windows. ###
### On mac, they'll be downloaded to ~/Users/<username>.cache/huggingface/hub/transformers ###
### To change the default location, you can set the environment variable TRANSFORMERS_CACHE to point to your desired location ###

def createDocStore():
    '''This will initialize the FAISS document store into which we store our documents and their corresponding vector embeddings'''
    
    document_store = FAISSDocumentStore(embedding_dim=128, faiss_index_factory_str="Flat", return_embedding=True)
    
    # save the document store index (keeps track of files and embeddings) and config, so that the store can be reloaded later
    document_store.save(index_path = "./docstore/my_index.faiss", config_path = "./docstore/my_config.json")

    return document_store

def loadDocStore():
    '''Loads the document store from hardcoded file locations - this can be changed if needed'''
    
    document_store = FAISSDocumentStore.load("./docstore/my_index.faiss","./docstore/my_config.json")
    return document_store

def populateDocStore(document_store):
    '''Here, we can populate our document store with Game of Thrones documentation!'''

    # create pre-processor to clean our docs before storing them
    pre_processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=400,
        split_respect_sentence_boundary=True,
        split_overlap=0)

    # now we grab the files and throw them in a folder called GOT within a folder called data
    doc_dir = "./data/GOT"
    # s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt12.zip"
    # fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Convert files to documents
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

    # Now, let's write the dicts containing documents to our DB.
    document_store.write_documents(docs)

def createRetriever(document_store):
    '''Create a retriever to grab context from the document store based on similarity to the query'''
    
    # create a retriever that embeds passages and queries according to the two specified models below
    # embedding is the process of mapping a query/passage to a vector, e.g. <1,124,4,12,35,...,3>
    
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
        passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
        use_gpu=True,
        embed_title=True
    )
    return retriever

def updateEmbeddings(document_store, retriever):
    '''Embeddings in the document store must be updated according to the embedding models of the retriever'''

    # update the embeddings so that all new documents since the last update get embeddings
    document_store.update_embeddings(retriever, update_existing_embeddings=False)

def createGenerator():
    '''This function creates our long-form question-answering generator, vblagoje/bart_lfqa and returns it to be used'''

    # this step should take some time on the first run; otherwise, it should be stored locally
    generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
    
    return generator

def createPipeline(generator, retriever):
    '''This function creates a retrieval pipeline, linking the retriever (which has been created to read from our document store)
    and the generator'''

    pipeline = GenerativeQAPipeline(generator, retriever)
    return pipeline


def askQuestion(question, pipeline):
    '''This function allows you to ask a question, and the generator will answer it using context fetched from document store by
    retriever'''
    response = pipeline.run(question, params={"Retriever": {"top_k": 3}})
    return response


### OVERALL EXECUTION SEQUENCE ###

# check if a document store exists
if not os.path.exists("docstore/my_index.faiss"):
    # we can assume there is no pre-existing document store; create one and populate it
    document_store = createDocStore()
    populateDocStore(document_store)
else:
    document_store = loadDocStore()

# now that we have a document store, let's create a retriever for it
retriever = createRetriever(document_store)
updateEmbeddings(document_store, retriever)
document_store.save('./docstore/my_index.faiss', './docstore/my_config.json')

# create generator
generator = createGenerator()

# finally, create overall pipeline
pipeline = createPipeline(generator, retriever)

# feel free to edit this part!
# send a question down the pipeline and grab the answer
res = askQuestion("Who is Arya Stark?", pipeline)

# print the result of the question
print(res)
