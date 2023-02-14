# Chat-Your-Data

Create a ChatGPT like experience over your custom docs using [LangChain](https://github.com/hwchase17/langchain). This repo can help you use models hosted on HuggingFace for embedding and for text generation.

The explanation of [this blog post](https://blog.langchain.dev/tutorial-chatgpt-over-your-data/) can help you understand the reason for different files and the steps involved. We will primarily replace OpenAI API with huggingface based models.

## Environment Variable
Please set huggingface token as `huggingfacehub_api_token=[your-token]`. Token can be generated in the HuggingFace account settings.


## Ingest data

Ingestion of data is done over the `state_of_the_union.txt` file. 
Therefore, the only thing that is needed is to be done to ingest data is run `python ingest_data.py`

## Query data
Custom prompts are used to ground the answers in the state of the union text file.

## Running the Application

By running `python app.py` from the command line you can easily interact with your ChatGPT over your own data.
