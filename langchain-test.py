from langchain.llms import OpenAI
# Create a LangChain client
llm = OpenAI(openai_api_key="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")

# Generate text
text = llm.predict("Write a poem about love.")

# Print the text
print(text)