import os
import json
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Configurar a chave de API da OpenAI

# JSON que você quer transformar
input_data = {
    "collection_id": "1234",
    "content": {
        "fields": [
            {
                "filed_name": "Nome",
                "field_values": [
                    {"value": "Alice", "status": "UP_TO_DATE"},
                    {"value": "Bob", "status": "OUTDATED"}
                ]
            },
            {
                "filed_name": "Idade",
                "field_values": [
                    {"value": "25", "status": "UP_TO_DATE"}
                ]
            }
        ]
    }
}

# Prompt para transformar o JSON
prompt_template = """
Dado o seguinte JSON de entrada:

{input_data}

Transforme-o para o formato JSON abaixo:
1. "page" deve ser o valor de "collection_id".
2. "components" deve ser uma lista de dicionários, cada um com:
   - "component" como o "filed_name" de cada campo.
   - "value" como o valor em "field_values" com status "UP_TO_DATE".

Apenas inclua valores com status "UP_TO_DATE".

Saída esperada:
"""

prompt = PromptTemplate(
    input_variables=["input_data"],
    template=prompt_template
)

# Criar a LLM Chain para processamento com o LangChain
llm = OpenAI(model="gpt-4o-mini") 
chain = LLMChain(llm=llm, prompt=prompt)

# Executar a transformação
input_data_json = json.dumps(input_data, indent=2)
response = chain.run(input_data=input_data_json) 

# Exibir o resultado transformado
print("JSON Transformado:")
print(response)
