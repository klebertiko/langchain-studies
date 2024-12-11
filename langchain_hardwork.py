from langchain.chains import TransformChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json

# Exemplo de JSON de entrada
original_json = {
    "data": {
        "user_info": {
            "user_id": 123,
            "user_name": "John Doe"
        },
        "location": {
            "city": "New York",
            "zip": "10001"
        },
        "orders": [
            {
                "order_id": 1,
                "product": "Laptop",
                "price": 1200
            },
            {
                "order_id": 2,
                "product": "Phone",
                "price": 800
            }
        ]
    }
}

# Definir o formato desejado para o JSON de saída
response_schemas = [
    ResponseSchema(name="user_id", description="The ID of the user"),
    ResponseSchema(name="user_name", description="The name of the user"),
    ResponseSchema(name="user_city", description="The city of the user"),
    ResponseSchema(name="orders", description="List of orders with order_id, product_name, and product_price")
]

# Configurar o JSON Output Parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Função para extrair dados dinamicamente
def extract_data(data, mapping):
    """
    Extrai dados dinamicamente de um JSON usando um mapeamento.

    :param data: JSON de entrada
    :param mapping: Dicionário de mapeamento que define como os dados devem ser extraídos
    :return: JSON transformado
    """
    output = {}

    for key, value in mapping.items():
        if isinstance(value, dict):
            # Se o valor for um dicionário, significa que precisamos navegar mais fundo no JSON
            path = value.get("path", [])
            default = value.get("default", None)
            current_data = data

            # Percorre o caminho no JSON
            for step in path:
                current_data = current_data.get(step, {}) if isinstance(current_data, dict) else None
                if current_data is None:
                    break

            # Adiciona o valor ao JSON de saída
            output[key] = current_data if current_data is not None else default

        elif isinstance(value, list):
            # Se o valor for uma lista, significa que precisamos iterar sobre uma lista de objetos
            path = value[0].get("path", [])
            current_data = data

            # Percorre o caminho no JSON
            for step in path:
                current_data = current_data.get(step, []) if isinstance(current_data, dict) else []

            # Extrai os itens da lista
            output[key] = []
            for item in current_data:
                extracted_item = {}
                for sub_key, sub_value in value[0].items():
                    if sub_key == "path":
                        continue
                    extracted_item[sub_key] = item.get(sub_value, None)
                output[key].append(extracted_item)

    return output


# Etapa 1: TransformChain para gerar o mapeamento dinamicamente com base no schema
def generate_mapping(inputs: dict) -> dict:
    """
    Função para gerar o mapeamento dinamicamente com base no schema.
    """
    # Prompt para a LLM gerar o mapeamento
    mapping_prompt = PromptTemplate(
        template="""
        Given the following JSON schema and the input JSON structure, generate a mapping to extract the data.

        Schema:
        {schema}

        Input JSON Structure:
        {json_input}

        Mapping Example:
        {{
            "user_id": {{"path": ["data", "user_info", "user_id"]}},
            "user_name": {{"path": ["data", "user_info", "user_name"]}},
            "user_city": {{"path": ["data", "location", "city"]}},
            "orders": [
                {{
                    "path": ["data", "orders"],
                    "order_id": "order_id",
                    "product_name": "product",
                    "product_price": "price"
                }}
            ]
        }}

        Mapping:
        """,
        input_variables=["schema", "json_input"]
    )

    # Converter o schema para string
    schema_str = json.dumps([schema.dict() for schema in response_schemas], indent=4)

    # Preencher o prompt com o schema e o JSON de entrada
    _input = mapping_prompt.format_prompt(schema=schema_str, json_input=json.dumps(inputs["json_input"], indent=4))

    # Configurar o modelo de linguagem (OpenAI GPT)
    llm = OpenAI(temperature=0.0)

    # Gerar o mapeamento
    mapping_output = llm(_input.to_string())

    # Converter o mapeamento de string para JSON
    mapping = json.loads(mapping_output)

    return {"mapping": mapping}


generate_mapping_chain = TransformChain(
    input_variables=["json_input"],
    output_variables=["mapping"],
    transform=generate_mapping
)

# Etapa 2: TransformChain para extrair dados dinamicamente
def transform_extract_data(inputs: dict) -> dict:
    """
    Função de transformação para extrair dados dinamicamente.
    """
    extracted_data = extract_data(inputs["json_input"], inputs["mapping"])
    return {"extracted_data": extracted_data}


extract_chain = TransformChain(
    input_variables=["json_input", "mapping"],
    output_variables=["extracted_data"],
    transform=transform_extract_data
)

# Etapa 3: TransformChain para converter o JSON extraído em string
def transform_json_to_string(inputs: dict) -> dict:
    """
    Função de transformação para converter o JSON extraído em string.
    """
    json_string = json.dumps(inputs["extracted_data"], indent=4)
    return {"json_string": json_string}


json_to_string_chain = TransformChain(
    input_variables=["extracted_data"],
    output_variables=["json_string"],
    transform=transform_json_to_string
)

# Etapa 4: Prompt para o modelo de linguagem
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Extract the following information from the JSON:\n{format_instructions}\n{json_string}\n",
    input_variables=["json_string"],
    partial_variables={"format_instructions": format_instructions}
)

# Configurar o modelo de linguagem (OpenAI GPT)
llm = OpenAI(temperature=0.0)

# Etapa 5: Chain para gerar a saída do modelo
def generate_output(inputs: dict) -> dict:
    """
    Função para gerar a saída do modelo de linguagem.
    """
    _input = prompt.format_prompt(json_string=inputs["json_string"])
    output = llm(_input.to_string())
    return {"model_output": output}


generate_output_chain = TransformChain(
    input_variables=["json_string"],
    output_variables=["model_output"],
    transform=generate_output
)

# Etapa 6: Chain para parsear a saída do modelo
def parse_output(inputs: dict) -> dict:
    """
    Função para parsear a saída do modelo.
    """
    parsed_output = output_parser.parse(inputs["model_output"])
    return {"parsed_output": parsed_output}


parse_output_chain = TransformChain(
    input_variables=["model_output"],
    output_variables=["parsed_output"],
    transform=parse_output
)

# Criar a SequentialChain para encadear todas as etapas
sequential_chain = SequentialChain(
    chains=[
        generate_mapping_chain,  # Etapa 1: Gerar mapeamento
        extract_chain,           # Etapa 2: Extrair dados
        json_to_string_chain,    # Etapa 3: Converter para string
        generate_output_chain,   # Etapa 4: Gerar saída do modelo
        parse_output_chain       # Etapa 5: Parsear saída
    ],
    input_variables=["json_input"],
    output_variables=["parsed_output"]
)

# Executar a chain com o JSON de entrada
result = sequential_chain({"json_input": original_json})

# Imprimir o JSON final
print(json.dumps(result["parsed_output"], indent=4))