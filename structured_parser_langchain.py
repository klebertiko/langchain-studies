from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
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

# Criar um template de prompt para o modelo de linguagem
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="Extract the following information from the JSON:\n{format_instructions}\n{json_input}\n",
    input_variables=["json_input"],
    partial_variables={"format_instructions": format_instructions}
)

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


# Mapeamento para extrair os dados
mapping = {
    "user_id": {"path": ["data", "user_info", "user_id"]},
    "user_name": {"path": ["data", "user_info", "user_name"]},
    "user_city": {"path": ["data", "location", "city"]},
    "orders": [
        {
            "path": ["data", "orders"],
            "order_id": "order_id",
            "product_name": "product",
            "product_price": "price"
        }
    ]
}

# Extrair os dados dinamicamente
extracted_data = extract_data(original_json, mapping)

# Converter o JSON extraído para string
json_input = json.dumps(extracted_data, indent=4)

# Preencher o prompt com o JSON de entrada
_input = prompt.format_prompt(json_input=json_input)

# Configurar o modelo de linguagem (OpenAI GPT)
llm = OpenAI(temperature=0.0)

# Gerar a saída do modelo
output = llm(_input.to_string())

# Parsear a saída para JSON
parsed_output = output_parser.parse(output)

# Imprimir o JSON final
print(json.dumps(parsed_output, indent=4))