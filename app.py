from flask import Flask, render_template, request, jsonify
from langchain.chains import TransformChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json
import re
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Configuração do modelo LlamaCpp
llm = LlamaCpp(
    model_path="models/codellama-7b.Q4_K_M.gguf",  # Caminho para o modelo
    n_gpu_layers=40,  # Número de camadas do modelo a serem carregadas na GPU
    n_batch=512,  # Tamanho do lote para processamento
    verbose=False,  # Desabilitar logs detalhados
)

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
    """
    output = {}

    for key, value in mapping.items():
        if isinstance(value, dict):
            # Tratar mapeamentos para valores simples
            path = value.get("path", [])
            default = value.get("default", None)
            current_data = data

            # Percorrer o caminho no JSON
            for step in path:
                if isinstance(current_data, list):
                    # Se o caminho aponta para uma lista, iterar sobre os itens
                    current_data = [item.get(step, {}) for item in current_data]
                else:
                    # Se o caminho aponta para um objeto, acessar diretamente
                    current_data = current_data.get(step, {})

                # Se o valor for None, interromper o loop
                if current_data is None:
                    break

            # Se o caminho resultou em uma lista, pegar o primeiro item
            if isinstance(current_data, list) and current_data:
                current_data = current_data[0]

            # Adicionar o valor ao JSON de saída
            output[key] = current_data if current_data is not None else default

        elif isinstance(value, list):
            # Tratar mapeamentos para listas de objetos
            path = value[0].get("path", [])
            current_data = data

            # Percorrer o caminho no JSON
            for step in path:
                if isinstance(current_data, list):
                    # Se o caminho aponta para uma lista, iterar sobre os itens
                    current_data = [item.get(step, {}) for item in current_data]
                else:
                    # Se o caminho aponta para um objeto, acessar diretamente
                    current_data = current_data.get(step, {})

                # Se o valor for None, interromper o loop
                if current_data is None:
                    break

            # Extrair os itens da lista
            output[key] = []
            for item in current_data:
                extracted_item = {}
                for sub_key, sub_value in value[0].items():
                    if sub_key == "path":
                        continue
                    extracted_item[sub_key] = item.get(sub_value, None)
                output[key].append(extracted_item)

    return output


# Função para limpar a saída do modelo
def clean_model_output(output: str) -> str:
    """
    Limpa a saída do modelo para garantir que seja um JSON válido.
    """
    logging.info(f"Saída bruta do modelo: {output}")  # Log da saída bruta

    # Remover texto adicional que não faz parte do JSON
    cleaned_output = re.sub(r"^[^{]*", "", output)
    cleaned_output = re.sub(r"[^}]*$", "", cleaned_output)

    # Tentar parsear o JSON
    try:
        json.loads(cleaned_output)
        logging.info(f"Saída limpa e válida: {cleaned_output}")  # Log da saída limpa
        return cleaned_output
    except json.JSONDecodeError as e:
        logging.error(f"Erro ao parsear JSON: {e}")
        raise ValueError("A saída do modelo não é um JSON válido.")


# Etapa 1: TransformChain para gerar o mapeamento dinamicamente com base no schema
def generate_mapping(inputs: dict) -> dict:
    """
    Função para gerar o mapeamento dinamicamente com base no schema.
    """
    mapping_prompt = PromptTemplate(
        template="""
        Given the JSON structure, generate a valid JSOM object mapping.

        JSON Structure:
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

    # Reduzir o schema para caber no limite de tokens
    schema_str = json.dumps([{"name": schema.name, "description": schema.description} for schema in response_schemas], indent=2)
    json_input_str = json.dumps(inputs["json_input"], indent=2)

    # Verificar o tamanho do prompt
    prompt_text = mapping_prompt.format(schema=schema_str, json_input=json_input_str)
    logging.info(f"Tamanho do prompt: {len(prompt_text.split())} tokens")

    # Enviar o prompt para a LLM
    logging.info(f"Prompt enviado para a LLM (generate_mapping): {prompt_text}")
    mapping_output = llm.invoke(prompt_text)

    logging.info(f"Resposta da LLM (generate_mapping): {mapping_output}")

    # Limpar a saída do modelo
    cleaned_output = clean_model_output(mapping_output)

    # Converter o mapeamento de string para JSON
    mapping = json.loads(cleaned_output)

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

# Etapa 5: Chain para gerar a saída do modelo
def generate_output(inputs: dict) -> dict:
    """
    Função para gerar a saída do modelo de linguagem.
    """
    _input = prompt.format_prompt(json_string=inputs["json_string"])

    logging.info(f"Prompt enviado para a LLM (generate_output): {_input.to_string()}")

    output = llm.invoke(_input.to_string())

    logging.info(f"Resposta da LLM (generate_output): {output}")

    # Limpar a saída do modelo
    cleaned_output = clean_model_output(output)

    return {"model_output": cleaned_output}


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

# Rota inicial
@app.route("/")
def index():
    return render_template("index.html")

# Rota para iniciar o processo
@app.route("/start", methods=["POST"])
def start_process():
    try:
        result = sequential_chain.invoke({"json_input": original_json})
        return jsonify({
            "status": "success",
            "output": result["parsed_output"]
        })
    except Exception as e:
        logging.error(f"Erro ao iniciar o processo: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# Rota para executar uma etapa específica
@app.route("/run_step", methods=["POST"])
def run_step():
    data = request.json
    step = data.get("step")

    try:
        if step == "generate_mapping":
            result = generate_mapping_chain.invoke({"json_input": original_json})
        elif step == "extract_data":
            result = extract_chain.invoke({"json_input": original_json, "mapping": result["mapping"]})
        elif step == "json_to_string":
            result = json_to_string_chain.invoke({"extracted_data": result["extracted_data"]})
        elif step == "generate_output":
            result = generate_output_chain.invoke({"json_string": result["json_string"]})
        elif step == "parse_output":
            result = parse_output_chain.invoke({"model_output": result["model_output"]})
        else:
            return jsonify({"status": "error", "message": "Etapa inválida."})

        return jsonify({"status": "success", "output": result})
    except Exception as e:
        logging.error(f"Erro ao executar a etapa {step}: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)