import os
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PIL import Image
from io import BytesIO
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# Function to validate the URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])  # Check if scheme and netloc are present
    except ValueError:
        return False

# Function to capture the header and save the screenshot and JSON
def capture_header_and_save(url):
    # Validate the URL
    if not is_valid_url(url):
        raise ValueError("Invalid URL. Please provide a valid URL starting with http:// or https://")

    # Configuração do Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Executar em modo headless
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1280,720")

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

    try:
        # Navegar para a URL
        driver.get(url)

        # Capturar o header
        header = driver.find_element(By.TAG_NAME, "header")

        # Tirar screenshot do header
        header_screenshot = header.screenshot_as_png
        header_image = Image.open(BytesIO(header_screenshot))

        # Criar pasta com o nome do site
        site_name = url.split("//")[-1].split("/")[0].replace(".", "_")
        folder_path = os.path.join("sites", site_name)
        os.makedirs(folder_path, exist_ok=True)

        # Salvar o screenshot
        screenshot_path = os.path.join(folder_path, "header_screenshot.png")
        header_image.save(screenshot_path)

        # Criar arquivo JSON com a estrutura e conteúdo do header
        header_json = {
            "tag": "header",
            "children": []
        }
        for child in header.find_elements(By.XPATH, "./*"):
            child_info = {
                "tag": child.tag_name,
                "text": child.text,
                "attributes": child.get_attribute("outerHTML")
            }
            header_json["children"].append(child_info)

        json_path = os.path.join(folder_path, "header_structure.json")
        with open(json_path, "w") as json_file:
            json.dump(header_json, json_file, indent=4)

        return screenshot_path, json_path

    finally:
        driver.quit()

# Função para chamar o llama.cpp e gerar o HTML
def generate_html_from_llama(screenshot_path, json_path):
    # Configurar o modelo LlamaCpp
    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
        n_gpu_layers=40,
        n_batch=512,
        verbose=False
    )

    # Template para o prompt
    template = """
    Given the following screenshot and JSON structure, generate an HTML file with CSS that represents the screenshot and fill with the contents in the json file:

    Screenshot Path: {screenshot_path}
    JSON Path: {json_path}

    HTML and CSS:
    """
    prompt = PromptTemplate(template=template, input_variables=["screenshot_path", "json_path"])

    # Criar a cadeia de LLM
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Executar a cadeia
    result = llm_chain.run(screenshot_path=screenshot_path, json_path=json_path)

    return result

# Função principal
def main():
    # Solicitar URL ao usuário
    url = input("Digite a URL do site: ")

    try:
        # Passo 1: Capturar o header e salvar
        screenshot_path, json_path = capture_header_and_save(url)

        # Passo 2: Gerar HTML com llama.cpp
        html_result = generate_html_from_llama(screenshot_path, json_path)

        # Salvar o HTML gerado
        site_name = url.split("//")[-1].split("/")[0].replace(".", "_")
        folder_path = os.path.join("sites", site_name)
        html_path = os.path.join(folder_path, "generated_header.html")
        with open(html_path, "w") as html_file:
            html_file.write(html_result)

        print(f"HTML gerado e salvo em: {html_path}")

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()