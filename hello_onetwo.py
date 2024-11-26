from onetwo import ot
from onetwo.builtins import llm
from onetwo.backends import openai_api
import os

# Confirm the currently installed OneTwo version.
print(f'Running OneTwo v{ot.__version__}.')

if not 'OPENAI_API_KEY' not in os.environ:
  raise ValueError(
      'The api key must be specified either here or in the environment.')

model_name = 'gpt-4o-mini'  # Specify the model of your choice.

# Create and register a connection to the OpenAI backend.
backend = openai_api.OpenAIAPI(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name=model_name,
    temperature=0.0,
    max_retries=12,
    cache_filename=get_cache_path(model_name, temperature=0.0),
)
backends[model_name] = backend
backend.register()
print('OpenAI API backend registered.')
