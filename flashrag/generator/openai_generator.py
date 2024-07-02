from dotenv import load_dotenv
import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
from typing import Callable
import numpy as np

import asyncio
from openai import AzureOpenAI
import tiktoken


def _get_key_from_key_vault(key_vault_name: str, secret_name: str) -> str:
    # NOTE: not used the now due to Azure's security policy
    print(f"Try get {secret_name} from {key_vault_name}...")
    from azure.keyvault.secrets import SecretClient
    from azure.identity import DefaultAzureCredential

    KVUri = f"https://{key_vault_name}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)
    retrieved_secret = client.get_secret(secret_name)
    return retrieved_secret.value


def check_azure_openai_api_key() -> None:
    # NOTE: not used the now due to Azure's security policy
    if os.environ.get("AZURE_OPENAI_API_KEY", None) is None:
        key_vault_name = os.environ.get("KEY_VAULT_NAME", None)
        secret_name = os.environ.get("AZURE_OPENAI_API_KEY_SECRET_NAME", None)
        assert key_vault_name is not None, f"Neither AZURE_OPENAI_API_KEY nor KEY_VAULT_NAME are set!"
        assert secret_name is not None, f"Neither AZURE_OPENAI_API_KEY nor KEY_VAULT_NAME are set!"

        os.environ["AZURE_OPENAI_API_KEY"] = _get_key_from_key_vault(key_vault_name, secret_name)
        print(f"AZURE_OPENAI_API_KEY set with {secret_name} accessed from {key_vault_name}.")
    return


def get_azure_active_directory_token_provider() -> Callable[[], str]:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

    return token_provider


class OpenaiGenerator:
    """Class for api-based openai models"""
    def __init__(self, config):
        load_success = load_dotenv(config["env_path"])
        assert load_success is True, f'Failed to load dot env file: {config["env_path"]}'
        
        self.model = config['generator_model']
        self.batch_size = config['generator_batch_size']
        self.generation_params = config['generation_params']

        self.openai_setting = config['openai_setting']
        # if self.openai_setting['api_key'] is None:
        #     self.openai_setting['api_key'] = os.getenv('OPENAI_API_KEY')


        self.client = AzureOpenAI(
            azure_ad_token_provider=get_azure_active_directory_token_provider(),
            # **self.openai_setting
        )
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def get_response(self, input: List, **params):
        response = self.client.chat.completions.create(
            model = self.model,
            messages=input,
            **params
        )
        return response.choices[0]

    '''asynchronous not supported for azureopenai'''
    # async def get_response(self, input: List, **params):
    #     response = await self.client.chat.completions.create(
    #         model = self.model,
    #         messages=input,
    #         **params
    #     )
    #     return response.choices[0].message.content

    # async def get_batch_response(self, input_list:List[List], batch_size, **params):
    #     total_input = [self.get_response(input, **params) for input in input_list]
    #     all_result = []
    #     for idx in tqdm(range(0, len(input_list), batch_size), desc='Generation process: '):
    #         batch_input = total_input[idx:idx+batch_size]
    #         batch_result = await asyncio.gather(*batch_input)
    #         all_result.extend(batch_result)

    #     return all_result
    
    def get_batch_response(self, input_list:List[List], batch_size, **params):
        all_result = [self.get_response(input, **params) for input in input_list]

        return all_result

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        if len(input_list) == 1:
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)

        if return_scores:
            if generation_params.get('logprobs') is not None:
                generation_params['logprobs'] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params['logprobs'] = True

        if generation_params.get('n') is not None:
            generation_params['n'] = 1
            warnings.warn("Set n to 1. It can minimize costs.")
        else:
            generation_params['n'] = 1

        # loop = asyncio.get_event_loop()
        # result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))
        result = self.get_batch_response(input_list, batch_size, **generation_params)

        # parse result into response text and logprob
        scores = []
        response_text =[]
        for res in result:
            response_text.append(res.message.content)
            if return_scores:
                score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                scores.append(score)
        if return_scores:
            return response_text, scores
        else:
            return response_text
