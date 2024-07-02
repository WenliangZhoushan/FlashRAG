import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--retriever_path', type=str)
args = parser.parse_args()

config_dict = {
                'data_dir': 'dataset/',
                'index_path': 'indexes/e5_Flat.index',
                'corpus_path': 'indexes/general_knowledge.jsonl',
                'faiss_gpu': True,
                'model2path': {'e5': 'intfloat/e5-base-v2', 'llama3-8B-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct'},
                'generator_model': 'llama3-8B-instruct',
                'framework': 'vllm',
                'retrieval_method': 'e5',
                'metrics': ['em','f1','sub_em'],
                'retrieval_topk': 10,
                'save_intermediate_data': True
            }

config = Config(config_dict = config_dict)

all_split = get_dataset(config)
test_data = all_split['test']
prompt_templete = PromptTemplate(
    config,
    system_prompt = "Answer the question based on the given document. \
                    Only give me the answer and do not output any other words. \
                    \nThe following are given documents.\n\n{reference}",
    user_prompt = "Question: {question}\nAnswer:"
)
pipeline = SequentialPipeline(config, prompt_template=prompt_templete)

output_dataset = pipeline.run(test_data, do_eval=True)
print("---generation output---")
print(output_dataset.pred)