import modal
import dotenv
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dotenv.load_dotenv()

DATASET = "satpalsr/chatml-translation-filter"
MODEL_DIR = "/model"
MODEL_NAME = "satpalsr/llama2-translation-filter-full"
MODEL_REVISION = "main"

def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt"],  # Using safetensors
        revision=model_revision,
    )
    move_cache()

def get_df_dict(dataset_name):
    dataset = load_dataset(dataset_name)
    df = pd.DataFrame(dataset['validation'])
    data_list = df.values.tolist()
    df_dict = {"q": [], "a": []}
    for i in range(len(data_list)):
        df_dict["q"].append(data_list[i][0][1]['value'])
        df_dict["a"].append(eval(data_list[i][0][2]['value']))

    return df_dict

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
    )
)
app = modal.App("vllm-translation-filter", image=image)

with image.imports():
    import vllm
    import time
    import pandas as pd
    from datasets import load_dataset

GPU_CONFIG = modal.gpu.A100(count=1)  # 40GB A100 by default
@app.cls(gpu=GPU_CONFIG)
class Model:
    @modal.enter()
    def load_model(self):
        # Tip: models that are not fully implemented by Hugging Face may require `trust_remote_code=true`.
        self.llm = vllm.LLM(MODEL_DIR, tensor_parallel_size=GPU_CONFIG.count)
        self.template = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
"""
    @modal.method()
    def generate(self, user_questions, temperature=0.75):
        prompts = [
            self.template.format(system='For a given question assess whether translating the potential answer to another language might yield an inaccurate response. Avoid translation in tasks related to coding problems, alliteration, idioms, paraphrasing text, word count, spelling correction, and other linguistic constructs or contextual nuances that may affect the accuracy of the answer. When translation is deemed unsuitable, output {"translate": False}. Otherwise, output {"translate": True}.', user=q) for q in user_questions
        ]

        sampling_params = vllm.SamplingParams(
            temperature=temperature,
            stop_token_ids=[32001],
        )
        start = time.monotonic_ns()
        result = self.llm.generate(prompts, sampling_params)
        duration_s = (time.monotonic_ns() - start) / 1e9
        num_tokens = 0

        COLOR = {
            "HEADER": "\033[95m",
            "BLUE": "\033[94m",
            "GREEN": "\033[92m",
            "RED": "\033[91m",
            "ENDC": "\033[0m",
        }

        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(
                f"{COLOR['HEADER']}{COLOR['GREEN']}{output.prompt}",
                f"\n{COLOR['BLUE']}{output.outputs[0].text}",
                "\n\n",
                sep=COLOR["ENDC"],
            )
            time.sleep(0.01)
        print(
            f"{COLOR['HEADER']}{COLOR['GREEN']}Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second on {GPU_CONFIG}.{COLOR['ENDC']}"
        )

        return result



@app.local_entrypoint()
def main():
    import pickle
    df_dict = get_df_dict(DATASET)
    questions = df_dict["q"]
    model = Model()
    result_with_var_temperature = []
    data = []
    for i in range(11):
        result = model.generate.remote(questions, i/10)
        result_with_var_temperature.append(result)
        for j in range(len(df_dict["q"])):
            text = result[j].outputs[0].text.strip()
            try:
                output_dict = eval(text)
                if i == 0:
                    data.append({"question": df_dict["q"][j], "answer": df_dict["a"][j], "output": {f"{i*10}%": {"value":output_dict["translate"], "status": "valid"}}})
                else:
                    data[j]["output"].update({f"{i*10}%": {"value":output_dict["translate"], "status": "valid"}})
            except:
                if i == 0:
                    data.append({"question": df_dict["q"][j], "answer": df_dict["a"][j], "output": {f"{i*10}%": {"value":text, "status": "invalid"}}})
                else:
                    data[j]["output"].update({f"{i*10}%": {"value":text, "status": "invalid"}})
                continue
    
    with open("result.pkl", "wb") as f:
        pickle.dump(result_with_var_temperature, f)

    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)
    dct = {
        'Total': [len(df_dict['q']) for i in range(11)],
        'Total Predicted Translate': [],
        'Total Predicted Don\'t Translate': [],
        'Total Actual Translate': [],
        'Total Actual Don\'t Translate': [],
        'Correct Translate': [],
        'Incorrect Translate': [],
        'Correct Don\'t Translate': [],
        'Incorrect Don\'t Translate': [],
        'Total Correct': [],
        'Total Incorrect': [],
        'No Output': [],
        'Achieved Accuracy': [],
        'Achieved Sensitivity': [],
        'Achieved Specificity': []
    }
    for i in range(11):
        print(f"Results for temperature {i*10}%")
        count_actual_true = 0
        count_actual_false = 0
        count_predicted_true = 0
        count_predicted_false = 0
        count_correct_true = 0
        count_incorrect_true = 0
        count_correct_false = 0
        count_incorrect_false = 0
        count_no_output = 0
        for j in range(len(df_dict["q"])):
            if df_dict["a"][j]['translate']:
                count_actual_true += 1
            else:
                count_actual_false += 1

            if data[j]["output"][f"{i*10}%"]["status"] == "valid":
                if data[j]["output"][f"{i*10}%"]["value"]:
                    if data[j]["output"][f"{i*10}%"]["value"] == df_dict["a"][j]['translate']:
                        count_correct_true += 1
                    else:
                        count_incorrect_true += 1
                else:
                    if data[j]["output"][f"{i*10}%"]["value"] == df_dict["a"][j]['translate']:
                        count_correct_false += 1
                    else:
                        count_incorrect_false += 1
            else:
                count_no_output += 1

        count_predicted_false = count_correct_false+ count_incorrect_false
        count_predicted_true = count_correct_true + count_incorrect_true
        dct['Total Predicted Translate'].append(count_predicted_true)
        dct['Total Predicted Don\'t Translate'].append(count_predicted_false)
        dct['Total Actual Translate'].append(count_actual_true)
        dct['Total Actual Don\'t Translate'].append(count_actual_false)
        dct['Correct Translate'].append(count_correct_true)
        dct['Incorrect Translate'].append(count_incorrect_true)
        dct['Correct Don\'t Translate'].append(count_correct_false)
        dct['Incorrect Don\'t Translate'].append(count_incorrect_false)
        dct['Total Correct'].append(count_correct_true+count_correct_false)
        dct['Total Incorrect'].append(count_incorrect_true+count_incorrect_false)
        dct['No Output'].append(count_no_output)
        dct['Achieved Accuracy'].append((count_correct_true+count_correct_false)/(len(df_dict['q'])-count_no_output))
        dct['Achieved Sensitivity'].append(count_correct_true/(count_correct_true + count_incorrect_false))
        dct['Achieved Specificity'].append(count_incorrect_false/(count_incorrect_false+count_incorrect_true))
        total = count_correct_true + count_incorrect_true + count_correct_false + count_incorrect_false

        cf_matrix = np.array([
            [count_correct_false, count_incorrect_true],
            [count_incorrect_false, count_correct_true]
        ])

        # Names for the matrix quadrants
        group_names = ["True Negative\nActual False and Predicted False", "False Positive\nActual False and Predicted True",
                    "False Negative\nActual True Predicted False", "True Positive\nActual True Predicted True"]

        # Counts in each quadrant
        group_counts = [f"{value:0.0f}" for value in cf_matrix.flatten()]

        # Percentages for each quadrant
        group_percentages = [f"{value:.2%}" for value in cf_matrix.flatten() / total]

        # Create labels by combining the names, counts, and percentages
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.array(labels).reshape(2, 2)

        # Create the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    df = pd.DataFrame(dct)
    df.to_csv("result.csv")
                
            
        
            
            

        
    


