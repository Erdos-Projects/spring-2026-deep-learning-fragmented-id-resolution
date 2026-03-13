import json
import uuid
import pandas as pd
import numpy as np
import os
from getpass import getpass
import time
import torch

# Import HuggingFace libraries
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("ERROR: Please install transformers and torch")
    print("Run: pip install transformers torch")
    raise


class LLM_augment_Local:
    """
    LLM augmentation using local HuggingFace models
    Loads model locally and runs inference on your device
    """
    
    # Default prompts (same as your original)
    DEFAULT_HARD_POSITIVE_PROMPT = (
        "You generate realistic variations of a DUPLICATE customer record for training a duplicate detection model.\n"
        "These are variations of the SAME PERSON with realistic data entry errors, address changes, or name variations.\n"
        "The user will send a JSON object with keys 'base_record' and 'n_variants'. "
        "'base_record' is one record; 'n_variants' is an integer. "
        "You MUST return exactly n_variants JSON records in a list.\n\n"
        "Constraints:\n"
        "- Each output record MUST represent the SAME person as base_record (it's a duplicate with variations).\n"
        "- Make realistic changes that would occur in duplicate records:\n"
        "  * Minor typos in names (e.g., 'Micheal' instead of 'Michael')\n"
        "  * Informal/short names (e.g., 'Bob' instead of 'Robert', 'Bill' instead of 'William')\n"
        "  * Slightly different addresses (moved to nearby address, address typos)\n"
        "  * Missing middle name/initial, or added middle initial\n"
        "  * Different address formatting (e.g., 'St.' vs 'Street')\n"
        "  * You MAY change birth_year by ±1 (data entry errors)\n"
        "- DO NOT drastically change the person's identity.\n"
        "Return ONLY a JSON list of exactly n_variants records, each with exactly the same keys as base_record."
    )

    DEFAULT_HARD_NEGATIVE_PROMPT = (
        "You generate HARD NEGATIVE examples - records of different people who look SIMILAR to the base record "
        "but are NOT the same person. These are confusing cases that a duplicate detection model should learn to distinguish.\n"
        "The user will send a JSON object with keys 'base_record' and 'n_variants'. "
        "'base_record' is one record; 'n_variants' is an integer. "
        "You MUST return exactly n_variants JSON records in a list.\n\n"
        "Constraints:\n"
        "- Each output record MUST be a DIFFERENT person from base_record (not a duplicate).\n"
        "- Generate confusing cases that look similar but aren't the same person:\n"
        "  * Twins or siblings: same last name, same address, but different first names\n"
        "  * Family members: same last name, mostly same address, but different age/birth year\n"
        "  * Name variations without identity match (e.g., 'John Smith Sr.' vs 'John Smith Jr.' - different people)\n"
        "  * Same/similar names but completely different addresses or ages\n"
        "  * Similar names (like 'James' and 'John') but different last names\n"
        "- These should be plausible hard negative cases that a naive model might confuse with duplicates.\n"
        "- ALWAYS keep at least one significant difference (name or address) that shows they're different people.\n"
        "Return ONLY a JSON list of exactly n_variants records, each with exactly the same keys as base_record."
    )

    def __init__(self, model_name: str = None, use_gpu: bool = True):
        """
        Initialize with local HuggingFace model
        
        Args:
            model_name: Model to use. Default: 'microsoft/Phi-3.5-mini-instruct'
                       Other options:
                       - 'HuggingFaceH4/zephyr-7b-beta' (~7B params, ~14GB)
                       - 'mistralai/Mistral-7B-Instruct-v0.3' (~7B params, ~14GB)
            
            use_gpu: Whether to use GPU (True) or CPU (False)
        """
        if model_name is None:
            model_name = 'microsoft/Phi-3.5-mini-instruct'
        
        # Set HuggingFace cache directory to temp location
        cache_dir = os.path.join(os.path.expanduser("~"), "AppData", "Local", "huggingface", "hub")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = cache_dir
        
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        print(f"Loading model: {model_name}...")
        print(f"Cache directory: {cache_dir}")
        print(f"Device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                cache_dir=cache_dir
            )
            self.model.to(self.device)
            
            print(f"✓ Model loaded successfully!")
            print(f"✓ Device: {self.device}")
        except Exception as e:
            print(f"ERROR loading model: {str(e)[:200]}")
            raise

    def generate_response(self, system_prompt: str, user_payload: dict,
                         max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generate response using local model"""
        
        user_prompt = json.dumps(user_payload, ensure_ascii=False)
        
        full_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{user_prompt}
<|im_end|>
<|im_start|>assistant
"""
        
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean response
            if "<|im_start|>assistant" in text:
                text = text.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in text:
                text = text.split("<|im_end|>")[0].strip()
            
            return text
        
        except Exception as e:
            print(f"  Model Error: {str(e)[:150]}")
            return ""

    def call_llm_for_variants(self, record: dict, augmentation_type: str,
                             system_prompt: str = None, n_variants: int = 1) -> list:
        """
        Generate variants of a record
        
        Args:
            record: Base record dict
            augmentation_type: "positive" (duplicates) or "negative" (non-duplicates)
            system_prompt: Custom prompt (optional)
            n_variants: Number of variants
        
        Returns:
            List of variant dicts
        """
        if augmentation_type not in ("positive", "negative"):
            raise ValueError("augmentation_type must be 'positive' or 'negative'")
        
        # Use default prompts
        if system_prompt is None:
            system_prompt = (self.DEFAULT_HARD_POSITIVE_PROMPT if augmentation_type == "positive"
                           else self.DEFAULT_HARD_NEGATIVE_PROMPT)
        
        user_payload = {"base_record": record, "n_variants": n_variants}
        response_text = self.generate_response(system_prompt, user_payload)

        # Parse JSON
        try:
            variants = json.loads(response_text)
        except json.JSONDecodeError:
            try:
                start = response_text.index("[")
                end = response_text.rindex("]") + 1
                variants = json.loads(response_text[start:end])
            except:
                print(f"  Failed to parse: {response_text[:200]}")
                return []

        # Clean variants
        clean_variants = []
        for v in variants:
            if not isinstance(v, dict):
                continue
            new_v = {key: v.get(key, record[key]) for key in record.keys()}
            clean_variants.append(new_v)

        return clean_variants[:n_variants]

    def perturb_dataframe_with_llm(self, df_base: pd.DataFrame, augmentation_type: str,
                                   system_prompt: str = None, frac: float = 0.01,
                                   random_state: int = 42) -> tuple:
        """
        Generate synthetic variants for a dataframe
        
        Returns: (df_seed, df_syn, df_aug, pair_mapping)
        """
        if augmentation_type not in ("positive", "negative"):
            raise ValueError("augmentation_type must be 'positive' or 'negative'")
        
        df_base = df_base.copy()
        if 'id' not in df_base.columns:
            df_base['id'] = [str(uuid.uuid4()) for _ in range(len(df_base))]
        
        df_seed = df_base.sample(frac=frac, random_state=random_state).reset_index(drop=True).copy()
        synthetic_rows = []
        pair_mapping = {}

        print(f"\nGenerating {augmentation_type} variants for {len(df_seed)} records...")
        
        for i, row in df_seed.iterrows():
            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(df_seed)}", end='\r')
            
            base_record = {
                "first_name": row["first_name"],
                "middle_name": row["middle_name"] if pd.notna(row["middle_name"]) else "",
                "last_name": row["last_name"],
                "name_suffix_lbl": row["name_suffix_lbl"] if pd.notna(row["name_suffix_lbl"]) else "",
                "res_street_address": row["res_street_address"],
                "res_city_desc": row["res_city_desc"],
                "zip_code": str(row["zip_code"]).split(".")[0] if pd.notna(row["zip_code"]) else "",
                "birth_year": int(row["birth_year"]) if pd.notna(row["birth_year"]) else None,
            }

            variants = self.call_llm_for_variants(base_record, augmentation_type, system_prompt, n_variants=1)

            for v in variants:
                new_row = row.to_dict()
                new_row.update(v)
                
                synthetic_id = str(uuid.uuid4())
                new_row["id"] = synthetic_id
                
                pair_mapping[row["id"]] = {
                    "synthetic_id": synthetic_id,
                    "augmentation_type": augmentation_type
                }

                new_row["label"] = 1 if augmentation_type == "positive" else 0
                new_row["source"] = f"synthetic_hard_{augmentation_type}"
                synthetic_rows.append(new_row)
        
        print(f"\n✓ Generated {len(synthetic_rows)} variants")

        df_syn = pd.DataFrame(synthetic_rows)
        df_seed = df_seed.copy()
        if "label" not in df_seed.columns:
            df_seed["label"] = 0
        if "source" not in df_seed.columns:
            df_seed["source"] = "real"

        df_aug = pd.concat([df_seed, df_syn], ignore_index=True)
        return df_seed, df_syn, df_aug, pair_mapping

    def create_pair_dataframes(self, df_augmented: pd.DataFrame, pair_mapping: dict) -> tuple:
        """Create paired dataframes"""
        duplicates = []
        non_duplicates = []
        
        for original_id, mapping_info in pair_mapping.items():
            synthetic_id = mapping_info["synthetic_id"]
            augmentation_type = mapping_info["augmentation_type"]
            
            orig = df_augmented[df_augmented["id"] == original_id]
            synth = df_augmented[df_augmented["id"] == synthetic_id]
            
            if orig.empty or synth.empty:
                continue
            
            orig = orig.iloc[0].to_dict()
            synth = synth.iloc[0].to_dict()
            
            pair_row = {}
            for col in orig.keys():
                if col != "id":
                    pair_row[f"record_1_{col}"] = orig[col]
            pair_row["record_1_id"] = original_id
            
            for col in synth.keys():
                if col != "id":
                    pair_row[f"record_2_{col}"] = synth[col]
            pair_row["record_2_id"] = synthetic_id
            
            pair_row["label"] = 1 if augmentation_type == "positive" else 0
            
            if augmentation_type == "positive":
                duplicates.append(pair_row)
            else:
                non_duplicates.append(pair_row)
        
        return (pd.DataFrame(duplicates) if duplicates else pd.DataFrame(),
                pd.DataFrame(non_duplicates) if non_duplicates else pd.DataFrame())


if __name__ == "__main__":
    print("Testing LLM_augment_Local with local models...")
    
    augmenter = LLM_augment_Local()
    
    test_record = {
        "first_name": "John",
        "middle_name": "A",
        "last_name": "Smith",
        "name_suffix_lbl": "",
        "res_street_address": "123 Main Street",
        "res_city_desc": "Springfield",
        "zip_code": "12345",
        "birth_year": 1980
    }
    
    print("\n[TEST] Generating hard positive...")
    pos = augmenter.call_llm_for_variants(test_record, "positive", n_variants=1)
    print("✓ Result:", pos)
    
    print("\n[TEST] Generating hard negative...")
    neg = augmenter.call_llm_for_variants(test_record, "negative", n_variants=1)

    print("✓ Result:", neg)
