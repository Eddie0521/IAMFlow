import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EntityStruct:

    entity: str
    attrs: List[str] = field(default_factory=list)
    global_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {"entity": self.entity, "attrs": self.attrs, "global_id": self.global_id}

    @classmethod
    def from_dict(cls, data: Dict) -> "EntityStruct":
        return cls(
            entity=data.get("entity", ""),
            attrs=data.get("attrs", []),
            global_id=data.get("global_id"),
        )


class LLMWrapper:

    def __init__(
        self,
        model_path: str = "../Qwen3-4B-Instruct-2507",
        device: str = None,
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.2,
        backend: Optional[str] = None,
        device_id: Optional[int] = None,
    ):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._device = device
        self._backend = str(backend or ("vllm" if use_vllm else "hf"))
        self._use_vllm = self._backend == "vllm"
        self._sampling_params = None
        self._gpu_memory_utilization = gpu_memory_utilization
        self._device_id = device_id

    def preload(self):
        self._load_model()
        if self._use_vllm and self._model is not None:
            print("[LLMWrapper] Running warmup inference...")
            warmup_prompt = "Hello"
            try:
                text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": warmup_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                sampling_params = self._sampling_params(max_tokens=1, temperature=0.01)
                _ = self._model.generate([text], sampling_params)
                print("[LLMWrapper] Warmup complete")
            except Exception as e:
                print(f"[LLMWrapper] Warmup failed (non-critical): {e}")

    def _load_model(self):
        if self._model is not None:
            return

        if self._use_vllm:
            self._load_vllm_model()
        else:
            self._load_hf_model()

    def _load_vllm_model(self):
        print(f"[LLMWrapper] Loading model with vLLM from {self.model_path}")
        print(f"[LLMWrapper] gpu_memory_utilization={self._gpu_memory_utilization}")
        if self._device_id is not None:
            print(f"[LLMWrapper] Pinning vLLM worker to GPU {self._device_id}")

        # Pin vLLM before importing it. vLLM inspects visible devices at import
        # and engine construction time, so changing CUDA_VISIBLE_DEVICES after
        # `import vllm` is too late on ROCm.
        saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self._device_id is not None:
            if saved_cvd is not None:
                physical_ids = [x.strip() for x in saved_cvd.split(",")]
                if self._device_id < len(physical_ids):
                    target_gpu = physical_ids[self._device_id]
                else:
                    target_gpu = str(self._device_id)
            else:
                target_gpu = str(self._device_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu

        try:
            from vllm import LLM, SamplingParams

            try:
                self._model = LLM(
                    model=self.model_path,
                    trust_remote_code=True,
                    dtype="bfloat16",
                    gpu_memory_utilization=self._gpu_memory_utilization,
                    max_model_len=1024,
                    enforce_eager=True,
                )
            finally:
                pass

            self._tokenizer = self._model.get_tokenizer()
            self._sampling_params = SamplingParams

            print(f"[LLMWrapper] vLLM model loaded successfully")

        except ImportError:
            print(f"[LLMWrapper] vLLM not installed, falling back to HuggingFace")
            self._use_vllm = False
            self._load_hf_model()
        finally:
            # Restore so subsequent CUDA operations see all GPUs.
            if self._device_id is not None:
                if saved_cvd is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def _load_hf_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._device is None:
            if self._device_id is not None and torch.cuda.is_available():
                self._device = f"cuda:{self._device_id}"
            else:
                self._device = (
                    "mps"
                    if torch.backends.mps.is_available()
                    else "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

        print(
            f"[LLMWrapper] Loading model with HuggingFace from {self.model_path} on {self._device}"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        is_cuda = str(self._device).startswith("cuda")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if is_cuda else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        if self._device != "cuda":
            # Specific device (e.g. "cuda:1") or non-CUDA -- move explicitly
            self._model = self._model.to(self._device)

        print(f"[LLMWrapper] HuggingFace model loaded successfully")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        self._load_model()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if self._use_vllm:
            return self._generate_vllm(text, max_new_tokens, temperature)
        else:
            return self._generate_hf(text, max_new_tokens, temperature)

    def _generate_vllm(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        sampling_params = self._sampling_params(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9 if temperature > 0 else 1.0,
        )

        outputs = self._model.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response

    def _generate_hf(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        model_inputs = self._tokenizer([prompt], return_tensors="pt").to(
            self._model.device
        )

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        response = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return response


class EntityStructExtractor:

    SYSTEM_PROMPT = """Extract human characters from the video prompt.

RULES:
1. "entities": ONLY human/person characters (man, woman, protagonist, etc.)
   - Extract ONLY visual/physical attributes: hair, clothing, accessories, body type, age, skin, facial features
   - DO NOT extract behavioral states (walking, nodding, reading, sitting) or emotions (quiet, contemplative, happy)
   - Keep entity names short

OUTPUT FORMAT (JSON object only, no explanation):
{"entities": [{"entity": "<name>", "attrs": ["<attr1>", "<attr2>"]}]}

If no humans found, entities should be []."""

    def __init__(
        self,
        llm: Optional[LLMWrapper] = None,
        model_path: str = "../Qwen3-4B-Instruct-2507",
    ):
        self.llm = llm or LLMWrapper(model_path)

    def extract(self, prompt: str) -> List[EntityStruct]:
        response = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=prompt,
            max_new_tokens=1024,
            temperature=0,
        )

        entities_data = self._parse_response(response)
        entities = [
            EntityStruct(
                entity=e.get("entity", ""), attrs=e.get("attrs", []), global_id=None
            )
            for e in entities_data
        ]
        return entities

    def _parse_response(self, response: str) -> List[Dict]:
        try:
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)
            response = response.strip()

            obj_start = response.find("{")
            obj_end = response.rfind("}")
            if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
                json_str = response[obj_start : obj_end + 1]
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "entities" in data:
                        entities = data.get("entities", [])
                        return entities if isinstance(entities, list) else []
                except json.JSONDecodeError:
                    pass

            arr_start = response.find("[")
            arr_end = response.rfind("]")
            if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
                json_str = response[arr_start : arr_end + 1]
                try:
                    arr = json.loads(json_str)
                    if isinstance(arr, list):
                        return arr
                except json.JSONDecodeError:
                    return self._extract_entities_fallback(json_str)

            parsed = json.loads(response)
            if isinstance(parsed, list):
                return parsed

            if isinstance(parsed, dict):
                if "entities" in parsed:
                    entities = parsed.get("entities", [])
                    if not isinstance(entities, list):
                        entities = []
                    return entities

                if "entity" in parsed:
                    return [
                        {
                            "entity": parsed.get("entity", ""),
                            "attrs": parsed.get("attrs", [])
                            if isinstance(parsed.get("attrs", []), list)
                            else [],
                        }
                    ]

            return self._extract_entities_fallback(response)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON, raw response: {response[:500]}...")
            return self._extract_entities_fallback(response)

    def _extract_entities_fallback(self, text: str) -> List[Dict]:
        entities = []

        entity_pattern = r'"entity"\s*:\s*"([^"]+)"'
        attrs_pattern = r'"attrs"\s*:\s*\[([^\]]*)\]'

        entity_matches = list(re.finditer(entity_pattern, text))
        attrs_matches = list(re.finditer(attrs_pattern, text))

        for i, entity_match in enumerate(entity_matches):
            entity_name = entity_match.group(1)

            entity_pos = entity_match.end()
            best_attrs = []

            for attrs_match in attrs_matches:
                if attrs_match.start() > entity_pos:
                    attrs_str = attrs_match.group(1)
                    try:
                        attrs_list = json.loads(f"[{attrs_str}]")
                        best_attrs = [str(a) for a in attrs_list if a]
                    except:
                        best_attrs = re.findall(r'"([^"]+)"', attrs_str)
                    break

            if entity_name and entity_name not in ["<name>", "name"]:
                entities.append({"entity": entity_name, "attrs": best_attrs})

        return entities


class GlobalIDManager:

    MATCHING_SYSTEM_PROMPT = """Match a new character to existing characters.

TASK: Given a new character description and existing character registry, determine if they refer to the same person.

MATCHING RULES:
1. Words like "protagonist", "main character", "he", "she" usually refer to previously introduced characters
2. Matching clothing or appearance attributes indicates the same person
3. Words like "another", "other", "new", "different" indicate a NEW person - return null

OUTPUT FORMAT (JSON only, no explanation):
{"matched_id": <number or null>}"""

    NEW_ENTITY_MARKERS = ["another", "other", "new", "different", "second", "third"]

    def __init__(
        self,
        llm: Optional[LLMWrapper] = None,
        model_path: str = "../Qwen3-4B-Instruct-2507",
    ):
        self.llm = llm or LLMWrapper(model_path)
        self._next_id = 1

    def assign_ids(
        self,
        entities: List[EntityStruct],
        global_registry: Dict[str, Dict],
        is_first_prompt: bool,
    ) -> List[EntityStruct]:
        if global_registry:
            max_id = max(int(k) for k in global_registry.keys())
            self._next_id = max(self._next_id, max_id + 1)

        if is_first_prompt:
            for entity in entities:
                entity.global_id = self._allocate_new_id()
        else:
            for entity in entities:
                matched_id = self._match_or_allocate(entity, global_registry)
                entity.global_id = matched_id

        return entities

    def _allocate_new_id(self) -> int:
        new_id = self._next_id
        self._next_id += 1
        return new_id

    def _match_or_allocate(
        self, entity: EntityStruct, global_registry: Dict[str, Dict]
    ) -> int:
        entity_lower = entity.entity.lower()

        is_explicitly_new = any(
            marker in entity_lower for marker in self.NEW_ENTITY_MARKERS
        )
        if is_explicitly_new:
            return self._allocate_new_id()

        if not global_registry:
            return self._allocate_new_id()

        entity_desc = (
            f"{entity.entity}: {', '.join(entity.attrs)}"
            if entity.attrs
            else entity.entity
        )

        registry_info = self._format_registry_for_llm(global_registry)

        user_prompt = f"""New character description:
"{entity_desc}"

Existing characters:
{registry_info}

Does the new character match any existing one? If yes, return the ID. If no, return null.
Output JSON only: {{"matched_id": <number or null>}}"""

        response = self.llm.generate(
            system_prompt=self.MATCHING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=256,
            temperature=0,
        )

        matched_id = self._parse_matching_response(response)

        if matched_id is not None and str(matched_id) in global_registry:
            return matched_id
        else:
            return self._allocate_new_id()

    def _format_registry_for_llm(self, global_registry: Dict[str, Dict]) -> str:
        lines = []
        for gid, info in global_registry.items():
            entities = info.get("all_entities", [])
            attrs = info.get("all_attrs", [])
            entity_names = "/".join(entities)
            attrs_str = ", ".join(attrs) if attrs else "no attributes"
            lines.append(f"ID {gid}: {entity_names}: {attrs_str}")
        return "\n".join(lines)

    def _parse_matching_response(self, response: str) -> Optional[int]:
        try:
            response = re.sub(r"```json\s*", "", response)
            response = re.sub(r"```\s*", "", response)
            response = response.strip()

            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = response[start : end + 1]
                data = json.loads(json_str)
                matched_id = data.get("matched_id")
                if matched_id is not None:
                    return int(matched_id)
            return None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


class LLMAgent:

    def __init__(
        self,
        model_path: str = "../Qwen3-4B-Instruct-2507",
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.2,
        backend: Optional[str] = None,
        device_id: Optional[int] = None,
    ):
        self.llm = LLMWrapper(
            model_path,
            use_vllm=use_vllm,
            gpu_memory_utilization=gpu_memory_utilization,
            backend=backend,
            device_id=device_id,
        )
        self.extractor = EntityStructExtractor(llm=self.llm)
        self.id_manager = GlobalIDManager(llm=self.llm)

    def preload(self):
        print("[LLMAgent] Preloading LLM model...")
        self.llm.preload()
        print("[LLMAgent] LLM model preloaded")

    def process_prompt(
        self, prompt: str, prompt_id: int, global_registry: Dict[str, Dict]
    ) -> Tuple[List[EntityStruct], Dict[str, Any]]:
        entities = self.extractor.extract(prompt)

        if not entities:
            return [], {}

        is_first_prompt = prompt_id == 1
        entities = self.id_manager.assign_ids(
            entities, global_registry, is_first_prompt
        )

        registry_update = self._build_registry_update(
            entities, prompt_id, global_registry
        )

        return entities, registry_update

    def _build_registry_update(
        self,
        entities: List[EntityStruct],
        prompt_id: int,
        existing_registry: Dict[str, Dict],
    ) -> Dict[str, Any]:
        update = {}

        for entity in entities:
            gid = str(entity.global_id)

            if gid in existing_registry:
                update[gid] = {
                    "action": "update",
                    "new_entity": entity.entity,
                    "new_attrs": entity.attrs,
                    "prompt_id": prompt_id,
                }
            else:
                entity_type = self._infer_entity_type(entity.entity)
                type_count = sum(
                    1
                    for k, v in existing_registry.items()
                    if v.get("name", "").startswith(entity_type)
                )
                type_count += sum(
                    1
                    for k, v in update.items()
                    if v.get("name", "").startswith(entity_type)
                )

                update[gid] = {
                    "action": "create",
                    "name": f"{entity_type}_{type_count + 1}",
                    "all_entities": [entity.entity],
                    "all_attrs": entity.attrs.copy(),
                    "instances": [
                        {
                            "prompt_id": prompt_id,
                            "entity": entity.entity,
                            "attrs": entity.attrs.copy(),
                        }
                    ],
                }

        return update

    def _infer_entity_type(self, entity_name: str) -> str:
        entity_lower = entity_name.lower()
        if any(w in entity_lower for w in ["woman", "girl", "lady", "female", "she"]):
            return "woman"
        elif any(
            w in entity_lower
            for w in ["man", "boy", "guy", "male", "he", "protagonist"]
        ):
            return "man"
        else:
            return "person"
