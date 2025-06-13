import os
import logging
import time
import psutil
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
    AutoModelForSeq2SeqLM
)

from src.utils.config import (
    DEFAULT_LLM_MODEL,
    SYSTEM_PROMPT,
    CACHE_DIR
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    
    def __init__(self, stop_strings: List[str], tokenizer):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.generated_text = ""
        self.call_count = 0
        logger.debug(f"Initialized CustomStoppingCriteria with stop strings: {stop_strings}")
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.call_count += 1
        
        # Decode the generated text
        generated = self.tokenizer.decode(input_ids[0])
        self.generated_text = generated
        logger.info("Generating text: " + generated.split("[/INST]")[1]) # TODO
        # Log every 10 calls to avoid spam
        if self.call_count % 10 == 0:
            logger.debug(f"Stopping criteria check #{self.call_count}: Current generated length: {len(generated)} chars")
        
        # Check if any stop string appears in the generated text
        for stop_string in self.stop_strings:
            if stop_string in generated:
                logger.info(f"Stopping criteria triggered by: '{stop_string}' at call #{self.call_count}")
                return True
        
        return False

class LLMManager:
    """Class for managing the LLM and generating responses."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        system_prompt: str = SYSTEM_PROMPT,
        cache_dir: str = CACHE_DIR,
        device: Optional[str] = None,
        load_in_bits: Optional[int] = None # New parameter for quantization (e.g., 4 or 8)
    ):
        """Initialize the LLM manager.
        
        Args:
            model_name: Name or path of the model to use
            system_prompt: System prompt to use for generation
            cache_dir: Directory to cache model weights
            device: Device to run the model on (auto-detected if None)
            load_in_bits: Load model in specified bits (4 or 8) for quantization. Requires bitsandbytes.
        """
        logger.info(f"Initializing LLMManager with:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Cache dir: {cache_dir}")
        logger.info(f"  - System prompt length: {len(system_prompt)} chars")
        if load_in_bits:
            logger.info(f"  - Quantization: {load_in_bits}-bit")
        else:
            logger.info(f"  - Quantization: None")

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.cache_dir = cache_dir
        self.load_in_bits = load_in_bits # Store the quantization choice
        
        # Log system information
        self._log_system_info()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug(f"Cache directory created/verified: {cache_dir}")
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            logger.info(f"Auto-detected device: {self.device}")
        else:
            self.device = device
            logger.info(f"Using specified device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        
        # Initialize the model
        self._initialize_model()
    
    def _log_system_info(self):
        """Log system information for debugging."""
        logger.info(f"System Information:")
        logger.info(f"  - Python version: {os.sys.version}")
        logger.info(f"  - PyTorch version: {torch.__version__}")
        logger.info(f"  - Available CPU cores: {psutil.cpu_count()}")
        logger.info(f"  - Available RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
        logger.info(f"  - Available disk space: {psutil.disk_usage('/').free / 1e9:.2f} GB")
        
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        start_time = time.time()
        logger.info(f"Starting model initialization: {self.model_name} on {self.device}")
        
        try:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token: # Make sure this is HUGGINGFACEHUB_API_TOKEN or consistent with your .env/Colab secrets
                logger.warning("HUGGINGFACE_TOKEN environment variable not found. Trying HUGGINGFACEHUB_API_TOKEN.")
                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            
            if not hf_token:
                 logger.error("Hugging Face token (HUGGINGFACE_TOKEN or HUGGINGFACEHUB_API_TOKEN) not found.")
                 raise ValueError("Hugging Face token not set.")
            logger.debug("Hugging Face token found.")

            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA specified but not available. Falling back to CPU.")
                self.device = "cpu"
            elif self.device == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                logger.warning("MPS specified but not available/built. Falling back to CPU.")
                self.device = "cpu"
            logger.info(f"Effective device for model and pipeline: {self.device}")

            # Log memory before loading (if CUDA or MPS)
            if self.device == "cuda":
                logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                           f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
            elif self.device == "mps":
                # Note: MPS memory tracking is less direct than CUDA.
                # torch.mps.current_allocated_memory() and torch.mps.driver_allocated_memory() can be used.
                logger.info("Checking MPS memory (note: detailed breakdown like CUDA's is not standard).")
                if hasattr(torch.mps, 'current_allocated_memory'):
                    logger.info(f"MPS current allocated memory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")


            # Load tokenizer
            tokenizer_start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                token=hf_token
            )
            if self.tokenizer.pad_token is None: # Important for some models/pipelines
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Tokenizer pad_token set to eos_token as it was None.")
            tokenizer_time = time.time() - tokenizer_start_time
            logger.info(f"Tokenizer loaded in {tokenizer_time:.2f} seconds. Vocab size: {self.tokenizer.vocab_size}")

            if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
                ModelClass = AutoModelForSeq2SeqLM
                pipeline_task = "text2text-generation"
                logger.info(f"Using AutoModelForSeq2SeqLM for {self.model_name}. Pipeline task: {pipeline_task}")
            else:
                ModelClass = AutoModelForCausalLM
                pipeline_task = "text-generation"
                logger.info(f"Using AutoModelForCausalLM for {self.model_name}. Pipeline task: {pipeline_task}")
            
            # Prepare model loading arguments
            model_load_args = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "token": hf_token,
                "torch_dtype": torch.float16 if (self.device == "cuda" or self.device == "mps") else torch.float32,
                "low_cpu_mem_usage": True if self.device != "cpu" else False
            }

            if self.load_in_bits == 8:
                model_load_args["load_in_8bit"] = True
                logger.info("Attempting to load model with 8-bit quantization.")
            elif self.load_in_bits == 4:
                model_load_args["load_in_4bit"] = True
                logger.info("Attempting to load model with 4-bit quantization.")
            # For more fine-grained 4-bit control (e.g., bnb_4bit_compute_dtype, bnb_4bit_quant_type),
            # you might need to import BitsAndBytesConfig from transformers and pass it to quantization_config.
            # However, load_in_4bit=True often uses good defaults.

            if self.device == "cuda" or self.device == "mps":
                model_load_args["device_map"] = "auto" 
            # For CPU, no device_map is typically needed; model loads to CPU by default.
            # device_map="auto" is important for bitsandbytes quantization to correctly handle device placement.
            
            logger.info(f"Loading model {self.model_name} with args: {model_load_args}")
            model_start_time = time.time()
            self.model = ModelClass.from_pretrained(self.model_name, **model_load_args)
            model_time = time.time() - model_start_time
            logger.info(f"Model loaded in {model_time:.2f} seconds.")

            if self.device == "cpu" and (not hasattr(self.model, 'hf_device_map') or not self.model.hf_device_map):
                 logger.info(f"Ensuring model is on CPU device: {self.device}")
                 self.model.to(self.device)
            elif self.device == "cuda" and model_load_args.get("device_map") != "auto":
                logger.info(f"CUDA device_map is not 'auto', model device: {self.model.device if hasattr(self.model, 'device') else 'N/A'}")
            elif self.device == "mps" and model_load_args.get("device_map") != "auto":
                logger.info(f"MPS device_map is not 'auto'. Forcing model to MPS: {self.device}")


            # Log memory after loading (if CUDA or MPS)
            if self.device == "cuda":
                logger.info(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                           f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
                logger.info(f"Model is on device: {self.model.device if hasattr(self.model, 'device') else 'N/A'}")
            elif self.device == "mps":
                logger.info(f"Model is on device: {self.model.device if hasattr(self.model, 'device') else 'N/A'}")
                if hasattr(torch.mps, 'current_allocated_memory'):
                    logger.info(f"MPS current allocated memory after load: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")

            logger.info(f"Creating Hugging Face pipeline with task: {pipeline_task}")
            pipeline_start_time = time.time()
            
            logger.info(f"Determining pipeline device argument. Model load args used: {model_load_args}")
            if model_load_args.get("device_map") == "auto":
                pipeline_device_arg = None
                logger.info("Model was loaded with device_map='auto' (e.g., for MPS/CUDA or quantization). "
                           "Pipeline 'device' argument set to None for inference by accelerate.")
            elif self.device == "cpu":
                # Model is on CPU. Pipeline infers from model, or device=None is fine.
                pipeline_device_arg = None
                logger.info("Model on CPU. Pipeline 'device' argument set to None.")
            else:
                # Fallback for explicit single-device setting when not using device_map="auto" and not on CPU.
                # This branch is less likely with the current model loading logic that prefers device_map="auto" for CUDA/MPS.
                pipeline_device_arg = self.device
                logger.info(f"Model not using device_map='auto' and not on CPU. Pipeline 'device' argument set to '{self.device}'.")
            self.pipe = pipeline(
                pipeline_task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device_arg
            )
            pipeline_time = time.time() - pipeline_start_time
            logger.info(f"Pipeline created in {pipeline_time:.2f} seconds. Pipeline device: {self.pipe.device}")

            total_time = time.time() - start_time
            logger.info(f"LLMManager initialized successfully in {total_time:.2f} seconds.")
            logger.info(f"Breakdown: Tokenizer: {tokenizer_time:.2f}s, Model: {model_time:.2f}s, Pipeline: {pipeline_time:.2f}s")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}", exc_info=True)
            raise
    
    def _format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the model.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        logger.debug(f"Formatting prompt for model type: {self.model_name}")
        logger.debug(f"Query length: {len(query)} chars")
        logger.debug(f"Context length: {len(context)} chars")
        
        # Format depends on the model type
        if "mistral" in self.model_name.lower():
            # Mistral-style prompt
            formatted_prompt = f"<s>[INST] {self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
            logger.debug("Using Mistral-style prompt formatting")
        elif "llama" in self.model_name.lower():
            formatted_prompt = f"<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\nContext:\n{context}\n\nQuestion: {query} [/INST]"
            logger.debug("Using LLaMA-style prompt formatting")
        elif "t5" in self.model_name.lower(): # Specific handling for T5
            formatted_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            logger.debug("Using T5-style prompt formatting")
        else:
            # Generic prompt (fallback, might need tuning for other model types)
            formatted_prompt = f"System: {self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            logger.debug("Using generic prompt formatting")
        
        logger.debug(f"Final prompt length: {len(formatted_prompt)} chars")
        return formatted_prompt
    
    def generate_response(
        self,
        query: str,
        context: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate a response to a query based on the given context.
        
        Args:
            query: User query
            context: Retrieved context
            max_length: Maximum *new* tokens to generate
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            
        Returns:
            Generated response
        """
        generation_start_time = time.time()
        logger.info(f"Starting response generation")
        
        # Format the prompt
        prompt_start = time.time()
        prompt = self._format_prompt(query, context)
        prompt_time = time.time() - prompt_start

        # === ENHANCED LOGGING ===
        logger.info(f"=== LLM GENERATION REQUEST ===")
        logger.info(f"QUERY: {query}")
        logger.info(f"CONTEXT LENGTH: {len(context)} chars")
        logger.info(f"CONTEXT PREVIEW: {context[:200]}{'...' if len(context) > 200 else ''}")
        logger.info(f"PROMPT LENGTH: {len(prompt)} chars")
        logger.info(f"PROMPT FORMATTING TIME: {prompt_time:.4f}s")
        logger.debug(f"FULL PROMPT:\n{prompt}")
        logger.info(f"CONTEXT:\n{context}")
        logger.info(f"GENERATION PARAMETERS:")
        logger.info(f"  - MAX_NEW_TOKENS: {max_length}")
        logger.info(f"  - TEMPERATURE: {temperature}")
        logger.info(f"  - TOP_P: {top_p}")
        logger.info(f"  - MODEL: {self.model_name}")
        logger.info(f"  - DEVICE: {self.device}")
        
        # Tokenize and log token info
        tokenize_start = time.time()
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt")
        tokenize_time = time.time() - tokenize_start

        # Log memory before generation
        if self.device == "cuda" and torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory before generation: {mem_before:.2f} GB")
        
        logger.info(f"===============================")
        # === END OF ENHANCED LOGGING ===
        
        try:
            # Define stopping criteria
            stop_strings = ["User:", "Human:", "<|endoftext|>", "</s>"]
            logger.debug(f"Using stop strings: {stop_strings}")
            stopping_criteria = StoppingCriteriaList([
                CustomStoppingCriteria(stop_strings, self.tokenizer)
            ])
            
            # Generate response
            generation_actual_start = time.time()
            logger.info(f"Starting actual text generation...")
            
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_length, 
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
            
            generation_actual_time = time.time() - generation_actual_start
            logger.info(f"Text generation completed in {generation_actual_time:.2f}s")
            
            # Log memory after generation
            if self.device == "cuda" and torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1e9
                logger.info(f"GPU memory after generation: {mem_after:.2f} GB")
                logger.info(f"Memory change: {mem_after - mem_before:.2f} GB")
            
            generated_text = outputs[0]["generated_text"]
            logger.info(f"=== GENERATION OUTPUT ===")
            logger.info(f"RAW OUTPUT LENGTH: {len(generated_text)} chars")
            logger.info(f"RAW OUTPUT:\\n{generated_text}")
            logger.info(f"========================")

            # Enhanced response extraction with detailed logging
            extraction_start = time.time()
            logger.info(f"Starting response extraction...")
            
            response = ""
            extraction_method = "unknown"
            
            # For Mistral/Llama, prompt ends with [/INST] and is part of generated_text
            if self.model_name.startswith("mistralai/") or "llama" in self.model_name.lower():
                extraction_method = "mistral/llama"
                logger.debug(f"Using {extraction_method} extraction method")
                # Check if the original prompt is part of the generated_text
                # A common pattern is that generated_text = prompt + answer
                inst_token = "[/INST]"
                # Ensure we're splitting based on the prompt's structure, not just any [/INST]
                # if generated_text.startswith(prompt): # This might be too strict if tokenizer differences exist
                # A more robust way for these models is to find the [/INST] that signifies the end of the prompt section
                
                # Find the position of the end of the prompt within the generated_text.
                # The prompt itself contains "[/INST]". We assume the LLM's output starts after that.
                prompt_inst_index = prompt.rfind(inst_token) # Find last [/INST] in the original prompt string
                
                if prompt_inst_index != -1:
                    # Expected end of prompt part in generated_text
                    expected_prompt_end_in_output = prompt_inst_index + len(inst_token)
                    
                    # Check if generated_text is long enough and if the initial part matches the prompt
                    if len(generated_text) > expected_prompt_end_in_output and \
                       generated_text[:expected_prompt_end_in_output] == prompt[:expected_prompt_end_in_output]:
                        response = generated_text[expected_prompt_end_in_output:].strip()
                        logger.debug(f"Removed Mistral/Llama prompt prefix based on exact match up to and including [/INST].")
                    else:
                        # Fallback: if exact prompt not found at start, try splitting by [/INST]
                        # This handles cases where model might not perfectly echo the input or has subtle token differences
                        parts = generated_text.split(inst_token, 1)
                        if len(parts) > 1:
                            response = parts[1].strip()
                            logger.debug(f"Removed Mistral/Llama prompt prefix by splitting on first occurrence of [/INST].")
                        else:
                            response = generated_text.strip() # No [/INST] found in output, take all
                            logger.warning(f"No [/INST] marker found in generated_text for Mistral/Llama. Using full output.")
                else:
                    # This case should ideally not be reached if prompts are correctly formatted for Mistral/Llama
                    response = generated_text.strip() 
                    logger.warning(f"Original prompt for Mistral/Llama did not contain [/INST]. Using full output.")

            elif "t5" in self.model_name.lower():
                extraction_method = "t5"
                logger.debug(f"Using {extraction_method} extraction method")
                # T5 models from `text2text-generation` pipeline usually output only the answer
                response = generated_text.strip() 
            else: 
                extraction_method = "generic"
                logger.debug(f"Using {extraction_method} (generic) extraction method by stripping prompt length.")
                # Generic fallback: attempt to strip original prompt string by its character length
                # This is the least robust method and prone to issues if tokenization alters perceived length.
                prompt_char_len = len(prompt)
                if len(generated_text) > prompt_char_len and generated_text.startswith(prompt[:100]): # Check first 100 chars of prompt
                    response = generated_text[prompt_char_len:].strip()
                else: 
                    response = generated_text.strip()
            
            extraction_time = time.time() - extraction_start
            
            # Remove any remaining stop strings from the response
            original_response_before_stop_string_cleaning = response
            original_response_length = len(response)

            for stop_string in stop_strings:
                if stop_string in response: # Check if stop_string is anywhere
                    response_before_split = response
                    response = response.split(stop_string)[0].strip()
                    if response != response_before_split:
                        logger.debug(f"Cleaned stop string '{stop_string}'. Response changed from '{response_before_split[:50]}...' to '{response[:50]}...'")
            
            if len(response) < original_response_length:
                 logger.info(f"Response length after stop string cleaning: {len(response)} (was {original_response_length})")
            
            total_generation_time = time.time() - generation_start_time
            
            logger.info(f"=== FINAL RESPONSE ===")
            logger.info(f"EXTRACTION METHOD: {extraction_method}")
            logger.info(f"EXTRACTION TIME: {extraction_time:.4f}s")
            logger.info(f"RESPONSE LENGTH: {len(response)} chars (was {original_response_length})")
            logger.info(f"TOTAL GENERATION TIME: {total_generation_time:.2f}s")
            logger.info(f"  - Prompt formatting: {prompt_time:.4f}s")
            logger.info(f"  - Input tokenization: {tokenize_time:.4f}s") 
            logger.info(f"  - Actual generation: {generation_actual_time:.2f}s")
            logger.info(f"  - Response extraction: {extraction_time:.4f}s")
            logger.info(f"TOKENS/SECOND: {input_tokens.shape[1] / generation_actual_time:.1f}")
            logger.info(f"FINAL RESPONSE:\n{response}")
            logger.info(f"=====================")
            
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            logger.error(f"Generation failed after {time.time() - generation_start_time:.2f}s")
            return "Sorry, I couldn't generate a response at this time."
