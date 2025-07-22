"""
Interface Hugging Face pour remplacer Ollama dans VulnRAG
"""

import logging
import time
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

logger = logging.getLogger(__name__)

class HuggingFaceInterface:
    """Interface pour les modèles Hugging Face"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        use_4bit: bool = True,
        use_8bit: bool = False,
        max_length: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.cache_dir = cache_dir
        
        # Configuration de quantification
        self.quantization_config = None
        if use_4bit:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif use_8bit:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Initialisation différée
        self.tokenizer = None
        self.model = None
        self._initialized = False
    
    def _initialize_model(self):
        """Initialise le modèle et le tokenizer"""
        if self._initialized:
            return
        
        logger.info(f"Initializing Hugging Face model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            
            # Ajouter le pad token si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Charger le modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                quantization_config=self.quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.quantization_config else None,
            )
            
            # Mettre le modèle en mode évaluation
            self.model.eval()
            
            self._initialized = True
            logger.info(f"Model initialized in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Génère du texte avec le modèle Hugging Face
        
        Args:
            prompt: Le prompt d'entrée
            max_new_tokens: Nombre maximum de nouveaux tokens
            temperature: Température pour la génération
            top_p: Top-p sampling
            do_sample: Si True, utilise le sampling stochastique
            
        Returns:
            Le texte généré
        """
        self._initialize_model()
        
        # Paramètres de génération
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature or self.temperature,
            "top_p": top_p or self.top_p,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        
        try:
            # Tokeniser le prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - max_new_tokens,
            )
            
            # Déplacer vers le bon device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Générer
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            
            # Décoder la réponse
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error: {str(e)}"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Interface de chat compatible avec l'API Ollama
        
        Args:
            messages: Liste de messages au format [{"role": "user", "content": "..."}]
            max_new_tokens: Nombre maximum de nouveaux tokens
            temperature: Température pour la génération
            
        Returns:
            La réponse du modèle
        """
        # Construire le prompt à partir des messages
        prompt = self._build_chat_prompt(messages)
        
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Construit un prompt de chat à partir des messages
        
        Args:
            messages: Liste de messages
            
        Returns:
            Le prompt formaté
        """
        if not messages:
            return ""
        
        # Format spécifique pour Qwen2.5
        if "qwen" in self.model_name.lower():
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            
            prompt += "<|im_start|>assistant\n"
            return prompt
        
        # Format générique pour d'autres modèles
        else:
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            prompt += "Assistant: "
            return prompt

# Fonction de compatibilité avec l'ancienne API Ollama
def generate(
    prompt: str,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    max_tokens: int = 512,
    temperature: float = 0.1,
    **kwargs
) -> str:
    """
    Fonction de compatibilité avec l'ancienne API Ollama
    
    Args:
        prompt: Le prompt d'entrée
        model: Nom du modèle Hugging Face
        max_tokens: Nombre maximum de tokens
        temperature: Température pour la génération
        
    Returns:
        Le texte généré
    """
    interface = HuggingFaceInterface(
        model_name=model,
        temperature=temperature,
    )
    
    return interface.generate(
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

# Mapping des modèles Ollama vers Hugging Face
MODEL_MAPPING = {
    "qwen2.5-coder:latest": "Qwen/Qwen2.5-7B-Instruct",
    "kirito1/qwen3-coder:latest": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}

def get_hf_model_name(ollama_model: str) -> str:
    """
    Convertit un nom de modèle Ollama en nom de modèle Hugging Face
    
    Args:
        ollama_model: Nom du modèle Ollama
        
    Returns:
        Nom du modèle Hugging Face
    """
    return MODEL_MAPPING.get(ollama_model, ollama_model) 