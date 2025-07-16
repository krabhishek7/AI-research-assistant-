"""Chat service for AI assistant functionality."""

import logging
from typing import Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling AI chat interactions."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = "microsoft/DialoGPT-small"
        self.initialized = False
    
    def _initialize_model(self):
        """Initialize the chat model."""
        if self.initialized:
            return
        
        try:
            logger.info(f"Loading chat model: {self.model_name}")
            
            # Try to use a text generation pipeline first (simpler)
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model="gpt2",
                    tokenizer="gpt2",
                    max_length=512,
                    truncation=True,
                    pad_token_id=50256
                )
                logger.info("Chat pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load pipeline: {e}")
                # Fallback to basic model
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.model = AutoModelForCausalLM.from_pretrained("gpt2")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Chat model loaded successfully")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error loading chat model: {str(e)}")
            raise
    
    def generate_response(self, message: str, context: Optional[str] = None, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate a response to a user message."""
        try:
            self._initialize_model()
            
            # Prepare the prompt
            if context:
                prompt = f"Research context: {context[:200]}...\n\nQuestion: {message}\nAnswer:"
            else:
                prompt = f"Research question: {message}\nAnswer:"
            
            # Generate response using pipeline if available
            if self.pipeline:
                try:
                    response = self.pipeline(
                        prompt,
                        max_length=len(prompt.split()) + max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.pipeline.tokenizer.pad_token_id
                    )
                    
                    generated_text = response[0]['generated_text']
                    
                    # Extract only the answer part
                    if "Answer:" in generated_text:
                        answer = generated_text.split("Answer:")[-1].strip()
                    else:
                        answer = generated_text.replace(prompt, "").strip()
                    
                    return answer if len(answer) > 10 else self._get_fallback_response(message)
                    
                except Exception as e:
                    logger.warning(f"Pipeline generation failed: {e}")
                    return self._get_fallback_response(message)
            
            # Fallback to model generation
            elif self.model and self.tokenizer:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=400)
                
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + min(max_length, 150),
                        temperature=temperature,
                        pad_token_id=self.tokenizer.pad_token_id,
                        do_sample=True,
                        top_p=0.9
                    )
                
                response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the answer part
                if "Answer:" in response_text:
                    answer = response_text.split("Answer:")[-1].strip()
                else:
                    answer = response_text.replace(prompt, "").strip()
                
                return answer if len(answer) > 10 else self._get_fallback_response(message)
            
            else:
                return self._get_fallback_response(message)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._get_fallback_response(message)
    
    def _get_fallback_response(self, message: str) -> str:
        """Get a fallback response based on the message content."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['quantum', 'physics', 'particle']):
            return "Quantum physics is a fascinating field that deals with the behavior of matter and energy at the atomic and subatomic scale. The fundamental principles include wave-particle duality, uncertainty principle, and quantum entanglement. Would you like me to explain any specific quantum concepts?"
        
        elif any(word in message_lower for word in ['machine learning', 'ai', 'neural', 'deep learning']):
            return "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Common techniques include supervised learning, unsupervised learning, and reinforcement learning. Neural networks are a key component of deep learning systems. What specific aspect would you like to explore?"
        
        elif any(word in message_lower for word in ['research', 'paper', 'study', 'methodology']):
            return "Research methodology is crucial for conducting valid scientific studies. Key components include defining research questions, choosing appropriate methods, collecting and analyzing data, and drawing valid conclusions. Different fields may use quantitative, qualitative, or mixed methods approaches. What research area are you interested in?"
        
        elif any(word in message_lower for word in ['citation', 'reference', 'bibliography']):
            return "Proper citation is essential in academic writing to give credit to original sources and allow readers to verify information. Common citation styles include APA, MLA, Chicago, and IEEE. Each style has specific formatting requirements for different types of sources. What citation style are you working with?"
        
        elif any(word in message_lower for word in ['summary', 'summarize', 'abstract']):
            return "Summarization involves condensing key information from longer texts while preserving essential meaning. Good summaries include main findings, methodology, and conclusions. Academic abstracts typically follow a structured format: background, methods, results, and conclusions. What type of content would you like help summarizing?"
        
        else:
            return "I'm here to help with your research questions! I can assist with understanding complex concepts, methodology, citations, and paper analysis. Could you provide more specific details about what you'd like to explore?"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the chat model."""
        return {
            "model_name": self.model_name,
            "initialized": self.initialized,
            "has_pipeline": self.pipeline is not None,
            "has_model": self.model is not None,
            "device": "cpu"
        }

# Global chat service instance
_chat_service = None

def get_chat_service() -> ChatService:
    """Get or create the chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service 