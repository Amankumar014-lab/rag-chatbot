"""
LLM Handler Module for RAG Pipeline
Uses Mistral 7B Instruct for generating task-oriented responses
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, Any, Optional


class LLMHandler:
    """Handle LLM inference with Mistral 7B Instruct"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit: bool = True,
        device: str = "auto"
    ):
        """
        Initialize LLM handler
        
        Args:
            model_name: HuggingFace model name
            load_in_4bit: Use 4-bit quantization for memory efficiency
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Quantization: {'4-bit' if load_in_4bit else 'None'}")
        
        # Configure quantization for free-tier compatibility
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",  # Auto-distribute between GPU/CPU
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Reduce CPU memory usage
                    max_memory={0: "6GB", "cpu": "12GB"}  # Adjust based on your system
                )
                print("Model loaded with 4-bit quantization and CPU offloading enabled")
            except ImportError:
                print("bitsandbytes not available. Loading model without quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=device,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        print("LLM Handler initialized successfully!")
    
    def create_prompt(
        self,
        query: str,
        context: str,
        conversation_history: str = "",
        system_message: Optional[str] = None
    ) -> str:
        """
        Create formatted prompt for Mistral Instruct with conversation history
        
        Args:
            query: User question
            context: Retrieved documents context
            conversation_history: Previous conversation for follow-up questions
            system_message: Optional system message (uses default if None)
            
        Returns:
            Formatted prompt string
        """
        if system_message is None:
            system_message = (
                "You are a helpful technical assistant that helps users perform specific tasks "
                "like fixing devices, configuring systems, or troubleshooting issues. "
                "Provide clear, step-by-step instructions based on the provided context. "
                "If the user asks a follow-up question, use the previous conversation to understand context. "
                "If the context doesn't contain enough information, say so clearly."
            )
        
        # Build prompt with conversation history
        history_section = ""
        if conversation_history:
            history_section = f"\n{conversation_history}\n"
        
        # Mistral Instruct format: <s>[INST] {prompt} [/INST]
        prompt = f"""<s>[INST] {system_message}
{history_section}
Context from repair guides:
{context}

User Question: {query}

Based on the context above and previous conversation (if any), provide a clear, step-by-step answer to help the user complete their task. [/INST]"""
        
        return prompt
    
    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using RAG context and conversation history
        
        Args:
            query: User question
            context: Retrieved document context
            conversation_history: Previous conversation context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        # Create prompt with conversation history
        prompt = self.create_prompt(query, context, conversation_history)
        
        # Generate response
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_full_text=False
        )
        
        response = outputs[0]['generated_text']
        
        # Clean up response
        response = response.strip()
        
        return response
    
    def generate_with_retrieval(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        conversation_history: str = "",
        max_new_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: generate response with retrieved context and conversation history
        
        Args:
            query: User question
            retrieval_result: Result from retriever.retrieve_and_format()
            conversation_history: Previous conversation for context
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with response, sources, and context
        """
        context = retrieval_result['context']
        sources = retrieval_result['sources']
        
        # Generate response with conversation history
        response = self.generate_response(
            query=query,
            context=context,
            conversation_history=conversation_history,
            max_new_tokens=max_new_tokens
        )
        
        return {
            'query': query,
            'response': response,
            'sources': sources,
            'context': context
        }


class SimpleLLMHandler:
    """Simplified LLM handler using smaller models or API-based approach"""
    
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialize with a smaller, faster model for testing
        
        Args:
            model_name: Model name (default: flan-t5-base for CPU-friendly inference)
        """
        print(f"Loading simplified model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.pipe = pipeline(
            "text2text-generation" if "t5" in model_name.lower() else "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        print("Simplified LLM Handler initialized!")
    
    def generate_response(self, query: str, context: str, conversation_history: str = "") -> str:
        """Generate response with context and conversation history"""
        
        history_section = ""
        if conversation_history:
            history_section = f"\n{conversation_history}\n"
        
        prompt = f"""Based on the following repair guides, answer the user's question.
{history_section}
Context:
{context[:1000]}

Question: {query}

Answer:"""
        
        outputs = self.pipe(prompt, max_length=512, do_sample=True, temperature=0.7)
        return outputs[0]['generated_text']


def main():
    """Test the LLM handler"""
    print("="*80)
    print("TESTING LLM HANDLER")
    print("="*80)
    
    # For testing, we'll create a mock context
    mock_context = """
[Document 1 - iPhone Screen Replacement | Category: Phone]
Title: iPhone Screen Replacement
Summary: Replace a cracked or broken iPhone screen
Step 1: Power off the iPhone completely by holding the power button
Step 2: Remove the two pentalobe screws near the charging port using P2 Pentalobe screwdriver
Step 3: Use a suction cup to gently lift the screen from the bottom edge
Step 4: Carefully disconnect the display cable connectors
Step 5: Remove the broken screen and install the new screen
Step 6: Reconnect all cables and test the new screen before sealing
Tools Required: P2 Pentalobe Screwdriver, Suction Cup, Spudger
"""
    
    test_query = "How do I replace my iPhone screen?"
    
    print("\nTest Query:", test_query)
    print("\nContext:", mock_context[:200], "...\n")
    
    print("-"*80)
    print("Note: For full testing with Mistral 7B, run with proper GPU setup.")
    print("Using SimpleLLMHandler for demonstration...")
    print("-"*80)
    
    try:
        # Use simplified handler for testing
        handler = SimpleLLMHandler()
        response = handler.generate_response(test_query, mock_context)
        
        print("\nGenerated Response:")
        print(response)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("This is expected if dependencies are not installed.")
        print("\nTo use the full LLMHandler:")
        print("1. Install: pip install transformers torch bitsandbytes accelerate")
        print("2. Ensure sufficient GPU memory or use quantization")


if __name__ == "__main__":
    main()
