"""
Gradio App Interface for Task-Oriented RAG Chatbot
"""

import gradio as gr
from retriever import Retriever
from llm_handler import LLMHandler, SimpleLLMHandler
from typing import Tuple, List
import os


class RAGChatbot:
    """Task-Oriented RAG Chatbot Application"""
    
    def __init__(self, use_simple_model: bool = False):
        """
        Initialize the chatbot
        
        Args:
            use_simple_model: Use SimpleLLMHandler instead of full Mistral 7B
        """
        print("Initializing RAG Chatbot...")
        
        # Initialize retriever
        print("\n1. Loading retriever...")
        self.retriever = Retriever(top_k=5)
        
        # Initialize LLM
        print("\n2. Loading LLM...")
        if use_simple_model:
            self.llm = SimpleLLMHandler()
        else:
            self.llm = LLMHandler(load_in_4bit=True)
        
        # Initialize conversation memory
        self.conversation_history = []
        self.use_simple_model = use_simple_model
        
        print("\n✓ Chatbot initialized successfully!")
    
    def answer_query(
        self,
        query: str,
        history: List[Tuple[str, str]] = None
    ) -> str:
        """
        Process user query with conversation history support
        
        Args:
            query: User question
            history: Chat history (list of [user_msg, bot_msg] pairs)
            
        Returns:
            Bot response string
        """
        if not query.strip():
            return "Please enter a question."
        
        # Retrieve relevant documents
        retrieval_result = self.retriever.retrieve_and_format(query, top_k=5)
        
        # Format conversation history for context
        conversation_context = self._format_conversation_history(history)
        
        # Generate response with conversation history
        if self.use_simple_model:
            response = self.llm.generate_response(
                query, 
                retrieval_result['context'],
                conversation_history=conversation_context
            )
        else:
            result = self.llm.generate_with_retrieval(
                query=query,
                retrieval_result=retrieval_result,
                conversation_history=conversation_context,
                max_new_tokens=512
            )
            response = result['response']
        
        # Format sources
        sources = retrieval_result['sources']
        sources_text = self._format_sources(sources)
        
        # Append sources to response
        full_response = f"{response}\n\n{sources_text}"
        
        # Store in conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'sources': sources
        })
        
        return full_response
    
    def _format_conversation_history(self, history: List[Tuple[str, str]] = None) -> str:
        """Format conversation history for LLM context"""
        if not history:
            return ""
        
        formatted = "Previous conversation:\n"
        for i, (user_msg, bot_msg) in enumerate(history[-3:], 1):  # Last 3 exchanges
            formatted += f"\nUser: {user_msg}\n"
            formatted += f"Assistant: {bot_msg}\n"
        
        return formatted
    
    def _format_sources(self, sources: List[dict]) -> str:
        """Format sources for display"""
        if not sources:
            return "No sources found."
        
        formatted = "\n---\n### 📚 Retrieved Sources:\n\n"
        for i, source in enumerate(sources, 1):
            formatted += f"**{i}. {source['title']}**\n"
            formatted += f"- Category: {source['category']}\n"
            formatted += f"- Relevance Score: {source['score']}\n"
            if source.get('url'):
                formatted += f"- [View Guide]({source['url']})\n"
            formatted += "\n"
        
        return formatted
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio chat interface with conversation history"""
        
        with gr.Blocks(title="Task-Oriented Repair Assistant", theme=gr.themes.Soft()) as interface:
            gr.Markdown(
                """
                # 🔧 Task-Oriented Repair Assistant with Chain of Thought
                
                Ask me anything about device repair, troubleshooting, or technical tasks!
                I'll provide step-by-step guidance and remember our conversation for follow-up questions.
                
                **Features:**
                - 💬 Conversational interface with memory
                - 🔗 Chain of thought - ask follow-up questions
                - 📚 Responses based on expert repair guides
                - 🔍 Source references for each answer
                """
            )
            
            # ChatInterface with history support
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_label=True,
                avatar_images=(None, "🤖"),
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask me anything about repairs or follow-up on previous answers...",
                    lines=2,
                    scale=4
                )
                submit = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Conversation")
                
            # Examples
            gr.Examples(
                examples=[
                    "How do I replace an iPhone screen?",
                    "What tools do I need for that?",
                    "My laptop won't turn on. What should I check?",
                    "How to replace a phone battery safely?",
                    "Can you explain step 3 in more detail?",
                    "What if I don't have that tool?"
                ],
                inputs=msg,
                label="Example Questions (including follow-ups)"
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                💡 **Tips:** 
                - Ask follow-up questions naturally - I remember our conversation!
                - Reference previous answers ("What about that step?", "Why is that necessary?")
                - Be specific for better results
                
                Data source: [MyFixit Dataset](https://github.com/microsoft/MyFixit-Dataset) - iFixit repair guides
                """
            )
            
            # Event handlers
            def respond(message, chat_history):
                """Handle user message and update chat"""
                # Get bot response with conversation context
                bot_message = self.answer_query(message, chat_history)
                
                # Append to chat history
                chat_history.append((message, bot_message))
                
                return "", chat_history
            
            def clear_conversation():
                """Clear chat history and internal memory"""
                self.conversation_history = []
                return [], ""
            
            # Submit button
            submit.click(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            # Enter key support
            msg.submit(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            # Clear button
            clear.click(
                fn=clear_conversation,
                outputs=[chatbot, msg]
            )
        
        return interface
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
        """
        Launch the Gradio interface
        
        Args:
            share: Create public link (useful for Colab)
            server_name: Server address
            server_port: Server port
        """
        interface = self.create_interface()
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True
        )


def main():
    """Launch the chatbot application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-Oriented RAG Chatbot")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simplified LLM model (faster, less accurate)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    
    args = parser.parse_args()
    
    # Check if index exists
    if not os.path.exists("faiss_index.bin"):
        print("="*80)
        print("ERROR: FAISS index not found!")
        print("="*80)
        print("\nPlease run the following steps first:")
        print("1. python data_processor.py  # Process documents")
        print("2. python embeddings.py      # Create embeddings and index")
        print("3. python app.py             # Launch chatbot")
        print("\nOr use: python main.py --build  # To run all steps automatically")
        return
    
    # Initialize and launch chatbot
    chatbot = RAGChatbot(use_simple_model=args.simple)
    
    print("\n" + "="*80)
    print("LAUNCHING GRADIO INTERFACE")
    print("="*80)
    
    chatbot.launch(
        share=args.share,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
