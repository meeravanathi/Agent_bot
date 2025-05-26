"""
LangChain Agent for E-commerce Customer Service (using Ollama LLaMA 3.1)
"""

import os
import traceback
from typing import Dict, List, Any
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from tools import get_tools  # Your custom tools

load_dotenv()

class EcommerceAgent:
    """E-commerce customer service agent using Ollama (LLaMA 3.1)"""

    def __init__(self):
        # Initialize Ollama LLM
        self.llm = self._initialize_llm()

        # Load tools
        self.tools = get_tools()
        print("Loaded tools:", [tool.name for tool in self.tools])

        # Memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )

        # Prompt setup
        self.prompt = self._create_prompt()

        # Agent setup
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
            verbose=True
        )

        # Agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True
        )

    def _initialize_llm(self):
        """Initialize Ollama LLaMA 3.1"""
        try:
            print("Initializing Ollama LLaMA 3.1...")

            llm = OllamaLLM(
                model="llama3.1",  # Ensure Ollama is running this model
                temperature=0.2,
                max_tokens=1024,
                top_p=0.9,
                request_timeout=60
            )

            # Test model connection
            _ = llm.invoke("Hello")
            print("✅ Ollama LLaMA 3.1 initialized successfully!")

            return llm

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize Ollama: {e}")

    def _create_prompt(self):
        """Create the system prompt for the agent"""

        template = """You are an AI customer service representative for an e-commerce platform. Your role is to help customers with their inquiries in a friendly, professional, and efficient manner.

**Your Capabilities:**
- Check order status, cancel orders, and process returns
- Search for products and provide detailed product information
- Access customer information and update preferences
- Get weather information for shipping estimates
- Provide product recommendations based on weather or preferences
- Make autonomous decisions about which tools to use
- Chain multiple tools together when needed

**Guidelines:**
1. **Be Proactive**: Anticipate customer needs and offer relevant information
2. **Be Contextual**: Remember previous conversation context and use it appropriately
3. **Be Autonomous**: Decide which tools to use based on customer queries without asking for permission
4. **Be Helpful**: If you can't directly solve a problem, offer alternatives or escalation paths
5. **Be Professional**: Maintain a friendly, helpful tone while being efficient
6. **Chain Tools**: Use multiple tools in sequence when it provides better customer service

**Tool Usage Examples:**
- If a customer asks about an order, check order status and optionally get weather for shipping updates
- If a customer wants to return something, first check order status, then process the return
- If a customer asks for product recommendations, consider using weather information to provide seasonal suggestions
- If updating customer preferences, confirm the changes and suggest relevant products

**Important Notes:**
- Always prioritize customer satisfaction
- If you're unsure about something, it's better to ask for clarification than make assumptions
- When handling cancellations or returns, explain the process clearly
- Provide order IDs, product IDs, and other reference numbers when relevant
- Be empathetic when dealing with complaints or issues
- Always include "Final Answer:" even if tools fail
- Be helpful and specific
- If tools fail, provide alternative solutions
- Never leave customer hanging without a response
You have access to the following tools:{tools}

 IMPORTANT INSTRUCTIONS:\n
 - For simple greetings, questions, or general conversation, respond directly without using tools
 - Only use tools when you need specific information (like product details, order status, etc.)
 - When you don't need tools, just provide a helpful response
            
            When you DO need to use tools, follow this format:
              Thought: [your reasoning about what to do] dont rerun
              Action: [tool name from: {tool_names}] dont rerun,if missing action , just give the final answer as the thought
              Final Answer: [your response to the user] 
            
            When you DON'T need tools, just respond naturally:
               Thought: [brief reasoning]
                Final Answer: [your helpful response]
            
                Current conversation:
            Human: {input}
            {agent_scratchpad}

if there is a mssing Action , return your thought as the final answer
Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

        return PromptTemplate(
            template=template,
            input_variables=["input", "chat_history", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )

    def process_message(self, message: str, customer_context: Dict[str, Any] = None) -> str:
        """Process a customer message and return response"""
        try:
            enhanced_message = message
            if customer_context:
                enhanced_message += f"\nCustomer Context: {customer_context}"

            chat_history = ""
            for msg in self.memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    chat_history += f"Human: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    chat_history += f"AI: {msg.content}\n"

            response = self.agent_executor.invoke({
                "input": enhanced_message,
                "chat_history": chat_history
            })

            self.memory.save_context(
                {"input": message},
                {"output": response["output"]}
            )

            return response["output"]

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            print(f"Error details: {traceback.format_exc()}")

            self.memory.save_context(
                {"input": message},
                {"output": error_msg}
            )
            return error_msg

    def reset_conversation(self):
        """Reset the conversation memory"""
        self.memory.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history"""
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history

    def cleanup(self):
        """Clean up resources"""
        print("🧹 Session cleaned up")


class CustomerContext:
    """Manage customer context and session information"""

    def __init__(self):
        self.sessions = {}

    def get_context(self, session_id: str) -> Dict[str, Any]:
        return self.sessions.get(session_id, {})

    def update_context(self, session_id: str, context: Dict[str, Any]):
        if session_id not in self.sessions:
            self.sessions[session_id] = {}
        self.sessions[session_id].update(context)

    def set_customer_id(self, session_id: str, customer_id: str):
        self.update_context(session_id, {"customer_id": customer_id})

    def set_customer_email(self, session_id: str, email: str):
        self.update_context(session_id, {"customer_email": email})


# Global context manager
customer_context_manager = CustomerContext()
