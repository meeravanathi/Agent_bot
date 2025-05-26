from langchain_ollama import OllamaLLM
import streamlit as st
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
import time
from agent import customer_context_manager, AgentExecutor

from tools import get_tools# Use your Ollama agent!
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
# Load environment variables
load_dotenv()

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="E-commerce Customer Service Bot",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        flex-direction: row-reverse;
        color: black;
    }
    .chat-message.bot {
        background-color: #f5f5f5;
        color: black;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
    }
    .chat-message.user .avatar { background-color: #2196f3; }
    .chat-message.bot .avatar { background-color: #4caf50; }
    .chat-message .message { flex: 1; padding: 0 10px; }
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: black;
    }
    .metric-card h3 {
        color: #4caf50;
        margin: 0;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        color: #666;
    }
    .ollama-badge {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .model-selector {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 10px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "agent_executor" not in st.session_state:
        try:
            # Initialize Ollama LLM
            llm = OllamaLLM(model="llama3.1", base_url="http://localhost:11434")
            
            # Get tools (you need to define this function)
            tools = get_tools()
            
            # Create prompt template for ReAct agent
            prompt = PromptTemplate(
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
                template=(
                   """You are an AI customer service representative for an e-commerce platform. Your role is to help customers with their inquiries in a friendly, professional, and efficient manner.

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
    weather related quesries strictly use weather tool only and give final answer based on the weather tool output        
            When you DO need to use tools, follow this format:
              Thought: [your reasoning about what to do] dont rerun
              Action: [tool name from: {tool_names}] 
               if no action or missing action , just give the final answer from your thought 
               else Observe the output of the tool and use it to form your response.
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
                )
            )
            
            # Create ReAct agent
            agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
            
            # Create agent executor
            st.session_state.agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
            
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.agent_executor = None


    if "customer_authenticated" not in st.session_state:
        st.session_state.customer_authenticated = False

    if "current_customer" not in st.session_state:
        st.session_state.current_customer = None

    if "current_model" not in st.session_state:
        st.session_state.current_model = "llama3.1"

def display_message(role, content, timestamp=None):
    """Display a chat message with styling"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%H:%M")

    avatar = "👤" if role == "user" else "🤖"
    css_class = "user" if role == "user" else "bot"

    st.markdown(f"""
    <div class="chat-message {css_class}">
        <div class="avatar">{avatar}</div>
        <div class="message">
            <div style="font-size: 0.8em; color: #666; margin-bottom: 5px;">
                {role.title()} • {timestamp}
            </div>
            <div>{content}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def process_user_message(prompt, context):
    """Process user message and get AI response"""
    try:
        with st.spinner("🤖 Ollama AI is thinking..."):
            # Only pass a dict with the required keys
           response = st.session_state.agent_executor.invoke({
    "input": prompt,
    "chat_history": context  # Only include this if your agent expects it!
})

        return response["output"] if isinstance(response, dict) and "output" in response else response
    except Exception as e:
        error_msg = f"⚠️ Sorry, I encountered an error: {str(e)}"
        # ... rest of your error handling ...
        return error_msg


def main():
    """Main Streamlit app"""
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("🛍️ Customer Service")
      
        st.markdown("""
        <div class="ollama-badge">
            🦙 Powered by Local Ollama
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        # Model Selection
        st.subheader("🤖 AI Model")


        # Current model info
        model_info = {
            "llama3.1": "🦙 LLaMA  - Most capable",
           
        }
        
     
        
        st.markdown("---")
        
        # Customer Login Section
        st.subheader("👤 Customer Login")

        if not st.session_state.customer_authenticated:
            login_method = st.radio("Login with:", ["Customer ID", "Email"])
            if login_method == "Customer ID":
                customer_id = st.text_input("Customer ID", placeholder="e.g., CUST001")
                if st.button("Login with ID", use_container_width=True):
                    if customer_id:
                        customer_context_manager.set_customer_id(st.session_state.session_id, customer_id)
                        st.session_state.current_customer = customer_id
                        st.session_state.customer_authenticated = True
                        st.rerun()
                    else:
                        st.warning("Please enter a Customer ID")
            else:
                email = st.text_input("Email", placeholder="your.email@example.com")
                if st.button("Login with Email", use_container_width=True):
                    if email and "@" in email:
                        customer_context_manager.set_customer_email(st.session_state.session_id, email)
                        st.session_state.current_customer = email
                        st.session_state.customer_authenticated = True
                        st.rerun()
                    else:
                        st.warning("Please enter a valid email address")
        else:
            st.success(f"✅ Logged in as:\n{st.session_state.current_customer}")
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.customer_authenticated = False
                st.session_state.current_customer = None
                st.rerun()

        st.markdown("---")
        st.subheader("⚡ Quick Actions")

        quick_actions = {
            "📦 Check Order Status": "I'd like to check the status of my order",
            "🚚 Track Package": "Can you help me track my package?",
            "↩️ Return Item": "I need to return an item",
            "🔍 Product Search": "I'm looking for products",
            "💡 Get Recommendations": "Can you recommend some products?",
            "🌤️ Weather Update": "What's the weather like for shipping?"
        }

        for action, prompt in quick_actions.items():
            if st.button(action, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                context = customer_context_manager.get_context(st.session_state.session_id)
                response = process_user_message(prompt, context)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

        st.markdown("---")
        st.subheader("📊 System Status")
        st.markdown("""
        <div class="status-card">
            <h4>🟢 All Systems Operational</h4>
            <p>✅ Order Management: Online<br>
            ✅ Product Database: Online<br>
            ✅ Weather Service: Online<br>
            ✅ Ollama API: Connected</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent.reset_conversation()
            st.success("Chat cleared!")
            time.sleep(1)
            st.rerun()

    # Main content area
    st.title("🤖 E-commerce Customer Service Assistant")
    st.markdown("Welcome! I'm powered by **local Ollama AI** 🦙. I'm here to help you with orders, products, returns, and more!")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card"><h3>🦙</h3><p>Local Ollama</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><h3>9</h3><p>Tools Available</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card"><h3>~1s</h3><p>Avg Response</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card"><h3>24/7</h3><p>Available</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Chat interface
    chat_container = st.container()
    
    # Display existing messages
    with chat_container:
        if not st.session_state.messages:
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="color: black">👋 Hello! How can I help you today?</h3>
                <p>I'm running on <strong>{st.session_state.current_model}</strong> via Ollama for private, fast responses!</p>
                <p>Try asking about orders, products, returns, or use the quick actions in the sidebar!</p>
            </div>
            """, unsafe_allow_html=True)
        
        for i, message in enumerate(st.session_state.messages):
            display_message(message["role"], message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here... 💬"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with chat_container:
            display_message("user", prompt)

        # Get context and process message
        context = customer_context_manager.get_context(st.session_state.session_id)
        response = process_user_message(prompt, context)

        # Add and display assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with chat_container:
            display_message("assistant", response)

    # Help section
    with st.expander("💡 Sample Conversations & Tips"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🛒 Order Management:**
            - "What's the status of order ORD001?"
            - "I need to cancel order ORD002"
            - "I want to return the items from order ORD003"
            
            **🔍 Product Search:**
            - "Show me wireless headphones"
            - "I'm looking for electronics under $100"
            - "What are the details of product PROD001?"
            """)
        
        with col2:
            st.markdown("""
            **👤 Customer Service:**
            - "What are my loyalty points?"
            - "Update my preferred categories to include Books"
            - "Recommend products based on today's weather"
            
            **🔗 Complex Queries:**
            - "Check my order status and recommend similar products"
            - "What's the weather like and suggest appropriate clothing"
            """)

    # Ollama-specific help
    with st.expander("🦙 About Ollama Integration"):
        st.markdown("""
        **Why Ollama?**
        - **Private & Secure**: All AI runs locally on your machine
        - **No API Keys Needed**: No cloud, no rate limits, no cost
        - **Fast**: Modern LLMs with GPU/CPU acceleration
        - **Flexible**: Easily switch between models

        **Model:**
        - **LLaMA 3.1 **: Most capable, best for complex queries
    

        **Setup:**
        1. [Install Ollama](https://ollama.com/download)
        2. Pull models: `ollama pull llama3.1` 
        3. Run this app – no keys or cloud required!
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em; padding: 1rem;">
        🤖 E-commerce Customer Service Bot • Powered by LangChain + Ollama 🦙 • 
        Private, Fast, Local AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
