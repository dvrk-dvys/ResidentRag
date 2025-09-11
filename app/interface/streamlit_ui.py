import streamlit as st
import uuid
from typing import Dict, List
import streamlit as st
from streamlit_chat import message
from streamlit_option_menu import option_menu
import plotly.express as px


class MedicalRAG_UI:
    def __init__(self, chat_assistant):
        self.assistant = chat_assistant
        self.setup_page_config()
        self.init_session_state()
        self.user_to_detail_map = {
                                    "Medical Researcher": 2,  # "Technical"
                                    "Healthcare Provider": 1,  # "Detailed"
                                    "Patient": 0              # "Simple"
        }

    def setup_page_config(self):
        st.set_page_config(
            page_title="Medical RAG Assistant",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def init_session_state(self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())

    def render_sidebar(self):
        with st.sidebar:
            st.header("🏥 Medical RAG")

            # User type selection
            user_type = st.selectbox(
                "I am a:",
                ["Medical Researcher", "Healthcare Provider", "Patient"],
                index = 1  # Default to "Healthcare Provider" (index 1)
            )

            if st.button("New Conversation"):
                self.reset_conversation()

            # Settings
            with st.expander("⚙️ Settings"):
                response_detail = st.selectbox(
                    "Response Detail:",
                    ["Simple", "Detailed", "Technical"],
                    index=self.user_to_detail_map.get(user_type, 1)  # Auto-set response detail based on user type but Default to "Detailed"
                )
                show_sources = st.checkbox("Show Sources", value=False)
        st.sidebar.text("© Jordan Alexander Harris")
        
        # Returns settings dict for use in chat class
        return {
            "user_type": user_type,
            "response_detail": response_detail,
            "show_sources": show_sources
        }


    def render_chat_interface(self, settings: dict):
        st.title("🏥 Medical RAG Assistant")
        st.caption("Ask medical questions with confident sources")

        # Display chat history
        for msg in st.session_state.chat_messages:
            self.render_message(msg)

        # Promp User input
        if prompt := st.chat_input("Ask a medical question..."):
            self.handle_user_input(prompt, settings)

    def render_message(self, message: Dict):
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

                # Show sources if available
                sources = message.get("sources")

                if sources:  # truthy -> non-empty list
                    with st.expander("📚 Sources"):
                        for s in sources:
                            if isinstance(s, dict):
                                st.write(f"• {s.get('title', '(untitled)')} — score: {s.get('score', 0):.4f}")
                            else:
                                st.write(f"• {s}")

        elif role == "tool_call":
            with st.chat_message("assistant"):
                with st.expander(f"🔧 {message['tool_name']}"):
                    st.json(message["output"])

    def handle_user_input(self, prompt: str, settings: dict):
        # Add user message to chat history above
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(user_msg)

        #user_type = settings.get("user_type", "Healthcare Provider")
        #response_detail = settings.get("response_detail")
        #user_type = st.session_state.get('user_type', 'Healthcare Provider')
        #override_detail = st.session_state.get('response_detail')

        # Get assistant response with current settings
        with st.spinner("Thinking..."):
            response = self.assistant.process_message(
                question=prompt,
                settings=settings
            )

        # Add assistant response
        assistant_msg = {
            "role": "assistant",
            "content": response["answer"],
            #"sources": response.get("sources", [])
        }
        if response.get("sources"):
            assistant_msg["sources"] = response["sources"]

        st.session_state.chat_messages.append(assistant_msg)

        st.rerun()

    def reset_conversation(self):
        st.session_state.chat_messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

    def render(self):
        settings = self.render_sidebar()
        self.render_chat_interface(settings)





            #selected = option_menu(
            #    menu_title="Medical RAG",
            #    options=["Chat", "Analytics", "Settings"],
            #    icons=["chat-dots", "graph-up", "gear"],
            #    menu_icon="hospital",
            #    default_index=0,
            #    styles={
            #        "container": {"padding": "0!important", "background-color": "#fafafa"},
            #        "icon": {"color": "orange", "font-size": "25px"},
            #        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            #        "nav-link-selected": {"background-color": "#02ab21"},
            #    }
            #)

            # Beautiful chat bubbles
            #message("Hello! I'm your medical assistant", key="assistant1")
            #message("What's the difference between Type 1 and Type 2 diabetes?", is_user=True, key="user1")