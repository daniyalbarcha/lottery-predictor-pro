import streamlit as st
import json
import os
from pathlib import Path

# Initialize session state for API keys if not already present
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        "openai_api_key": "",
        "anthropic_api_key": "",
        "grok_api_key": ""
    }

# Set page configuration
st.set_page_config(page_title="Settings - Lottery Predictor Pro", page_icon="⚙️")

# Custom CSS
st.markdown("""
    <style>
    .api-key-input {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .settings-header {
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .save-button {
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_api_keys():
    """Load API keys from session state or secrets"""
    try:
        # First try to get from session state
        if 'api_keys' in st.session_state:
            return st.session_state.api_keys
        # Then try to get from secrets
        return st.secrets.api_keys
    except Exception as e:
        st.warning("Unable to load API keys from secrets. Using empty keys.")
        return st.session_state.api_keys  # Return the initialized empty keys

def save_api_keys(keys):
    """Save API keys to session state"""
    try:
        st.session_state.api_keys = keys
    except Exception as e:
        st.error(f"Error saving API keys to session state. Please try again.")
        raise e

def main():
    try:
        st.title("⚙️ Settings")
        
        # Load existing API keys
        api_keys = load_api_keys()
        
        st.header("🔑 API Key Management")
        st.info("""
        Enter your API keys for each AI model. These keys will be stored for your current session.
        Don't have API keys? Visit:
        - OpenAI (GPT): https://platform.openai.com
        - Anthropic (Claude): https://console.anthropic.com
        - Grok: https://grok.x.ai
        """)
        
        # GPT API Key
        st.subheader("OpenAI (GPT) Settings")
        with st.expander("Configure GPT API Key", expanded=True):
            gpt_key = st.text_input(
                "OpenAI API Key",
                value=api_keys.get("openai_api_key", ""),
                type="password",
                help="Enter your OpenAI API key for GPT access"
            )
            st.caption("Your OpenAI API key is stored securely and never shared.")
        
        # Claude API Key
        st.subheader("Anthropic (Claude) Settings")
        with st.expander("Configure Claude API Key", expanded=True):
            claude_key = st.text_input(
                "Anthropic API Key",
                value=api_keys.get("anthropic_api_key", ""),
                type="password",
                help="Enter your Anthropic API key for Claude access"
            )
            st.caption("Your Anthropic API key is stored securely and never shared.")
        
        # Grok API Key
        st.subheader("Grok Settings")
        with st.expander("Configure Grok API Key", expanded=True):
            grok_key = st.text_input(
                "Grok API Key",
                value=api_keys.get("grok_api_key", ""),
                type="password",
                help="Enter your Grok API key"
            )
            st.caption("Your Grok API key is stored securely and never shared.")
        
        # Save Button
        if st.button("💾 Save API Keys", type="primary"):
            new_keys = {
                "openai_api_key": gpt_key,
                "anthropic_api_key": claude_key,
                "grok_api_key": grok_key
            }
            save_api_keys(new_keys)
            st.success("✅ API keys saved successfully!")
            
            # Show warning if any key is missing
            missing_keys = [k for k, v in new_keys.items() if not v]
            if missing_keys:
                st.warning(f"⚠️ Missing API keys for: {', '.join(k.split('_')[0].title() for k in missing_keys)}")
        
        # Usage Instructions
        st.header("📖 Usage Instructions")
        st.markdown("""
        1. **Enter API Keys**: Paste your API keys in the fields above
        2. **Save Changes**: Click the 'Save API Keys' button
        3. **Verify Status**: Green success message confirms keys are saved
        4. **Return to Predictions**: Use the sidebar to return to predictions
        
        **Note**: API keys are required for:
        - GPT predictions
        - Claude predictions
        - Grok predictions
        
        Without valid API keys, the AI prediction features will not be available.
        """)
        
        # Security Notice
        st.header("🔒 Security Information")
        st.markdown("""
        Your API keys are:
        - Stored securely in your session
        - Never transmitted to any third party
        - Only used for model API calls
        - Cleared when you close the browser
        """)
    except Exception as e:
        st.error("An error occurred while loading the settings page. Please try refreshing the page.")
        st.exception(e)

if __name__ == "__main__":
    main() 