import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import time
import uuid
import os
import openai

# Set page configuration
st.set_page_config(
    page_title="Mindmap Chat",
    layout="wide"
)

###############################################################################
# UTILITIES
###############################################################################

def generate_node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())

def call_llm_api(user_query: str, context: str) -> str:
    """
    Calls the OpenAI ChatCompletion API using the GPT-4o model.
    Ensure that your API key is set in st.secrets["OPENAI_API_KEY"] or as an environment variable.
    
    IMPORTANT: If you're using openai>=1.0.0, please run `openai migrate` to update your codebase,
    or pin your installation to openai==0.28 if you want to keep the old interface.
    """
    # Set API key from st.secrets or environment variable
    openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Use the appropriate model (adjust if needed)
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7
        )
        # Access the content from the response using the new dictionary style
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return "Error fetching response from LLM."

def trigger_rerun():
    """Force a rerun by updating query parameters."""
    # Use st.experimental_set_query_params instead of st.query_params
    st.experimental_set_query_params(rerun=str(uuid.uuid4()))

###############################################################################
# INITIALIZATION
###############################################################################

if "mindmap" not in st.session_state:
    # Initialize mindmap with a root node.
    root_id = generate_node_id()
    st.session_state.mindmap = {
        "nodes": {
            root_id: {
                "id": root_id,
                "title": "Root Topic",
                "content": [],  # conversation turns: {"user": ..., "assistant": ...}
                "parent": None,
                "children": [],
                "branch_name": "Root",
            }
        },
        "active_node": root_id,
        "show_mindmap": False,
    }

###############################################################################
# CORE FUNCTIONS
###############################################################################

def get_active_node():
    """Return the currently active node's dictionary."""
    active_id = st.session_state.mindmap["active_node"]
    return st.session_state.mindmap["nodes"][active_id]

def create_branch(branch_name: str):
    """
    Create a new child node under the current active node with the given branch name.
    """
    parent_id = st.session_state.mindmap["active_node"]
    new_id = generate_node_id()
    st.session_state.mindmap["nodes"][new_id] = {
        "id": new_id,
        "title": branch_name,
        "content": [],
        "parent": parent_id,
        "children": [],
        "branch_name": branch_name
    }
    st.session_state.mindmap["nodes"][parent_id]["children"].append(new_id)
    st.session_state.mindmap["active_node"] = new_id
    trigger_rerun()

def return_to_parent():
    """
    Return to the parent node of the current active node.
    """
    active_id = st.session_state.mindmap["active_node"]
    node = st.session_state.mindmap["nodes"][active_id]
    if node["parent"] is not None:
        st.session_state.mindmap["active_node"] = node["parent"]
        trigger_rerun()
    else:
        st.warning("Already at the root node. Cannot go higher.")

def add_chat_message(user_query: str):
    """
    Add a chat message to the active node by calling the LLM API.
    """
    active_node = get_active_node()
    
    # Build context from the conversation history in the active node
    full_context = ""
    for turn in active_node["content"]:
        full_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    
    assistant_reply = call_llm_api(user_query, full_context)
    active_node["content"].append({"user": user_query, "assistant": assistant_reply})
    trigger_rerun()

###############################################################################
# VISUALIZATION
###############################################################################

def render_mindmap():
    """
    Render a graph visualization of the mindmap using streamlit_agraph.
    """
    nodes_data = []
    edges_data = []
    
    for node_id, node_info in st.session_state.mindmap["nodes"].items():
        label = node_info["branch_name"]
        color = "#FFCC00" if node_id == st.session_state.mindmap["active_node"] else "#99CCFF"
        nodes_data.append(
            Node(
                id=node_id,
                label=label,
                shape="circle",
                size=25,
                color=color
            )
        )
        for child_id in node_info["children"]:
            edges_data.append(Edge(source=node_id, target=child_id))
    
    config = Config(
        width=800,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True
    )
    
    agraph(nodes=nodes_data, edges=edges_data, config=config)

###############################################################################
# UI LAYOUT
###############################################################################

def main():
    st.title("Mindmap LLM Chat")

    # Display the active branch
    active_node = get_active_node()
    st.subheader(f"Current Branch: {active_node['branch_name']}")
    
    # Display conversation history
    if active_node["content"]:
        for turn in active_node["content"]:
            st.markdown(f"**User**: {turn['user']}")
            st.markdown(f"**Assistant**: {turn['assistant']}")
            st.markdown("---")
    else:
        st.info("No conversation yet in this branch.")

    # Chat form
    with st.form("chat_form"):
        user_query = st.text_area("Ask a question or add to this conversation:", key="user_input", height=100)
        submitted = st.form_submit_button("Submit")
        if submitted and user_query.strip():
            add_chat_message(user_query.strip())
    
    # Branch navigation buttons
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        branch_name = st.text_input("Branch name", key="branch_name")
        if st.button("Create Branch"):
            if branch_name.strip():
                create_branch(branch_name.strip())
            else:
                st.warning("Please enter a branch name before creating a branch!")
    with c2:
        if st.button("Return to Parent"):
            return_to_parent()
    with c3:
        if st.button("Zoom Out (Show Mindmap)"):
            st.session_state.show_mindmap = not st.session_state.get("show_mindmap", False)
            trigger_rerun()

    st.markdown("---")
    
    # Optionally render the mindmap
    if st.session_state.get("show_mindmap", False):
        st.subheader("Mindmap Overview")
        render_mindmap()
        if st.button("Hide Mindmap"):
            st.session_state.show_mindmap = False
            trigger_rerun()

if __name__ == "__main__":
    main()
