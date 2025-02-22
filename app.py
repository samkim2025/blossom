import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import time
import uuid
import os
import openai

# Set the page configuration
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
    """
    # Set your API key from Streamlit secrets or environment variables
    openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using the GPT-4o model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context},
                {"role": "user", "content": user_query}
            ],
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return "Error fetching response from LLM."

def trigger_rerun():
    """Force a rerun by updating query parameters."""
    st.query_params(rerun=str(uuid.uuid4()))

###############################################################################
# INITIALIZATION
###############################################################################

if "mindmap" not in st.session_state:
    # Store mindmap data in a dictionary:
    # mindmap["nodes"] = { node_id: { "id", "title", "content", "parent", "children", "branch_name" } }
    # mindmap["active_node"] = the node_id we are currently zoomed into / chatting in
    root_id = generate_node_id()
    st.session_state.mindmap = {
        "nodes": {
            root_id: {
                "id": root_id,
                "title": "Root Topic",
                "content": [],  # list of conversation turns: {"user": ..., "assistant": ...}
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
    """Return the dictionary of the currently active node."""
    active_id = st.session_state.mindmap["active_node"]
    return st.session_state.mindmap["nodes"][active_id]

def create_branch(branch_name: str):
    """
    Create a new child node under the current active node, representing a branch with a user-defined name.
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
    # Link the new branch to its parent
    st.session_state.mindmap["nodes"][parent_id]["children"].append(new_id)
    # Set the new branch as the active node
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
    Simulate an LLM chat: the user provides a query, we build context from the active branch,
    fetch a response from the GPT-4o API, and store the conversation turn.
    """
    active_node = get_active_node()
    
    # Build context from all previous conversation turns in the active branch
    full_context = ""
    for turn in active_node["content"]:
        full_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    
    # Call the LLM API with the context and user query
    assistant_reply = call_llm_api(user_query, full_context)
    # Append the new conversation turn
    active_node["content"].append({"user": user_query, "assistant": assistant_reply})
    trigger_rerun()

###############################################################################
# VISUALIZATION
###############################################################################

def render_mindmap():
    """
    Render a graph visualization of the entire mindmap using streamlit_agraph.
    """
    nodes_data = []
    edges_data = []
    
    for node_id, node_info in st.session_state.mindmap["nodes"].items():
        label = node_info["branch_name"]
        # Highlight the active node with a distinct color
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

    # Display the currently active node
    active_node = get_active_node()
    st.subheader(f"Current Branch: {active_node['branch_name']}")
    
    # Show conversation history for the active branch
    if active_node["content"]:
        for turn in active_node["content"]:
            st.markdown(f"**User**: {turn['user']}")
            st.markdown(f"**Assistant**: {turn['assistant']}")
            st.markdown("---")
    else:
        st.info("No conversation yet in this branch.")

    # -- Chat input form
    with st.form("chat_form"):
        user_query = st.text_area("Ask a question or add to this conversation:", key="user_input", height=100)
        submitted = st.form_submit_button("Submit")
        if submitted and user_query.strip():
            add_chat_message(user_query.strip())
    
    # -- Branching and navigation buttons
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
    
    # -- Optionally display the mindmap visualization
    if st.session_state.get("show_mindmap", False):
        st.subheader("Mindmap Overview")
        render_mindmap()
        if st.button("Hide Mindmap"):
            st.session_state.show_mindmap = False
            trigger_rerun()

if __name__ == "__main__":
    main()
