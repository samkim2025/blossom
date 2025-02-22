import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import time
import uuid

# -- If you plan to use OpenAI or another LLM, uncomment and replace placeholders:
import openai

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
    Placeholder function to call your LLM of choice.
    Replace the body with your actual LLM logic (OpenAI, Anthropic, etc.).
    This function receives the user query plus any relevant conversation context.
    """
    # Example with OpenAI (uncomment if needed):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    response = openai.ChatCompletion.create(
         model="gpt-3.5-turbo",
         messages=[
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": context},
             {"role": "user", "content": user_query}
         ]
     )
     return response.choices[0].message["content"].strip()
    
    # For demonstration, just echo back:
    time.sleep(1)  # Simulate some latency
    return f"LLM response to '{user_query}' (context length: {len(context)} chars)."

def trigger_rerun():
    """Force a rerun by updating query parameters."""
    st.experimental_set_query_params(rerun=str(uuid.uuid4()))

###############################################################################
# INITIALIZATION
###############################################################################

if "mindmap" not in st.session_state:
    # We'll store mindmap data in a dict:
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
    Create a new child node under the current active node, 
    representing a branch with a user-defined name.
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
    # Link from parent to child
    st.session_state.mindmap["nodes"][parent_id]["children"].append(new_id)
    # Move focus (active_node) into the new branch
    st.session_state.mindmap["active_node"] = new_id
    trigger_rerun()

def return_to_parent():
    """
    Return to the parent node of the current active node.
    The branch remains in the data structure but we are no longer "in" it.
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
    Simulate an LLM chat: user provides a query, 
    we fetch context from the active branch, get a response, and store it.
    """
    active_node = get_active_node()
    
    # Build a simple context from all content in the active branch
    full_context = ""
    for turn in active_node["content"]:
        full_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    
    # Call the LLM (placeholder)
    assistant_reply = call_llm_api(user_query, full_context)
    # Store new conversation turn
    active_node["content"].append({"user": user_query, "assistant": assistant_reply})
    trigger_rerun()

###############################################################################
# VISUALIZATION
###############################################################################

def render_mindmap():
    """
    Show a graph visualization of the entire mindmap using streamlit_agraph.
    """
    # Collect nodes and edges
    nodes_data = []
    edges_data = []
    
    for node_id, node_info in st.session_state.mindmap["nodes"].items():
        label = node_info["branch_name"]
        # Highlight the active node
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
    
    # Show conversation content
    if active_node["content"]:
        for turn in active_node["content"]:
            st.markdown(f"**User**: {turn['user']}")
            st.markdown(f"**Assistant**: {turn['assistant']}")
            st.markdown("---")
    else:
        st.info("No conversation yet in this branch.")

    # -- Chat form
    with st.form("chat_form"):
        user_query = st.text_area("Ask a question or add to this conversation:", key="user_input", height=100)
        submitted = st.form_submit_button("Submit")
        if submitted and user_query.strip():
            add_chat_message(user_query.strip())
    
    # -- Buttons for branching and returning
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
    
    # -- Optionally show the entire mindmap
    if st.session_state.get("show_mindmap", False):
        st.subheader("Mindmap Overview")
        render_mindmap()
        if st.button("Hide Mindmap"):
            st.session_state.show_mindmap = False
            trigger_rerun()

if __name__ == "__main__":
    main()
