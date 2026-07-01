"""
visualize_workflow.py

Generates a visual diagram of the News Agent's LangGraph workflow
(defined in agent.py) so you can inspect the graph structure without
running the full agent.

Usage:
    python visualize_workflow.py

Outputs (created in the current directory):
    workflow_diagram.mmd   - Mermaid diagram source (always created)
    workflow_diagram.png   - Rendered PNG (created if possible)
    workflow_ascii.txt     - ASCII rendering (fallback, always created)

Notes:
    - PNG rendering via `draw_mermaid_png()` calls the free mermaid.ink
      service and requires internet access. If that's unavailable, this
      script falls back to a local Graphviz render (if `pygraphviz` or
      `graphviz` + `pydot` are installed), and finally to a plain ASCII
      diagram printed to the console and saved to workflow_ascii.txt.
"""

import sys

from agent import app  # the compiled langgraph workflow from agent.py


def save_mermaid_source(graph, path: str = "workflow_diagram.mmd") -> str:
    """Save the Mermaid diagram source to a file and return it."""
    mermaid_src = graph.draw_mermaid()
    with open(path, "w") as f:
        f.write(mermaid_src)
    print(f"✅ Mermaid source saved to {path}")
    return mermaid_src


def try_save_png(graph, path: str = "workflow_diagram.png") -> bool:
    """Attempt to render a PNG. Returns True on success."""
    # 1) Try the built-in mermaid.ink renderer (needs internet access)
    try:
        png_bytes = graph.draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ PNG diagram saved to {path} (via mermaid.ink)")
        return True
    except Exception as e:
        print(f"⚠️  Could not render PNG via mermaid.ink: {e}")

    # 2) Try local Graphviz rendering
    try:
        png_bytes = graph.draw_png()
        with open(path, "wb") as f:
            f.write(png_bytes)
        print(f"✅ PNG diagram saved to {path} (via local Graphviz)")
        return True
    except Exception as e:
        print(f"⚠️  Could not render PNG via local Graphviz: {e}")

    return False


def save_ascii(graph, path: str = "workflow_ascii.txt") -> None:
    """Print and save an ASCII rendering of the graph."""
    try:
        ascii_art = graph.draw_ascii()
        print("\n" + ascii_art)
        with open(path, "w") as f:
            f.write(ascii_art)
        print(f"✅ ASCII diagram saved to {path}")
    except Exception as e:
        print(f"⚠️  Could not render ASCII diagram: {e}")


def main():
    graph = app.get_graph()

    print("Nodes:", list(graph.nodes.keys()))
    print()

    mermaid_src = save_mermaid_source(graph)
    print("\n--- Mermaid diagram source ---")
    print(mermaid_src)
    print("--- (paste into https://mermaid.live to view) ---\n")

    png_ok = try_save_png(graph)

    if not png_ok:
        save_ascii(graph)


if __name__ == "__main__":
    sys.exit(main())
