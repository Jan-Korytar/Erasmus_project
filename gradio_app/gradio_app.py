import gradio as gr


# Dummy model function — replace this with your real model logic
def process_text(text):
    # Example: uppercase the input
    return f"Processed: {text.upper()}"


# Set up the interface
interface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs="text",
    title="Text Processor",
    description="Enter a text description of the pokemon and receive an output, note that this model is quite limited."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
