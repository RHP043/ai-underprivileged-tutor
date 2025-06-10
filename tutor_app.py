import os
import gradio as gr
from openai import OpenAI
import tempfile
from datetime import datetime
import traceback # For better error logging

# Try importing fpdf - provide better error message if missing
try:
    # Use FPDF class directly from the library
    from fpdf import FPDF
except ImportError:
    print("WARNING: FPDF library is not installed. PDF generation will be disabled. Install using: pip install fpdf2")
    # Use None to indicate fpdf is not available
    FPDF = None # Use class name for check later

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API Key not found. Set OPENAI_API_KEY in your environment variables.")

client = OpenAI(api_key=API_KEY)

# --- Prompts (kept exactly as in your original code) ---
SYSTEM_PROMPT = """
You are a world‑class, patient virtual STEM tutor for K–12 students—specifically those in low-income areas—focused on delivering top‑tier, one‑on‑one support.

Goals:
1. Ensure mastery before moving on.
2. Use clear, everyday language; define any new term.
3. Make lessons interactive, hands‑on, and tied to real life.
4. Celebrate effort and guide gently through challenges.
5. Recommend low‑tech or free tools (paper, household items, free online apps).

Style:
- Break concepts into simple, bite‑sized steps and ask "What's your next thought?"
- Speak like a supportive coach: "Great work! Let's try the next step."
- Use relatable examples and at‑home experiments.
- Keep sentences short—one idea each.
- Adapt: simplify when stuck; offer challenges when they excel.

Stay On‑Topic:
- If off‑track, say: "I'm here for STEM—what math or science question shall we tackle?"
- Tie any off‑topic comment back to STEM relevance.
- Decline non‑STEM requests: "I focus on science, technology, engineering, and math—what can I help you learn today?"

Start by asking:
"Hi! What STEM topic or school subject would you like help with today? If unsure, tell me your grade and I'll suggest some options."
"""

SUGGESTED_RESPONSES_PROMPT = """
You are an AI that helps students formulate thoughtful responses to their tutor.

Generate 3 different potential student responses to the tutor's last message. Make these responses:
1. Show different levels of understanding (basic, intermediate, curious/advanced)
2. Be brief (1-3 sentences each)
3. Be natural and conversational, as a student would actually speak
4. Include specific follow-up questions or requests for clarification when appropriate
5. Sometimes express partial understanding or confusion that will help the tutor know where to focus

Only generate the 3 responses, with no preamble or explanation. Format as:
1. [first response]
2. [second response]
3. [third response]

Example responses might include:
- Showing understanding: "Oh I see, so the mitochondria creates energy for the cell. How much energy can one cell make?"
- Asking for clarification: "I'm still confused about how photosynthesis works. Can you explain it differently?"
- Sharing progress: "I tried solving it and got x = 7. Is that right?"
"""

SUMMARY_SYSTEM_PROMPT = """
You are a world‑class AI study‑guide generator. Transform the entire tutoring conversation into a professional, PDF‑ready study guide.

Core Tasks:
1. Identify the main subject and key concepts.
2. Extract essential definitions, formulas, and examples.
3. Organize content into clear sections:
   - H1 for the guide title. Use '# Title'
   - H2 for major topics. Use '## Topic'
   - H3 for subtopics. Use '### Subtopic'
4. Use bullet points for definitions, formulas, and examples. Use '- ' or '* '
5. Indicate visuals with placeholders:
   [Insert diagram: brief description]
6. Reserve space for student notes under each section:
   Notes: ___________________________

Formatting Rules:
- Blank line before every heading.
- 1.5 line spacing (simulate with single blank lines between items).
- Consistent font style implied by plain text.
- No extra commentary—output only the formatted guide.

Begin output with:
# Study Guide: [Topic Title]
"""

# --- Core Functions (with functional fixes) ---

# Transcribe audio to text (Simplified)
def transcribe_audio(audio_filepath):
    if audio_filepath is None:
        return "" # Return empty string if no audio
    try:
        # audio_filepath is the path provided by Gradio's Audio component type="filepath"
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        # Gradio manages the temp file from the Audio component
        return transcription.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        traceback.print_exc() # Print full traceback for debugging
        # Return error message to be displayed in chat
        return f"Error transcribing audio: {str(e)}"

# Generate speech from text (Using tempfile)
def generate_speech(text, voice="nova"):
    try:
        # Use a temporary file for the speech output
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_speech_file:
            speech_file_path = tmp_speech_file.name

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice, # alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        response.stream_to_file(speech_file_path)
        return speech_file_path
    except Exception as e:
        print(f"Error generating speech: {e}")
        traceback.print_exc()
        return None

# Get AI tutor response
def ai_tutor(user_input, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for human, ai in history:
        if human: messages.append({"role": "user", "content": human})
        if ai: messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": user_input})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o", # Or gpt-4, gpt-3.5-turbo
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        print(f"Error getting AI completion: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}\n\nPlease check your API key/quota and connection."

# Generate suggested student responses based on tutor's message
def generate_suggested_responses(tutor_message):
    if not tutor_message or tutor_message.startswith("Error"):
        return []
    messages = [
        {"role": "system", "content": SUGGESTED_RESPONSES_PROMPT},
        {"role": "user", "content": f"Generate 3 student response suggestions for this tutor message:\n\n{tutor_message}"}
    ]
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Cheaper model
            messages=messages,
            max_tokens=150,
            temperature=0.8
        )
        suggestions_text = completion.choices[0].message.content
        suggestions = []
        lines = suggestions_text.strip().split('\n')
        for line in lines:
            line_strip = line.strip()
            if line_strip and line_strip[0].isdigit() and '. ' in line_strip:
                parts = line_strip.split('. ', 1)
                if len(parts) == 2 and parts[1].strip():
                    suggestions.append(parts[1].strip())
        if not suggestions and len(lines) >= 1:
            suggestions = [line.strip() for line in lines if line.strip()][:3]
        return suggestions[:3]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        traceback.print_exc()
        return ["Error generating suggestions."]

# Generate study guide content from conversation history
def generate_study_guide_content(chat_history):
    if not chat_history:
        return "No conversation history to create a study guide from."
    messages = [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT}]
    conversation_text = "### TUTORING SESSION TRANSCRIPT ###\n\n"
    for human, ai in chat_history:
        conversation_text += f"STUDENT: {human}\n\n"
        if ai: conversation_text += f"TUTOR: {ai}\n\n"
    messages.append({"role": "user", "content": f"Please create a concise study guide based on this tutoring conversation:\n\n{conversation_text}"})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=2000,
            temperature=0.5
        )
        study_guide_content = completion.choices[0].message.content
        if not study_guide_content.strip().startswith("#"):
             study_guide_content = "# Study Guide: Tutoring Session\n\n" + study_guide_content
        return study_guide_content
    except Exception as e:
        print(f"Error generating study guide content: {e}")
        traceback.print_exc()
        return f"Error generating study guide: {str(e)}"

# Create a text file version of the study guide
def create_text_file(study_guide_text):
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as temp_file:
            file_path = temp_file.name
            temp_file.write("STEM TUTOR - STUDY GUIDE\n")
            temp_file.write("=" * 30 + "\n\n")
            temp_file.write(study_guide_text)
            temp_file.write("\n\n" + "=" * 30 + "\n")
            temp_file.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        return file_path
    except Exception as e:
        print(f"Error creating text file: {e}")
        traceback.print_exc()
        return None

# Convert study guide text (Markdown-like) to PDF (Improved Error Handling)
def create_pdf(study_guide_text):
    if not FPDF: return None
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", "", 12)
        line_height = 6 # Base line height

        lines = study_guide_text.strip().split("\n")

        # Simplified processing loop with error handling per line
        for line in lines:
            try:
                # Attempt to encode with latin-1, skip if fails (common PDF limitation)
                line.encode('latin-1', 'ignore') # Ignore problematic chars

                line_strip = line.strip()
                if line.startswith("# "):
                    pdf.set_font("Arial", "B", 16)
                    pdf.ln(line_height)
                    pdf.multi_cell(0, line_height, line_strip[2:])
                    pdf.ln(line_height * 0.5)
                elif line.startswith("## "):
                    pdf.set_font("Arial", "B", 14)
                    pdf.ln(line_height * 0.5)
                    pdf.multi_cell(0, line_height, line_strip[3:])
                    pdf.ln(line_height * 0.5)
                elif line.startswith("### "):
                    pdf.set_font("Arial", "BI", 12) # Bold Italic for H3
                    pdf.ln(line_height * 0.5)
                    pdf.multi_cell(0, line_height, line_strip[4:])
                    pdf.ln(line_height * 0.5)
                elif line_strip.startswith(("- ", "* ")):
                    pdf.set_font("Arial", "", 12)
                    pdf.set_x(15)
                    pdf.cell(5, line_height, "•", ln=0)
                    pdf.multi_cell(0, line_height, line_strip[2:])
                    pdf.set_x(10)
                elif line_strip and line_strip[0].isdigit() and ". " in line_strip:
                     parts = line_strip.split(". ", 1)
                     if len(parts) == 2 and parts[0].isdigit():
                         pdf.set_font("Arial", "", 12)
                         pdf.set_x(15)
                         pdf.cell(5, line_height, f"{parts[0]}.", ln=0)
                         pdf.multi_cell(0, line_height, parts[1])
                         pdf.set_x(10)
                     else: # Treat as regular text if not a standard numbered list item
                         pdf.set_font("Arial", "", 12)
                         pdf.multi_cell(0, line_height, line)
                elif line_strip: # Regular text
                    pdf.set_font("Arial", "", 12)
                    pdf.multi_cell(0, line_height, line) # Use original line to preserve leading spaces if any
                else: # Blank line
                    pdf.ln(line_height * 0.5)

            except Exception as line_error:
                 print(f"Skipping PDF line due to error: {line_error} - Line: '{line_strip[:50]}...'")
                 # Optionally add a placeholder in the PDF for skipped lines
                 try:
                     pdf.set_font("Arial", "I", 8)
                     pdf.set_text_color(255, 0, 0)
                     pdf.multi_cell(0, line_height, "[Skipped line due to processing error]")
                     pdf.set_text_color(0, 0, 0)
                 except: pass # Ignore if error placeholder also fails

        # Save the PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            pdf_path = temp_file.name
            pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        print(f"Error creating PDF: {e}")
        traceback.print_exc()
        return None

# Function to save the study guide (tries PDF first, falls back to text)
def save_study_guide(study_guide_text):
    pdf_path = create_pdf(study_guide_text)
    if pdf_path:
        return pdf_path, "Study guide created as PDF!"
    else:
        print("Falling back to text file for study guide.")
        text_path = create_text_file(study_guide_text)
        if text_path:
            return text_path, "Study guide created as Text (PDF failed)."
        else:
            return None, "Error creating study guide file (PDF/Text)."

# --- Gradio Specific Functions (Updated Logic) ---

# Use a suggestion in the text input
def use_suggestion(suggestion):
    # When a suggestion button is clicked, its value is passed as 'suggestion'
    return suggestion # Return it to update the main textbox 'msg'

# Update suggestion buttons based on list of suggestions
def update_suggestions(suggestions):
    updates = []
    for i in range(3): # Max 3 suggestion buttons
        if i < len(suggestions) and suggestions[i] and not suggestions[i].startswith("Error"):
            # Make button visible and set its value
            updates.append(gr.Button(value=suggestions[i], visible=True))
        else:
            # Hide button if no suggestion or if it's an error message
            updates.append(gr.Button(visible=False))
    # Return the update objects for the 3 buttons
    return updates[0], updates[1], updates[2]

# Combined function to process text input, get response, generate audio/suggestions, and update UI
def process_text_and_update(message, chat_history, voice_enabled):
    if not message.strip(): # Ignore empty messages
        s1, s2, s3 = update_suggestions([]) # Clear suggestions if input is cleared
        # Return updates: clear msg, keep history, no audio, clear suggestions
        return "", chat_history, None, s1, s2, s3

    # 1. Get AI response
    bot_message = ai_tutor(message, chat_history)
    # 2. Update history
    chat_history.append((message, bot_message))
    # 3. Generate audio (if enabled and no error)
    audio_output_path = None
    if voice_enabled and not bot_message.startswith("Error"):
        audio_output_path = generate_speech(bot_message)
    # 4. Generate suggestions (based on valid bot message)
    suggestions = generate_suggested_responses(bot_message)
    # 5. Get suggestion button updates
    s1_update, s2_update, s3_update = update_suggestions(suggestions)
    # 6. Return all updates for Gradio outputs
    #    Clear msg, update chatbot, update audio_output, update 3 suggestion buttons
    return "", chat_history, audio_output_path, s1_update, s2_update, s3_update

# Combined function to process audio input, get response, generate audio/suggestions, and update UI
def process_audio_and_update(audio_filepath, chat_history, voice_enabled):
    if audio_filepath is None: # No audio input provided
        s1, s2, s3 = update_suggestions([])
        # Return updates: clear audio_input, keep history, no audio output, clear suggestions
        return None, chat_history, None, s1, s2, s3

    # 1. Transcribe Audio
    user_text = transcribe_audio(audio_filepath)

    # Handle transcription error or empty transcription
    if not user_text or user_text.startswith("Error"):
        error_msg_display = user_text if user_text else "Audio could not be transcribed or was empty."
        chat_history.append(("(Audio input)", error_msg_display)) # Show indication of audio + error
        s1, s2, s3 = update_suggestions([])
        # Return updates: clear audio_input, update chat, no audio output, clear suggestions
        return None, chat_history, None, s1, s2, s3

    # --- Proceed if transcription is successful ---
    # 2. Get AI response
    bot_message = ai_tutor(user_text, chat_history)
    # 3. Update history (use transcribed text as user input)
    chat_history.append((user_text, bot_message))
    # 4. Generate audio (if enabled and no error)
    audio_output_path = None
    if voice_enabled and not bot_message.startswith("Error"):
        audio_output_path = generate_speech(bot_message)
    # 5. Generate suggestions
    suggestions = generate_suggested_responses(bot_message)
    # 6. Get suggestion button updates
    s1_update, s2_update, s3_update = update_suggestions(suggestions)
    # 7. Return all updates
    #    Clear audio_input, update chat, update audio output, update suggestions
    return None, chat_history, audio_output_path, s1_update, s2_update, s3_update


# Function to create and provide the study guide file (updated output handling)
def create_and_download_study_guide(chat_history):
    if not chat_history:
        # Provide feedback via Markdown, hide File component
        return gr.File.update(value=None, visible=False), gr.Markdown.update(value="*Please have a conversation first.*")

    try:
        study_guide_text = generate_study_guide_content(chat_history)
        if study_guide_text.startswith("Error"):
             return gr.File.update(value=None, visible=False), gr.Markdown.update(value=f"*Error generating content: {study_guide_text}*")

        file_path, status_message = save_study_guide(study_guide_text)

        if file_path:
            # Provide file path and make File component visible, update status message
            return gr.File.update(value=file_path, visible=True), gr.Markdown.update(value=f"*{status_message}*")
        else:
            # Saving failed, show error message, hide File component
            return gr.File.update(value=None, visible=False), gr.Markdown.update(value=f"*Error: {status_message}*")

    except Exception as e:
        print(f"Error in create_and_download_study_guide: {e}")
        traceback.print_exc()
        return gr.File.update(value=None, visible=False), gr.Markdown.update(value=f"*An unexpected error occurred creating the guide.*")


# Clear chat and related UI elements (Updated outputs)
def clear_chat():
    # Return updates for: chatbot, audio_output, msg, suggestion buttons (x3), study_guide_download, study_guide_output(Markdown)
    # Need 8 return values to match the 8 outputs in clear_btn.click
    return (
        [],                      # chatbot history
        None,                    # audio_output value
        "",                      # msg textbox
        gr.Button(visible=False),# suggestion1
        gr.Button(visible=False),# suggestion2
        gr.Button(visible=False),# suggestion3
        gr.File(value=None, visible=False), # study_guide_download
        ""                       # study_guide_output (Markdown text)
    )


# Create your original custom theme
def black_orange_theme():
    # Copied directly from your original code
    return gr.Theme(
        primary_hue="orange",
        secondary_hue="gray",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Arial"), "sans-serif"], # Use gr.themes.GoogleFont for clarity if needed, or just list strings
        font_mono=[gr.themes.GoogleFont("Courier New"), "monospace"],
    )


# --- Gradio Interface (Restored to original structure) ---
with gr.Blocks(theme=black_orange_theme()) as demo: # Using your theme
    with gr.Column(scale=1): # This outer column might not be strictly necessary but was in original
        gr.Markdown("# AI STEM Tutor")
        gr.Markdown("Ask any STEM question and get help from your virtual tutor!")

    with gr.Row():
        with gr.Column(scale=3): # Left column for chat
            chatbot = gr.Chatbot(height=500, show_label=False) # Original parameters

            # Text input and buttons (Original structure)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your question here...",
                    show_label=False,
                    container=False,
                    scale=8 # Original scale
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1) # Original scale/variant

            # Response Suggestions Section (Original structure)
            suggestion_container = gr.Markdown("### Suggested Responses") # Original Markdown title

            # Create suggestion buttons that start as invisible (Original names/visibility)
            # We will update these buttons directly now
            suggestion1 = gr.Button("Suggestion 1", visible=False)
            suggestion2 = gr.Button("Suggestion 2", visible=False)
            suggestion3 = gr.Button("Suggestion 3", visible=False)

            # Initial example questions (Original structure/examples)
            examples = gr.Examples(
                examples=[
                    "Can you explain fractions?",
                    "How does photosynthesis work?",
                    "What is Newton's First Law?",
                    "Help me understand basic algebra",
                    "What's the difference between a chemical and physical change?"
                ],
                inputs=msg,
                label="Example Questions" # Original label
            )

        with gr.Column(scale=1): # Right column for controls
            # Voice input/output section (Original structure)
            gr.Markdown("### Voice Interaction")
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath", # Keep this type
                label="Voice Input", # Original label
            )

            audio_output = gr.Audio(
                label="Voice Output", # Original label
                autoplay=True, # Keep autoplay, output path will be None if disabled
                # Removed visible=True, default is usually True
            )

            # Controls (Original structure)
            with gr.Row():
                # Checkbox is now functional
                voice_toggle = gr.Checkbox(label="Voice Output", value=True) # Original label/value
                clear_btn = gr.Button("Clear Chat", variant="stop") # Original label/variant

            # Study Guide Section (Original structure)
            gr.Markdown("### Study Resources")
            study_guide_btn = gr.Button("Create Study Guide", variant="primary") # Original label/variant
            # Markdown for status messages, File for download link
            study_guide_output = gr.Markdown("") # For status messages
            study_guide_download = gr.File(label="Download Study Guide", visible=False) # Start hidden


    # --- Event Listeners (Corrected Logic) ---

    # Text input processing (Enter key)
    msg.submit(
        process_text_and_update,
        inputs=[msg, chatbot, voice_toggle],
        # Outputs match the return values of process_text_and_update
        outputs=[msg, chatbot, audio_output, suggestion1, suggestion2, suggestion3]
    )

    # Text input processing (Send button)
    submit_btn.click(
        process_text_and_update,
        inputs=[msg, chatbot, voice_toggle],
        outputs=[msg, chatbot, audio_output, suggestion1, suggestion2, suggestion3]
    )

    # Audio input processing (when recording finishes)
    audio_input.change(
        process_audio_and_update,
        inputs=[audio_input, chatbot, voice_toggle],
        # Outputs match the return values of process_audio_and_update
        # Note: audio_input is cleared by returning None to it
        outputs=[audio_input, chatbot, audio_output, suggestion1, suggestion2, suggestion3]
    )

    # Connect suggestion buttons to input box (Original logic was correct here)
    # Clicking a suggestion button calls use_suggestion with the button's value,
    # and the return value updates the 'msg' textbox.
    suggestion1.click(fn=use_suggestion, inputs=suggestion1, outputs=msg)
    suggestion2.click(fn=use_suggestion, inputs=suggestion2, outputs=msg)
    suggestion3.click(fn=use_suggestion, inputs=suggestion3, outputs=msg)

    # Study guide generation
    study_guide_btn.click(
        create_and_download_study_guide,
        inputs=[chatbot],
        # Outputs match the return values of create_and_download_study_guide
        outputs=[study_guide_download, study_guide_output] # File component and Markdown status
    )

    # Clear button action (Corrected outputs list)
    clear_btn.click(
        clear_chat,
        inputs=None,
        # List all components that clear_chat returns updates for
        outputs=[chatbot, audio_output, msg, suggestion1, suggestion2, suggestion3, study_guide_download, study_guide_output],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=True) # Keep debug=True for testing