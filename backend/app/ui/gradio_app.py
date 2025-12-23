import os
import tempfile
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime 

from ..core.config import settings
from ..utils.file_parsers import extract_text_from_pdf, extract_text_from_docx, extract_text_from_doc, parse_uploaded_file
from ..core.llm import get_llm
from ..workflows.resume_graph import build_resume_graph
from ..services.workflow_service import stream_resume_workflow
from ..services.resume_service import prepare_resume_state



llm = get_llm()

final_workflow = build_resume_graph()


def _normalize_chat_messages(msgs):
    """Normalize various message formats into a list of 2-item lists [user, assistant].
    Accepts tuples, lists, dicts, and langchain message objects.
    """
    out = []
    for m in msgs:
        # Already a 2-tuple/list
        if isinstance(m, (list, tuple)) and len(m) == 2:
            out.append([m[0], m[1]])
            continue

        # langchain message objects
        if hasattr(m, "content") and hasattr(m, "__class__"):
            if isinstance(m, HumanMessage):
                out.append([m.content, None])
            else:
                out.append([None, m.content])
            continue

        # dicts like {'role': 'assistant', 'content': '...'}
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                out.append([content, None])
            else:
                out.append([None, content])
            continue

        # Fallback: coerce to string as assistant message
        out.append([None, str(m)])

    return out

def parse_uploaded_file(file):
    """Helper to parse uploaded file content"""
    if file is None:
        return ""
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file.name, 'r', encoding='latin-1') as f:
            return f.read()

def run_workflow(job_description_raw: str, resume_file, resume_raw_content: str, resume_format: str):
    """Run the workflow with user inputs"""
    # Combine file and text input for resume
    file_content = parse_uploaded_file(resume_file)
    final_resume_content = file_content + "\n" + resume_raw_content if file_content else resume_raw_content
    
    initial_state = {
        "messages": [HumanMessage(content="Optimize my resume!")],
        "job_description_raw": job_description_raw,
        "resume_raw_content": final_resume_content,
        "resume_format": resume_format,
        "job_description_text": "",
        "resume_plain_text": "",
        "extracted_keywords": [],
        "analysis_report": "",
        "edited_resume_content": "",
        "human_feedback": "proceed",  # Auto-proceed in web version
        "next_agent": "",
        "task_complete": False,
        "current_task": "",
        "old_ats_score": None,
        "new_ats_score": None
    }
    
    # Run workflow
    output_state = None
    thread_id = "user_session"  # In a real app, generate unique session ID
    
    messages = []
    for s in stream_resume_workflow(initial_state, thread_id):
        for key, value in s.items():
            if key != "__end__":
                output_state = value
                # Collect messages for output
                if "messages" in value and len(value["messages"]) > len(initial_state["messages"]):
                    new_msgs = value["messages"][len(initial_state["messages"]):]
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage):
                            messages.append((None, msg.content))
                        elif isinstance(msg, SystemMessage):
                            messages.append((None, f"System: {msg.content}"))
    
    # Process final output
    if output_state:
        final_report = output_state.get("edited_resume_content", "No output generated")
        ats_improvement = f"ATS Score Improved from {output_state.get('old_ats_score', 'N/A')}% to {output_state.get('new_ats_score', 'N/A')}%"
        
        # Add final messages
        messages.append((None, "Workflow completed!"))
        messages.append((None, ats_improvement))
        
        # Save to temporary file for download
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(final_report)
            temp_path = f.name
        
        return messages, final_report, temp_path
    
    return [(None, "Error: No output generated")], "Error: No output generated", None

custom_css = """
/* Dashboard-compact theme */
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #10b981;
    --dark: #0f172a;
    --darker: #020617;
    --light: #f1f5f9;
}

.gradio-container {
    font-family: 'Inter', sans-serif;
    background-color: var(--darker) !important;
    color: var(--light) !important;
    max-width: 1400px !important;
    margin: auto !important;
}

/* Compact header */
.header {padding: 1.2rem; margin-bottom: 1rem;}
.header-title {font-size: 2rem;}

/* Dashboard grid tighter */
.dashboard-grid {grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;}

/* Smaller card padding */
.card {padding: 1rem; border-radius: 10px;}
.metric-value {font-size: 2rem;}

/* Input section smaller */
.input-section textarea, .input-section input {padding: 0.5rem 0.75rem !important;}
.file-upload {padding: 1rem !important;}

/* Tabs smaller height */
.tab-btn {padding: 0.3rem 0.7rem; font-size: 0.9rem;}

/* Reduce big column height */
#jd_input, #resume_input {min-height: 80px !important; max-height: 150px;}

/* Chatbot compact */
.chatbot {height: 240px !important;}

/* Output scroll area smaller */
.output {max-height: 350px;}

.bento-box {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.card {
    flex: 1 1 48%;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 1rem;
}

/* Bento box items */
.bento-box-item {
    flex: 1;
    min-width: 48%;
}

/* Metrics bento grid */
.bento-metrics .dashboard-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
}

.bento-metrics .card {
    padding: 0.8rem;
}

/* Make Upload + Paste equal height */
.file-upload, .bento-box-item textarea {
    height: 100% !important;
}

/* Compact chatbot */
.chatbot {border-radius: 8px; height: 250px !important;}

/* Keep card height balanced and compact */
.card {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1rem;
}

/* Ensure tabs fill available space */
.card .tabitem {
    min-height: 280px; /* keeps Paste & Upload equal height */
}

/* Resume format radio spacing */
.resume-format-radio {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.1);
}


"""

def create_dashboard_metrics(old_score=None, new_score=None, keywords_count=0, improvement=0):
    """Create dashboard-style metrics with improved design"""
    
    old_display = f"{old_score}%" if old_score is not None else "--"
    new_display = f"{new_score}%" if new_score is not None else "--"
    improvement_display = f"+{improvement}%" if improvement > 0 else f"{improvement}%" if improvement < 0 else "0%"
    
    improvement_icon = "📈" if improvement > 0 else "📊" if improvement == 0 else "📉"
    improvement_class = "text-green-400" if improvement > 0 else "text-yellow-400" if improvement == 0 else "text-red-400"
    
    return f"""
    <div class="dashboard-grid">
        <div class="card">
            <div class="metric">
                <div class="metric-label">Original ATS Score</div>
                <div class="metric-value">{old_display}</div>
                <div class="metric-change">Before optimization</div>
            </div>
        </div>
        
        <div class="card">
            <div class="metric">
                <div class="metric-label">Optimized ATS Score</div>
                <div class="metric-value">{new_display}</div>
                <div class="metric-change">After optimization</div>
            </div>
        </div>
        
        <div class="card">
            <div class="metric">
                <div class="metric-label">Improvement</div>
                <div class="metric-value {improvement_class}">{improvement_display}</div>
                <div class="metric-change">{improvement_icon} {'Better' if improvement > 0 else 'Same' if improvement == 0 else 'Review needed'}</div>
            </div>
        </div>
        
        <div class="card">
            <div class="metric">
                <div class="metric-label">Keywords Matched</div>
                <div class="metric-value">{keywords_count}</div>
                <div class="metric-change">From job description</div>
            </div>
        </div>
    </div>
    """

def safe_enhanced_run_workflow(
    job_description_raw,
    resume_file,
    resume_raw_content,
    resume_format
):


    # ---- helper for guaranteed return shape ----
    def _fail(message: str):
        return (
            _normalize_chat_messages([(None, f"❌ {message}")]),
            "",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            create_dashboard_metrics(),
            get_status_message("🔴 Failed")
        )

    # ---- STEP 1: Prepare initial state via service ----
    try:
        initial_state = prepare_resume_state(
            job_description_raw=job_description_raw,
            resume_file=resume_file,
            resume_raw_content=resume_raw_content,
            resume_format=resume_format
        )
    except ValueError as e:
        return _fail(str(e))
    except Exception as e:
        return _fail("Failed to prepare resume data.")

    # ---- STEP 2: Run streaming workflow via service ----
    messages = []
    output_state = None
    thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    messages.append((None, "🚀 Starting resume optimization..."))

    try:
        for event in stream_resume_workflow(initial_state, thread_id):
            for key, state in event.items():
                if key == "__end__":
                    continue

                output_state = state

                for msg in state.get("messages", []):
                    if isinstance(msg, AIMessage):
                        messages.append((None, msg.content))

    except Exception as e:
        return _fail("Workflow execution failed.")

    # ---- STEP 3: Validate final state ----
    if not output_state:
        return _fail("Workflow completed but produced no output.")

    # ---- STEP 4: Extract outputs ----
    optimized_resume = output_state.get("edited_resume_content", "")
    cover_letter = output_state.get("cover_letter_text", "")

    old_score = output_state.get("old_ats_score")
    new_score = output_state.get("new_ats_score")
    keywords = output_state.get("extracted_keywords", [])

    improvement = (
        new_score - old_score
        if old_score is not None and new_score is not None
        else 0
    )

    metrics_html = create_dashboard_metrics(
        old_score=old_score,
        new_score=new_score,
        keywords_count=len(keywords),
        improvement=improvement
    )

    # ---- STEP 5: FINAL GUARANTEED RETURN (7 outputs) ----
    return (
        _normalize_chat_messages(messages),
        optimized_resume or "No optimized resume generated.",
        cover_letter or "Cover letter not generated.",
        gr.update(visible=bool(optimized_resume)),
        gr.update(visible=bool(cover_letter)),
        metrics_html,
        get_status_message("🟢 Optimization completed")
    )


   
def get_status_message(status):
    """Get styled status message"""
    if status.startswith("🟢"):
        return f'<div class="status status-success">{status}</div>'
    elif status.startswith("🟡"):
        return f'<div class="status status-warning">{status}</div>'
    elif status.startswith("🔴"):
        return f'<div class="status status-error">{status}</div>'
    else:
        return f'<div class="status status-info">{status}</div>'

# Enhanced Gradio Interface with Dashboard Design
with gr.Blocks(
    title="AI Resume Optimizer Pro", 
    css=custom_css,
    theme=gr.themes.Default(primary_hue="indigo")
) as demo:
    
    # Header Section
    with gr.Column(elem_classes=["header"]):
        gr.HTML("""
        <h1 class="header-title">AI Resume Optimizer Pro</h1>
        <p class="header-subtitle">Transform your resume with AI-powered ATS optimization and professional enhancement</p>
        """)
    
    # Status indicator
    status_display = gr.HTML(
        get_status_message("🔵 Ready to optimize your resume"),
        elem_id="status-display"
    )
    
    with gr.Row():
        # LEFT - Bento style input card
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.HTML('<div class="card-title"><i class="fas fa-edit"></i> Input Your Details</div>')
            
            # Job Description + Resume Format row
            with gr.Row():
                jd_input = gr.Textbox(
                    label="Job Description / URL",
                    placeholder="Paste JD or URL...",
                    lines=6,
                    elem_classes=["bento-box-item"]
                )

            with gr.Row():
                with gr.Column(scale=1, elem_classes=["card"]):
                    gr.HTML('<div class="card-title"><i class="fas fa-file-alt"></i> Resume Input</div>')
                    
                    with gr.Tabs():
                        with gr.Tab("📤 Upload Resume"):
                            resume_file = gr.File(
                                label="Upload Resume",
                                file_types=[".pdf", ".doc", ".docx", ".txt", ".md", ".tex"],
                                elem_classes=["file-upload"]
                            )
                        with gr.Tab("✏️ Paste Resume"):
                            resume_input = gr.Textbox(
                                label="Paste Resume Content",
                                placeholder="Paste your resume here...",
                                lines=12,
                            )
                    
                    # Resume format picker inside same card
                    format_radio = gr.Radio(
                        choices=["auto", "markdown", "pdf", "docx"],
                        label="Resume Format",
                        value="markdown",
                        elem_classes=["resume-format-radio"]
                    )

                    with gr.Row():
                        submit_btn = gr.Button("Optimize My Resume", variant="primary")
                        clear_btn = gr.Button("Clear All")

        # RIGHT - Bento Stats Grid
        with gr.Column(scale=1.2):
            metrics_dashboard = gr.HTML(
                create_dashboard_metrics(),
                elem_id="metrics-dashboard",
                elem_classes=["bento-metrics"]
            )
            
            # Optimization Progress higher up
            with gr.Column(elem_classes=["card"]):
                gr.HTML('<div class="card-title"><i class="fas fa-tasks"></i> Optimization Progress</div>')
                chatbot = gr.Chatbot(
                    value=[],
                    height=250,
                    show_label=False,
                    elem_classes=["chatbot"]
                )
            
    # Results Section
    with gr.Row():
        # Optimized Resume
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["card", "output"]):
                gr.HTML('<div class="output-title"><i class="fas fa-file-alt"></i> Optimized Resume</div>')
                optimized_resume_output = gr.Markdown(
                    "Your optimized resume will appear here once processing is complete...",
                    elem_classes=["markdown"]
                )
                resume_download = gr.DownloadButton(
                    "Download Resume",
                    visible=False,
                    elem_classes=["btn", "btn-secondary"]
                )
        
        # Cover Letter
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["card", "output"]):
                gr.HTML('<div class="output-title"><i class="fas fa-envelope"></i> Cover Letter</div>')
                cover_letter_output = gr.Markdown(
                    "Your personalized cover letter will be generated automatically...",
                    elem_classes=["markdown"]
                )
                cover_letter_download = gr.DownloadButton(
                    "Download Cover Letter",
                    visible=False,
                    elem_classes=["btn", "btn-secondary"]
                )
    
    # Event Handlers
    def clear_all():
        return (
            "",  # job description
            None,  # resume file
            "",  # resume content
            "markdown",  # format
            [],  # chatbot
            "Your optimized resume will appear here once processing is complete...",  # resume output
            "Your personalized cover letter will be generated automatically...",  # cover letter output
            gr.update(visible=False),  # resume download
            gr.update(visible=False),  # cover letter download
            create_dashboard_metrics(),  # metrics dashboard
            get_status_message("🔵 Ready to optimize your resume")  # status
        )
    
    # Connect event handlers
    submit_btn.click(
        fn=safe_enhanced_run_workflow,
        inputs=[jd_input, resume_file, resume_input, format_radio],
        outputs=[
            chatbot, 
            optimized_resume_output, 
            cover_letter_output,
            resume_download, 
            cover_letter_download,
            metrics_dashboard,
            status_display
        ],
        api_name="optimize_resume"
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[
            jd_input,
            resume_file,
            resume_input,
            format_radio,
            chatbot,
            optimized_resume_output,
            cover_letter_output,
            resume_download,
            cover_letter_download,
            metrics_dashboard,
            status_display
        ]
    )


