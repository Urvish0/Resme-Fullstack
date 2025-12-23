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

def safe_enhanced_run_workflow(job_description_raw: str, resume_file, resume_raw_content: str, resume_format: str):
    """Enhanced workflow runner with proper error handling and type checking"""
    
    try:
        # Handle None values and ensure strings
        job_description_raw = str(job_description_raw) if job_description_raw is not None else ""
        resume_raw_content = str(resume_raw_content) if resume_raw_content is not None else ""
        
        # Combine file and text input for resume
        # file_content = ""
        # if resume_file is not None:
        #     file_ext = os.path.splitext(resume_file.name)[-1].lower()

        #     if file_ext == ".md" or file_ext == ".txt":
        #         resume_content = resume_file.read().decode("utf-8", errors="ignore")

        #     elif file_ext == ".tex":
        #         resume_content = extract_text_from_latex(resume_file)

        #     elif file_ext == ".pdf":
        #         resume_content = extract_text_from_pdf(resume_file)

        #     elif file_ext == ".docx":
        #         resume_content = extract_text_from_docx(resume_file)

        #     elif file_ext == ".doc":
        #         resume_content = extract_text_from_doc(resume_file)

        #     else:
        #         raise ValueError(f"Unsupported file format: {file_ext}")
        
        file_content = ""
        resume_content = ""
        if resume_file is not None:
            file_ext = os.path.splitext(resume_file.name)[-1].lower()
            path = getattr(resume_file, "name", None)

            try:
                if file_ext in [".md", ".txt"]:
                    # plain text / markdown
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        resume_content = f.read()

                elif file_ext == ".pdf":
                    # pass path to PDF extractor
                    resume_content = extract_text_from_pdf(path)

                elif file_ext == ".docx":
                    resume_content = extract_text_from_docx(path)

                elif file_ext == ".doc":
                    resume_content = extract_text_from_doc(path)

                else:
                    # unsupported -> leave empty and let validation handle it
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        resume_content = f.read()

            except Exception as e:
                resume_content = ""
                print(f"Error reading uploaded file '{getattr(resume_file, 'name', 'unknown')}': {e}")

            # set file_content so later concatenation uses it
            file_content = resume_content or ""
        
        
        # Safely concatenate strings
        final_resume_content = ""
        if file_content and resume_raw_content:
            final_resume_content = file_content + "\n" + resume_raw_content
        elif file_content:
            final_resume_content = file_content
        elif resume_raw_content:
            final_resume_content = resume_raw_content
        
        # Validation
        if not job_description_raw.strip():
            return (
                _normalize_chat_messages([(None, "❌ Please provide a job description.")]), 
                "Please enter a job description to continue.",
                "Please provide job details first.",
                gr.update(visible=False),
                gr.update(visible=False),
                create_dashboard_metrics(),
                get_status_message("🔴 Waiting for job description")
            )
        
        if not final_resume_content.strip():
            return (
                _normalize_chat_messages([(None, "❌ Please provide your resume content.")]), 
                "Please upload or paste your resume content.",
                "Please provide your resume first.", 
                gr.update(visible=False),
                gr.update(visible=False),
                create_dashboard_metrics(),
                get_status_message("🔴 Waiting for resume content")
            )
        
        # Initialize state with safe defaults
        initial_state = {
            "messages": [HumanMessage(content="Optimize my resume!")],
            "job_description_raw": job_description_raw,
            "resume_raw_content": final_resume_content,
            "resume_format": resume_format or "markdown",
            "job_description_text": "",
            "resume_plain_text": "",
            "extracted_keywords": [],
            "analysis_report": "",
            "edited_resume_content": "",
            "human_feedback": "proceed",
            "next_agent": "",
            "task_complete": False,
            "current_task": "",
            "old_ats_score": None,
            "new_ats_score": None,
            "cover_letter_text": "",
            "cover_letter_markdown": ""
        }
        
        # Run workflow
        output_state = None
        thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        messages = []
        step_counter = 0
        current_status = get_status_message("🟡 Processing...")
        
        messages.append((None, "🚀 Starting resume optimization..."))
        
        for s in stream_resume_workflow(initial_state, thread_id):
            for key, value in s.items():
                if key != "__end__":
                    output_state = value
                    step_counter += 1
                    
                    # Update current task
                    current_task = value.get("current_task", "Processing...")
                    current_status = get_status_message(f"🟡 Step {step_counter}: {current_task}")
                    messages.append((None, f"✓ {current_task}"))
                    
                    # Add relevant system messages
                    if "messages" in value and len(value["messages"]) > len(initial_state["messages"]):
                        new_msgs = value["messages"][len(initial_state["messages"]):]
                        for msg in new_msgs:
                            if isinstance(msg, AIMessage):
                                if "Node:" in msg.content and " - " in msg.content:
                                    clean_msg = msg.content.split(" - ")[-1]
                                    if not clean_msg.startswith("Sub-task:"):
                                        messages.append((None, f"📋 {clean_msg}"))
                            initial_state["messages"] = value["messages"]
        
        # Process final output
        if output_state and output_state.get("task_complete"):
            optimized_resume = output_state.get("edited_resume_content", "No resume generated")
            cover_letter = output_state.get("cover_letter_text", "")
            old_score = output_state.get("old_ats_score")
            new_score = output_state.get("new_ats_score")
            keywords_count = len(output_state.get("extracted_keywords", []))
            
            # Calculate improvement
            improvement = 0
            if old_score is not None and new_score is not None:
                improvement = new_score - old_score
            
            # Add success message
            messages.append((None, "🎉 **Optimization Complete!**"))
            current_status = get_status_message("🟢 Optimization completed successfully!")
            
            if improvement > 0:
                messages.append((None, f"📈 Your ATS score improved by **{improvement}%**"))
            elif old_score and new_score:
                messages.append((None, f"📊 ATS analysis complete (Score: {new_score}%)"))
            
            # Save files
            resume_file_path = None
            cover_letter_path = None
            
            try:
                if optimized_resume and optimized_resume != "No resume generated":
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                        f.write(optimized_resume)
                        resume_file_path = f.name
                
                if cover_letter and cover_letter.strip():
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                        f.write(cover_letter)
                        cover_letter_path = f.name
                        
            except Exception as e:
                messages.append((None, f"⚠️ Files generated but download may not work: {str(e)}"))
            
            # Create dashboard metrics
            metrics_html = create_dashboard_metrics(old_score, new_score, keywords_count, improvement)
            
            return (
                _normalize_chat_messages(messages),
                optimized_resume,
                cover_letter or "Cover letter will be generated with your resume optimization.",
                gr.update(visible=bool(resume_file_path)),
                gr.update(visible=bool(cover_letter_path)),
                metrics_html,
                current_status
            )
        
        else:
            current_status = get_status_message("🔴 Optimization failed")
            return (
                _normalize_chat_messages(messages + [(None, "❌ Optimization completed but no output was generated.")]),
                "The optimization process completed but didn't generate output. Please try again with different content.",
                "No cover letter was generated.",
                gr.update(visible=False),
                gr.update(visible=False),
                create_dashboard_metrics(),
                current_status
            )
            
    except Exception as e:
        error_msg = str(e)
        current_status = get_status_message(f"🔴 Error: {error_msg}")
        return (
            _normalize_chat_messages([(None, f"❌ An error occurred: {error_msg}")]),
            f"An error occurred during optimization: {error_msg}",
            "Cover letter generation failed due to an error.",
            gr.update(visible=False),
            gr.update(visible=False),
            create_dashboard_metrics(),
            current_status
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


