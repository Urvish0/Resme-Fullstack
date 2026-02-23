const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export interface OptimizeRequest {
    job_description: string;
    resume_text: string;
    resume_format: string;
}
export interface ColdEmailInfo {
    sender_name?: string;
    sender_email?: string;
    recipient_name?: string;
    recipient_email?: string;
}

export interface OptimizeRequestExtended extends OptimizeRequest {
    services?: string[];
    cold_email?: ColdEmailInfo;
    // Flat fields for backend compatibility
    cold_email_sender_name?: string;
    cold_email_sender_email?: string;
    cold_email_recipient_name?: string;
    cold_email_recipient_email?: string;
    cold_email_company_name?: string;
    cold_email_target_role?: string;
}

export interface OptimizeAsyncResponse {
    job_id: string;
    status: string;
    source?: string;
}

export type JobStatusResponse = {
    status: "pending" | "running" | "awaiting_feedback" | "success" | "failed"
    result?: {
      optimized_resume?: string
      cover_letter?: string
      cold_email?: string
      old_ats_score?: number
      new_ats_score?: number
      reflection_report?: string
      resume_json?: Record<string, unknown>
      // HITL fields (present when status === "awaiting_feedback")
      analysis_report?: string
      thread_id?: string
      extracted_keywords?: string[]
    }
    error?: string
}

// Submit async optimization job
export async function optimizeResumeAsync(
    payload: OptimizeRequestExtended,
    idempotencyKey: string
): Promise<OptimizeAsyncResponse> {
    const res = await fetch(`${API_BASE_URL}/optimize/async`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotencyKey,
        },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        throw new Error(`Failed to submit job: ${res.status}`);
    }

    return await res.json();
}

// Get job status
export async function getJobStatus(
    jobId: string
): Promise<JobStatusResponse>{
    const res = await fetch(`${API_BASE_URL}/optimize/status/${jobId}`, {
        cache: "no-store",
    })
    
    if (!res.ok) {
        throw new Error("Failed to fetch job status")
    }
    
    return res.json()
}

// Upload resume file
export async function uploadResumeFile(file: File): Promise<{ filename: string; text: string }> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE_URL}/optimize/upload`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        let message = "Upload failed";
        try {
            const err = await res.json();
            message = err.detail ?? message;
        } catch {}
        throw new Error(message);
    }

    return res.json();
}

// Fetch JD from URL
export async function fetchJDFromUrl(url: string): Promise<{ job_description: string }> {
    const res = await fetch(`${API_BASE_URL}/optimize/jd-from-url`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
    });

    if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to extract JD");
    }

    return res.json();
}

// Download resume as PDF
export async function downloadResumePDF(
    jobId: string,
    template: "modern" | "classic" | "minimalist" = "modern"
): Promise<void> {
    const res = await fetch(
        `${API_BASE_URL}/optimize/pdf/${jobId}?template=${template}`
    );

    if (!res.ok) {
        let detail = "PDF download failed";
        try {
            const err = await res.json();
            detail = err.detail ?? detail;
        } catch {}
        throw new Error(detail);
    }

    // Extract filename from Content-Disposition header
    const disposition = res.headers.get("Content-Disposition");
    const filenameMatch = disposition?.match(/filename="(.+)"/);
    const filename = filenameMatch?.[1] ?? `resume_${template}.pdf`;

    // Trigger browser download
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Phase 6.2 HITL: Submit feedback for a paused workflow
export async function submitHITLFeedback(
    jobId: string,
    feedback: string
): Promise<{ job_id: string; status: string; message: string }> {
    const res = await fetch(`${API_BASE_URL}/optimize/feedback/${jobId}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ feedback }),
    });

    if (!res.ok) {
        let detail = "Failed to submit feedback";
        try {
            const err = await res.json();
            detail = err.detail ?? detail;
        } catch {}
        throw new Error(detail);
    }

    return res.json();
}
