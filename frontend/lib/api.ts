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
    status: "pending" | "running" | "success" | "failed"
    result?: {
      optimized_resume?: string
      cover_letter?: string
      cold_email?: string
      old_ats_score?: number
      new_ats_score?: number
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