const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export interface OptimizeRequest {
    job_description: string;
    resume_text: string;
    resume_format: string;
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
      old_ats_score?: number
      new_ats_score?: number
    }
    error?: string
}

// Submit async optimization job
export async function optimizeResumeAsync(
    payload: OptimizeRequest,
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