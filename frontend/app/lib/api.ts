const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL; 

export async function startOptimizeJob(payload: {
    job_description: string;
    resume_text: string;
    resume_format: string; 
}){

    const res = await fetch(`${API_BASE}/optimize/async`,{
        method: 'POST',
        headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": crypto.randomUUID(),
        },
        body: JSON.stringify(payload),
    });

    if(!res.ok){
        throw new Error("Failed to start optimization job");
    }

    return res.json();
}

export async function getJobStatus(jobId: string){
    const res = await fetch(`${API_BASE}/optimize/status/${jobId}`);

    if(!res.ok){
        throw new Error("Failed to fetch job status");
    }

    return res.json();
}